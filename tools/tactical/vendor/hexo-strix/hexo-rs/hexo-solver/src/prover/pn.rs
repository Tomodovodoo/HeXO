//! Shared proof-number machinery: the `INF` sentinel + saturating pn/dn algebra,
//! a byte-budgeted direct-mapped transposition table with effort-based
//! replacement, and the classic in-memory best-first PN search used as PDS-PN's
//! level-2 evaluator.
//!
//! Proof number `pn` estimates the effort to prove a node; disproof number `dn`
//! is the dual. Shared descendants can be counted by more than one branch.
//! At an OR node `pn = min` over children,
//! `dn = sum`; at an AND node they swap. A proved node is `(0, INF)`, a disproved
//! node `(INF, 0)`, an unexpanded unknown `(1, 1)`.

use super::kernel::{AndEval, KernelCtx, Node, OrEval};
use crate::forcing::{CellSet2, Limits, WinDepthHints};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BinaryHeap;
use std::rc::Rc;

/// The proof/disproof "infinity". Kept well below `u32::MAX` so saturating sums of
/// several children never wrap; any pn/dn `>= INF` is treated as infinite.
pub(crate) const INF: u32 = 1 << 30;

/// Saturating add capped at `INF` (child pn/dn sums at OR-disproof / AND-proof).
#[inline]
pub(crate) fn sat_add(a: u32, b: u32) -> u32 {
    a.saturating_add(b).min(INF)
}

/// df-pn `1 + ε` threshold inflation (Pawlewicz & Lew 2007): search the current
/// best child until its number exceeds the runner-up by a factor `(1 + ε)`,
/// which sharply cuts transposition-table re-expansions vs. the plain `+1` rule.
#[inline]
pub(crate) fn one_plus_eps(second_best: u32, eps: f64) -> u32 {
    if second_best >= INF {
        return INF;
    }
    let inflated = ((second_best as f64 + 1.0) * (1.0 + eps)).ceil();
    if inflated >= INF as f64 { INF } else { inflated as u32 }
}

/// Horizon-aware immediate evaluation. `remaining == Some(n)` means at most
/// `n` attacker turns remain, including an immediate completion at this OR node.
/// `None` retains the ordinary unbounded PDS-PN semantics.
pub(crate) fn eval_child_at(
    k: &mut KernelCtx,
    node: Node,
    remaining: Option<u8>,
) -> (u32, u32, bool) {
    if let Some((_, won, _)) = k.exact_graph(node, remaining) {
        return if won { (0, INF, true) } else { (INF, 0, true) };
    }
    let estimate = match node {
        // OR children are seeded from `or_estimate`, not full move generation:
        // most are never expanded, and generation dominated the search.
        Node::Or { placements } => match k.or_estimate(placements) {
            None if remaining != Some(0) => (0, INF, true),
            None => (INF, 0, true),
            Some(0) => (INF, 0, true),
            Some(_) if remaining.is_some_and(|turns| turns < 2) => (INF, 0, true),
            Some(n) => (1, n.clamp(1, INF - 1), false),
        },
        Node::And if remaining == Some(0) => (INF, 0, true),
        Node::And => match k.and_eval() {
            AndEval::AttackerWin => (0, INF, true),
            AndEval::Loss => (INF, 0, true),
            AndEval::Covers(c) => ((c.len() as u32).clamp(1, INF - 1), 1, false),
        },
    };
    // A nonterminal candidate gets its stamp lookup when selected for expansion.
    // Before discarding a forcing dead end, still allow a local strategy to close
    // it: the stamp model can prove quiet continuations the forcing tree omits.
    if estimate.1 == 0 && let Some((_, won, _)) = k.exact(node, remaining) {
        return if won { (0, INF, true) } else { (INF, 0, true) };
    }
    estimate
}

#[inline]
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// A transposition-table key folding the board Zobrist hash together with the
/// node kind + placements, so an OR and AND node at the same board never collide.
#[inline]
pub(crate) fn node_key(hash: u64, node: Node) -> u64 {
    let (is_or, placements) = node.tag();
    let tag = ((placements as u64) << 1) | (is_or as u64);
    hash ^ mix(tag ^ 0xD1B5_4A32_D192_ED03)
}

/// Horizon-aware TT key. Unbounded callers retain the historical key exactly;
/// bounded callers salt in the remaining attacker turns so facts proved at one
/// threshold can never close a node at a smaller threshold.
#[inline]
pub(crate) fn node_key_at(hash: u64, node: Node, remaining: Option<u8>) -> u64 {
    let key = node_key(hash, node);
    match remaining {
        None => key,
        Some(turns) => key ^ mix(0xA076_1D64_78BD_642F ^ turns as u64),
    }
}

/// Consuming a non-terminal attacker move spends one attacker turn. Unbounded
/// searches remain unbounded.
#[inline]
pub(crate) fn after_attacker(remaining: Option<u8>) -> Option<u8> {
    remaining.map(|turns| turns.saturating_sub(1))
}

#[derive(Clone, Copy)]
struct Slot {
    key: u64,
    pn: u32,
    dn: u32,
    /// Subtree effort (nodes expanded) that produced this entry; the replacement
    /// key — a colliding write keeps whichever entry cost more to compute.
    work: u32,
    occupied: bool,
}

/// A **2-way set-associative**, fixed-byte-budget transposition table (the paper's
/// "TwoBig" scheme). Correctness of the proof searches never depends on the TT (it
/// only accelerates re-visits), so a bounded evicting table is sound. But a 1-way
/// direct-mapped table has a sharp collision cliff: two hot keys that differ only
/// in a dropped index bit map to the same slot and evict each other every visit,
/// and losing a proven leaf freezes the parent's proof number at a non-progress
/// fixed point. Two ways per bucket let such a pair coexist, which is the
/// difference between converging in thousands of nodes and thrashing to the budget
/// on this domain. Replacement within a full bucket: never evict a resolved
/// (`pn==0 || dn==0`) entry for an unresolved one; otherwise evict the smaller
/// `work` (the cheaper-to-recompute subtree).
const WAYS: usize = 2;

pub(crate) struct ProofTt {
    slots: Vec<Slot>,
    /// Bucket-index mask (over `nbuckets`, a power of two).
    mask: usize,
    hits: u64,
    stored: u64,
}

impl ProofTt {
    /// Size to the largest power-of-two bucket count whose `WAYS` slots fit in
    /// `tt_mb` megabytes (minimum 1024 buckets so tiny configs still function).
    pub(crate) fn new(tt_mb: usize) -> ProofTt {
        let budget = tt_mb.max(1) * 1024 * 1024;
        let entry = std::mem::size_of::<Slot>() * WAYS;
        let raw = (budget / entry).max(1024);
        let nbuckets = (raw.next_power_of_two() / 2).max(1024);
        ProofTt {
            slots: vec![Slot { key: 0, pn: 0, dn: 0, work: 0, occupied: false }; nbuckets * WAYS],
            mask: nbuckets - 1,
            hits: 0,
            stored: 0,
        }
    }

    #[inline]
    fn base(&self, key: u64) -> usize {
        ((key as usize) & self.mask) * WAYS
    }

    /// Look up `(pn, dn)` for a node key; `None` on miss (unknown → `(1, 1)`).
    #[inline]
    pub(crate) fn probe(&mut self, key: u64) -> Option<(u32, u32)> {
        let found = self.peek(key);
        self.hits += u64::from(found.is_some());
        found
    }

    /// Reporting reads must not count as search transposition hits.
    #[inline]
    pub(crate) fn peek(&self, key: u64) -> Option<(u32, u32)> {
        let b = self.base(key);
        for w in 0..WAYS {
            let s = self.slots[b + w];
            if s.occupied && s.key == key {
                return Some((s.pn, s.dn));
            }
        }
        None
    }

    /// Store `(pn, dn)` with the subtree `work` into the key's bucket: reuse the
    /// slot already holding this key, else an empty slot, else evict the bucket's
    /// weakest slot (an unresolved slot before a resolved one; then least `work`).
    #[inline]
    pub(crate) fn store(&mut self, key: u64, pn: u32, dn: u32, work: u32) {
        let b = self.base(key);
        // Same-key or empty slot: write there.
        let mut victim = b;
        let mut victim_rank = u64::MAX; // higher = more worth evicting
        for w in 0..WAYS {
            let s = self.slots[b + w];
            if !s.occupied || s.key == key {
                if s.occupied && (s.pn == 0 || s.dn == 0) && pn != 0 && dn != 0 {
                    return;
                }
                self.slots[b + w] = Slot { key, pn, dn, work, occupied: true };
                self.stored += 1;
                return;
            }
            // Eviction rank: unresolved entries (rank has high bit clear) are
            // evicted before resolved ones; within a class, smaller work first.
            let resolved = (s.pn == 0 || s.dn == 0) as u64;
            let rank = ((1 - resolved) << 32) | (u32::MAX - s.work.min(u32::MAX)) as u64;
            if rank > victim_rank || victim_rank == u64::MAX {
                victim_rank = rank;
                victim = b + w;
            }
        }
        // Only displace the chosen victim if the incoming entry is at least as
        // worth keeping: resolved always displaces unresolved; else compare work.
        let v = self.slots[victim];
        let incoming_resolved = pn == 0 || dn == 0;
        let victim_resolved = v.pn == 0 || v.dn == 0;
        let replace = match (incoming_resolved, victim_resolved) {
            (true, false) => true,
            (false, true) => false,
            _ => work >= v.work,
        };
        if replace {
            self.slots[victim] = Slot { key, pn, dn, work, occupied: true };
            self.stored += 1;
        }
    }

    pub(crate) fn hits(&self) -> u64 {
        self.hits
    }

    pub(crate) fn has_entries(&self) -> bool {
        self.stored > 0
    }

    /// Rehash into the requested byte budget. Shrinking may evict entries;
    /// the same resolved-first/work replacement rule applies as during search.
    pub(crate) fn resize(&mut self, tt_mb: usize) {
        let mut next = Self::new(tt_mb);
        if next.slots.len() == self.slots.len() {
            return;
        }
        for slot in &self.slots {
            if slot.occupied {
                next.store(slot.key, slot.pn, slot.dn, slot.work);
            }
        }
        next.hits = self.hits;
        next.stored = self.stored;
        *self = next;
    }

    pub(crate) fn bytes(&self) -> u64 {
        (self.slots.len() * std::mem::size_of::<Slot>()) as u64
    }
}

// --------------------------------------------------------------------------
// Classic in-memory best-first PN search (PDS-PN level 2 / PN²-style evaluator).
// --------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
struct PnEdge {
    child: usize,
    mv: CellSet2,
}

/// A node of the in-memory PN graph. Children are lazily generated on first
/// expansion; terminals carry `(0, INF)` / `(INF, 0)`.
struct PnNode {
    node: Node,
    /// Remaining attacker turns at this exact node (`None` = unbounded).
    remaining: Option<u8>,
    pn: u32,
    dn: u32,
    expanded: bool,
    /// Indices into the arena; empty until expanded.
    children: Vec<PnEdge>,
    parents: Vec<usize>,
    depth: usize,
    pending: bool,
    terminal: bool,
    /// The node's `node_key` (board hash folded with kind/placements), captured
    /// when the board was at this node's position. Lets a proved level-2 subtree
    /// export its proven nodes into the level-1 proven-set for PV reconstruction.
    key: u64,
    work: u32,
}

/// Bounded best-first PN over a graph of turn-context keys. Ordinary calls start
/// fresh; resident slice mode can continue an unresolved graph. Expanded
/// estimates and settled subproofs enter level 1's bounded table.
pub(crate) struct PnSearch {
    arena: Vec<PnNode>,
    index: FxHashMap<u64, usize>,
    backups: BinaryHeap<(usize, usize)>,
    backup_start: Option<usize>,
    max_nodes: u64,
    expansions: u64,
    /// Certificate-derived win-depth hints (guided probes only). A hit closes a
    /// node at its current horizon without expansion, mirroring the level-1
    /// `Dfpn::hint_val` check.
    hints: Option<Rc<WinDepthHints>>,
    /// Wall-clock deadline and cancel flag, checked before expansion and while
    /// propagating results through transposed parents.
    limits: Limits,
}

impl PnSearch {
    pub(crate) fn new(max_nodes: u64) -> PnSearch {
        PnSearch {
            arena: Vec::new(),
            index: FxHashMap::default(),
            backups: BinaryHeap::new(),
            backup_start: None,
            max_nodes: max_nodes.max(1),
            expansions: 0,
            hints: None,
            limits: Limits::default(),
        }
    }

    /// Stop expanding once `limits` expire; the root keeps its unresolved numbers.
    pub(crate) fn set_limits(&mut self, limits: Limits) {
        self.limits = limits;
    }

    /// Attach certificate-derived win-depth hints for a guided probe.
    pub(crate) fn set_hints(&mut self, hints: Option<Rc<WinDepthHints>>) {
        self.hints = hints;
    }

    /// Override the per-call node cap (used by adaptive leaf-budget schemes).
    pub(crate) fn set_max_nodes(&mut self, max_nodes: u64) {
        self.max_nodes = max_nodes.max(1);
    }

    /// Number of nodes expanded by the most recent [`Self::search`] call.
    pub(crate) fn expansions(&self) -> u64 {
        self.expansions
    }

    pub(crate) fn retained_nodes(&self) -> usize { self.arena.len() }

    /// Capacity accounting for the bounded resident cache, excluding allocator
    /// metadata. Hash storage is conservatively counted at twice its capacity.
    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.arena.capacity() * std::mem::size_of::<PnNode>()
            + self.index.capacity() * 2 * (std::mem::size_of::<(u64, usize)>() + 1)
            + self.backups.capacity() * std::mem::size_of::<(usize, usize)>()
            + self.arena.iter().map(|n| n.children.capacity() * std::mem::size_of::<PnEdge>()
                + n.parents.capacity() * std::mem::size_of::<usize>()).sum::<usize>()
    }

    /// Run PN search from `root_node` (board in `k` must be at that position).
    /// Returns `(pn, dn)` of the root when it is (dis)proved or the node cap is
    /// hit. The board is restored to the root position on return.
    pub(crate) fn search(&mut self, k: &mut KernelCtx, root_node: Node) -> (u32, u32) {
        self.search_at(k, root_node, None, None)
    }

    /// Horizon-aware PN² search used by depth-bounded PDS-PN.
    pub(crate) fn search_at(
        &mut self,
        k: &mut KernelCtx,
        root_node: Node,
        remaining: Option<u8>,
        tt: Option<&mut ProofTt>,
    ) -> (u32, u32) {
        self.arena.clear();
        self.index.clear();
        self.backups.clear();
        self.backup_start = None;
        self.arena.push(PnNode {
            node: root_node,
            remaining,
            pn: 1,
            dn: 1,
            expanded: false,
            children: Vec::new(),
            parents: Vec::new(),
            depth: 0,
            pending: false,
            terminal: false,
            key: node_key_at(k.hash(), root_node, remaining),
            work: 0,
        });
        self.index.insert(self.arena[0].key, 0);
        self.continue_at(k, root_node, remaining, tt)
    }

    /// Continue a retained frontier with fresh limits and a fresh work meter.
    /// Every deferred backup must finish before selecting another leaf.
    pub(crate) fn continue_at(
        &mut self,
        k: &mut KernelCtx,
        root_node: Node,
        remaining: Option<u8>,
        mut tt: Option<&mut ProofTt>,
    ) -> (u32, u32) {
        assert_eq!(self.arena[0].key, node_key_at(k.hash(), root_node, remaining));
        self.expansions = 0;
        if !self.drain_backups(&mut tt) {
            return (self.arena[0].pn, self.arena[0].dn);
        }
        let mut applied: Vec<PnEdge> = Vec::new();
        let mut selected = Vec::new();
        loop {
            let root = &self.arena[0];
            if root.pn == 0 || root.dn == 0 {
                break;
            }
            if self.expansions >= self.max_nodes || self.limits.charge() || self.limits.expired() {
                break;
            }
            // Select in the tree first. Consecutive leaves often share a long
            // prefix; leave those stones and their incremental windows in place.
            selected.clear();
            let mut cur = 0usize;
            loop {
                let n = &self.arena[cur];
                if !n.expanded || n.children.is_empty() {
                    break;
                }
                // OR: follow min-pn child; AND: follow min-dn child.
                let is_or = n.node.is_or();
                let mut best = n.children[0];
                let mut best_val = u32::MAX;
                for &c in &n.children {
                    let cc = &self.arena[c.child];
                    let v = if is_or { cc.pn } else { cc.dn };
                    if v < best_val {
                        best_val = v;
                        best = c;
                    }
                }
                cur = best.child;
                selected.push(best);
            }
            let common = applied.iter().zip(&selected).take_while(|(a, b)| a == b).count();
            for edge in applied[common..].iter().rev() {
                k.unplace(&edge.mv);
            }
            for edge in &selected[common..] {
                if !self.arena[edge.child].node.is_or() {
                    k.place_attacker(&edge.mv);
                } else {
                    k.place_defender(&edge.mv);
                }
            }
            std::mem::swap(&mut applied, &mut selected);
            // Expand `cur`.
            self.expand(k, cur, &mut tt);
            self.expansions += 1;
            // Back up the proof numbers; restore only the differing board suffix
            // on the next iteration, including if an ancestor just settled.
            if !self.backup(cur, &mut tt) {
                break;
            }
        }
        if self.arena[0].pn != 0 && !applied.is_empty() {
            k.offer_neural(self.arena[applied.last().unwrap().child].node, 1);
        }
        for edge in applied.iter().rev() {
            k.unplace(&edge.mv);
        }
        if let Some(t) = tt {
            for (i, n) in self.arena.iter().enumerate() {
                if i % 256 == 0 && self.limits.expired() { break; }
                if n.expanded {
                    t.store(n.key, n.pn, n.dn, n.work);
                }
            }
        }
        let (pn, dn) = (self.arena[0].pn, self.arena[0].dn);
        (pn, dn)
    }

    /// Export every proved (`pn == 0`) node's key from the current graph into
    /// `out`, including when the root remains unresolved. The level-1 driver's
    /// proven-set gains the level-2 subtree's proven positions — which is what lets
    /// PV reconstruction descend through a node that PDS-PN closed via level-2 PN
    /// (whose tree is otherwise discarded). Sound: every such node is a genuine
    /// forced win under the shared kernel rules.
    pub(crate) fn collect_proven(&self, out: &mut FxHashSet<u64>) {
        // Mandatory cleanup even after cancellation: backup may already have
        // cached positive facts. Retaining one without its witness would make
        // later resident queries unable to reconstruct the reused proof.
        for n in &self.arena {
            if n.pn == 0 {
                out.insert(n.key);
            }
        }
    }

    /// Apply the terminal test / child generation for node `cur` (board is at its
    /// position). Children are **immediately evaluated** with the "1 and n"
    /// initialization — each child is classified on generation, so terminals get
    /// `(0, INF)` / `(INF, 0)` at once and internal nodes are seeded from their
    /// branching factor rather than the shapeless `(1, 1)`. This is what makes a
    /// bounded PN search useful within `max_nodes`.
    fn expand(&mut self, k: &mut KernelCtx, cur: usize, tt: &mut Option<&mut ProofTt>) {
        let node = self.arena[cur].node;
        let remaining = self.arena[cur].remaining;
        self.arena[cur].expanded = true;
        if let Some((pn, dn)) = tt.as_deref_mut().and_then(|t| t.probe(self.arena[cur].key))
            && (pn == 0 || dn == 0)
        {
            return self.set_terminal(cur, pn, dn);
        }
        if let Some((_, won, _)) = k.exact(node, remaining) {
            return if won { self.set_terminal(cur, 0, INF) } else { self.set_terminal(cur, INF, 0) };
        }
        // (child_node, move) list, and whether children are reached by an attacker
        // move (OR parent) or a defender cover (AND parent).
        let (child_node, moves, parent_is_or): (Node, Vec<CellSet2>, bool) = match node {
            Node::Or { placements } => match k.or_eval(placements) {
                OrEval::WinNow if remaining != Some(0) => return self.set_terminal(cur, 0, INF),
                OrEval::WinNow => return self.set_terminal(cur, INF, 0),
                OrEval::Loss => return self.set_terminal(cur, INF, 0),
                OrEval::Moves(_) if remaining.is_some_and(|turns| turns < 2) => {
                    return self.set_terminal(cur, INF, 0);
                }
                OrEval::Moves(mvs) => (Node::And, mvs.to_vec(), true),
            },
            Node::And if remaining == Some(0) => return self.set_terminal(cur, INF, 0),
            Node::And => match k.and_eval() {
                AndEval::AttackerWin => return self.set_terminal(cur, 0, INF),
                AndEval::Loss => return self.set_terminal(cur, INF, 0),
                AndEval::Covers(covers) => (Node::Or { placements: 2 }, covers, false),
            },
        };
        let child_remaining = if parent_is_or { after_attacker(remaining) } else { remaining };
        for mv in moves {
            // Certificate-derived hint cutoff before the ordinary immediate
            // evaluation: a guided probe closes nodes the certificate already
            // resolved at this horizon without expanding them.
            let hash = k.child_hash(&mv, parent_is_or);
            let key = node_key_at(hash, child_node, child_remaining);
            let idx = if let Some(&idx) = self.index.get(&key) {
                debug_assert_eq!(self.arena[idx].depth, self.arena[cur].depth+1);
                idx
            } else {
                let mut evaluate = || {
                    if parent_is_or { k.place_attacker(&mv); } else { k.place_defender(&mv); }
                    let value = eval_child_at(k, child_node, child_remaining);
                    k.unplace(&mv);
                    value
                };
                let cached = tt.as_deref_mut().and_then(|t| t.probe(key));
                let (pn, dn, terminal) = if let Some((pn, dn)) = cached {
                    (pn, dn, pn == 0 || dn == 0)
                } else if let (Some(hints), Some(turns)) =
                    (&self.hints, child_remaining)
                {
                    let (is_or, placements) = child_node.tag();
                    if hints.proves_within(hash, is_or, placements, turns) {
                        (0, INF, true)
                    } else if hints.disproves_within(hash, is_or, placements, turns) {
                        (INF, 0, true)
                    } else {
                        evaluate()
                    }
                } else {
                    evaluate()
                };
                let idx = self.arena.len();
                self.arena.push(PnNode {
                    node: child_node,
                    remaining: child_remaining,
                    pn,
                    dn,
                    expanded: false,
                    children: Vec::new(),
                    parents: Vec::new(),
                    depth: self.arena[cur].depth+1,
                    pending: false,
                    terminal,
                    key,
                    work: 0,
                });
                self.index.insert(key, idx);
                idx
            };
            self.arena[idx].parents.push(cur);
            self.arena[cur].children.push(PnEdge { child: idx, mv });
            let (pn, dn) = (self.arena[idx].pn, self.arena[idx].dn);
            if (parent_is_or && pn == 0) || (!parent_is_or && dn == 0) {
                break;
            }
        }
        self.recompute(cur);
    }

    fn set_terminal(&mut self, cur: usize, pn: u32, dn: u32) {
        let n = &mut self.arena[cur];
        n.pn = pn;
        n.dn = dn;
        n.terminal = true;
    }

    /// Recompute a node's pn/dn from its children (OR: pn=min, dn=sum; AND swap).
    fn recompute(&mut self, cur: usize) -> bool {
        if self.arena[cur].terminal {
            return true;
        }
        let is_or = self.arena[cur].node.is_or();
        let children = &self.arena[cur].children;
        if children.is_empty() {
            return true;
        }
        let (mut min_pn, mut sum_pn) = (INF, 0u32);
        let (mut min_dn, mut sum_dn) = (INF, 0u32);
        for (i, &c) in children.iter().enumerate() {
            if i % 256 == 0 && self.limits.expired() { return false; }
            let cc = &self.arena[c.child];
            min_pn = min_pn.min(cc.pn);
            sum_pn = sat_add(sum_pn, cc.pn);
            min_dn = min_dn.min(cc.dn);
            sum_dn = sat_add(sum_dn, cc.dn);
        }
        let n = &mut self.arena[cur];
        if is_or {
            n.pn = min_pn;
            n.dn = sum_dn;
        } else {
            n.pn = sum_pn;
            n.dn = min_dn;
        }
        true
    }

    /// Every transposed parent sees a changed child. Stones only accumulate, so
    /// decreasing depth visits a parent after all affected children, once.
    fn backup(&mut self, start: usize, tt: &mut Option<&mut ProofTt>) -> bool {
        debug_assert!(self.backups.is_empty());
        self.backups.clear();
        self.backups.push((self.arena[start].depth, start));
        self.arena[start].pending = true;
        self.backup_start = Some(start);
        self.drain_backups(tt)
    }

    fn drain_backups(&mut self, tt: &mut Option<&mut ProofTt>) -> bool {
        while let Some(&(_, cur)) = self.backups.peek() {
            if self.limits.expired() { return false; }
            self.backups.pop();
            self.arena[cur].pending = false;
            let before = (self.arena[cur].pn, self.arena[cur].dn);
            if !self.recompute(cur) {
                self.backups.push((self.arena[cur].depth, cur));
                self.arena[cur].pending = true;
                return false;
            }
            self.arena[cur].work = self.arena[cur].work.saturating_add(1);
            let n = &self.arena[cur];
            if (n.pn == 0 || n.dn == 0) && let Some(t) = tt.as_deref_mut() {
                t.store(n.key, n.pn, n.dn, n.work);
            }
            if Some(cur) != self.backup_start && before == (n.pn, n.dn) { continue; }
            for i in 0..self.arena[cur].parents.len() {
                if i % 256 == 0 && self.limits.expired() {
                    // Recompute has committed, but not every parent was queued.
                    // Force propagation when this node is retried, even unchanged.
                    self.backup_start = Some(cur);
                    self.backups.push((self.arena[cur].depth, cur));
                    self.arena[cur].pending = true;
                    return false;
                }
                let p = self.arena[cur].parents[i];
                if !self.arena[p].pending {
                    self.arena[p].pending = true;
                    self.backups.push((self.arena[p].depth, p));
                }
            }
        }
        self.backup_start = None;
        true
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::types::Player;

    #[test]
    fn shared_search_restores_the_board_after_budget_cuts() {
        let history = [(0,0),(0,8),(2,8),(1,0),(2,0),(4,8),(6,8)];
        let stones: Vec<_> = history.into_iter().enumerate().map(|(i, p)|
            (p, if (i+1)/2%2 == 0 { Player::P1 } else { Player::P2 })).collect();
        let mut k = KernelCtx::new_wide(&stones, Player::P1, 6, 8, true).unwrap();
        let original = k.canonical_stones();
        let hash = k.hash();
        let mv = CellSet2::two((8,8), (9,8));
        for attacker in [false, true] {
            let expected = k.child_hash(&mv, attacker);
            if attacker { k.place_attacker(&mv); } else { k.place_defender(&mv); }
            assert_eq!(k.hash(), expected);
            k.unplace(&mv);
            assert_eq!(k.hash(), hash);
        }
        let mut shared = false;
        for limit in [1, 7, 64, 256, 1000] {
            let mut search = PnSearch::new(limit);
            let mut table = ProofTt::new(1);
            search.search_at(&mut k, Node::Or { placements: 2 }, None, Some(&mut table));
            assert_eq!(k.canonical_stones(), original, "budget {limit}");
            assert_eq!(k.hash(), hash);
            assert!(search.expansions() <= limit);
            shared |= search.arena.iter().any(|n| n.parents.len() > 1);
        }
        assert!(shared, "real turn transpositions are shared");
    }

    #[test]
    fn stale_estimates_cannot_erase_settled_table_entries() {
        let mut table = ProofTt::new(1);
        for (key, pn, dn) in [(7, 0, INF), (11, INF, 0)] {
            table.store(key, pn, dn, 1);
            table.store(key, 1, 1, u32::MAX);
            assert_eq!(table.peek(key), Some((pn, dn)));
        }
    }

    #[test]
    fn cancelled_parent_propagation_leaves_root_unresolved() {
        use std::sync::{Arc, atomic::AtomicBool};
        use std::time::Instant;
        for limits in [
            Limits { cancel: Some(Arc::new(AtomicBool::new(true))), ..Limits::default() },
            Limits { deadline: Some(Instant::now()), ..Limits::default() },
        ] {
            let mut search = PnSearch::new(1000);
            for id in 0..4 {
                search.arena.push(PnNode {
                    node: if id == 1 || id == 2 { Node::And } else { Node::Or { placements: 2 } },
                    remaining: None, pn: 1, dn: 1, expanded: true,
                    children: Vec::new(), parents: Vec::new(), depth: if id == 3 { 2 } else { usize::from(id != 0) },
                    pending: false, terminal: id == 3, key: id as u64, work: 0,
                });
            }
            for (parent, child) in [(0,1), (0,2), (1,3), (2,3)] {
                search.arena[parent].children.push(PnEdge { child, mv: CellSet2::one((child as i32,0)) });
                search.arena[child].parents.push(parent);
            }
            search.set_terminal(3, 0, INF);
            search.set_limits(limits);
            let mut table = ProofTt::new(1);
            assert!(!search.backup(3, &mut Some(&mut table)));
            assert_eq!((search.arena[0].pn, search.arena[0].dn), (1,1));
            assert!(search.arena.iter().all(|n| n.work == 0));
            search.set_limits(Limits::default());
            assert!(search.drain_backups(&mut Some(&mut table)));
            assert_eq!((search.arena[0].pn, search.arena[0].dn), (0,INF));
            search.set_limits(Limits { cancel: Some(Arc::new(AtomicBool::new(true))), ..Limits::default() });
            let mut witnesses = FxHashSet::default();
            search.collect_proven(&mut witnesses);
            for key in 0..4 {
                assert_eq!(table.peek(key), Some((0,INF)));
                assert!(witnesses.contains(&key), "cached proof {key} lost its witness after cancellation");
            }
        }
    }

    #[test]
    fn a_retained_frontier_advances_with_a_new_meter_and_restores_the_board() {
        let history = [(0,0),(0,8),(2,8),(1,0),(2,0),(4,8),(6,8)];
        let stones: Vec<_> = history.into_iter().enumerate().map(|(i, p)|
            (p, if (i+1)/2%2 == 0 { Player::P1 } else { Player::P2 })).collect();
        let mut k = KernelCtx::new_wide(&stones, Player::P1, 6, 8, true).unwrap();
        let original = k.canonical_stones();
        let root = Node::Or { placements: 2 };
        let mut search = PnSearch::new(1);
        let mut table = ProofTt::new(1);
        search.search_at(&mut k, root, None, Some(&mut table));
        let mut expanded = search.arena.iter().filter(|n| n.expanded).count();
        assert!(search.retained_nodes() > 1);
        for _ in 0..8 {
            if search.arena[0].pn == 0 || search.arena[0].dn == 0 { break; }
            let meter = crate::forcing::Meter::new(1);
            search.set_limits(Limits { meter: Some(meter.clone()), ..Default::default() });
            search.continue_at(&mut k, root, None, Some(&mut table));
            assert_eq!(search.expansions(), 1, "the new slice advances an unexpanded leaf");
            assert_eq!(meter.spent(), 1, "old expansions do not charge fresh work");
            let next = search.arena.iter().filter(|n| n.expanded).count();
            assert!(next > expanded);
            expanded = next;
            assert_eq!(k.canonical_stones(), original);
        }
        assert!(search.retained_bytes() >= search.retained_nodes() * std::mem::size_of::<PnNode>());
    }

    #[test]
    fn resuming_partly_queued_parents_reaches_every_shared_ancestor() {
        let mut search = PnSearch::new(1);
        for id in 0..=701 {
            search.arena.push(PnNode { node: if id == 0 || id == 701 { Node::Or { placements: 2 } } else { Node::And },
                remaining: None,pn:1,dn:1,expanded:true,children:Vec::new(),parents:Vec::new(),
                depth:if id == 701 {2} else {usize::from(id != 0)},pending:false,terminal:id == 701,key:id as u64,work:0 });
        }
        for parent in 1..=700 {
            search.arena[0].children.push(PnEdge { child:parent,mv:CellSet2::one((parent as i32,0)) });
            search.arena[parent].parents.push(0);
            search.arena[parent].children.push(PnEdge { child:701,mv:CellSet2::one((0,1)) });
            search.arena[701].parents.push(parent);
        }
        search.set_terminal(701,0,INF);
        // State at a cancellation after committing the child and queuing only
        // the first parent block. Retrying must not skip unchanged child values.
        search.backup_start=Some(701);
        search.arena[701].pending=true;
        search.backups.push((2,701));
        for parent in 1..=256 {
            search.arena[parent].pending=true;
            search.backups.push((1,parent));
        }
        let mut table=ProofTt::new(1);
        assert!(search.drain_backups(&mut Some(&mut table)));
        assert!(search.backups.is_empty());
        for id in 0..=701 {
            assert_eq!((search.arena[id].pn,search.arena[id].dn),(0,INF));
            assert!(!search.arena[id].pending);
            assert_eq!(table.peek(id as u64),Some((0,INF)));
        }
    }
}

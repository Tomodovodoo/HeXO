//! Compact all-defense certificates for PDS-PN wins.
//!
//! The search contributes only the set of positions it proved (`pn == 0`).
//! Reconstruction walks the shared forcing kernel from a fresh root: an OR node
//! records every attacker action already proved and retained by the normal search,
//! ranked by worst-case attacker turns, while an AND node records every exact
//! non-losing cover. The verifier is search-independent: it replays the board and
//! recomputes terminal status, attacker moves, depths, ranks, and the complete
//! cover set. It never trusts proof numbers, TT entries, or branch counts from the
//! search.

use super::io::Position;
use super::kernel::{AndEval, KernelCtx, Node, OrEval};
use super::pn::{after_attacker, node_key_at};
use crate::forcing::{CellSet2, WinDepthHints};
use hexo_engine::types::{Coord, Player};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

pub const CERTIFICATE_VERSION: u32 = 1;
const DEFAULT_VERIFY_NODE_LIMIT: usize = 5_000_000;

/// One action edge in the proof DAG.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofResponse {
    pub action: Vec<Coord>,
    pub child: u32,
}

/// A compact proof node. Node IDs are indices into [`ProofCertificate::nodes`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProofNode {
    /// The attacker completes a six within the placements remaining this turn.
    ImmediateWin { action: Vec<Coord> },
    /// Proved forcing attacker actions (OR choices), shortest verified bound
    /// first. `action`/`child` remain the primary choice for compatibility with
    /// version-1 certificates emitted before alternatives were retained.
    AttackerMove {
        action: Vec<Coord>,
        child: u32,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        alternatives: Vec<ProofResponse>,
    },
    /// Every exact non-losing two-cell defender cover (AND branches).
    DefenderReplies { responses: Vec<ProofResponse> },
    /// The attacker has a threat family with transversal number at least three;
    /// no two defender placements can stop the completion next attacker turn.
    Unstoppable {
        /// Inclusion-minimal next-turn completions whose minimum cover needs at
        /// least three placements. Empty only in legacy version-1 certificates.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        threats: Vec<Vec<Coord>>,
    },
}

/// Self-describing proof graph. The root position is supplied separately so a
/// report and a browser download can share the same certificate representation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofCertificate {
    pub version: u32,
    pub width: String,
    pub root: u32,
    pub nodes: Vec<ProofNode>,
}

impl ProofCertificate {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("proof certificate serializes")
    }
}

/// Measurements computed by the independent replay verifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofSummary {
    pub dag_nodes: u64,
    pub proof_edges: u64,
    /// Worst-case attacker turns after choosing the shortest ranked alternative
    /// retained at every attacker node. This is a bound for the proof DAG, not a
    /// claim that the mate is globally shortest.
    pub max_attacker_turns: u32,
}

/// An attacker-to-move state replayed from a verified certificate. The guided
/// optimizer may independently tighten `certified_depth`; the certificate itself
/// remains unchanged.
pub(crate) struct GuidedOrNode {
    pub id: u32,
    pub hash: u64,
    pub stones: Vec<(Coord, Player)>,
    pub placements: u8,
    pub certified_depth: u32,
    pub primary: bool,
}

/// Reconstruct an all-defense DAG from PDS-PN's monotone set of proved node keys.
pub(crate) fn reconstruct(
    pos: &Position,
    wide: bool,
    proven: &mut FxHashSet<u64>,
) -> Result<ProofCertificate, String> {
    reconstruct_at(pos, wide, proven, None, None)
}

/// Reconstruct the strategy proved at one exact attacker-turn horizon. Unlike
/// an ordinary certificate, every lookup includes the remaining-turn salt, so
/// the resulting DAG contains the depth-bounded strategy rather than whichever
/// unbounded proof happened to establish the initial upper bound.
pub(crate) fn reconstruct_bounded(
    pos: &Position,
    wide: bool,
    proven: &mut FxHashSet<u64>,
    remaining: u8,
) -> Result<ProofCertificate, String> {
    reconstruct_at(pos, wide, proven, Some(remaining), None)
}

/// Guided variant: nodes the search closed via certificate win-depth hints have
/// no children in the proven set, so the reconstruction consults `hints` to
/// close their children at the probe's horizon (monotone positive facts only).
pub(crate) fn reconstruct_bounded_guided(
    pos: &Position,
    wide: bool,
    proven: &mut FxHashSet<u64>,
    remaining: u8,
    hints: &WinDepthHints,
) -> Result<ProofCertificate, String> {
    reconstruct_at(pos, wide, proven, Some(remaining), Some(hints))
}

fn reconstruct_at(
    pos: &Position,
    wide: bool,
    proven: &mut FxHashSet<u64>,
    remaining: Option<u8>,
    hints: Option<&WinDepthHints>,
) -> Result<ProofCertificate, String> {
    let k = KernelCtx::new_wide(
        &pos.stones,
        pos.attacker,
        pos.config.win_length,
        pos.config.placement_radius,
        wide,
    )
    .ok_or("could not rebuild the proof kernel")?;
    let root_node = Node::Or {
        placements: pos.placements_remaining,
    };
    let root_key = node_key_at(k.hash(), root_node, remaining);
    if !proven.contains(&root_key) {
        return Err("PDS-PN proved the root but did not retain its proof key".to_string());
    }
    let mut builder = ProofBuilder {
        k,
        proven,
        hints,
        memo: FxHashMap::default(),
        nodes: Vec::new(),
        depths: Vec::new(),
    };
    let root = builder.emit(root_node, remaining)?;
    Ok(ProofCertificate {
        version: CERTIFICATE_VERSION,
        width: if wide { "wide" } else { "tight" }.to_string(),
        root,
        nodes: builder.nodes,
    })
}

struct ProofBuilder<'a> {
    k: KernelCtx,
    proven: &'a mut FxHashSet<u64>,
    hints: Option<&'a WinDepthHints>,
    memo: FxHashMap<u64, u32>,
    nodes: Vec<ProofNode>,
    depths: Vec<u32>,
}

impl ProofBuilder<'_> {
    /// True when the certificate's win-depth hints prove the node (whose position
    /// the board currently reflects) at the given remaining-turn horizon. Only
    /// positive facts are consulted — a miss means "search it", never "close it".
    #[inline]
    fn hint_proven(&self, node: Node, remaining: Option<u8>) -> bool {
        let (Some(hints), Some(turns)) = (self.hints, remaining) else {
            return false;
        };
        let (is_or, placements) = node.tag();
        hints.proves_within(self.k.hash(), is_or, placements, turns)
    }

    fn emit(&mut self, node: Node, remaining: Option<u8>) -> Result<u32, String> {
        let key = node_key_at(self.k.hash(), node, remaining);
        if let Some(&id) = self.memo.get(&key) {
            return Ok(id);
        }
        if !self.proven.contains(&key) {
            return Err(format!("missing proved node {key:016x}"));
        }

        let (proof_node, depth) = match node {
            Node::Or { placements } => match self.k.or_eval(placements) {
                OrEval::WinNow if remaining != Some(0) => {
                    let action = self
                        .k
                        .win_now_cells(placements)
                        .ok_or("win-now node had no completion")?;
                    (
                        ProofNode::ImmediateWin {
                            action: action.cells().to_vec(),
                        },
                        1,
                    )
                }
                OrEval::WinNow | OrEval::Loss => {
                    return Err("proved OR node re-evaluated beyond its horizon".to_string());
                }
                OrEval::Moves(_) if remaining.is_some_and(|turns| turns < 2) => {
                    return Err("proved OR node has no non-terminal turn left".to_string());
                }
                OrEval::Moves(moves) => {
                    let mut choices = Vec::new();
                    let child_remaining = after_attacker(remaining);
                    for (move_index, action) in moves.iter().enumerate() {
                        self.k.place_attacker(action);
                        let child_node = Node::And;
                        let child_key = node_key_at(self.k.hash(), child_node, child_remaining);
                        let built = if self.proven.contains(&child_key) {
                            Some(self.emit(child_node, child_remaining))
                        } else if self.hint_proven(child_node, child_remaining) {
                            self.proven.insert(child_key);
                            Some(self.emit(child_node, child_remaining))
                        } else {
                            None
                        };
                        self.k.unplace(action);
                        if let Some(built) = built {
                            let child = built?;
                            choices.push((
                                move_index,
                                ProofResponse {
                                    action: action.cells().to_vec(),
                                    child,
                                },
                                1u32.saturating_add(self.depths[child as usize]),
                            ));
                        }
                    }
                    choices.sort_by_key(|(move_index, _, depth)| (*depth, *move_index));
                    let (_, primary, depth) = choices
                        .first()
                        .cloned()
                        .ok_or("proved OR node has no reconstructable proved child")?;
                    let alternatives = choices
                        .into_iter()
                        .skip(1)
                        .map(|(_, response, _)| response)
                        .collect();
                    (
                        ProofNode::AttackerMove {
                            action: primary.action,
                            child: primary.child,
                            alternatives,
                        },
                        depth,
                    )
                }
            },
            Node::And if remaining == Some(0) => {
                return Err("proved AND node has no future attacker turn".to_string());
            }
            Node::And => match self.k.and_eval() {
                AndEval::AttackerWin => (
                    ProofNode::Unstoppable {
                        threats: self
                            .k
                            .unstoppable_witness()
                            .into_iter()
                            .map(|completion| completion.cells().to_vec())
                            .collect(),
                    },
                    1,
                ),
                AndEval::Loss => return Err("proved AND node re-evaluated as a loss".to_string()),
                AndEval::Covers(covers) => {
                    let mut responses = Vec::with_capacity(covers.len());
                    let mut depth = 0;
                    for cover in covers {
                        self.k.place_defender(&cover);
                        let child_node = Node::Or { placements: 2 };
                        let child_key = node_key_at(self.k.hash(), child_node, remaining);
                        let built = if self.proven.contains(&child_key) {
                            self.emit(child_node, remaining)
                        } else if self.hint_proven(child_node, remaining) {
                            self.proven.insert(child_key);
                            self.emit(child_node, remaining)
                        } else {
                            Err(format!("missing proved defense child {child_key:016x}"))
                        };
                        self.k.unplace(&cover);
                        let child = built?;
                        depth = depth.max(self.depths[child as usize]);
                        responses.push(ProofResponse {
                            action: cover.cells().to_vec(),
                            child,
                        });
                    }
                    (ProofNode::DefenderReplies { responses }, depth)
                }
            },
        };

        let id = u32::try_from(self.nodes.len())
            .map_err(|_| "proof DAG contains more than u32::MAX nodes".to_string())?;
        self.nodes.push(proof_node);
        self.depths.push(depth);
        self.memo.insert(key, id);
        Ok(id)
    }
}

/// Replay the certificate's minimax principal variation: the shortest retained
/// attack at every OR node and a maximum-delay reply at every AND node. Because
/// `verify` recomputes the complete defense sets and node depths first, a line
/// returned here is a concrete witness attaining the certificate's worst-case
/// attacker-turn bound (ties retain deterministic certificate order).
pub(crate) fn worst_case_pv(
    pos: &Position,
    certificate: &ProofCertificate,
) -> Result<Vec<Coord>, String> {
    replay_pv(pos, certificate, true)
}

/// Replay the certificate's shortest line: the primary (shortest retained)
/// attack at every OR node and a minimum-delay reply at every AND node. The
/// result is a concrete witness that wins in the fewest attacker turns the
/// certificate can exhibit — the natural "example winning line" for the UI.
/// The certificate's worst-case bound is unchanged; this only picks the
/// defender's fastest-losing replies instead of the longest-delay ones.
pub(crate) fn shortest_pv(
    pos: &Position,
    certificate: &ProofCertificate,
) -> Result<Vec<Coord>, String> {
    replay_pv(pos, certificate, false)
}

/// Shared DAG replay for [`worst_case_pv`] / [`shortest_pv`]. `worst_case`
/// selects the maximum-delay defender reply at AND nodes (the certificate's
/// certified bound); otherwise the minimum-delay reply (the shortest line).
fn replay_pv(
    pos: &Position,
    certificate: &ProofCertificate,
    worst_case: bool,
) -> Result<Vec<Coord>, String> {
    verify(pos, certificate)?;

    fn depth(
        certificate: &ProofCertificate,
        id: u32,
        memo: &mut FxHashMap<u32, u32>,
    ) -> Result<u32, String> {
        if let Some(&known) = memo.get(&id) {
            return Ok(known);
        }
        let node = certificate
            .nodes
            .get(id as usize)
            .ok_or_else(|| format!("proof node {id} is out of range"))?;
        let value = match node {
            ProofNode::ImmediateWin { .. } | ProofNode::Unstoppable { .. } => 1,
            ProofNode::AttackerMove { child, .. } => {
                1u32.saturating_add(depth(certificate, *child, memo)?)
            }
            ProofNode::DefenderReplies { responses } => responses
                .iter()
                .map(|response| depth(certificate, response.child, memo))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .ok_or("verified defender node had no responses")?,
        };
        memo.insert(id, value);
        Ok(value)
    }

    let mut depths = FxHashMap::default();
    depth(certificate, certificate.root, &mut depths)?;
    let mut k = KernelCtx::new_wide(
        &pos.stones,
        pos.attacker,
        pos.config.win_length,
        pos.config.placement_radius,
        certificate.width == "wide",
    )
    .ok_or("could not build the replay kernel")?;
    let mut id = certificate.root;
    let mut pv = Vec::new();
    loop {
        let node = certificate
            .nodes
            .get(id as usize)
            .ok_or_else(|| format!("proof node {id} is out of range"))?;
        match node {
            ProofNode::ImmediateWin { action } => {
                let cells = CellSet2::from_cells(action);
                k.place_attacker(&cells);
                pv.extend_from_slice(cells.cells());
                break;
            }
            ProofNode::AttackerMove { action, child, .. } => {
                let cells = CellSet2::from_cells(action);
                k.place_attacker(&cells);
                pv.extend_from_slice(cells.cells());
                id = *child;
            }
            ProofNode::DefenderReplies { responses } => {
                let response = if worst_case {
                    responses
                        .iter()
                        .max_by_key(|response| depths.get(&response.child).copied().unwrap_or(0))
                } else {
                    responses
                        .iter()
                        .min_by_key(|response| depths.get(&response.child).copied().unwrap_or(0))
                }
                .ok_or("verified defender node had no responses")?;
                let cells = CellSet2::from_cells(&response.action);
                k.place_defender(&cells);
                pv.extend_from_slice(cells.cells());
                id = response.child;
            }
            ProofNode::Unstoppable { .. } => {
                let defense = k.futile_pair();
                k.place_defender(&defense);
                pv.extend_from_slice(defense.cells());
                let completion = k
                    .win_now_cells(2)
                    .ok_or("unstoppable node had no replayable completion")?;
                k.place_attacker(&completion);
                pv.extend_from_slice(completion.cells());
                break;
            }
        }
    }
    Ok(pv)
}

/// Verify a certificate from scratch against the supplied root position.
pub fn verify(pos: &Position, certificate: &ProofCertificate) -> Result<ProofSummary, String> {
    verify_with_limit(pos, certificate, DEFAULT_VERIFY_NODE_LIMIT)
}

pub fn verify_with_limit(
    pos: &Position,
    certificate: &ProofCertificate,
    node_limit: usize,
) -> Result<ProofSummary, String> {
    verify_with_limit_and_hints(pos, certificate, node_limit, false)
        .map(|(summary, _, _)| summary)
}

/// Verify a certificate and retain its positive winning-depth facts for IDTT.
/// Invalid certificates return no hints at all.
pub(crate) fn verify_with_hints(
    pos: &Position,
    certificate: &ProofCertificate,
) -> Result<(ProofSummary, WinDepthHints, Vec<GuidedOrNode>), String> {
    verify_with_limit_and_hints(pos, certificate, DEFAULT_VERIFY_NODE_LIMIT, true)
}

fn verify_with_limit_and_hints(
    pos: &Position,
    certificate: &ProofCertificate,
    node_limit: usize,
    collect_guidance: bool,
) -> Result<(ProofSummary, WinDepthHints, Vec<GuidedOrNode>), String> {
    if certificate.version != CERTIFICATE_VERSION {
        return Err(format!(
            "unsupported proof certificate version {}",
            certificate.version
        ));
    }
    if certificate.nodes.is_empty() {
        return Err("proof certificate has no nodes".to_string());
    }
    if certificate.nodes.len() > node_limit {
        return Err(format!(
            "proof certificate has {} nodes, exceeding verifier limit {node_limit}",
            certificate.nodes.len()
        ));
    }
    let wide = match certificate.width.as_str() {
        "tight" => false,
        "wide" => true,
        other => return Err(format!("invalid proof width {other:?}")),
    };
    let k = KernelCtx::new_wide(
        &pos.stones,
        pos.attacker,
        pos.config.win_length,
        pos.config.placement_radius,
        wide,
    )
    .ok_or("could not build the proof-verification kernel")?;
    let mut verifier = ProofVerifier {
        certificate,
        k,
        seen_states: FxHashMap::default(),
        verified_depths: FxHashMap::default(),
        visiting: FxHashSet::default(),
        reached: FxHashSet::default(),
        edges: 0,
        hints: WinDepthHints::default(),
        or_nodes: Vec::new(),
        collect_guidance,
    };
    let root_node = Node::Or {
        placements: pos.placements_remaining,
    };
    let depth = verifier.walk(certificate.root, root_node)?;
    if verifier.reached.len() != certificate.nodes.len() {
        return Err(format!(
            "proof certificate contains {} unreachable node(s)",
            certificate.nodes.len() - verifier.reached.len()
        ));
    }
    if collect_guidance {
        let primary = primary_reachable(certificate);
        for node in &mut verifier.or_nodes {
            node.primary = primary.contains(&node.id);
        }
    }
    let summary = ProofSummary {
        dag_nodes: certificate.nodes.len() as u64,
        proof_edges: verifier.edges,
        max_attacker_turns: depth,
    };
    Ok((summary, verifier.hints, verifier.or_nodes))
}

struct ProofVerifier<'a> {
    certificate: &'a ProofCertificate,
    k: KernelCtx,
    seen_states: FxHashMap<u32, (Node, Vec<(Coord, Player)>)>,
    verified_depths: FxHashMap<u32, u32>,
    visiting: FxHashSet<u32>,
    reached: FxHashSet<u32>,
    edges: u64,
    hints: WinDepthHints,
    or_nodes: Vec<GuidedOrNode>,
    collect_guidance: bool,
}

impl ProofVerifier<'_> {
    fn walk(&mut self, id: u32, expected: Node) -> Result<u32, String> {
        let proof = self
            .certificate
            .nodes
            .get(id as usize)
            .ok_or_else(|| format!("proof node {id} is out of range"))?
            .clone();
        let state = self.k.canonical_stones();
        let hash = self.k.hash();
        let (is_or, placements) = expected.tag();
        if let Some((prior_node, prior_state)) = self.seen_states.get(&id) {
            if *prior_node != expected || *prior_state != state {
                return Err(format!(
                    "proof node {id} is reused for a different position"
                ));
            }
            return self
                .verified_depths
                .get(&id)
                .copied()
                .ok_or_else(|| format!("proof DAG contains a cycle through node {id}"));
        }
        if !self.visiting.insert(id) {
            return Err(format!("proof DAG contains a cycle through node {id}"));
        }
        let guide_state = self.collect_guidance.then(|| state.clone());
        self.seen_states.insert(id, (expected, state));
        self.reached.insert(id);

        let depth = match (expected, proof) {
            (Node::Or { placements }, ProofNode::ImmediateWin { action }) => {
                let claimed = parse_action(&action)?;
                let actual = self
                    .k
                    .win_now_cells(placements)
                    .ok_or_else(|| format!("node {id}: immediate-win claim is false"))?;
                if claimed != actual {
                    return Err(format!(
                        "node {id}: immediate-win action is not the recomputed completion"
                    ));
                }
                1
            }
            (
                Node::Or { placements },
                ProofNode::AttackerMove {
                    action,
                    child,
                    alternatives,
                },
            ) => {
                let moves = match self.k.or_eval(placements) {
                    OrEval::Moves(moves) => moves,
                    OrEval::WinNow => {
                        return Err(format!(
                            "node {id}: used a forcing move where an immediate win exists"
                        ));
                    }
                    OrEval::Loss => {
                        return Err(format!("node {id}: attacker node has no forcing move"));
                    }
                };
                let mut choices = Vec::with_capacity(1 + alternatives.len());
                choices.push(ProofResponse { action, child });
                choices.extend(alternatives);
                let mut supplied = FxHashSet::default();
                let mut previous_depth = None;
                let mut primary_depth = None;
                let mut ranked_actions = Vec::with_capacity(choices.len());
                for response in choices {
                    let action = parse_action(&response.action)?;
                    if !supplied.insert(action) {
                        return Err(format!("node {id}: duplicate attacker alternative"));
                    }
                    if !moves.contains(&action) {
                        return Err(format!(
                            "node {id}: attacker action is not generated by the certified width"
                        ));
                    }
                    self.k.place_attacker(&action);
                    self.edges += 1;
                    let child_depth = self.walk(response.child, Node::And);
                    self.k.unplace(&action);
                    let choice_depth = 1u32.saturating_add(child_depth?);
                    ranked_actions.push(action);
                    if previous_depth.is_some_and(|prior| prior > choice_depth) {
                        return Err(format!(
                            "node {id}: attacker alternatives are not ranked by worst-case depth"
                        ));
                    }
                    previous_depth = Some(choice_depth);
                    primary_depth.get_or_insert(choice_depth);
                }
                if self.collect_guidance {
                    self.hints.set_attacker_order(hash, placements, ranked_actions);
                }
                primary_depth.ok_or_else(|| format!("node {id}: attacker node has no choice"))?
            }
            (Node::And, ProofNode::DefenderReplies { responses }) => {
                let covers = match self.k.and_eval() {
                    AndEval::Covers(covers) => covers,
                    AndEval::AttackerWin => {
                        return Err(format!(
                            "node {id}: enumerated replies to an unstoppable fork"
                        ));
                    }
                    AndEval::Loss => {
                        return Err(format!("node {id}: defender refutes the claimed proof"));
                    }
                };
                let mut supplied = FxHashMap::default();
                for response in responses {
                    let action = parse_action(&response.action)?;
                    if supplied.insert(action, response.child).is_some() {
                        return Err(format!("node {id}: duplicate defender response"));
                    }
                }
                if supplied.len() != covers.len()
                    || covers.iter().any(|cover| !supplied.contains_key(cover))
                {
                    return Err(format!(
                        "node {id}: defender response set is incomplete or has extras"
                    ));
                }
                let mut max_depth = 0;
                let mut ranked_covers = Vec::with_capacity(covers.len());
                for cover in covers {
                    let child = supplied[&cover];
                    self.k.place_defender(&cover);
                    self.edges += 1;
                    let child_depth = self.walk(child, Node::Or { placements: 2 });
                    self.k.unplace(&cover);
                    let child_depth = child_depth?;
                    max_depth = max_depth.max(child_depth);
                    ranked_covers.push((child_depth, cover));
                }
                ranked_covers.sort_by_key(|(depth, cover)| (std::cmp::Reverse(*depth), *cover));
                if self.collect_guidance {
                    self.hints.set_defender_order(
                        hash,
                        ranked_covers.into_iter().map(|(_, cover)| cover).collect(),
                    );
                }
                max_depth
            }
            (Node::And, ProofNode::Unstoppable { threats }) => match self.k.and_eval() {
                AndEval::AttackerWin => {
                    if !threats.is_empty() {
                        let supplied = threats
                            .iter()
                            .map(|action| parse_action(action))
                            .collect::<Result<Vec<_>, _>>()?;
                        let actual = self.k.unstoppable_witness();
                        if supplied != actual {
                            return Err(format!(
                                "node {id}: unstoppable threat witness does not match the position"
                            ));
                        }
                    }
                    1
                }
                _ => return Err(format!("node {id}: unstoppable-fork claim is false")),
            },
            (Node::Or { .. }, _) => {
                return Err(format!("node {id}: defender-shaped proof at attacker node"));
            }
            (Node::And, _) => {
                return Err(format!("node {id}: attacker-shaped proof at defender node"));
            }
        };

        self.visiting.remove(&id);
        self.verified_depths.insert(id, depth);
        if self.collect_guidance {
            self.hints.insert(hash, is_or, placements, depth);
        }
        if is_or && self.collect_guidance {
            self.or_nodes.push(GuidedOrNode {
                id,
                hash,
                stones: guide_state.expect("guided OR node retains its exact state"),
                placements,
                certified_depth: depth,
                primary: false,
            });
        }
        Ok(depth)
    }
}

/// Nodes in the certificate's recommended all-defense strategy: primary attack
/// at OR nodes, every response at AND nodes.
fn primary_reachable(certificate: &ProofCertificate) -> FxHashSet<u32> {
    let mut reached = FxHashSet::default();
    let mut pending = vec![certificate.root];
    while let Some(id) = pending.pop() {
        if !reached.insert(id) {
            continue;
        }
        let Some(node) = certificate.nodes.get(id as usize) else { continue };
        match node {
            ProofNode::AttackerMove { child, .. } => pending.push(*child),
            ProofNode::DefenderReplies { responses } => {
                pending.extend(responses.iter().map(|response| response.child));
            }
            ProofNode::ImmediateWin { .. } | ProofNode::Unstoppable { .. } => {}
        }
    }
    reached
}

fn parse_action(cells: &[Coord]) -> Result<CellSet2, String> {
    if cells.is_empty() || cells.len() > 2 {
        return Err("proof action must contain one or two cells".to_string());
    }
    if cells.len() == 2 && cells[0] == cells[1] {
        return Err("proof action repeats a cell".to_string());
    }
    Ok(CellSet2::from_cells(cells))
}

#[cfg(test)]
mod tests {
    use super::ProofNode;

    #[test]
    fn legacy_attacker_node_defaults_to_no_alternatives() {
        let node: ProofNode =
            serde_json::from_str(r#"{"kind":"attacker_move","action":[[1,2],[3,4]],"child":7}"#)
                .expect("legacy version-1 attacker node must deserialize");
        match node {
            ProofNode::AttackerMove { alternatives, .. } => assert!(alternatives.is_empty()),
            _ => panic!("wrong proof-node shape"),
        }
    }

    #[test]
    fn legacy_unstoppable_node_defaults_to_no_threat_witness() {
        let node: ProofNode = serde_json::from_str(r#"{"kind":"unstoppable"}"#)
            .expect("legacy version-1 unstoppable node must deserialize");
        match node {
            ProofNode::Unstoppable { threats } => assert!(threats.is_empty()),
            _ => panic!("wrong proof-node shape"),
        }
    }
}

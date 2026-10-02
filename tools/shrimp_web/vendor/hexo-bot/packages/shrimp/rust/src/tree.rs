//! shrimp PUCT tree.
//!
//! Divergence machinery gated by the `Divergences` toggles:
//!
//! - per-edge sum-of-squares accumulator (LCB selection; inactive when
//!   `lcb_move_selection` is off)
//! - per-node/per-edge (ml_sum, ml_weight) moves-left stats (inactive when
//!   `moves_left_utility` is off)
//!
//! Every engine-legal move is a candidate; there is no candidate crop, so TSS
//! injection covers the full legal set.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use hexo_engine::{
    apply_placement, pack_coord, unpack_coord, GameOutcome, HexCoord, HexoState as RustHexoState,
    PackedCoord, Placement, Player,
};
use hexo_utils::StateHash;

use crate::cache::{state_hash, RustEvaluation};
use crate::state::move_error;
use crate::threats_shared as threats;

/// Edge Q is `value_sum / visits` on the symmetric interval [-1, 1] (win/loss
/// utility, no per-node normalization). The exploration (U) and value (Q) terms
/// are added on this same scale. `Q_UTILITY_WIDTH` is the width of that interval
/// (2.0) and is not read in the selection math; it documents the c_puct scale.
pub const Q_UTILITY_WIDTH: f32 = 2.0;
/// Alternate reference utility width (1.4). Not read in the selection math.
pub const KATAGO_UTILITY_WIDTH: f32 = 1.4;

/// Divergence toggles and constants. `parity()` forces the divergences off;
/// `production()` turns them on.
#[derive(Clone, Copy, Debug)]
pub struct Divergences {
    pub lcb_move_selection: bool,
    pub lcb_z: f32,
    pub lcb_min_visits: u32,
    pub lcb_visit_fraction: f32,
    pub early_stop: bool,
    pub full_visit_floor: f32,
    pub moves_left_utility: bool,
    pub ml_weight: f32,
    pub ml_scale: f32,
    pub ml_q_gate: f32,
    /// When on, the moves-left bonus also fires on the losing side
    /// (Q < -gate); when off, it fires only on the winning side (Q > gate).
    pub ml_two_sided: bool,
    /// Final root-move decisiveness tie-break among near-LCB-tied moves.
    pub ml_final_pick: bool,
    /// LCB band (in value units) defining "near-tied" for the final pick.
    pub ml_final_pick_band: f32,
    /// When true, nucleus widening accumulates cumulative mass in f64 and
    /// short-circuits to "take all" when `widening.mass >= 1.0`. When false,
    /// accumulate in f32 with no `mass == 1.0` short-circuit. See
    /// `nucleus_count_values`.
    pub nucleus_f64: bool,
    /// When true, the new (visits==0) child selection score includes the FPU
    /// baseline `value_or_fpu(parent_value, fpu_reduction)` (fpu + U). When
    /// false, the score is U-only (`prior * exploration_scale`). See
    /// `new_child_score`. Intended to be enabled together with `lazy_widening`.
    pub new_child_fpu: bool,
    /// When true, widening eligibility is a live `peek_next_candidate().is_some()`
    /// check (an unexpanded candidate exists). When false, the frozen comparison
    /// `edges.len() < max_eligible_children`.
    pub lazy_widening: bool,
    /// When true, cache the clean post-temp root priors and reset from them
    /// before re-applying temperature on reused/promoted roots. When false,
    /// re-apply in place.
    pub clean_root_prior_cache: bool,
    // === Gumbel AlphaZero divergences (Danihelka et al. 2022). ===
    // All default off in both parity() and production(); `gumbel()` turns the
    // four mechanism bools on.
    /// Improved-policy target export: train the policy head against
    /// π'(a)=softmax(logits+σ(completedQ)). Off => visit-count target.
    pub gumbel_target: bool,
    /// Gumbel-Top-k root candidate sampling. On => the top-m candidate set
    /// drives root diversity. Off => the PUCT root.
    pub gumbel_root: bool,
    /// Sequential-Halving root budget allocation over the Gumbel-Top-m
    /// candidates (root only). Requires `gumbel_root`. Off => normal PUCT visits.
    pub gumbel_sequential_halving: bool,
    /// Deterministic non-root selection: visit-regularized rule
    /// argmax[π'(a) − N(a)/(1+ΣN)], π'=softmax(logits+σ(completedQ)), with no
    /// PUCT/c_puct/FPU/widening at interior nodes. Off => PUCT interior.
    pub gumbel_nonroot_select: bool,
    /// σ transform constants: σ(q)=(c_visit+max_b N(b))·c_scale·q.
    pub gumbel_c_visit: f32,
    pub gumbel_c_scale: f32,
    /// Export-only σ scale override for the improved-policy target π'. When
    /// `Some`, this c_scale replaces `gumbel_c_scale` in the σ call inside
    /// gumbel_target_policy ONLY; the SH ranking and interior selection keep
    /// `gumbel_c_scale`. `None` => the target uses `gumbel_c_scale` (unchanged).
    pub gumbel_target_c_scale: Option<f32>,
    /// Candidate count m for Gumbel-Top-k (clamped to n_legal at the root).
    pub gumbel_m: u32,
    /// Draw temperature τ applied to the LOGIT (only) in the Gumbel-Top-m draw
    /// comparator in init_gumbel_root: sa = logit/τ + g(a). This samples the
    /// candidate set from softmax(logits/τ) without replacement (Gumbel-top-k),
    /// widening candidates at τ>1. Affects ONLY the draw sort — the SH σ ranking,
    /// exported target, and TSS force-include all read raw logits. τ<=0 or 1.0 =>
    /// exactly today's behavior.
    pub gumbel_draw_temperature: f32,
    /// Target support floor: actions with N(a) < this are excluded from the
    /// target softmax support (then renormalized over survivors).
    pub gumbel_target_min_visits: u32,
    /// Play-policy quota prune: at an active SH root with temperature > 0 the
    /// PLAYED move samples the delta-visit histogram with round-0 quota losers
    /// zeroed (they carry schedule mass, not quality mass). Recorded targets
    /// are untouched. Off => sample the raw histogram (legacy behavior).
    pub gumbel_play_prune: bool,
}

impl Divergences {
    pub fn parity() -> Self {
        Self {
            lcb_move_selection: false,
            lcb_z: 1.6,
            lcb_min_visits: 8,
            lcb_visit_fraction: 0.1,
            early_stop: false,
            full_visit_floor: 0.75,
            moves_left_utility: false,
            ml_weight: 0.03,
            ml_scale: 32.0,
            ml_q_gate: 0.6,
            ml_two_sided: false,
            ml_final_pick: false,
            ml_final_pick_band: 0.05,
            nucleus_f64: false,
            new_child_fpu: false,
            lazy_widening: false,
            clean_root_prior_cache: false,
            // Gumbel mechanisms off in parity.
            gumbel_target: false,
            gumbel_root: false,
            gumbel_sequential_halving: false,
            gumbel_nonroot_select: false,
            gumbel_c_visit: 50.0,
            gumbel_c_scale: 1.0,
            gumbel_target_c_scale: None,
            gumbel_m: 16,
            gumbel_draw_temperature: 1.0,
            gumbel_target_min_visits: 1,
            gumbel_play_prune: false,
        }
    }

    pub fn production() -> Self {
        Self {
            lcb_move_selection: true,
            early_stop: true,
            moves_left_utility: true,
            ml_two_sided: true,
            ml_final_pick: true,
            nucleus_f64: true,
            new_child_fpu: true,
            lazy_widening: true,
            clean_root_prior_cache: true,
            ..Self::parity()
        }
    }

    /// Gumbel AlphaZero profile (Danihelka et al. 2022): starts from
    /// `production()` and turns the four Gumbel mechanism bools on, with the
    /// σ/m scalars at their defaults.
    pub fn gumbel() -> Self {
        Self {
            gumbel_target: true,
            gumbel_root: true,
            gumbel_sequential_halving: true,
            gumbel_nonroot_select: true,
            ..Self::production()
        }
    }
}

#[derive(Clone, Debug)]
pub struct RustEdge {
    pub action_id: PackedCoord,
    pub action: HexCoord,
    pub prior: f32,
    pub visits: u32,
    pub value_sum: f32,
    /// Sum of squared real backup values (LCB sigma; virtual losses excluded).
    pub value_sq_sum: f32,
    /// Moves-left stats accumulated on real backups.
    pub ml_sum: f32,
    pub ml_weight: f32,
    pub pending: u32,
    pub child: Option<usize>,
    pub forced: bool,
}

impl RustEdge {
    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    fn value_or_fpu(&self, parent_value: f32, fpu_reduction: f32) -> f32 {
        if self.visits == 0 {
            parent_value - fpu_reduction
        } else {
            self.value()
        }
    }

    pub fn ml_mean(&self) -> Option<f32> {
        if self.ml_weight > 0.0 {
            Some(self.ml_sum / self.ml_weight)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct RustPriorCandidate {
    pub action_id: PackedCoord,
    pub prior: f32,
}

impl RustPriorCandidate {
    fn into_edge(self) -> RustEdge {
        RustEdge {
            action_id: self.action_id,
            action: unpack_coord(self.action_id),
            prior: self.prior,
            visits: 0,
            value_sum: 0.0,
            value_sq_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            pending: 0,
            child: None,
            forced: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Widening {
    pub mass: f32,
    pub min_children: usize,
    pub max_children: usize,
}

#[derive(Clone, Debug)]
pub enum NodePriors {
    /// Interior nodes share the cache's descending normalized prior vector;
    /// the next unexpanded candidate is `priors[edges.len()]`.
    Shared(Arc<RustEvaluation>),
    /// Owned ascending candidate list (highest prior popped from the back).
    Owned(Vec<RustPriorCandidate>),
}

#[derive(Clone, Debug)]
pub struct RustNode {
    pub state_hash: StateHash,
    pub player: Player,
    pub eval_value: f32,
    /// Evaluator moves-left decode for this node's state, in decisions.
    pub eval_ml: Option<f32>,
    pub visits: u32,
    pub value_sum: f32,
    pub ml_sum: f32,
    pub ml_weight: f32,
    pub edges: Vec<RustEdge>,
    pub priors: NodePriors,
    pub max_eligible_children: usize,
    /// Raw pre-softmax policy logits for this node's legal actions, keyed by
    /// action_id. `None` unless the evaluation carried logits for this node.
    /// Stored raw: never temperature-shaped, normalized, or Dirichlet-noised, so
    /// the σ/Gumbel/completedQ paths read true logits. Carried onto both root and
    /// interior nodes so the Gumbel mechanisms can look up `logits(a)`.
    pub root_logits: Option<HashMap<PackedCoord, f32>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RustSearchDiagnostics {
    pub node_count: usize,
    pub active_edge_count: usize,
    pub hidden_prior_count: usize,
    pub root_active_edges: usize,
    pub root_hidden_priors: usize,
    pub max_active_edges_per_node: usize,
    pub max_hidden_priors_per_node: usize,
    pub active_edge_bytes: usize,
    pub hidden_prior_bytes: usize,
    pub shared_prior_nodes: usize,
    pub shared_prior_refs: usize,
}

impl RustNode {
    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            self.eval_value
        } else {
            self.value_sum / self.visits as f32
        }
    }

    pub fn ml_mean(&self) -> Option<f32> {
        if self.ml_weight > 0.0 {
            Some(self.ml_sum / self.ml_weight)
        } else {
            self.eval_ml
        }
    }

    fn has_actions(&self) -> bool {
        !self.edges.is_empty() || self.remaining_prior_count() > 0
    }

    pub fn remaining_prior_count(&self) -> usize {
        match &self.priors {
            NodePriors::Shared(eval) => eval.priors.len().saturating_sub(self.edges.len()),
            NodePriors::Owned(unexpanded) => unexpanded.len(),
        }
    }

    fn peek_next_candidate(&self) -> Option<(PackedCoord, f32)> {
        match &self.priors {
            NodePriors::Shared(eval) => eval.priors.get(self.edges.len()).copied(),
            NodePriors::Owned(unexpanded) => unexpanded
                .last()
                .map(|candidate| (candidate.action_id, candidate.prior)),
        }
    }

    fn materialize_next_candidate(&mut self) -> RustEdge {
        match &mut self.priors {
            NodePriors::Owned(unexpanded) => unexpanded
                .pop()
                .expect("last prior candidate exists")
                .into_edge(),
            NodePriors::Shared(eval) => {
                let (action_id, prior) = eval.priors[self.edges.len()];
                RustPriorCandidate { action_id, prior }.into_edge()
            }
        }
    }

    pub fn remaining_priors(&self) -> Vec<(PackedCoord, f32)> {
        match &self.priors {
            NodePriors::Shared(eval) => eval.priors[self.edges.len().min(eval.priors.len())..]
                .iter()
                .copied()
                .collect(),
            NodePriors::Owned(unexpanded) => unexpanded
                .iter()
                .map(|candidate| (candidate.action_id, candidate.prior))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RustSearch {
    pub root_state: RustHexoState,
    pub root_hash: StateHash,
    pub nodes: Vec<RustNode>,
    pub node_table: HashMap<StateHash, usize>,
    pub target_visits: u32,
    pub completed_visits: u32,
    fpu_reduction: f32,
    root_fpu_reduction: f32,
    widening: Widening,
    pub tss_enabled: bool,
    pub divergences: Divergences,
    /// Set when an early-stop fired for this search (telemetry).
    pub early_stopped: bool,
    /// Clean post-temp root priors (policy after the at-most-once
    /// root-policy-temperature step), keyed by action_id. When
    /// `clean_root_prior_cache` is on, a reused/promoted root resets its
    /// edge/candidate priors from this cache before re-applying temperature.
    /// Lazily populated on first root setup; None until then.
    clean_root_priors: Option<HashMap<PackedCoord, f32>>,
    active_edge_count: usize,
    max_active_edges_per_node: usize,
    /// Per-root Gumbel-Top-k candidate set + Sequential-Halving state. `Some`
    /// only when `gumbel_root` is on for a Full move; `None` otherwise, in which
    /// case the PUCT root path runs.
    gumbel_root: Option<GumbelRootState>,
}

pub struct RustSelectedLeaf {
    pub path: Vec<(usize, usize)>,
    pub state: RustHexoState,
    pub state_hash: StateHash,
    pub parent_node: usize,
    pub edge_index: usize,
    pub terminal: Option<GameOutcome>,
    pub existing_node: Option<usize>,
}

pub struct RustLeaf {
    pub root_index: usize,
    pub parent_node: usize,
    pub edge_index: usize,
    pub path: Vec<(usize, usize)>,
    pub state: RustHexoState,
    pub state_hash: StateHash,
}

impl RustSearch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        root_state: RustHexoState,
        evaluation: &RustEvaluation,
        target_visits: u32,
        fpu_reduction: f32,
        root_fpu_reduction: f32,
        root_policy_temperature: f32,
        widening: Widening,
        tss_enabled: bool,
        divergences: Divergences,
    ) -> PyResult<Self> {
        let root_hash = state_hash(&root_state);
        let (root_node, clean_root_priors) = owned_root_from_evaluation(
            root_hash,
            &root_state,
            evaluation,
            Some(root_policy_temperature),
            widening,
            tss_enabled,
            divergences,
        )?;
        let mut node_table = HashMap::new();
        node_table.insert(root_hash, 0);
        Ok(Self {
            root_state,
            root_hash,
            nodes: vec![root_node],
            node_table,
            target_visits,
            completed_visits: 0,
            fpu_reduction,
            root_fpu_reduction,
            widening,
            tss_enabled,
            divergences,
            early_stopped: false,
            clean_root_priors,
            active_edge_count: 0,
            max_active_edges_per_node: 0,
            gumbel_root: None,
        })
    }

    pub fn set_tss_enabled(&mut self, enabled: bool) {
        self.tss_enabled = enabled;
    }

    pub fn set_root_fpu_reduction(&mut self, value: f32) {
        self.root_fpu_reduction = value;
    }

    pub fn set_divergences(&mut self, divergences: Divergences) {
        self.divergences = divergences;
    }

    /// Apply root-policy softmax temperature to the root priors, before noise,
    /// at most once per root lifetime.
    ///
    /// With `clean_root_prior_cache` on: if the cache is already populated for
    /// this root, reset the edge/candidate priors from it and return without
    /// re-powering (temperature is already baked into the cached priors). If the
    /// cache is empty (a freshly promoted root), apply temperature once and
    /// capture the result into the cache.
    pub fn apply_root_policy_temperature(&mut self, temperature: f32) {
        if self.divergences.clean_root_prior_cache {
            if self.clean_root_priors.is_some() {
                self.reset_root_priors_from_clean_cache();
                return;
            }
            // Promoted root with no cache yet: fall through, apply temperature
            // once, and capture below.
        }
        if !temperature.is_finite() || temperature <= 0.0 || (temperature - 1.0).abs() < 1.0e-6 {
            // A no-op temperature still seeds the clean cache on a promoted root
            // so the subsequent noise mix resets from this baseline.
            if self.divergences.clean_root_prior_cache && self.clean_root_priors.is_none() {
                self.capture_clean_root_priors();
            }
            return;
        }
        self.ensure_root_owned();
        let root = &mut self.nodes[0];
        let NodePriors::Owned(unexpanded) = &mut root.priors else {
            return;
        };
        let inverse = 1.0 / temperature;
        let mut total = 0.0f32;
        for edge in root.edges.iter_mut() {
            if edge.prior.is_finite() && edge.prior > 0.0 {
                edge.prior = edge.prior.powf(inverse);
                total += edge.prior;
            }
        }
        for candidate in unexpanded.iter_mut() {
            if candidate.prior.is_finite() && candidate.prior > 0.0 {
                candidate.prior = candidate.prior.powf(inverse);
                total += candidate.prior;
            }
        }
        if total > 0.0 {
            for edge in root.edges.iter_mut() {
                if edge.prior.is_finite() && edge.prior > 0.0 {
                    edge.prior /= total;
                }
            }
            for candidate in unexpanded.iter_mut() {
                if candidate.prior.is_finite() && candidate.prior > 0.0 {
                    candidate.prior /= total;
                }
            }
        }
        // Capture the post-temp, pre-noise priors for a promoted root that had
        // no cache yet (the fall-through case above).
        if self.divergences.clean_root_prior_cache && self.clean_root_priors.is_none() {
            self.capture_clean_root_priors();
        }
    }

    /// Snapshot the current root edge/candidate priors into the clean cache.
    fn capture_clean_root_priors(&mut self) {
        let root = &self.nodes[0];
        let mut cache: HashMap<PackedCoord, f32> =
            HashMap::with_capacity(root.edges.len() + root.remaining_prior_count());
        for edge in &root.edges {
            cache.insert(edge.action_id, edge.prior);
        }
        if let NodePriors::Owned(unexpanded) = &root.priors {
            for candidate in unexpanded {
                cache.insert(candidate.action_id, candidate.prior);
            }
        }
        self.clean_root_priors = Some(cache);
    }

    /// Reset the root edge/candidate priors back to the clean (post-temp,
    /// pre-noise) values cached for this root.
    fn reset_root_priors_from_clean_cache(&mut self) {
        let Some(cache) = self.clean_root_priors.clone() else {
            return;
        };
        self.ensure_root_owned();
        let root = &mut self.nodes[0];
        for edge in root.edges.iter_mut() {
            if let Some(&clean) = cache.get(&edge.action_id) {
                edge.prior = clean;
            }
        }
        if let NodePriors::Owned(unexpanded) = &mut root.priors {
            for candidate in unexpanded.iter_mut() {
                if let Some(&clean) = cache.get(&candidate.action_id) {
                    candidate.prior = clean;
                }
            }
        }
    }

    fn ensure_root_owned(&mut self) {
        let root = &mut self.nodes[0];
        let owned = match &root.priors {
            NodePriors::Owned(_) => return,
            NodePriors::Shared(eval) => {
                let start = root.edges.len().min(eval.priors.len());
                let mut unexpanded: Vec<RustPriorCandidate> = eval.priors[start..]
                    .iter()
                    .map(|&(action_id, prior)| RustPriorCandidate { action_id, prior })
                    .collect();
                unexpanded.reverse();
                unexpanded.shrink_to_fit();
                unexpanded
            }
        };
        root.priors = NodePriors::Owned(owned);
    }

    pub fn root_edges_empty(&self) -> bool {
        !self.nodes[0].has_actions()
    }

    pub fn needs_visits(&self) -> bool {
        self.completed_visits < self.target_visits && !self.root_edges_empty()
    }

    pub fn remaining_visits(&self) -> u32 {
        self.target_visits.saturating_sub(self.completed_visits)
    }

    pub fn set_additional_visits(&mut self, visits: u32) {
        self.target_visits = self.completed_visits.saturating_add(visits);
        self.early_stopped = false;
    }

    pub fn root(&self) -> &RustNode {
        debug_assert_eq!(self.nodes[0].state_hash, self.root_hash);
        &self.nodes[0]
    }

    pub fn root_edge_visits(&self) -> Vec<(PackedCoord, u32)> {
        self.root()
            .edges
            .iter()
            .map(|edge| (edge.action_id, edge.visits))
            .collect()
    }

    /// True when a Gumbel-Top-k root candidate set is active on this search.
    pub fn has_gumbel_root(&self) -> bool {
        self.gumbel_root.is_some()
    }

    /// Round-0 per-candidate visit quota of the active Gumbel SH root
    /// (`floor(budget / (R * m_initial))`, min 1), or None without an active
    /// sequential-halving state. The play-policy quota prune uses it as its
    /// cut line: a candidate whose delta visits never exceeded this quota was
    /// eliminated without surviving a halving.
    pub fn gumbel_play_quota(&self) -> Option<u32> {
        let state = self.gumbel_root.as_ref()?;
        if !state.sequential_halving {
            return None;
        }
        let m = state.gumbel.len().max(1) as u32;
        let r = state.num_rounds.max(1);
        Some((state.budget / (r * m)).max(1))
    }

    /// Tear down any Gumbel root state (e.g. when a slot transitions to a
    /// non-Full move on reuse). Idempotent.
    pub fn clear_gumbel_root(&mut self) {
        self.gumbel_root = None;
    }

    /// Build the Gumbel-Top-k root candidate set and seed the Sequential-Halving
    /// schedule. Draws g(a)~Gumbel(0,1) per legal root action from `seed`, takes
    /// the top `min(m, n_legal)` by `logits(a)+g(a)`, and lays out the SH rounds
    /// over `budget` visits. Clears state and returns without building unless
    /// `gumbel_root` is on and the root carries raw logits. `sequential_halving`
    /// selects SH vs the intermediate "top-m + PUCT" mode.
    pub fn init_gumbel_root(&mut self, seed: u64, budget: u32) {
        self.gumbel_root = None;
        if !self.divergences.gumbel_root {
            return;
        }
        // Raw per-action logits are required for the Gumbel-max draw; without
        // them the state stays None and the PUCT root path runs.
        let Some(logit_map) = self.nodes[0].root_logits.clone() else {
            return;
        };
        // Survivors materialize on demand from the owned candidate tail
        // (gumbel_root_edge_index); a promoted root can still be Shared, so
        // convert it up front.
        self.ensure_root_owned();
        // Legal root action ids: edges (nucleus) + any unexpanded candidates, so
        // the Gumbel draw covers the full legal set.
        let mut action_ids: Vec<PackedCoord> =
            self.nodes[0].edges.iter().map(|e| e.action_id).collect();
        for (action_id, _) in self.nodes[0].remaining_priors() {
            action_ids.push(action_id);
        }
        if action_ids.is_empty() {
            return;
        }
        // g(a) per action from a dedicated per-action sub-stream of `seed`.
        let mut gumbel: HashMap<PackedCoord, f32> = HashMap::with_capacity(action_ids.len());
        for &action_id in &action_ids {
            let g = gumbel_draw(seed.wrapping_add(action_id as u64));
            gumbel.insert(action_id, g);
        }
        // Top-m by logits(a)+g(a) (Gumbel-max sampling-without-replacement).
        // Under SH, m is budget-calibrated: the configured gumbel_m is sized
        // for the selfplay full budget, and reusing it at smaller budgets
        // (eval matches, quick-gate evals) starves the round-0 quota below
        // GUMBEL_MIN_ROUND0_VISITS per candidate.
        let mut m = (self.divergences.gumbel_m as usize)
            .min(action_ids.len())
            .max(1);
        if self.divergences.gumbel_sequential_halving {
            m = GumbelRootState::budget_calibrated_m(m, budget)
                .min(action_ids.len())
                .max(1);
        }
        // Draw temperature τ divides the LOGIT only in the top-m sort, sampling
        // the candidate set from softmax(logits/τ) without replacement. τ<=0 or
        // 1.0 leaves the draw at logit+g (today's behavior); the SH σ ranking,
        // exported target, and TSS force-include below all keep raw logits.
        let draw_tau = self.divergences.gumbel_draw_temperature;
        let draw_tau = if draw_tau > 0.0 { draw_tau } else { 1.0 };
        action_ids.sort_by(|&a, &b| {
            let la = logit_map.get(&a).copied().unwrap_or(0.0) / draw_tau;
            let lb = logit_map.get(&b).copied().unwrap_or(0.0) / draw_tau;
            let sa = la + gumbel[&a];
            let sb = lb + gumbel[&b];
            // Descending by score; ties broken by action_id for determinism.
            sb.partial_cmp(&sa)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        // Force-include every legal TSS tactical cell in the candidate set even
        // if its (logit+g) Gumbel score ranks below the top-m. Tactical cells are
        // added to the m budget, not counted against it.
        let tactical: HashSet<PackedCoord> = if self.tss_enabled {
            threats::tactical_cells(&self.root_state)
                .into_iter()
                .map(pack_coord)
                .collect()
        } else {
            HashSet::new()
        };
        let mut survivors: Vec<PackedCoord> = Vec::with_capacity(m + tactical.len());
        let mut chosen: HashSet<PackedCoord> = HashSet::with_capacity(m + tactical.len());
        // 1. top-m by Gumbel score (action_ids already sorted descending).
        for &a in action_ids.iter().take(m) {
            if chosen.insert(a) {
                survivors.push(a);
            }
        }
        // 2. force-include any legal tactical cell not already in the top-m.
        for &a in &action_ids {
            if tactical.contains(&a) && chosen.insert(a) {
                survivors.push(a);
            }
        }
        // Restrict the kept g/logits maps to the survivor set.
        let mut g_kept: HashMap<PackedCoord, f32> = HashMap::with_capacity(survivors.len());
        let mut l_kept: HashMap<PackedCoord, f32> = HashMap::with_capacity(survivors.len());
        for &a in &survivors {
            g_kept.insert(a, gumbel[&a]);
            l_kept.insert(a, logit_map.get(&a).copied().unwrap_or(0.0));
        }
        let sequential_halving = self.divergences.gumbel_sequential_halving;
        let num_rounds = if sequential_halving {
            GumbelRootState::rounds_for(survivors.len())
        } else {
            0
        };
        // Move-entry snapshot of every root-edge visit count. Read here (before
        // any of this move's visits land) so this move's σ-scale can be built
        // from delta visits regardless of what the reused subtree carried in.
        let entry_visits: HashMap<PackedCoord, u32> = self
            .nodes[0]
            .edges
            .iter()
            .map(|e| (e.action_id, e.visits))
            .collect();
        let mut state = GumbelRootState {
            survivors,
            gumbel: g_kept,
            logits: l_kept,
            budget,
            num_rounds,
            round: 0,
            round_cap: HashMap::new(),
            entry_visits,
            sequential_halving,
        };
        if sequential_halving {
            self.seed_gumbel_round_caps(&mut state);
        }
        self.gumbel_root = Some(state);
    }

    /// Compute the per-survivor cumulative visit caps for the current round:
    /// each survivor must reach `entry_visits + floor(n/(R·|A_r|))` (min 1).
    /// Reads each survivor's current root-edge visit count as its round-entry
    /// baseline.
    fn seed_gumbel_round_caps(&self, state: &mut GumbelRootState) {
        let entry: HashMap<PackedCoord, u32> = self
            .nodes[0]
            .edges
            .iter()
            .map(|e| (e.action_id, e.visits))
            .collect();
        state.round_cap.clear();
        // A single final survivor gets an unbounded cap (u32::MAX), so the slot
        // keeps visiting it until the move's target_visits is reached.
        if state.survivors.len() <= 1 {
            for &a in &state.survivors {
                state.round_cap.insert(a, u32::MAX);
            }
            return;
        }
        let a_r = state.survivors.len() as u32;
        let r = state.num_rounds.max(1);
        // Equal per-survivor quota for this round: floor(n/(R·|A_r|)).
        let per = (state.budget / (r * a_r)).max(1);
        for &a in &state.survivors {
            let base = entry.get(&a).copied().unwrap_or(0);
            state.round_cap.insert(a, base.saturating_add(per));
        }
    }

    /// Intra-slot barrier: if every surviving candidate has met its current-round
    /// cap, rank survivors by `g(a)+logits(a)+σ(completedQ(a))`, keep the top
    /// `ceil(|A_r|/2)`, advance the round, and re-seed caps. Returns without
    /// change unless a SH Gumbel root is active and the barrier condition holds.
    /// Returns true when a halving fired.
    pub fn maybe_advance_gumbel_round(&mut self) -> bool {
        let Some(state) = self.gumbel_root.as_ref() else {
            return false;
        };
        if !state.sequential_halving || state.survivors.len() <= 1 || state.round >= state.num_rounds
        {
            return false;
        }
        // Intra-slot barrier: ALL survivors must have reached their round cap.
        let visits: HashMap<PackedCoord, u32> = self
            .nodes[0]
            .edges
            .iter()
            .map(|e| (e.action_id, e.visits))
            .collect();
        let all_met = state.survivors.iter().all(|a| {
            let v = visits.get(a).copied().unwrap_or(0);
            let cap = state.round_cap.get(a).copied().unwrap_or(0);
            v >= cap
        });
        if !all_met {
            return false;
        }
        // Rank survivors by the SH score g(a)+logits(a)+σ(completedQ(a)).
        let root = &self.nodes[0];
        let logit_map = root.root_logits.clone().unwrap_or_default();
        let (completed, _v_mix) = gumbel_completed_q(root, &logit_map);
        let c_visit = self.divergences.gumbel_c_visit;
        let c_scale = self.divergences.gumbel_c_scale;
        let state = self.gumbel_root.as_ref().expect("checked above");
        // σ scale = THIS MOVE's max delta visits (current − move-entry baseline),
        // NOT the cumulative count: on a reused root the inherited visits would
        // otherwise apply a late-round σ multiplier to survivors whose Q rests on
        // a handful of fresh look-aheads. On a fresh root every entry baseline is
        // 0, so this is numerically identical to the cumulative max.
        let entry = &state.entry_visits;
        let max_n = root
            .edges
            .iter()
            .map(|e| e.visits.saturating_sub(entry.get(&e.action_id).copied().unwrap_or(0)))
            .max()
            .unwrap_or(0);
        let mut ranked: Vec<(PackedCoord, f32)> = state
            .survivors
            .iter()
            .map(|&a| {
                let g = state.gumbel.get(&a).copied().unwrap_or(0.0);
                let l = state.logits.get(&a).copied().unwrap_or(0.0);
                // Unsearched survivors fall back to v_mix via `completed`; if a
                // survivor never materialized an edge it uses v_mix too.
                let q = completed.get(&a).copied().unwrap_or(_v_mix);
                let score = g + l + gumbel_sigma(q, max_n, c_visit, c_scale);
                (a, score)
            })
            .collect();
        // Descending by score; deterministic action_id tie-break.
        ranked.sort_by(|&(a, sa), &(b, sb)| {
            sb.partial_cmp(&sa)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let keep = ((ranked.len() + 1) / 2).max(1);
        ranked.truncate(keep);
        let survivors: Vec<PackedCoord> = ranked.into_iter().map(|(a, _)| a).collect();
        let state = self.gumbel_root.as_mut().expect("checked above");
        state.survivors = survivors;
        state.round = state.round.saturating_add(1);
        // Re-seed caps for the new round (against the new, smaller survivor set).
        let mut new_state = state.clone();
        self.seed_gumbel_round_caps(&mut new_state);
        *self.gumbel_root.as_mut().expect("checked above") = new_state;
        true
    }

    pub fn add_node_from_eval(
        &mut self,
        state: &RustHexoState,
        hash: StateHash,
        evaluation: Arc<RustEvaluation>,
    ) -> PyResult<usize> {
        if let Some(existing) = self.node_table.get(&hash).copied() {
            return Ok(existing);
        }
        let id = self.nodes.len();
        // TSS expansion injection. Every tactical cell is engine-legal and the
        // candidate set is the full legal set, so all tactical cells are injected.
        let tactical = if self.tss_enabled {
            threats::tactical_cells(state)
        } else {
            Vec::new()
        };
        let nucleus_f64 = self.divergences.nucleus_f64;
        let node = if tactical.is_empty() {
            shared_from_cache(hash, state, evaluation, self.widening, nucleus_f64)
        } else {
            owned_with_injection_from_eval(
                hash,
                state,
                &evaluation,
                self.widening,
                &tactical,
                nucleus_f64,
            )
        };
        let injected_edges = node.edges.len();
        self.nodes.push(node);
        self.node_table.insert(hash, id);
        self.active_edge_count += injected_edges;
        self.max_active_edges_per_node = self.max_active_edges_per_node.max(injected_edges);
        Ok(id)
    }

    /// Moves-left selection bonus for one edge. Returns 0 when
    /// `moves_left_utility` is off, when the edge has no visits, or when either
    /// moves-left mean is absent. Otherwise
    /// `-w * s(Q_e) * tanh((M_e - M_node) / m_scale)`, where s = +1 for
    /// Q_e > ml_q_gate, s = -1 when two-sided and Q_e < -ml_q_gate, and s = 0
    /// for |Q_e| <= ml_q_gate. Delegates to `crate::search::debug_ml_bonus`.
    fn ml_bonus(&self, node: &RustNode, edge: &RustEdge) -> f32 {
        if !self.divergences.moves_left_utility {
            return 0.0;
        }
        if edge.visits == 0 {
            return 0.0;
        }
        let (Some(m_edge), Some(m_node)) = (edge.ml_mean(), node.ml_mean()) else {
            return 0.0;
        };
        crate::search::debug_ml_bonus(
            edge.value(),
            m_edge,
            m_node,
            self.divergences.ml_weight,
            self.divergences.ml_scale,
            self.divergences.ml_q_gate,
            self.divergences.ml_two_sided,
        )
    }

    pub fn select_pending_leaf(&mut self, c_puct: f32) -> PyResult<Option<RustSelectedLeaf>> {
        let mut state = self.root_state.clone();
        let mut node_id = 0usize;
        let mut path = Vec::new();
        let mut last_parent = None;
        let mut last_edge = None;
        let mut current_hash = self.root_hash;

        loop {
            let Some(edge_index) = self.select_or_materialize_edge(node_id, c_puct) else {
                let Some(parent_node) = last_parent else {
                    return Ok(None);
                };
                let edge_index = last_edge.expect("edge index exists with parent");
                return Ok(Some(RustSelectedLeaf {
                    path,
                    state,
                    state_hash: current_hash,
                    parent_node,
                    edge_index,
                    terminal: None,
                    existing_node: Some(node_id),
                }));
            };

            let edge = &self.nodes[node_id].edges[edge_index];
            if edge.pending > 0 && edge.child.is_none() {
                return Ok(None);
            }

            let action = edge.action;
            let child = edge.child;
            apply_placement(&mut state, Placement { coord: action }).map_err(move_error)?;
            current_hash = state_hash(&state);
            path.push((node_id, edge_index));
            last_parent = Some(node_id);
            last_edge = Some(edge_index);

            if let Some(child_id) = child {
                node_id = child_id;
                continue;
            }

            if let Some(child_id) = self.node_table.get(&current_hash).copied() {
                self.nodes[node_id].edges[edge_index].child = Some(child_id);
                return Ok(Some(RustSelectedLeaf {
                    path,
                    state,
                    state_hash: current_hash,
                    parent_node: node_id,
                    edge_index,
                    terminal: None,
                    existing_node: Some(child_id),
                }));
            }

            return Ok(Some(RustSelectedLeaf {
                path,
                state: state.clone(),
                state_hash: current_hash,
                parent_node: node_id,
                edge_index,
                terminal: state.terminal(),
                existing_node: None,
            }));
        }
    }

    fn select_or_materialize_edge(&mut self, node_id: usize, c_puct: f32) -> Option<usize> {
        // At a Gumbel-Top-k root, selection is constrained to the surviving
        // candidate set and (under SH) the current round's visit caps, replacing
        // the PUCT root path. Interior nodes (node_id != 0) are not handled here.
        if node_id == 0 && self.gumbel_root.is_some() {
            return self.select_gumbel_root_edge(c_puct);
        }
        // TSS forced edges get a guaranteed first visit before normal PUCT.
        for (index, edge) in self.nodes[node_id].edges.iter().enumerate() {
            if edge.forced && edge.visits == 0 && !(edge.pending > 0 && edge.child.is_none()) {
                return Some(index);
            }
        }

        // Deterministic non-root selection. At interior nodes (node_id != 0),
        // when `gumbel_nonroot_select` is on, child selection is the
        // visit-regularized rule argmax_a [ π'(a) − N(a)/(1+ΣN) ] with
        // π'=softmax(logits+σ(completedQ)), with no PUCT/c_puct/FPU/widening
        // scoring. The root (node_id == 0) does not reach here; it is handled by
        // the Gumbel-Top-k branch above or the PUCT path. Flag off => PUCT.
        if self.divergences.gumbel_nonroot_select && node_id != 0 {
            return self.select_gumbel_nonroot_edge(node_id);
        }

        let node = &self.nodes[node_id];
        let exploration_scale = c_puct * (node.visits.max(1) as f32).sqrt();
        let parent_value = node.value();
        let fpu_reduction = if node_id == 0 {
            self.root_fpu_reduction
        } else {
            self.fpu_reduction
        };
        let mut best: Option<(usize, f32, u32, PackedCoord)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = edge.value_or_fpu(parent_value, fpu_reduction)
                + edge.prior * exploration_scale / (1.0 + edge.visits as f32)
                + self.ml_bonus(node, edge);
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                best = Some(candidate);
            }
        }

        // Widening eligibility. With lazy_widening: a live check that an
        // unexpanded candidate exists. Otherwise: the frozen comparison
        // `edges.len() < max_eligible_children` (max_eligible_children is set at
        // node creation and not re-derived).
        let can_widen = if self.divergences.lazy_widening {
            self.nodes[node_id].peek_next_candidate().is_some()
        } else {
            self.nodes[node_id].edges.len() < self.nodes[node_id].max_eligible_children
        };
        if can_widen {
            if let Some((action_id, prior)) = self.nodes[node_id].peek_next_candidate() {
                // New-child selection score (see `new_child_score`).
                let score = new_child_score(
                    parent_value,
                    fpu_reduction,
                    prior,
                    exploration_scale,
                    self.divergences.new_child_fpu,
                );
                let candidate_key = (usize::MAX, score, 0, action_id);
                let replace = match best {
                    Some(current) => compare_edge_score(candidate_key, current) == Ordering::Greater,
                    None => true,
                };
                if replace {
                    let edge_index = self.nodes[node_id].edges.len();
                    let edge = self.nodes[node_id].materialize_next_candidate();
                    self.nodes[node_id].edges.push(edge);
                    self.record_materialized_edge(node_id);
                    return Some(edge_index);
                }
            }
        }

        best.map(|item| item.0)
    }

    /// Deterministic non-root child selection (Danihelka et al. 2022). Interior
    /// nodes pick the visit-regularized action
    ///
    /// ```text
    ///   argmax_a [ π'(a) − N(a) / (1 + Σ_b N(b)) ]
    /// ```
    ///
    /// where `π'(a) = softmax_a( logits(a) + σ(completedQ(a)) )` is the node-level
    /// improved policy over the node's candidate set (all materialized edges plus
    /// the next-materializable prior), `N(a)` is the action's visit count, and
    /// `Σ_b N(b)` is the total visit count over that candidate set. This drives
    /// the empirical visit distribution toward π' (proportional allocation). No
    /// PUCT, no c_puct, no FPU, no widening-score gate. completedQ(a) = Q(a) for
    /// visited edges, else the visit-weighted node-value fallback `v_mix`. The σ
    /// transform uses this node's own `max_b N(b)`.
    ///
    /// A still-unmaterialized candidate (the next prior in the owned/shared tail)
    /// participates in the π' softmax at its completedQ == v_mix (it is unvisited)
    /// and at N==0, so its visit-regularized score is just π'(candidate). It is
    /// materialized only when it wins the argmax, using the peek/expand machinery.
    ///
    /// Uses `compare_edge_score` `(index, score, visits, action_id)` so ties
    /// break deterministically (fewer visits, then smaller action_id).
    fn select_gumbel_nonroot_edge(&mut self, node_id: usize) -> Option<usize> {
        let c_visit = self.divergences.gumbel_c_visit;
        let c_scale = self.divergences.gumbel_c_scale;
        // logits(a) for this node's legal actions (raw). Absent => logit 0.
        let logit_map = self.nodes[node_id].root_logits.clone().unwrap_or_default();
        let node = &self.nodes[node_id];
        // completedQ map + v_mix fallback (visit-weighted node value).
        let (completed, v_mix) = gumbel_completed_q(node, &logit_map);
        // σ scaling uses this node's own max child visit count.
        let max_n = node.edges.iter().map(|e| e.visits).max().unwrap_or(0);

        // Build the node-level improved policy π'(a) = softmax(logits + σ(Q*))
        // over the candidate set: every materialized edge in order, then the
        // next-materializable candidate (if any) as a virtual trailing slot. The
        // softmax spans the full candidate support so π' is a proper policy.
        let next_candidate = node.peek_next_candidate();
        let n_edges = node.edges.len();
        let mut scores: Vec<f32> = Vec::with_capacity(n_edges + 1);
        for edge in &node.edges {
            let l = logit_map.get(&edge.action_id).copied().unwrap_or(0.0);
            let q = completed.get(&edge.action_id).copied().unwrap_or(v_mix);
            scores.push(l + gumbel_sigma(q, max_n, c_visit, c_scale));
        }
        if let Some((action_id, _prior)) = next_candidate {
            let l = logit_map.get(&action_id).copied().unwrap_or(0.0);
            // The candidate is unvisited ⇒ completedQ == v_mix.
            scores.push(l + gumbel_sigma(v_mix, max_n, c_visit, c_scale));
        }
        let improved_pi = gumbel_softmax(&scores);

        // Σ_b N(b) over the candidate set (the next candidate contributes 0).
        let total_visits: u64 = node.edges.iter().map(|e| e.visits as u64).sum();
        let reg_denom = 1.0 + total_visits as f32;

        // Visit-regularized score for each materialized edge:
        //   π'(a) − N(a)/(1+ΣN). Skip edges with an in-flight expansion exactly
        // as the PUCT path does.
        let mut best: Option<(usize, f32, u32, PackedCoord)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = improved_pi[index] - (edge.visits as f32) / reg_denom;
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                best = Some(candidate);
            }
        }

        // Widening: the next-materializable candidate competes at its π' score
        // (its N==0 => visit-regularized score is just π'(candidate)), in place of
        // the lazy-widening FPU/PUCT gate. Materialization uses the peek/expand
        // machinery.
        if let Some((action_id, _prior)) = next_candidate {
            let score = improved_pi[n_edges]; // trailing softmax slot; N==0.
            let candidate_key = (usize::MAX, score, 0, action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate_key, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                let edge_index = self.nodes[node_id].edges.len();
                let edge = self.nodes[node_id].materialize_next_candidate();
                self.nodes[node_id].edges.push(edge);
                self.record_materialized_edge(node_id);
                return Some(edge_index);
            }
        }

        best.map(|item| item.0)
    }

    /// Root edge selection under an active Gumbel-Top-k candidate set. Selection
    /// is constrained to the surviving candidates; no PUCT exploration term, no
    /// nucleus widening.
    ///
    /// SH mode: allocate visits equally per round by selecting the survivor with
    /// the largest deficit below its current round cap (ties -> fewest visits ->
    /// action_id). When every survivor has met its cap, return None; the slot
    /// makes no further root progress until the intra-slot barrier
    /// (`maybe_advance_gumbel_round`) halves and re-seeds the next round.
    ///
    /// Intermediate mode (gumbel_root on, SH off): visits allocate among the
    /// survivors via the PUCT score, over the top-m set only.
    fn select_gumbel_root_edge(&mut self, c_puct: f32) -> Option<usize> {
        let state = self.gumbel_root.as_ref()?;
        let survivor_set: HashSet<PackedCoord> = state.survivors.iter().copied().collect();
        let sequential_halving = state.sequential_halving;
        let round_cap = state.round_cap.clone();

        // Materialize any survivor that is still an unexpanded prior so it can be
        // visited. (The Gumbel draw covers the full legal set, so a survivor may
        // sit in the owned-candidate tail rather than the nucleus edges.)
        // We materialize on demand: only when a survivor is the chosen action and
        // has no edge yet. First, build the current edge index per survivor.
        if sequential_halving {
            // Equal-allocation: choose the survivor with the largest (cap - visits)
            // deficit that is not currently pending-without-child. Prefer existing
            // edges; materialize a survivor with no edge only when it is the pick.
            let mut best_action: Option<(PackedCoord, u32)> = None; // (action, deficit)
            for &action in &survivor_set {
                let cap = round_cap.get(&action).copied().unwrap_or(0);
                // Current edge state for this survivor (if materialized).
                let edge = self.nodes[0]
                    .edges
                    .iter()
                    .find(|e| e.action_id == action);
                if let Some(edge) = edge {
                    if edge.pending > 0 && edge.child.is_none() {
                        continue; // already in flight; respect equal allocation
                    }
                    if edge.visits >= cap {
                        continue; // met its round cap
                    }
                    let deficit = cap.saturating_sub(edge.visits);
                    let replace = match best_action {
                        Some((ba, bd)) => deficit > bd || (deficit == bd && action < ba),
                        None => true,
                    };
                    if replace {
                        best_action = Some((action, deficit));
                    }
                } else {
                    // Unexpanded survivor: visits == 0, deficit == cap (>=1).
                    let deficit = cap.max(1);
                    let replace = match best_action {
                        Some((ba, bd)) => deficit > bd || (deficit == bd && action < ba),
                        None => true,
                    };
                    if replace {
                        best_action = Some((action, deficit));
                    }
                }
            }
            let (action, _) = best_action?;
            return self.gumbel_root_edge_index(action);
        }

        // Intermediate mode: PUCT among survivors only.
        let node = &self.nodes[0];
        let exploration_scale = c_puct * (node.visits.max(1) as f32).sqrt();
        let parent_value = node.value();
        let fpu_reduction = self.root_fpu_reduction;
        let mut best: Option<(usize, f32, u32, PackedCoord)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if !survivor_set.contains(&edge.action_id) {
                continue;
            }
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = edge.value_or_fpu(parent_value, fpu_reduction)
                + edge.prior * exploration_scale / (1.0 + edge.visits as f32)
                + self.ml_bonus(node, edge);
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                best = Some(candidate);
            }
        }
        // Materialize an unexpanded survivor if PUCT prefers a new child. The
        // new-child score is the standard widening score, gated to the survivor
        // set, so a survivor sitting in the owned tail can get its first visit.
        let mut best_new: Option<(PackedCoord, f32)> = None;
        for (action_id, prior) in node.remaining_priors() {
            if !survivor_set.contains(&action_id) {
                continue;
            }
            let score = new_child_score(
                parent_value,
                fpu_reduction,
                prior,
                exploration_scale,
                self.divergences.new_child_fpu,
            );
            let replace = match best_new {
                Some((ba, bs)) => score > bs || (score == bs && action_id < ba),
                None => true,
            };
            if replace {
                best_new = Some((action_id, score));
            }
        }
        if let Some((new_action, new_score)) = best_new {
            let take_new = match best {
                Some((_, best_score, _, _)) => new_score > best_score,
                None => true,
            };
            if take_new {
                return self.gumbel_root_edge_index(new_action);
            }
        }
        best.map(|item| item.0)
    }

    /// Return the edge index for `action`, materializing it from the owned root
    /// candidate tail if it has not been expanded yet. Returns None only if the
    /// action is neither an edge nor an owned candidate (should not happen for a
    /// survivor drawn from the legal set).
    fn gumbel_root_edge_index(&mut self, action: PackedCoord) -> Option<usize> {
        if let Some(index) = self.nodes[0]
            .edges
            .iter()
            .position(|e| e.action_id == action)
        {
            return Some(index);
        }
        // Materialize the specific owned candidate (root is always Owned).
        if let NodePriors::Owned(unexpanded) = &mut self.nodes[0].priors {
            if let Some(pos) = unexpanded.iter().position(|c| c.action_id == action) {
                let candidate = unexpanded.remove(pos);
                let edge_index = self.nodes[0].edges.len();
                self.nodes[0].edges.push(candidate.into_edge());
                self.record_materialized_edge(0);
                return Some(edge_index);
            }
        }
        None
    }

    pub fn apply_virtual_visit(&mut self, path: &[(usize, usize)], virtual_loss: f32) {
        self.completed_visits = self.completed_visits.saturating_add(1);
        for &(node_id, edge_index) in path {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].value_sum -= virtual_loss;
            self.nodes[node_id].edges[edge_index].visits += 1;
            self.nodes[node_id].edges[edge_index].value_sum -= virtual_loss;
        }
    }

    /// Real backup (adds back the virtual loss). `leaf_ml` is the moves-left
    /// estimate at the leaf in decisions (0 at a terminal). Per-step distance is
    /// added below. ML stats are side-agnostic.
    pub fn backup_virtual(
        &mut self,
        path: &[(usize, usize)],
        leaf_player: Player,
        leaf_value: f32,
        virtual_loss: f32,
        leaf_ml: Option<f32>,
    ) {
        let depth = path.len();
        for (step, &(node_id, edge_index)) in path.iter().enumerate() {
            let value = if self.nodes[node_id].player == leaf_player {
                leaf_value
            } else {
                -leaf_value
            };
            let node = &mut self.nodes[node_id];
            node.value_sum += value + virtual_loss;
            if let Some(ml) = leaf_ml {
                let ml_here = ml + (depth - step) as f32;
                node.ml_sum += ml_here;
                node.ml_weight += 1.0;
                let edge = &mut node.edges[edge_index];
                edge.value_sum += value + virtual_loss;
                edge.value_sq_sum += value * value;
                edge.ml_sum += ml_here - 1.0; // edge's child is one decision deeper
                edge.ml_weight += 1.0;
            } else {
                let edge = &mut node.edges[edge_index];
                edge.value_sum += value + virtual_loss;
                edge.value_sq_sum += value * value;
            }
        }
    }

    pub fn mark_pending(&mut self, node_id: usize, edge_index: usize, delta: i32) {
        let edge = &mut self.nodes[node_id].edges[edge_index];
        if delta >= 0 {
            edge.pending = edge.pending.saturating_add(delta as u32);
        } else {
            edge.pending = edge.pending.saturating_sub((-delta) as u32);
        }
    }

    fn record_materialized_edge(&mut self, node_id: usize) {
        self.active_edge_count += 1;
        self.max_active_edges_per_node = self
            .max_active_edges_per_node
            .max(self.nodes[node_id].edges.len());
    }

    pub fn diagnostics(&self) -> RustSearchDiagnostics {
        let mut hidden_prior_count = 0usize;
        let mut max_hidden_priors_per_node = 0usize;
        let mut shared_prior_nodes = 0usize;
        let mut shared_prior_refs = 0usize;
        for node in &self.nodes {
            match &node.priors {
                NodePriors::Owned(unexpanded) => {
                    hidden_prior_count += unexpanded.len();
                    max_hidden_priors_per_node = max_hidden_priors_per_node.max(unexpanded.len());
                }
                NodePriors::Shared(_) => {
                    shared_prior_nodes += 1;
                    shared_prior_refs += node.remaining_prior_count();
                }
            }
        }
        RustSearchDiagnostics {
            node_count: self.nodes.len(),
            active_edge_count: self.active_edge_count,
            hidden_prior_count,
            root_active_edges: self.nodes.first().map(|node| node.edges.len()).unwrap_or(0),
            root_hidden_priors: self
                .nodes
                .first()
                .map(|node| node.remaining_prior_count())
                .unwrap_or(0),
            max_active_edges_per_node: self.max_active_edges_per_node,
            max_hidden_priors_per_node,
            active_edge_bytes: self.active_edge_count * std::mem::size_of::<RustEdge>(),
            hidden_prior_bytes: hidden_prior_count * std::mem::size_of::<RustPriorCandidate>(),
            shared_prior_nodes,
            shared_prior_refs,
        }
    }

    pub fn advance_root(&mut self, action_id: PackedCoord) -> PyResult<bool> {
        let Some((edge_index, edge)) = self
            .nodes
            .first()
            .and_then(|node| {
                node.edges
                    .iter()
                    .enumerate()
                    .find(|(_, edge)| edge.action_id == action_id)
            })
            .map(|(index, edge)| (index, edge.clone()))
        else {
            return Ok(false);
        };
        let Some(child_id) = edge.child else {
            return Ok(false);
        };

        let mut new_root_state = self.root_state.clone();
        apply_placement(&mut new_root_state, Placement { coord: edge.action })
            .map_err(move_error)?;
        if new_root_state.terminal().is_some() {
            return Ok(false);
        }

        // The old root's player, captured BEFORE the subtree clone replaces
        // self.nodes. edge.value_sum is in the OLD root's player perspective;
        // whether the promoted root shares that perspective decides the sign.
        let old_player = self.nodes[0].player;

        let mut old_to_new = HashMap::new();
        let mut nodes = Vec::new();
        clone_subtree_nodes(child_id, &self.nodes, &mut old_to_new, &mut nodes);
        if nodes.is_empty() {
            return Ok(false);
        }

        let root_hash = state_hash(&new_root_state);
        nodes[0].state_hash = root_hash;
        if edge.visits > nodes[0].visits {
            nodes[0].visits = edge.visits;
            // edge.value_sum carries the child value in the OLD root's player
            // perspective. When the promoted root is the OTHER player (normal
            // turn pass) the perspective flips, so negate; on a same-player
            // promotion (FirstStone -> SecondStone keeps current_player) the
            // perspective already matches and negating would flip the sign.
            // This seeds the promoted root's FPU baseline and first reported
            // value; it is diluted (not overwritten) by fresh backups.
            nodes[0].value_sum = if nodes[0].player == old_player {
                edge.value_sum
            } else {
                -edge.value_sum
            };
        }
        let mut node_table = HashMap::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            node_table.insert(node.state_hash, index);
        }

        self.root_state = new_root_state;
        self.root_hash = root_hash;
        self.nodes = nodes;
        self.node_table = node_table;
        self.target_visits = 0;
        self.completed_visits = self.nodes[0]
            .edges
            .iter()
            .fold(self.nodes[0].visits, |total, edge| total.max(edge.visits));
        // The promoted root is a new root: discard the previous root's clean
        // prior cache so the next temperature/noise application re-captures the
        // promoted root's own clean (post-temp, pre-noise) priors.
        self.clean_root_priors = None;
        self.recompute_accounting();
        let _ = edge_index;
        Ok(true)
    }

    fn recompute_accounting(&mut self) {
        self.active_edge_count = 0;
        self.max_active_edges_per_node = 0;
        for node in &self.nodes {
            let active = node.edges.len();
            self.active_edge_count += active;
            self.max_active_edges_per_node = self.max_active_edges_per_node.max(active);
        }
    }
}

fn clone_subtree_nodes(
    old_id: usize,
    old_nodes: &[RustNode],
    old_to_new: &mut HashMap<usize, usize>,
    new_nodes: &mut Vec<RustNode>,
) -> usize {
    if let Some(new_id) = old_to_new.get(&old_id).copied() {
        return new_id;
    }
    let new_id = new_nodes.len();
    old_to_new.insert(old_id, new_id);
    let mut node = old_nodes[old_id].clone();
    for edge in &mut node.edges {
        edge.child = None;
    }
    new_nodes.push(node);

    for (edge_index, old_edge) in old_nodes[old_id].edges.iter().enumerate() {
        if let Some(old_child) = old_edge.child {
            let new_child = clone_subtree_nodes(old_child, old_nodes, old_to_new, new_nodes);
            new_nodes[new_id].edges[edge_index].child = Some(new_child);
        }
    }
    new_id
}

/// Split candidates into forced (tactical) edges and the remaining candidate
/// list, and return the widening cap. All tactical cells are treated as
/// candidates (no crop). The cap is the nucleus plus the number of tactical
/// cells that fall outside the nucleus.
fn split_tactical(
    candidates: Vec<RustPriorCandidate>,
    tactical: &[HexCoord],
    nucleus: usize,
) -> (Vec<RustEdge>, Vec<RustPriorCandidate>, usize) {
    if tactical.is_empty() {
        return (Vec::new(), candidates, nucleus);
    }
    let tac: HashSet<PackedCoord> = tactical.iter().map(|c| pack_coord(*c)).collect();
    let mut by_prior: Vec<usize> = (0..candidates.len()).collect();
    by_prior.sort_by(|&a, &c| {
        candidates[c]
            .prior
            .partial_cmp(&candidates[a].prior)
            .unwrap_or(Ordering::Equal)
    });
    let nucleus_set: HashSet<PackedCoord> = by_prior
        .iter()
        .take(nucleus)
        .map(|&i| candidates[i].action_id)
        .collect();
    let mut forced = Vec::new();
    let mut rest = Vec::with_capacity(candidates.len());
    let mut extra_beyond_nucleus = 0usize;
    for candidate in candidates {
        if tac.contains(&candidate.action_id) {
            if !nucleus_set.contains(&candidate.action_id) {
                extra_beyond_nucleus += 1;
            }
            let mut edge = candidate.into_edge();
            edge.forced = true;
            forced.push(edge);
        } else {
            rest.push(candidate);
        }
    }
    let cap = nucleus + extra_beyond_nucleus;
    (forced, rest, cap)
}

/// Build the raw-logit lookup carried onto a `RustNode` from an evaluation's
/// optional logits. When the evaluation has no logits this returns `None`, so no
/// Gumbel/σ path can read logits.
fn node_logits_from_evaluation(
    evaluation: &RustEvaluation,
) -> Option<HashMap<PackedCoord, f32>> {
    evaluation
        .logits
        .as_ref()
        .map(|pairs| pairs.iter().copied().collect())
}

fn owned_with_injection_from_eval(
    state_hash_value: StateHash,
    state: &RustHexoState,
    evaluation: &RustEvaluation,
    widening: Widening,
    tactical: &[HexCoord],
    nucleus_f64: bool,
) -> RustNode {
    let nucleus = nucleus_count_pairs(&evaluation.priors, widening, nucleus_f64);
    let mut candidates: Vec<RustPriorCandidate> = evaluation
        .priors
        .iter()
        .map(|&(action_id, prior)| RustPriorCandidate { action_id, prior })
        .collect();
    candidates.reverse();
    let (edges, rest, max_eligible_children) = split_tactical(candidates, tactical, nucleus);
    RustNode {
        state_hash: state_hash_value,
        player: state.current_player(),
        eval_value: evaluation.value,
        eval_ml: evaluation.moves_left,
        visits: 0,
        value_sum: 0.0,
        ml_sum: 0.0,
        ml_weight: 0.0,
        edges,
        priors: NodePriors::Owned(rest),
        max_eligible_children,
        root_logits: node_logits_from_evaluation(evaluation),
    }
}

fn shared_from_cache(
    state_hash_value: StateHash,
    state: &RustHexoState,
    evaluation: Arc<RustEvaluation>,
    widening: Widening,
    nucleus_f64: bool,
) -> RustNode {
    let max_eligible_children = nucleus_count_pairs(&evaluation.priors, widening, nucleus_f64);
    // Build the logit map BEFORE moving the Arc into NodePriors::Shared.
    let root_logits = node_logits_from_evaluation(&evaluation);
    RustNode {
        state_hash: state_hash_value,
        player: state.current_player(),
        eval_value: evaluation.value,
        eval_ml: evaluation.moves_left,
        visits: 0,
        value_sum: 0.0,
        ml_sum: 0.0,
        ml_weight: 0.0,
        edges: Vec::new(),
        priors: NodePriors::Shared(evaluation),
        max_eligible_children,
        root_logits,
    }
}

/// Build the owned root node. Returns the node plus the clean post-temp prior
/// cache (keyed by action_id) when `clean_root_prior_cache` is on, else None.
fn owned_root_from_evaluation(
    state_hash_value: StateHash,
    state: &RustHexoState,
    evaluation: &RustEvaluation,
    root_policy_temperature: Option<f32>,
    widening: Widening,
    tss_enabled: bool,
    divergences: Divergences,
) -> PyResult<(RustNode, Option<HashMap<PackedCoord, f32>>)> {
    let mut candidates: Vec<_> = evaluation
        .priors
        .iter()
        .map(|(action_id, prior)| RustPriorCandidate {
            action_id: *action_id,
            prior: *prior,
        })
        .collect();
    candidates.sort_by(compare_prior_candidate);
    let mut seen_actions = HashSet::new();
    candidates.retain(|candidate| seen_actions.insert(candidate.action_id));
    if let Some(temperature) = root_policy_temperature {
        apply_root_policy_temperature_to(&mut candidates, temperature);
    }
    normalize_candidate_priors(&mut candidates)?;
    // Cache the clean post-temp priors for reused/promoted roots.
    let clean_cache = if divergences.clean_root_prior_cache {
        Some(
            candidates
                .iter()
                .map(|candidate| (candidate.action_id, candidate.prior))
                .collect::<HashMap<_, _>>(),
        )
    } else {
        None
    };
    candidates.sort_by(compare_prior_candidate);
    candidates.reverse();
    let nucleus = nucleus_count(&candidates, widening, divergences.nucleus_f64);
    let tactical = if tss_enabled {
        threats::tactical_cells(state)
    } else {
        Vec::new()
    };
    let (edges, mut candidates, max_eligible_children) =
        split_tactical(candidates, &tactical, nucleus);
    candidates.shrink_to_fit();
    Ok((
        RustNode {
            state_hash: state_hash_value,
            player: state.current_player(),
            eval_value: evaluation.value,
            eval_ml: evaluation.moves_left,
            visits: 0,
            value_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            edges,
            priors: NodePriors::Owned(candidates),
            max_eligible_children,
            // Raw logits keyed by action_id (pre-temperature).
            root_logits: node_logits_from_evaluation(evaluation),
        },
        clean_cache,
    ))
}

fn nucleus_count(candidates: &[RustPriorCandidate], widening: Widening, f64_mode: bool) -> usize {
    nucleus_count_values(
        candidates.iter().map(|candidate| candidate.prior).collect(),
        widening,
        f64_mode,
    )
}

fn nucleus_count_pairs(
    priors: &[(PackedCoord, f32)],
    widening: Widening,
    f64_mode: bool,
) -> usize {
    nucleus_count_values(
        priors.iter().map(|(_, prior)| *prior).collect(),
        widening,
        f64_mode,
    )
}

/// Count the nucleus (the top-prior children whose cumulative mass first reaches
/// `widening.mass`), clamped to [min_children, max_children].
///
/// When `f64_mode` is true, accumulate cumulative mass in f64 and short-circuit
/// to `hi` (take all eligible) when `widening.mass >= 1.0`. When false,
/// accumulate in f32 with no `mass == 1.0` short-circuit.
fn nucleus_count_values(mut priors: Vec<f32>, widening: Widening, f64_mode: bool) -> usize {
    let total = priors.len();
    if total == 0 {
        return 0;
    }
    let lo = widening.min_children.max(1).min(total);
    let hi = widening.max_children.max(lo).min(total);
    if lo >= hi {
        return hi;
    }
    if f64_mode && widening.mass >= 1.0 {
        // Take every eligible child: for a normalized policy the cumulative mass
        // cannot exceed 1.0.
        return hi;
    }
    priors.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let mut count = 0usize;
    if f64_mode {
        let mut cumulative = 0.0f64;
        for prior in priors {
            cumulative += prior as f64;
            count += 1;
            if cumulative >= widening.mass as f64 {
                break;
            }
        }
    } else {
        let mut cumulative = 0.0f32;
        for prior in priors {
            cumulative += prior;
            count += 1;
            if cumulative >= widening.mass {
                break;
            }
        }
    }
    count.clamp(lo, hi)
}

fn apply_root_policy_temperature_to(candidates: &mut [RustPriorCandidate], temperature: f32) {
    if !temperature.is_finite() || temperature <= 0.0 || (temperature - 1.0).abs() < 1.0e-6 {
        return;
    }
    let inverse = 1.0 / temperature;
    for candidate in candidates.iter_mut() {
        if candidate.prior.is_finite() && candidate.prior > 0.0 {
            candidate.prior = candidate.prior.powf(inverse);
        }
    }
}

fn compare_prior_candidate(left: &RustPriorCandidate, right: &RustPriorCandidate) -> Ordering {
    right
        .prior
        .partial_cmp(&left.prior)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.action_id.cmp(&right.action_id))
}

fn normalize_candidate_priors(candidates: &mut [RustPriorCandidate]) -> PyResult<()> {
    let mut total = 0.0f32;
    for candidate in candidates.iter() {
        if !candidate.prior.is_finite() || candidate.prior < 0.0 {
            return Err(PyValueError::new_err(format!(
                "prior for action {} must be finite and >= 0",
                candidate.action_id
            )));
        }
        total += candidate.prior;
    }
    if candidates.is_empty() {
        return Ok(());
    }
    if total <= 0.0 {
        return Err(PyValueError::new_err(
            "candidate priors must contain positive mass",
        ));
    }
    for candidate in candidates {
        candidate.prior /= total;
    }
    Ok(())
}

/// PUCT selection value for a new (visits==0) candidate child.
///
/// - `new_child_fpu` on: `(parent_value - fpu_reduction) + U`. A new child has
///   visits==0, so the U denominator is (1 + 0) = 1 and
///   `U = prior * exploration_scale`. This matches the score
///   `RustEdge::value_or_fpu` yields for an existing 0-visit edge with the same
///   prior.
/// - `new_child_fpu` off: U-only `prior * exploration_scale`.
fn new_child_score(
    parent_value: f32,
    fpu_reduction: f32,
    prior: f32,
    exploration_scale: f32,
    new_child_fpu: bool,
) -> f32 {
    let u = prior * exploration_scale;
    if new_child_fpu {
        (parent_value - fpu_reduction) + u
    } else {
        u
    }
}

fn compare_edge_score(
    left: (usize, f32, u32, PackedCoord),
    right: (usize, f32, u32, PackedCoord),
) -> Ordering {
    left.1
        .partial_cmp(&right.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| right.2.cmp(&left.2))
        .then_with(|| right.3.cmp(&left.3))
}

pub fn terminal_value(outcome: GameOutcome, player: Player) -> f32 {
    if outcome.winner == player {
        1.0
    } else {
        -1.0
    }
}

pub fn random_unit(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

// === Gumbel AlphaZero shared math (Danihelka et al. 2022) ===
// These helpers are used by the Gumbel-Top-k root + Sequential Halving, the
// deterministic non-root selection, and the improved target. They are pure
// functions of the tree state and have no effect unless a Gumbel flag routes
// through them.

/// The σ transform: σ(q)=(c_visit + max_b N(b))·c_scale·q. `max_n` is the
/// maximum child visit count at the node σ is applied at. Q is already on
/// [-1,1] here, so σ is a monotone-in-q scaling (σ(0)=0).
pub fn gumbel_sigma(q: f32, max_n: u32, c_visit: f32, c_scale: f32) -> f32 {
    (c_visit + max_n as f32) * c_scale * q
}

/// A single Gumbel(0,1) draw from a deterministic per-action seed. The
/// inverse-CDF transform g = -ln(-ln(u)) on a uniform u in the open interval
/// (0,1). The uniform is clamped strictly inside (0,1) so the double log never
/// hits ±inf (random_unit can return exactly 0.0).
pub fn gumbel_draw(seed: u64) -> f32 {
    let mut u = random_unit(seed);
    // Clamp to the open interval; the ulp-scale floor/ceil keep g finite.
    const EPS: f64 = 1.0e-12;
    if u < EPS {
        u = EPS;
    } else if u > 1.0 - EPS {
        u = 1.0 - EPS;
    }
    (-(-u.ln()).ln()) as f32
}

/// completedQ + the v_mix unvisited fallback, computed once per node.
///
/// For each candidate action_id this returns `Q(a)` if it has visits, else the
/// shared `v_mix` fallback for unvisited actions. `v_mix` is the visit-count
/// weighted interpolation between the node's own value estimate `v̂` and the
/// prior-weighted average Q over visited children (Danihelka et al. 2022,
/// completed_Q):
///
/// ```text
///   v_mix = ( v̂ + (Σ_b N(b)) · (Σ_b π(b)·Q(b)) / (Σ_b π(b)) ) / ( 1 + Σ_b N(b) )
/// ```
///
/// over visited b only, where `π(b)=softmax(logits)(b)` is the network prior and
/// `v̂ = node.value()` (the node's backed-up value, or its eval_value when the
/// node itself is unvisited). At low visit counts the `v̂` term dominates; as
/// ΣN→∞ it approaches the visited-child average. When no child is visited
/// (ΣN==0 or the visited-π mass is 0) v_mix is exactly `v̂`.
///
/// `logits` maps action_id -> raw policy logit; absent actions get logit 0.
/// Returns the per-(action_id) completedQ map plus the `v_mix` scalar (so
/// callers can extend it to candidates not in `edges`).
pub fn gumbel_completed_q(
    node: &RustNode,
    logits: &HashMap<PackedCoord, f32>,
) -> (HashMap<PackedCoord, f32>, f32) {
    // Prior-weighted average Q over the visited children: softmax(logits) over
    // the visited support, weighting their Q. Numerically-stable shifted form;
    // denominator is the visited-π mass (not the full action set). `sum_n` is the
    // total child visit count Σ_b N(b) over the same visited support.
    let mut max_logit = f32::NEG_INFINITY;
    for edge in &node.edges {
        if edge.visits > 0 {
            let l = logits.get(&edge.action_id).copied().unwrap_or(0.0);
            if l > max_logit {
                max_logit = l;
            }
        }
    }
    let mut weighted_q = 0.0f32;
    let mut weight_sum = 0.0f32;
    let mut sum_n = 0u64;
    if max_logit.is_finite() {
        for edge in &node.edges {
            if edge.visits == 0 {
                continue;
            }
            let l = logits.get(&edge.action_id).copied().unwrap_or(0.0);
            let w = (l - max_logit).exp();
            weighted_q += w * edge.value();
            weight_sum += w;
            sum_n += edge.visits as u64;
        }
    }
    // v_mix: visit-count-weighted blend of v̂ (= node.value()) and the
    // prior-weighted visited-child average. When no child is visited this reduces
    // to v̂.
    let v_node = node.value();
    let v_mix = if weight_sum > 0.0 && sum_n > 0 {
        let visited_avg = weighted_q / weight_sum;
        let n = sum_n as f32;
        (v_node + n * visited_avg) / (1.0 + n)
    } else {
        v_node
    };
    let mut completed = HashMap::with_capacity(node.edges.len());
    for edge in &node.edges {
        let q = if edge.visits > 0 { edge.value() } else { v_mix };
        completed.insert(edge.action_id, q);
    }
    (completed, v_mix)
}

/// Numerically-stable softmax of a slice into a fresh Vec that sums to 1. Used
/// by the SH ranking and the target build. Empty input => empty output.
pub fn gumbel_softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // All -inf (no support): degenerate uniform to avoid NaNs.
        let u = 1.0 / scores.len() as f32;
        return vec![u; scores.len()];
    }
    let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

/// Per-slot Gumbel-Top-k + Sequential-Halving state for one root. Lives on
/// `RustSearch`. `survivors` shrinks as halving rounds advance; `gumbel` holds
/// the per-candidate g(a) draw for SH ranking; `round_cap[a]` is the cumulative
/// visit cap candidate `a` must reach in the current round before the intra-slot
/// barrier permits a halving. Constructed only when `gumbel_root` is on and the
/// move is a Full move.
#[derive(Clone, Debug)]
pub struct GumbelRootState {
    /// Surviving candidate action_ids for the current SH round (shrinks by ~half
    /// each round; in the no-SH intermediate mode this stays the full top-m set).
    pub survivors: Vec<PackedCoord>,
    /// g(a) ~ Gumbel(0,1) per original top-m candidate (kept across rounds for
    /// the SH rank `g(a)+logits(a)+σ(completedQ(a))`).
    pub gumbel: HashMap<PackedCoord, f32>,
    /// Raw logits per candidate (snapshot of the root's raw logits).
    pub logits: HashMap<PackedCoord, f32>,
    /// Total per-move visit budget `n` to allocate via SH.
    pub budget: u32,
    /// Number of SH rounds R = ceil(log2(m)).
    pub num_rounds: u32,
    /// 0-based current round index.
    pub round: u32,
    /// Cumulative visit cap each surviving candidate must reach by the END of
    /// the current round (its visits at round entry + this round's per-survivor
    /// quota). Keyed by action_id; only survivors appear.
    pub round_cap: HashMap<PackedCoord, u32>,
    /// Per-root-edge visit counts snapshotted at THIS MOVE's Gumbel-root init
    /// (move entry). On a reused root these are the inherited counts from the
    /// previous move's subtree; on a fresh root they are all 0. The SH ranking
    /// and the exported gumbel target derive their σ-scale `max_n` from this
    /// move's DELTA visits (current − entry) so a reuse-inflated cumulative
    /// count cannot apply a late-round σ multiplier to early-round survivors.
    pub entry_visits: HashMap<PackedCoord, u32>,
    /// Whether SH halving is active. When false (gumbel_root on, SH off) the
    /// candidate set is fixed to the top-m and visits allocate by PUCT among
    /// them (no rounds, no caps).
    pub sequential_halving: bool,
}

/// Minimum SH round-0 visits each candidate must afford before the tournament
/// is considered calibrated for the move's visit budget. A candidate scored
/// from fewer look-aheads than this carries too little Q signal to rank
/// tactical replies, so `init_gumbel_root` shrinks the candidate count instead
/// of starving the quota.
const GUMBEL_MIN_ROUND0_VISITS: u32 = 4;

impl GumbelRootState {
    /// Ceil(log2(m)) rounds. m<=1 => 0 rounds (nothing to halve).
    fn rounds_for(m: usize) -> u32 {
        if m <= 1 {
            0
        } else {
            (usize::BITS - (m - 1).leading_zeros()).max(1)
        }
    }

    /// Largest candidate count `m' <= m` (walking the halving ladder
    /// m -> ceil(m/2) -> ...) whose SH round-0 quota
    /// `floor(budget/(R(m')*m'))` reaches GUMBEL_MIN_ROUND0_VISITS. A single
    /// configured `gumbel_m` is sized for the selfplay full budget; smaller
    /// budgets (eval matches, quick-gate evals) reuse the same config, and
    /// without this clamp their round-0 quota collapses (32 candidates at 512
    /// visits = 3 look-aheads each). Never returns below 2 — a two-candidate
    /// tournament is the minimum meaningful SH; the legal-action clamp is the
    /// caller's.
    fn budget_calibrated_m(m: usize, budget: u32) -> usize {
        let mut mm = m;
        while mm > 2 {
            let r = Self::rounds_for(mm).max(1);
            if budget / (r * mm as u32) >= GUMBEL_MIN_ROUND0_VISITS {
                break;
            }
            mm = (mm + 1) / 2;
        }
        mm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::RustEvaluation;

    fn widening(mass: f32, min_c: usize, max_c: usize) -> Widening {
        Widening {
            mass,
            min_children: min_c,
            max_children: max_c,
        }
    }

    // === nucleus f64 short-circuit + f64/f32 agreement ===

    #[test]
    fn nucleus_f64_sentinel_returns_total_at_mass_one() {
        let priors = vec![1.0f32 / 50.0; 50];
        let w = widening(1.0, 2, 50);
        // mass >= 1.0 short-circuits to hi == total (50).
        assert_eq!(nucleus_count_values(priors.clone(), w, true), 50);
    }

    #[test]
    fn nucleus_f64_truncation_vs_f32() {
        // 1000 equal priors of 1/1000 each. The f64 path with the mass==1.0
        // short-circuit returns the full cap (hi == 1000).
        let priors = vec![1.0f32 / 1000.0; 1000];
        let w = widening(1.0, 2, 1000);
        let n_f64 = nucleus_count_values(priors.clone(), w, true);
        assert_eq!(n_f64, 1000, "f64 short-circuit takes all at mass==1.0");
    }

    #[test]
    fn nucleus_f32_matches_f64_for_normal_mass() {
        // Sharply-peaked policy with mass 0.95: f32 and f64 modes agree.
        let priors = vec![0.5f32, 0.3, 0.15, 0.04, 0.01];
        let w = widening(0.95, 2, 5);
        let n_f32 = nucleus_count_values(priors.clone(), w, false);
        let n_f64 = nucleus_count_values(priors, w, true);
        assert_eq!(n_f32, n_f64);
        // 0.5+0.3+0.15 = 0.95 reaches mass at the 3rd child.
        assert_eq!(n_f32, 3);
    }

    #[test]
    fn nucleus_respects_min_and_max_clamp() {
        let priors = vec![0.9f32, 0.05, 0.03, 0.02];
        // mass 0.95 would pick 2, but min_children=3 clamps up.
        let n = nucleus_count_values(priors.clone(), widening(0.95, 3, 4), true);
        assert_eq!(n, 3);
        // max_children=1 clamps down (lo>=hi path returns hi).
        let n2 = nucleus_count_values(priors, widening(0.95, 1, 1), true);
        assert_eq!(n2, 1);
    }

    // === new-child FPU candidate scoring ===

    #[test]
    fn new_child_score_fpu_vs_u_only() {
        let parent_value = -0.4f32;
        let fpu_reduction = 0.2f32;
        let prior = 0.3f32;
        let scale = 2.0f32;
        let u = prior * scale; // 0.6
        // fpu off: U-only.
        let off = new_child_score(parent_value, fpu_reduction, prior, scale, false);
        assert!((off - u).abs() < 1e-9);
        // fpu on: (parent_value - fpu_reduction) + U.
        let on = new_child_score(parent_value, fpu_reduction, prior, scale, true);
        assert!((on - ((parent_value - fpu_reduction) + u)).abs() < 1e-9);
        // The FPU baseline shifts the score by exactly (parent_value - fpu_red).
        assert!((on - off - (parent_value - fpu_reduction)).abs() < 1e-9);
    }

    #[test]
    fn new_child_score_matches_existing_zero_visit_edge() {
        // A materialized candidate (visits==0) must score IDENTICALLY to an
        // already-present unvisited edge with the same prior, when new_child_fpu
        // is on — i.e. value_or_fpu + U with the same denominator (1+0)=1.
        let parent_value = 0.25f32;
        let fpu = 0.2f32;
        let prior = 0.4f32;
        let scale = 1.7f32;
        let edge = RustEdge {
            action_id: 0,
            action: unpack_coord(0),
            prior,
            visits: 0,
            value_sum: 0.0,
            value_sq_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            pending: 0,
            child: None,
            forced: false,
        };
        let existing = edge.value_or_fpu(parent_value, fpu) + prior * scale / (1.0 + 0.0);
        let candidate = new_child_score(parent_value, fpu, prior, scale, true);
        assert!((existing - candidate).abs() < 1e-6, "{existing} vs {candidate}");
    }

    fn eval_with_priors(priors: Vec<(PackedCoord, f32)>, value: f32) -> RustEvaluation {
        RustEvaluation {
            value,
            legal_action_count: priors.len(),
            priors,
            moves_left: None,
            logits: None,
        }
    }

    // === parity() / production() divergence defaults ===

    #[test]
    fn parity_disables_all_divergences() {
        let p = Divergences::parity();
        assert!(!p.nucleus_f64);
        assert!(!p.new_child_fpu);
        assert!(!p.lazy_widening);
        assert!(!p.clean_root_prior_cache);
    }

    #[test]
    fn production_enables_all_divergences() {
        let p = Divergences::production();
        assert!(p.nucleus_f64);
        assert!(p.new_child_fpu);
        assert!(p.lazy_widening);
        assert!(p.clean_root_prior_cache);
    }

    // === Gumbel: raw-logit plumbing onto the tree ===

    /// An evaluation carrying the optional raw-logit column.
    fn eval_with_priors_and_logits(
        priors: Vec<(PackedCoord, f32)>,
        logits: Vec<(PackedCoord, f32)>,
        value: f32,
    ) -> RustEvaluation {
        RustEvaluation {
            value,
            legal_action_count: priors.len(),
            priors,
            moves_left: None,
            logits: Some(logits),
        }
    }

    fn build_search_from_eval(eval: &RustEvaluation, divergences: Divergences) -> RustSearch {
        RustSearch::new(
            RustHexoState::new(),
            eval,
            64,
            0.2,
            0.2,
            1.0,
            widening(0.95, 2, 8),
            false,
            divergences,
        )
        .unwrap()
    }

    #[test]
    fn root_logits_absent_when_eval_carries_none() {
        // Production/parity: no logits requested ⇒ eval.logits is None ⇒ the
        // root node carries no logit map and no Gumbel/σ path can read one.
        let eval = eval_with_priors(vec![(0, 0.5), (1, 0.3), (2, 0.2)], 0.1);
        assert!(eval.logits.is_none());
        let search = build_search_from_eval(&eval, Divergences::production());
        assert!(
            search.root().root_logits.is_none(),
            "root_logits must be None when the evaluation omits raw logits"
        );
    }

    #[test]
    fn root_logits_present_and_aligned_by_action_id() {
        // Gumbel path: eval carries raw logits aligned to the legal set. The
        // root must carry them keyed by action_id, RAW (no temperature/noise),
        // regardless of the descending-prior reordering applied to priors.
        let priors = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        // Logits in a DIFFERENT order than the prior-sorted set, with a negative
        // value (logits are unconstrained in sign, unlike priors).
        let logits = vec![(2, -1.25), (0, 3.5), (1, 0.0)];
        let eval = eval_with_priors_and_logits(priors, logits, 0.1);
        let search = build_search_from_eval(&eval, Divergences::production());
        let map = search
            .root()
            .root_logits
            .as_ref()
            .expect("root_logits must be Some when the evaluation carries logits");
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&0).copied(), Some(3.5));
        assert_eq!(map.get(&1).copied(), Some(0.0));
        assert_eq!(map.get(&2).copied(), Some(-1.25));
    }

    // === Gumbel: Gumbel-Top-k root + Sequential Halving ===

    /// gumbel() profile: a Full-move Gumbel root carries logits.
    fn gumbel_eval(n: usize) -> RustEvaluation {
        // Distinct priors (descending) and distinct logits per action id.
        let priors: Vec<(PackedCoord, f32)> = (0..n)
            .map(|i| (i as PackedCoord, (n - i) as f32))
            .collect();
        let total: f32 = priors.iter().map(|(_, p)| *p).sum();
        let priors: Vec<(PackedCoord, f32)> =
            priors.into_iter().map(|(a, p)| (a, p / total)).collect();
        let logits: Vec<(PackedCoord, f32)> = (0..n)
            .map(|i| (i as PackedCoord, (n - i) as f32 * 0.5 - 1.0))
            .collect();
        RustEvaluation {
            value: 0.0,
            legal_action_count: n,
            priors,
            moves_left: None,
            logits: Some(logits),
        }
    }

    #[test]
    fn sigma_transform_is_monotone() {
        // σ(q)=(c_visit+max_n)·c_scale·q ; σ(0)=0 ; monotone increasing in q.
        assert_eq!(gumbel_sigma(0.0, 7, 50.0, 1.0), 0.0);
        assert_eq!(gumbel_sigma(1.0, 0, 50.0, 1.0), 50.0);
        assert_eq!(gumbel_sigma(1.0, 10, 50.0, 1.0), 60.0);
        // boundaries q = ±1
        assert_eq!(gumbel_sigma(-1.0, 50, 50.0, 1.0), -100.0);
        // monotone: σ(q1) < σ(q2) for q1 < q2 at fixed (max_n, consts).
        let lo = gumbel_sigma(-0.3, 5, 50.0, 1.0);
        let hi = gumbel_sigma(0.6, 5, 50.0, 1.0);
        assert!(lo < hi);
    }

    #[test]
    fn gumbel_draw_is_finite_across_seeds() {
        for s in 0u64..10_000 {
            let g = gumbel_draw(s);
            assert!(g.is_finite(), "gumbel draw must be finite for seed {s}");
        }
    }

    #[test]
    fn rounds_for_is_ceil_log2() {
        assert_eq!(GumbelRootState::rounds_for(1), 0);
        assert_eq!(GumbelRootState::rounds_for(2), 1);
        assert_eq!(GumbelRootState::rounds_for(3), 2);
        assert_eq!(GumbelRootState::rounds_for(4), 2);
        assert_eq!(GumbelRootState::rounds_for(8), 3);
        assert_eq!(GumbelRootState::rounds_for(16), 4);
        assert_eq!(GumbelRootState::rounds_for(17), 5);
    }

    #[test]
    fn init_gumbel_root_noop_without_flag_or_logits() {
        // Flag off ⇒ no Gumbel state even with logits present.
        let eval = gumbel_eval(8);
        let mut s = build_search_from_eval(&eval, Divergences::production());
        s.init_gumbel_root(123, 64);
        assert!(!s.has_gumbel_root(), "no gumbel state when gumbel_root off");

        // Flag on but logits absent ⇒ falls back to PUCT (no state).
        let eval_nologit = eval_with_priors(vec![(0, 0.5), (1, 0.3), (2, 0.2)], 0.0);
        let mut s2 = build_search_from_eval(&eval_nologit, Divergences::gumbel());
        s2.init_gumbel_root(123, 64);
        assert!(
            !s2.has_gumbel_root(),
            "no gumbel state when raw logits are absent"
        );
    }

    #[test]
    fn gumbel_topk_selects_top_m_by_logits_plus_g() {
        // m candidates = top-m of logits(a)+g(a). Build the same draw the search
        // uses and assert the survivor set matches.
        let n = 12;
        let eval = gumbel_eval(n);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 5;
        let mut s = build_search_from_eval(&eval, dv);
        let seed = 0xABCD_1234u64;
        s.init_gumbel_root(seed, 256);
        assert!(s.has_gumbel_root());
        let state = s.gumbel_root.as_ref().unwrap();
        assert_eq!(state.survivors.len(), 5, "m clamped to gumbel_m");

        // Reference: logits+g over the full legal set, take top-5.
        let logit_map: std::collections::HashMap<PackedCoord, f32> =
            eval.logits.clone().unwrap().into_iter().collect();
        let mut scored: Vec<(PackedCoord, f32)> = (0..n as PackedCoord)
            .map(|a| {
                let g = gumbel_draw(seed.wrapping_add(a as u64));
                (a, logit_map[&a] + g)
            })
            .collect();
        scored.sort_by(|&(a, sa), &(b, sb)| {
            sb.partial_cmp(&sa).unwrap().then_with(|| a.cmp(&b))
        });
        let expected: std::collections::HashSet<PackedCoord> =
            scored.into_iter().take(5).map(|(a, _)| a).collect();
        let got: std::collections::HashSet<PackedCoord> =
            state.survivors.iter().copied().collect();
        assert_eq!(got, expected, "survivor set must be top-m by logits+g");
    }

    #[test]
    fn gumbel_m_budget_calibration_walks_halving_ladder() {
        // Selfplay full budget: 1024/(5*32)=6 >= 4, m untouched.
        assert_eq!(GumbelRootState::budget_calibrated_m(32, 1024), 32);
        // Eval budget: 512/(5*32)=3 < 4 -> halve once: 512/(4*16)=8.
        assert_eq!(GumbelRootState::budget_calibrated_m(32, 512), 16);
        // Quick-gate budget: 128 -> 8 (128/(3*8)=5).
        assert_eq!(GumbelRootState::budget_calibrated_m(32, 128), 8);
        // Floor: never below a two-candidate tournament.
        assert_eq!(GumbelRootState::budget_calibrated_m(32, 1), 2);
        // Already-affordable m is a no-op at any of these budgets.
        assert_eq!(GumbelRootState::budget_calibrated_m(8, 96), 8);
    }

    #[test]
    fn init_gumbel_root_calibrates_m_to_budget() {
        // 40 legal actions, configured m=32. At the selfplay budget the full
        // 32 survive; at the eval budget the candidate set halves to 16 so
        // round-0 still affords >= GUMBEL_MIN_ROUND0_VISITS per candidate.
        let eval = gumbel_eval(40);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 32;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(99, 1024);
        assert_eq!(s.gumbel_root.as_ref().unwrap().survivors.len(), 32);
        s.init_gumbel_root(99, 512);
        let state = s.gumbel_root.as_ref().unwrap();
        assert_eq!(state.survivors.len(), 16, "eval budget must shrink m");
        let per = 512 / (state.num_rounds * 16);
        assert!(per >= 4, "round-0 quota under the calibration floor: {per}");
    }

    #[test]
    fn sh_round_caps_are_equal_allocation() {
        // m=8, budget=256, R=3 ⇒ per-survivor round-0 quota = 256/(3*8)=10.
        let eval = gumbel_eval(8);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 8;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(42, 256);
        let state = s.gumbel_root.as_ref().unwrap();
        assert_eq!(state.num_rounds, 3);
        assert_eq!(state.survivors.len(), 8);
        let per = 256 / (3 * 8); // floor = 10
        // At round entry all survivor visits are 0, so cap == per.
        for &a in &state.survivors {
            assert_eq!(state.round_cap.get(&a).copied(), Some(per));
        }
    }

    #[test]
    fn sh_halving_keeps_ceil_half_and_advances() {
        // Drive a Gumbel root to its cap and assert one halving keeps ceil(m/2).
        let eval = gumbel_eval(8);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 8;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(7, 256);
        let m0 = s.gumbel_root.as_ref().unwrap().survivors.len();
        assert_eq!(m0, 8);

        // No halving before any survivor meets its cap.
        assert!(!s.maybe_advance_gumbel_round());

        // Force every survivor's root-edge visits up to its cap (materialize +
        // bump visits directly — this is the intra-slot barrier condition).
        let caps: Vec<(PackedCoord, u32)> = s
            .gumbel_root
            .as_ref()
            .unwrap()
            .round_cap
            .iter()
            .map(|(&a, &c)| (a, c))
            .collect();
        for (a, cap) in caps {
            let idx = s.gumbel_root_edge_index(a).expect("survivor materializes");
            s.nodes[0].edges[idx].visits = cap;
        }
        // Barrier now satisfied ⇒ exactly one halving to ceil(8/2)=4.
        assert!(s.maybe_advance_gumbel_round());
        let state = s.gumbel_root.as_ref().unwrap();
        assert_eq!(state.round, 1);
        assert_eq!(state.survivors.len(), 4, "halving keeps ceil(m/2)");
        // New caps were re-seeded against the survivors' current visits.
        for &a in &state.survivors {
            let v = s.nodes[0].edges.iter().find(|e| e.action_id == a).unwrap().visits;
            let cap = state.round_cap.get(&a).copied().unwrap();
            assert!(cap > v, "new round cap must be above current visits");
        }
    }

    #[test]
    fn completed_q_visited_is_edge_value_unvisited_is_v_mix() {
        // Hand-built: two visited edges (with Q via value_sum/visits) and one
        // unvisited. completedQ(visited)=value(); completedQ(unvisited)=v_mix.
        let eval = gumbel_eval(3);
        let mut s = build_search_from_eval(&eval, Divergences::gumbel());
        // Materialize all three as root edges and set visit stats.
        for a in 0u32..3 {
            let _ = s.gumbel_root_edge_index(a); // ensure edge exists
        }
        // edge 0: 4 visits, value_sum 2.0 ⇒ Q=0.5 ; edge 1: 1 visit, sum -0.5 ⇒ -0.5
        // edge 2: 0 visits ⇒ unvisited.
        {
            let e0 = s.nodes[0].edges.iter_mut().find(|e| e.action_id == 0).unwrap();
            e0.visits = 4;
            e0.value_sum = 2.0;
        }
        {
            let e1 = s.nodes[0].edges.iter_mut().find(|e| e.action_id == 1).unwrap();
            e1.visits = 1;
            e1.value_sum = -0.5;
        }
        let logit_map = s.nodes[0].root_logits.clone().unwrap();
        let (completed, v_mix) = gumbel_completed_q(&s.nodes[0], &logit_map);
        assert!((completed[&0] - 0.5).abs() < 1e-6);
        assert!((completed[&1] - (-0.5)).abs() < 1e-6);
        // unvisited action 2 == v_mix (prior-softmax weighted over visited 0,1).
        assert!((completed[&2] - v_mix).abs() < 1e-6);
        // v_mix must lie within [min,max] of the visited Qs (-0.5..0.5).
        assert!(v_mix >= -0.5 - 1e-6 && v_mix <= 0.5 + 1e-6);
    }

    // === Gumbel: Divergences profile gating ===

    #[test]
    fn gumbel_profile_flips_four_bools_on_with_default_scalars() {
        // gumbel() = production() with the four Gumbel bools on and the σ /
        // candidate scalars at their defaults.
        let g = Divergences::gumbel();
        assert!(g.gumbel_target);
        assert!(g.gumbel_root);
        assert!(g.gumbel_sequential_halving);
        assert!(g.gumbel_nonroot_select);
        assert_eq!(g.gumbel_c_visit, 50.0);
        assert_eq!(g.gumbel_c_scale, 1.0);
        assert_eq!(g.gumbel_m, 16);
        assert_eq!(g.gumbel_target_min_visits, 1);
        // gumbel() inherits the production() divergence set unchanged.
        assert!(g.nucleus_f64);
        assert!(g.clean_root_prior_cache);
    }

    #[test]
    fn parity_and_production_disable_all_gumbel_bools() {
        // With all four Gumbel bools false, every Gumbel path is bypassed. Both
        // parity() and production() hold the four bools off; scalars at the
        // defaults.
        for d in [Divergences::parity(), Divergences::production()] {
            assert!(!d.gumbel_target);
            assert!(!d.gumbel_root);
            assert!(!d.gumbel_sequential_halving);
            assert!(!d.gumbel_nonroot_select);
            assert_eq!(d.gumbel_c_visit, 50.0);
            assert_eq!(d.gumbel_c_scale, 1.0);
            assert_eq!(d.gumbel_m, 16);
            assert_eq!(d.gumbel_target_min_visits, 1);
        }
    }

    // === Gumbel: completedQ support floor excludes un-searched actions ===

    #[test]
    fn completed_q_floor_excludes_unsearched_from_target_support() {
        // The target softmax support is the set of edges with visits >=
        // gumbel_target_min_visits. completedQ itself still defines a value for
        // every edge (visited=Q, unvisited=v_mix), but a floored-out action must
        // not enter the softmax support — verified here by composing the same
        // logits+σ(completedQ) score the exporter uses and excluding floored ids.
        let eval = gumbel_eval(4);
        let mut s = build_search_from_eval(&eval, Divergences::gumbel());
        for a in 0u32..4 {
            let _ = s.gumbel_root_edge_index(a);
        }
        // visited: 0 (3 visits) and 2 (5 visits); unsearched: 1 and 3 (0 visits).
        for (a, v, sum) in [(0u32, 3u32, 1.5f32), (2, 5, -2.0)] {
            let e = s.nodes[0].edges.iter_mut().find(|e| e.action_id == a).unwrap();
            e.visits = v;
            e.value_sum = sum;
        }
        let min_visits = 1u32;
        let support: Vec<PackedCoord> = s.nodes[0]
            .edges
            .iter()
            .filter(|e| e.visits >= min_visits)
            .map(|e| e.action_id)
            .collect();
        assert_eq!(support, vec![0, 2], "floor keeps only searched actions");
        // The floored-out actions 1 and 3 carry a completedQ (== v_mix) but are
        // NOT members of the softmax support.
        let logit_map = s.nodes[0].root_logits.clone().unwrap();
        let (completed, v_mix) = gumbel_completed_q(&s.nodes[0], &logit_map);
        assert!((completed[&1] - v_mix).abs() < 1e-6);
        assert!((completed[&3] - v_mix).abs() < 1e-6);
        assert!(!support.contains(&1) && !support.contains(&3));
    }

    // === Gumbel: target softmax == softmax(logits + σ(completedQ)) ===

    #[test]
    fn target_softmax_matches_logits_plus_sigma_completed_q() {
        // Build a hand tree, compose the target as gumbel_target_policy does
        // (logits + σ(completedQ) over the floored support, then softmax), and
        // assert it sums to 1 and equals an independent reference computation.
        let eval = gumbel_eval(3);
        let mut s = build_search_from_eval(&eval, Divergences::gumbel());
        for a in 0u32..3 {
            let _ = s.gumbel_root_edge_index(a);
        }
        // edge 0: 8 visits, sum 4.0 ⇒ Q=0.5 ; edge 1: 2 visits, sum -0.6 ⇒ -0.3
        // edge 2: 4 visits, sum 0.0 ⇒ 0.0. All meet the floor (>=1).
        for (a, v, sum) in [(0u32, 8u32, 4.0f32), (1, 2, -0.6), (2, 4, 0.0)] {
            let e = s.nodes[0].edges.iter_mut().find(|e| e.action_id == a).unwrap();
            e.visits = v;
            e.value_sum = sum;
        }
        let c_visit = 50.0f32;
        let c_scale = 1.0f32;
        let logit_map = s.nodes[0].root_logits.clone().unwrap();
        let (completed, _v_mix) = gumbel_completed_q(&s.nodes[0], &logit_map);
        let max_n = s.nodes[0].edges.iter().map(|e| e.visits).max().unwrap();
        assert_eq!(max_n, 8);

        // Support in ascending action_id order (mirrors the exporter).
        let mut support: Vec<PackedCoord> = s.nodes[0]
            .edges
            .iter()
            .filter(|e| e.visits >= 1)
            .map(|e| e.action_id)
            .collect();
        support.sort_unstable();
        let scores: Vec<f32> = support
            .iter()
            .map(|a| {
                let l = logit_map[a];
                l + gumbel_sigma(completed[a], max_n, c_visit, c_scale)
            })
            .collect();
        let weights = gumbel_softmax(&scores);

        // Sums to 1 over support.
        let total: f32 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "target must sum to 1, got {total}");

        // Independent reference: explicit stable softmax of the same scores.
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&x| (x - m).exp()).collect();
        let z: f32 = exps.iter().sum();
        for (w, e) in weights.iter().zip(exps.iter()) {
            assert!((w - e / z).abs() < 1e-6);
        }
        // Highest score (action 0: largest logit AND largest Q) gets the largest
        // weight — monotone sanity on the composed target.
        let argmax = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(support[argmax], 0);
    }

    // === Gumbel: Gumbel-Top-k == sampling-without-replacement ===

    /// Chi-squared goodness-of-fit upper-tail survival probability P(X^2 >= stat)
    /// for `df` degrees of freedom, via the regularized upper incomplete gamma
    /// Q(df/2, stat/2). Pure (no deps) so the test stays self-contained.
    fn chi2_sf(stat: f64, df: f64) -> f64 {
        if stat <= 0.0 {
            return 1.0;
        }
        let a = df / 2.0;
        let x = stat / 2.0;
        // Lanczos ln-gamma (z >= 0.5 always holds here: a = df/2 >= 0.5 for the
        // chi-square dfs we use, so the reflection branch is unnecessary).
        fn ln_gamma(z: f64) -> f64 {
            const C: [f64; 8] = [
                676.520_368_121_885_1,
                -1_259.139_216_722_402_8,
                771.323_428_777_653_1,
                -176.615_029_162_140_6,
                12.507_343_278_686_905,
                -0.138_571_095_265_720_12,
                9.984_369_578_019_572e-6,
                1.505_632_735_149_311_6e-7,
            ];
            let g = 7.0;
            let z = z - 1.0;
            let mut x = 0.999_999_999_999_809_93;
            for (i, &c) in C.iter().enumerate() {
                x += c / (z + (i as f64) + 1.0);
            }
            let t = z + g + 0.5;
            0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
        }
        // Regularized lower incomplete gamma P(a,x); SF = 1 - P for the upper tail
        // of the gamma == chi-square upper tail.
        // Series expansion (good for x < a+1), else continued fraction.
        let gln = ln_gamma(a);
        if x < a + 1.0 {
            let mut ap = a;
            let mut sum = 1.0 / a;
            let mut del = sum;
            for _ in 0..500 {
                ap += 1.0;
                del *= x / ap;
                sum += del;
                if del.abs() < sum.abs() * 1e-15 {
                    break;
                }
            }
            let p = sum * (-x + a * x.ln() - gln).exp();
            1.0 - p
        } else {
            // Lentz continued fraction for Q(a,x).
            let tiny = 1e-300;
            let mut b = x + 1.0 - a;
            let mut c = 1.0 / tiny;
            let mut d = 1.0 / b;
            let mut h = d;
            for i in 1..500 {
                let an = -(i as f64) * (i as f64 - a);
                b += 2.0;
                d = an * d + b;
                if d.abs() < tiny {
                    d = tiny;
                }
                c = b + an / c;
                if c.abs() < tiny {
                    c = tiny;
                }
                d = 1.0 / d;
                let del = d * c;
                h *= del;
                if (del - 1.0).abs() < 1e-15 {
                    break;
                }
            }
            (-x + a * x.ln() - gln).exp() * h
        }
    }

    fn softmax_f64(logits: &[f32]) -> Vec<f64> {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = logits.iter().map(|&l| ((l as f64) - m).exp()).collect();
        let z: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / z).collect()
    }

    /// One Gumbel draw per action with the same per-action seed discipline the
    /// search uses: g(a) = gumbel_draw(seed.wrapping_add(action_id)).
    fn gumbel_argsort(logits: &[f32], seed: u64) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(a, &l)| (a, l + gumbel_draw(seed.wrapping_add(a as u64))))
            .collect();
        scored.sort_by(|x, y| {
            y.1.partial_cmp(&x.1)
                .unwrap()
                .then_with(|| x.0.cmp(&y.0))
        });
        scored.into_iter().map(|(a, _)| a).collect()
    }

    #[test]
    fn gumbel_topk_is_sampling_without_replacement_chi2_gate() {
        // Top-m of (logits + Gumbel(0,1)) is sampling without replacement from
        // softmax(logits). Gate: over >=100 random logit vectors (varied
        // length/entropy), draw for >=10_000 seeds each, compare empirical
        // first-pick freqs to softmax(logits) via chi-squared GoF at alpha=0.01;
        // allow <=5% of vectors to fail. Then the second pick, conditioned on the
        // first, must match softmax renormalized over the remaining support (same
        // gate).
        const N_VECTORS: usize = 120;
        const N_SEEDS: u64 = 12_000;
        const ALPHA: f64 = 0.01;

        // Deterministic LCG to generate the logit vectors (no rand dep needed).
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut first_fail = 0usize;
        let mut second_fail = 0usize;
        for v in 0..N_VECTORS {
            // Length 2..=8 and a scale that varies entropy (small scale ⇒ near
            // uniform, large scale ⇒ peaked).
            let n = 2 + (v % 7); // 2..=8
            let scale = 0.5 + (v as f64 % 5.0) * 1.2;
            let logits: Vec<f32> = (0..n)
                .map(|_| ((next() - 0.5) * 2.0 * scale) as f32)
                .collect();
            let probs = softmax_f64(&logits);

            // Space the seeds far apart so seed.wrapping_add(action_id) ranges of
            // adjacent trials never overlap (clean per-action seed independence).
            let stride: u64 = 1 << 20;
            // First-pick counts, plus second-pick counts conditioned on first==j*.
            let mut first_counts = vec![0u64; n];
            // For the most-probable first pick, accumulate second-pick freqs.
            let star = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let mut second_counts = vec![0u64; n];
            let mut second_total = 0u64;
            for t in 0..N_SEEDS {
                let seed = (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (t * stride);
                let order = gumbel_argsort(&logits, seed);
                first_counts[order[0]] += 1;
                if order[0] == star {
                    second_counts[order[1]] += 1;
                    second_total += 1;
                }
            }

            // First-pick chi-squared GoF vs softmax(logits).
            let total = N_SEEDS as f64;
            let mut stat = 0.0;
            for j in 0..n {
                let exp = probs[j] * total;
                if exp > 0.0 {
                    let obs = first_counts[j] as f64;
                    stat += (obs - exp) * (obs - exp) / exp;
                }
            }
            let p = chi2_sf(stat, (n - 1) as f64);
            if p <= ALPHA {
                first_fail += 1;
            }

            // Second-pick: conditioned on first==star, the remaining support is
            // softmax(logits) renormalized after removing `star`.
            let denom: f64 = 1.0 - probs[star];
            if denom > 1e-9 && second_total > 50 {
                let mut stat2 = 0.0;
                let st = second_total as f64;
                for j in 0..n {
                    if j == star {
                        // star can't be picked second; expect ~0.
                        continue;
                    }
                    let exp = (probs[j] / denom) * st;
                    if exp > 0.0 {
                        let obs = second_counts[j] as f64;
                        stat2 += (obs - exp) * (obs - exp) / exp;
                    }
                }
                // df = (support size after removal) - 1 = (n-1) - 1.
                let df = (n - 2).max(1) as f64;
                let p2 = chi2_sf(stat2, df);
                if p2 <= ALPHA {
                    second_fail += 1;
                }
                // star must essentially never be drawn second.
                assert!(
                    second_counts[star] as f64 <= 0.01 * st + 5.0,
                    "star {star} drawn second too often: {} / {st}",
                    second_counts[star]
                );
            }
        }

        let first_rate = first_fail as f64 / N_VECTORS as f64;
        let second_rate = second_fail as f64 / N_VECTORS as f64;
        assert!(
            first_rate <= 0.05,
            "first-pick GoF fail rate {first_rate:.3} exceeds 5% ({first_fail}/{N_VECTORS})"
        );
        assert!(
            second_rate <= 0.05,
            "second-pick GoF fail rate {second_rate:.3} exceeds 5% ({second_fail}/{N_VECTORS})"
        );
    }

    // === advance_root promoted-root value seeding (same/different player) ===

    /// Build a search on `root_state` with a single materialized root edge for
    /// `action`, wired to a child node whose player is `child_player`. The edge
    /// carries `edge_value_sum` over `edge_visits` (parent-perspective Q). This
    /// is the minimal shape advance_root needs to exercise its value seeding.
    fn search_with_child_edge(
        root_state: RustHexoState,
        action: PackedCoord,
        child_player: Player,
        edge_value_sum: f32,
        edge_visits: u32,
    ) -> RustSearch {
        // eval prior on `action` so RustSearch::new materializes it as a root
        // edge; a second distinct legal action keeps the root non-degenerate.
        let eval = eval_with_priors(vec![(action, 0.7), (action + 1, 0.3)], 0.0);
        let mut s = RustSearch::new(
            root_state,
            &eval,
            64,
            0.2,
            0.2,
            1.0,
            widening(0.95, 2, 8),
            false,
            Divergences::production(),
        )
        .expect("search builds");
        // Materialize `action` as a root edge (root edges are lazily expanded
        // from the owned candidate tail).
        let edge_index = s
            .gumbel_root_edge_index(action)
            .expect("action materializes as a root edge");
        // Append a child node for `action`; give it its own edges so the
        // promoted root is a normal (non-empty) node after clone.
        let child_id = s.nodes.len();
        let mut child = s.nodes[0].clone();
        child.player = child_player;
        child.visits = edge_visits.saturating_sub(1);
        child.value_sum = 0.0;
        s.nodes.push(child);
        let edge = &mut s.nodes[0].edges[edge_index];
        edge.child = Some(child_id);
        edge.visits = edge_visits;
        edge.value_sum = edge_value_sum;
        s
    }

    #[test]
    fn advance_root_same_player_promotion_keeps_edge_value_sign() {
        // Root at FirstStone (reached by the Opening ZERO move): current_player
        // is Player1 and stays Player1 through FirstStone -> SecondStone, so the
        // promoted child is the SAME player. edge.value_sum is already in that
        // player's perspective, so the seed must NOT be negated.
        let mut root_state = RustHexoState::new();
        apply_placement(&mut root_state, Placement { coord: HexCoord::ZERO })
            .expect("opening move");
        assert_eq!(root_state.current_player(), Player::Player1);
        let action = pack_coord(HexCoord::new(1, 0));
        // Positive edge Q (parent/Player1 perspective) must stay positive.
        let mut s = search_with_child_edge(root_state, action, Player::Player1, 6.0, 8);
        assert!(s.advance_root(action).expect("advance ok"));
        assert_eq!(s.nodes[0].player, Player::Player1);
        assert!(
            s.nodes[0].value() > 0.0,
            "same-player promotion must keep the edge value sign, got {}",
            s.nodes[0].value()
        );
        // Exact: value_sum seeded to +edge.value_sum, visits to edge.visits.
        assert!((s.nodes[0].value() - 6.0 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn advance_root_different_player_promotion_negates_edge_value() {
        // Root at Opening: current_player is Player0; advancing ZERO yields a
        // FirstStone child whose player is Player1 (DIFFERENT). edge.value_sum
        // is in Player0's perspective, so the seed MUST be negated into the
        // child's perspective.
        let root_state = RustHexoState::new();
        assert_eq!(root_state.current_player(), Player::Player0);
        let action = pack_coord(HexCoord::ZERO);
        // Positive edge Q (parent/Player0 perspective) must flip to negative.
        let mut s = search_with_child_edge(root_state, action, Player::Player1, 6.0, 8);
        assert!(s.advance_root(action).expect("advance ok"));
        assert_eq!(s.nodes[0].player, Player::Player1);
        assert!(
            s.nodes[0].value() < 0.0,
            "different-player promotion must negate the edge value, got {}",
            s.nodes[0].value()
        );
        assert!((s.nodes[0].value() - (-6.0 / 8.0)).abs() < 1e-6);
    }

    #[test]
    fn gumbel_sh_ranking_max_n_uses_this_moves_delta_visits() {
        // BUG 3: on a reused root the σ scale must be THIS move's delta visits,
        // not the cumulative (reuse-inflated) max. Snapshot a large entry
        // baseline, then verify the ranking's max_n reflects only fresh deltas.
        let eval = gumbel_eval(8);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 8;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(11, 256);
        // Materialize every survivor as a root edge (edges are lazily expanded).
        let survivors: Vec<PackedCoord> =
            s.gumbel_root.as_ref().unwrap().survivors.clone();
        for &a in &survivors {
            let _ = s.gumbel_root_edge_index(a).expect("survivor materializes");
        }
        // Simulate a fully-reused root: inflate every materialized edge's
        // cumulative visits by 500 over the move-entry baseline, and set the
        // entry snapshot to those same current counts (every action inherited
        // its whole subtree from the previous move, so this move's delta is 0).
        for e in s.nodes[0].edges.iter_mut() {
            e.visits += 500;
        }
        {
            let entry: HashMap<PackedCoord, u32> = s.nodes[0]
                .edges
                .iter()
                .map(|e| (e.action_id, e.visits))
                .collect();
            s.gumbel_root.as_mut().unwrap().entry_visits = entry;
        }
        // With entry == current, this move's delta max_n is 0; assert the
        // ranking computes σ off delta (max_n 0), not the inflated cumulative.
        let state = s.gumbel_root.as_ref().unwrap();
        let entry = &state.entry_visits;
        let delta_max = s.nodes[0]
            .edges
            .iter()
            .map(|e| e.visits.saturating_sub(entry.get(&e.action_id).copied().unwrap_or(0)))
            .max()
            .unwrap_or(0);
        let cumulative_max = s.nodes[0].edges.iter().map(|e| e.visits).max().unwrap_or(0);
        assert_eq!(delta_max, 0, "delta max_n must subtract the entry baseline");
        assert!(cumulative_max >= 500, "cumulative would be reuse-inflated");
    }

    // === Gumbel: draw temperature τ on the top-m sort (lever 2) ===

    /// Reference top-m by the draw comparator `logit/τ + g(a)` (τ<=0 => 1.0),
    /// mirroring init_gumbel_root's sort exactly (descending, action_id tie-break).
    fn draw_topm(logits: &[(PackedCoord, f32)], seed: u64, tau: f32, m: usize) -> Vec<PackedCoord> {
        let tau = if tau > 0.0 { tau } else { 1.0 };
        let mut scored: Vec<(PackedCoord, f32)> = logits
            .iter()
            .map(|&(a, l)| (a, l / tau + gumbel_draw(seed.wrapping_add(a as u64))))
            .collect();
        scored.sort_by(|&(a, sa), &(b, sb)| {
            sb.partial_cmp(&sa).unwrap().then_with(|| a.cmp(&b))
        });
        scored.into_iter().take(m).map(|(a, _)| a).collect()
    }

    #[test]
    fn draw_temperature_unset_and_one_match_todays_top_m() {
        // τ unset (gumbel() default 1.0) and an explicit τ=1.0 must both reproduce
        // the raw logit+g draw: same survivor set as the reference with no divide.
        let eval = gumbel_eval(12);
        let logits = eval.logits.clone().unwrap();
        let seed = 0x51EED_u64;
        for tau in [None, Some(1.0f32)] {
            let mut dv = Divergences::gumbel();
            dv.gumbel_m = 5;
            if let Some(t) = tau {
                dv.gumbel_draw_temperature = t;
            }
            assert_eq!(dv.gumbel_draw_temperature, 1.0, "default/explicit τ is 1.0");
            let mut s = build_search_from_eval(&eval, dv);
            s.init_gumbel_root(seed, 256);
            let got: HashSet<PackedCoord> =
                s.gumbel_root.as_ref().unwrap().survivors.iter().copied().collect();
            let expected: HashSet<PackedCoord> =
                draw_topm(&logits, seed, 1.0, 5).into_iter().collect();
            assert_eq!(got, expected, "τ=1 draw must equal today's logit+g top-m");
        }
    }

    #[test]
    fn draw_temperature_widens_candidate_set_and_preserves_raw_ranking_logits() {
        // Peaked logits with a big head gap that τ=4 collapses below the Gumbel
        // noise scale: at τ=1 the head dominates a low-noise tail candidate, but
        // τ=4 shrinks logit/τ enough that a different action clears the top-m cut.
        // Construct provably: action 0 logit 40 (always survives), then a tight
        // cluster [6,5,4,3,2,1,0] whose ordering τ=4 (÷4 => [1.5,1.25,1,...])
        // reshuffles against g(a). Search over seeds for a provable difference.
        let logits: Vec<(PackedCoord, f32)> = vec![
            (0, 40.0),
            (1, 6.0),
            (2, 5.0),
            (3, 4.0),
            (4, 3.0),
            (5, 2.0),
            (6, 1.0),
            (7, 0.0),
        ];
        let priors: Vec<(PackedCoord, f32)> =
            logits.iter().map(|&(a, _)| (a, 1.0 / logits.len() as f32)).collect();
        let m = 4usize;

        // Find a seed where the τ=1 and τ=4 candidate sets provably differ.
        let seed = (0u64..10_000)
            .find(|&s| {
                let a: HashSet<_> = draw_topm(&logits, s, 1.0, m).into_iter().collect();
                let b: HashSet<_> = draw_topm(&logits, s, 4.0, m).into_iter().collect();
                a != b
            })
            .expect("a τ=1 vs τ=4 candidate-set difference must exist");

        // τ=4: the search's survivor set must equal the τ=4 reference and DIFFER
        // from the τ=1 reference (the widening actually took effect).
        let eval = eval_with_priors_and_logits(priors.clone(), logits.clone(), 0.0);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = m as u32;
        // SH would budget-calibrate m down; disable SH so the raw top-m is kept and
        // the draw-temperature effect is isolated (target/select paths unaffected).
        dv.gumbel_sequential_halving = false;
        dv.gumbel_draw_temperature = 4.0;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(seed, 256);
        let got: HashSet<PackedCoord> =
            s.gumbel_root.as_ref().unwrap().survivors.iter().copied().collect();
        let ref_tau4: HashSet<PackedCoord> = draw_topm(&logits, seed, 4.0, m).into_iter().collect();
        let ref_tau1: HashSet<PackedCoord> = draw_topm(&logits, seed, 1.0, m).into_iter().collect();
        assert_eq!(got, ref_tau4, "τ=4 survivor set must match the τ=4 draw");
        assert_ne!(got, ref_tau1, "τ=4 must change the candidate set vs τ=1");

        // The SH σ-ranking reads state.logits, which must remain the RAW logits
        // (undivided) — the τ divide is confined to the draw sort. Assert every
        // survivor's stored ranking logit equals its raw eval logit, not logit/τ.
        let raw: HashMap<PackedCoord, f32> = logits.iter().copied().collect();
        let state = s.gumbel_root.as_ref().unwrap();
        for (&a, &stored) in state.logits.iter() {
            assert_eq!(
                stored, raw[&a],
                "SH ranking logit for {a} must be raw ({}), not logit/τ",
                raw[&a]
            );
        }
    }

    #[test]
    fn gumbel_entry_visits_zero_on_fresh_root_delta_equals_cumulative() {
        // Invariant: on a fresh (non-reused) root the entry snapshot is all 0,
        // so delta visits == cumulative visits and BUG-3's fix is a no-op there.
        let eval = gumbel_eval(8);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 8;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(3, 256);
        for (&_a, &v) in s.gumbel_root.as_ref().unwrap().entry_visits.iter() {
            assert_eq!(v, 0, "fresh-root entry baseline must be 0");
        }
        // Bump some edges; delta must equal cumulative when the baseline is 0.
        for (i, e) in s.nodes[0].edges.iter_mut().enumerate() {
            e.visits = i as u32 + 1;
        }
        let entry = &s.gumbel_root.as_ref().unwrap().entry_visits;
        for e in s.nodes[0].edges.iter() {
            let delta = e.visits.saturating_sub(entry.get(&e.action_id).copied().unwrap_or(0));
            assert_eq!(delta, e.visits, "delta must equal cumulative on a fresh root");
        }
    }
}

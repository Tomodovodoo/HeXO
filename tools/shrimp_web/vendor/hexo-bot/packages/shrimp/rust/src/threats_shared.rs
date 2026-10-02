//! threats_shared.rs — phase-aware Connect6/Hexo threat analysis (Threat-Space
//! Search). The shared threat-space-search core, vendored into shrimp. Pure
//! board geometry over the engine's incremental `WindowStore`; no graph/feature
//! construction, no network.
//!
//! This module is the single source of TSS threat semantics — one definition of
//! "what is a threat / win-now / forced loss". It is consumed by shrimp's
//! search via `crate::threats_shared`:
//!   - the tactical-candidate INJECTION at node expansion,
//!   - the phase-aware hitting-set leaf value OVERRIDE,
//!   - the tactical move-selection GUARD at the root.
//!
//! The threat model is verified against the real `hexo_engine`:
//!
//!   * A THREAT is an active (single-colour) length-6 window with count >= 4
//!     (engine: `WindowEntry::threat_player`).
//!   * PER-NODE != PER-TURN. The MCTS expands ONE placement per node. Let
//!     `B = placements_remaining_in_turn` (2 at FirstStone, 1 at Opening /
//!     SecondStone). A count-5 wins with one placement (any B); a count-4 wins
//!     only with two placements left (B == 2, FirstStone) — at SecondStone a
//!     count-4 is NOT win-now (TEST G).
//!   * DEFENSE is a HITTING SET, not "fill all gaps": one defender stone in ANY
//!     empty of an opponent threat window two-colours it (kills it). The side to
//!     move is a 1-ply forced LOSS iff it has no own win this node AND the
//!     minimum number of cells hitting every opponent >=4 window's empties
//!     exceeds `B` (TESTS C/D/H/I refuted the set-cover framing).
//!
//! Threat cells are intrinsically LOCAL: every >=4 window holds >= 4 stones, so
//! each of its empties sits within a length-6 window of those stones (hex distance
//! <= 5), well inside the engine's `LEGAL_RADIUS == 8` placement range. So every
//! tactical cell is always a legal move; shrimp (infinite-candidate) injects
//! all of them at the call site.
//!
//! D6-safe: D6 maps windows/owners/empties bijectively; every output here is the
//! image set / a phase-derived (D6-fixed) scalar.

use hexo_engine::{HexCoord, HexoState as RustHexoState, TurnPhase};

/// Placements remaining in the current turn for the side to move:
/// 2 at FirstStone, 1 at Opening and SecondStone. This is the budget `B` that
/// parameterizes every "win-now / forced" statement.
pub(crate) fn placements_remaining(state: &RustHexoState) -> u8 {
    match state.phase() {
        TurnPhase::FirstStone => 2,
        TurnPhase::Opening | TurnPhase::SecondStone { .. } => 1,
    }
}

/// Result of a 1-ply phase-aware threat analysis at one node/leaf.
pub(crate) struct ThreatAnalysis {
    /// Placements remaining this turn (the budget B).
    pub(crate) b: u8,
    /// The side to move can complete a window with its B placements this turn
    /// (own count-5 for any B, or own count-4 when B == 2).
    pub(crate) own_win_now: bool,
    /// Minimum number of cells hitting >=1 empty of EVERY opponent >=4 window,
    /// capped at B: `Some(k)` with `k <= B` if defensible, `None` if it needs
    /// more than B (a 1-ply forced loss when there is no own win). `Some(0)`
    /// when there are no opponent threats.
    pub(crate) min_hitting_set: Option<u8>,
    /// Number of active opponent >=4 windows (the must-answer threats).
    pub(crate) opp_threat_count: usize,
}

impl ThreatAnalysis {
    /// 1-ply forced LOSS for the side to move: no own win this node AND the
    /// opponent's threats cannot all be hit with B placements.
    pub(crate) fn forced_loss(&self) -> bool {
        !self.own_win_now && self.min_hitting_set.is_none()
    }

    /// Phase-aware leaf verdict for the side to move: `Some(1.0)` proven win,
    /// `Some(-1.0)` proven loss, `None` no 1-ply proof (let net/search decide).
    /// HARD WIN is checked first (the side to move moves first).
    pub(crate) fn verdict(&self) -> Option<f32> {
        if self.own_win_now {
            Some(1.0)
        } else if self.min_hitting_set.is_none() {
            Some(-1.0)
        } else {
            None
        }
    }
}

/// Smallest number of cells hitting >=1 element of every set in `sets`, capped
/// at `budget`. `Some(0)` if `sets` is empty; `Some(k)` for the least `k <=
/// budget` that hits all; `None` if more than `budget` are needed. `budget` is
/// at most 2 in Hexo (a single turn places <= 2 stones), so the k=1/k=2 cases
/// below are exhaustive.
fn min_hitting_set(sets: &[Vec<HexCoord>], budget: u8) -> Option<u8> {
    if sets.is_empty() {
        return Some(0);
    }
    // A >=4 window has count in {4,5} < 6, so it always has >=1 empty; an empty
    // set here would mean an unhittable (already-won) window — treat as a miss.
    if sets.iter().any(|s| s.is_empty()) {
        return None;
    }
    let mut universe: Vec<HexCoord> = Vec::new();
    for s in sets {
        for &c in s {
            if !universe.contains(&c) {
                universe.push(c);
            }
        }
    }
    if budget >= 1 {
        // k = 1: a single cell common to every window's empties.
        for &c in &universe {
            if sets.iter().all(|s| s.contains(&c)) {
                return Some(1);
            }
        }
    }
    if budget >= 2 {
        // k = 2: any pair of cells covering every window.
        for i in 0..universe.len() {
            for j in (i + 1)..universe.len() {
                let (a, b) = (universe[i], universe[j]);
                if sets.iter().all(|s| s.contains(&a) || s.contains(&b)) {
                    return Some(2);
                }
            }
        }
    }
    None
}

/// Phase-aware 1-ply threat analysis for the side to move at `state`. Single
/// pass over the live threat windows (this runs at every searched leaf in the
/// override, so it avoids the two separate `threat_entries` scans).
pub(crate) fn analyze(state: &RustHexoState) -> ThreatAnalysis {
    let b = placements_remaining(state);
    // Threat-free short-circuit (the common case). With no active >= 4 window the
    // loop below buckets nothing: own_win_now stays false and `opp_empties` stays
    // empty, so `min_hitting_set(&[], b) == Some(0)` and `opp_threat_count == 0`.
    // This returns exactly that, skipping the O(all-touched-windows) scan. The
    // `has_threats()` index is an exact mirror of the scan, so the result is
    // bit-identical.
    if !state.board().windows().has_threats() {
        return ThreatAnalysis {
            b,
            own_win_now: false,
            min_hitting_set: Some(0),
            opp_threat_count: 0,
        };
    }
    let me = state.current_player();
    let mut own_win_now = false;
    let mut opp_empties: Vec<Vec<HexCoord>> = Vec::new();
    for (player, entry) in state.board().windows().threats() {
        if player == me {
            // own win-now: count-5 (1 placement) any B; count-4 only at B==2.
            match entry.count(me) {
                5 => own_win_now = true,
                4 if b >= 2 => own_win_now = true,
                _ => {}
            }
        } else {
            opp_empties.push(entry.empty_cells());
        }
    }
    let opp_threat_count = opp_empties.len();
    let min_hitting_set = min_hitting_set(&opp_empties, b);
    ThreatAnalysis {
        b,
        own_win_now,
        min_hitting_set,
        opp_threat_count,
    }
}

/// The TACTICAL SET T(state) injected as guaranteed-expanded children at a node
/// with any active >=4 threat: own winning completions (phase-aware) UNION the
/// empties of every opponent >=4 window. Deduplicated. Empty when the node has no
/// >=4 threat (the common path — injection is a no-op there).
///
/// SINGLE PASS over the live threat windows (perf): buckets own/opp in one
/// `threats()` pass (own cells first, then opponent empties, with first-occurrence
/// dedup).
///
/// These are full-board engine coordinates; shrimp (infinite-candidate)
/// injects all of them.
pub(crate) fn tactical_cells(state: &RustHexoState) -> Vec<HexCoord> {
    // Threat-free short-circuit (the common case): with no active >= 4 window the
    // scan below collects nothing, so the tactical set is empty. Identical result,
    // no full-window scan. `has_threats()` exactly mirrors the scan.
    if !state.board().windows().has_threats() {
        return Vec::new();
    }
    let b = placements_remaining(state);
    let me = state.current_player();
    let mut own: Vec<HexCoord> = Vec::new();
    let mut opp: Vec<HexCoord> = Vec::new();
    for (player, entry) in state.board().windows().threats() {
        if player == me {
            // own winning completions: count-5 (1 placement) any B; count-4 at B==2.
            match entry.count(me) {
                5 => push_unique(&mut own, entry.empty_cells()),
                4 if b >= 2 => push_unique(&mut own, entry.empty_cells()),
                _ => {}
            }
        } else {
            push_unique(&mut opp, entry.empty_cells());
        }
    }
    push_unique(&mut own, opp); // own cells first, then opponent empties (deduped)
    own
}

/// Append the items of `src` onto `dst`, skipping any already present, so `dst`
/// keeps insertion order with no duplicates. `src` is consumed.
fn push_unique(dst: &mut Vec<HexCoord>, src: Vec<HexCoord>) {
    for c in src {
        if !dst.contains(&c) {
            dst.push(c);
        }
    }
}

// --- shared PyO3 diagnostic builder ------------------------------------------
//
// One builder, used by every lineage's `*_threat_analysis` pyfunction, so the
// diagnostic surface is identical across lineages by construction (the parity
// tests compare two pyfunctions that funnel through this single function).

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

/// Build the phase-aware threat-analysis diagnostic dict for a live engine state.
/// Drives the TSS regression fixtures from Python and lets self-play instrument
/// how often the override/injection fire.
#[cfg(feature = "python")]
pub(crate) fn analysis_pydict<'py>(
    py: Python<'py>,
    state: &RustHexoState,
) -> PyResult<Bound<'py, PyDict>> {
    let a = analyze(state);
    let d = PyDict::new(py);
    d.set_item("b", a.b)?;
    d.set_item("own_win_now", a.own_win_now)?;
    d.set_item("opp_threat_count", a.opp_threat_count)?;
    d.set_item(
        "min_hitting_set",
        a.min_hitting_set.map(|v| v as i64).unwrap_or(-1),
    )?;
    d.set_item("forced_loss", a.forced_loss())?;
    d.set_item("verdict", a.verdict())?;
    let cells: Vec<(i16, i16)> = tactical_cells(state)
        .into_iter()
        .map(|c| (c.q, c.r))
        .collect();
    d.set_item("tactical_cells", cells)?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::HexCoord;

    fn c(q: i16, r: i16) -> HexCoord {
        HexCoord { q, r }
    }

    #[test]
    fn empty_sets_need_zero_hits() {
        assert_eq!(min_hitting_set(&[], 1), Some(0));
        assert_eq!(min_hitting_set(&[], 2), Some(0));
    }

    #[test]
    fn single_window_hit_by_one() {
        let sets = vec![vec![c(0, 0), c(1, 0)]];
        assert_eq!(min_hitting_set(&sets, 1), Some(1));
        assert_eq!(min_hitting_set(&sets, 2), Some(1));
    }

    #[test]
    fn two_windows_sharing_a_cell_hit_by_one() {
        // Both windows share (0,0): a single stone hits both -> k = 1.
        let sets = vec![vec![c(0, 0), c(1, 0)], vec![c(0, 0), c(5, 5)]];
        assert_eq!(min_hitting_set(&sets, 1), Some(1));
    }

    #[test]
    fn two_disjoint_windows_need_two() {
        // Disjoint empties: a single stone cannot hit both. k = 2 with budget 2,
        // but None with budget 1 (a 1-ply forced loss when B == 1).
        let sets = vec![vec![c(0, 0), c(1, 0)], vec![c(9, 9), c(10, 9)]];
        assert_eq!(min_hitting_set(&sets, 2), Some(2));
        assert_eq!(min_hitting_set(&sets, 1), None);
    }

    #[test]
    fn three_disjoint_windows_exceed_budget_two() {
        let sets = vec![
            vec![c(0, 0), c(1, 0)],
            vec![c(9, 9), c(10, 9)],
            vec![c(-5, -5), c(-4, -5)],
        ];
        assert_eq!(min_hitting_set(&sets, 2), None);
    }

    #[test]
    fn unhittable_already_won_window_is_a_miss() {
        // An empty set models a count-6 (already won) window: never hittable.
        let sets = vec![vec![c(0, 0)], vec![]];
        assert_eq!(min_hitting_set(&sets, 2), None);
    }

    #[test]
    fn verdict_prefers_win_over_loss() {
        let win = ThreatAnalysis {
            b: 1,
            own_win_now: true,
            min_hitting_set: None,
            opp_threat_count: 1,
        };
        assert_eq!(win.verdict(), Some(1.0));
        assert!(!win.forced_loss());

        let loss = ThreatAnalysis {
            b: 1,
            own_win_now: false,
            min_hitting_set: None,
            opp_threat_count: 2,
        };
        assert_eq!(loss.verdict(), Some(-1.0));
        assert!(loss.forced_loss());

        let quiet = ThreatAnalysis {
            b: 1,
            own_win_now: false,
            min_hitting_set: Some(1),
            opp_threat_count: 1,
        };
        assert_eq!(quiet.verdict(), None);
        assert!(!quiet.forced_loss());
    }
}

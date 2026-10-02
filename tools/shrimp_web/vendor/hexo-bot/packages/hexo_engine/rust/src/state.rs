//! Game state and phase-aware move application.
//!
//! This is the heart of the rule engine. Hexo turns are represented
//! autoregressively:
//! - `Opening`: Player 0 places the center stone.
//! - `FirstStone`: current player places the first stone of a normal turn.
//! - `SecondStone`: the same player places the second stone, then turn passes.
//!
//! A win is checked after every single placement. If the first stone of a
//! two-stone turn wins, the second stone is never played.

use super::board::{Board, BoardDelta};
use super::coord::HexCoord;
use super::error::{MoveError, StateLoadError};
use super::legal::{pack_coord, PackedCoord};
use super::rules::is_legal_placement;
use super::snapshot::{StateSnapshot, HEXO_STATE_SNAPSHOT_VERSION};
use super::tactics::WindowUpdate;
use serde::{Deserialize, Serialize};

/// Player identifier and stone owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Player {
    Player0,
    Player1,
}

impl Player {
    /// Return the opponent.
    pub fn other(self) -> Self {
        match self {
            Self::Player0 => Self::Player1,
            Self::Player1 => Self::Player0,
        }
    }

    /// Stable zero-based index for arrays and tensors.
    pub fn index(self) -> usize {
        match self {
            Self::Player0 => 0,
            Self::Player1 => 1,
        }
    }
}

/// Where the current player is inside the autoregressive turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TurnPhase {
    /// Game start. Only Player 0 at `(0, 0)` is legal.
    Opening,
    /// First placement of a normal two-stone turn.
    FirstStone,
    /// Second placement of the same turn; stores the first coordinate so the
    /// same cell cannot be reused and encoders can mark it.
    SecondStone { first: HexCoord },
}

/// One single-stone action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Placement {
    pub coord: HexCoord,
}

/// Terminal result. Hexo has no normal draw under the current rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameOutcome {
    /// Winning player.
    pub winner: Player,
    /// Number of stones placed when the game ended.
    pub placements: u32,
}

/// Flat history record for encoders and training samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementRecord {
    /// Player who placed the stone.
    pub player: Player,
    /// Coordinate that was placed.
    pub coord: HexCoord,
    /// Phase before the stone was placed.
    pub phase: TurnPhase,
    /// One-based placement count after this stone is applied.
    pub placement_index: u32,
}

/// Human-sized record of the most recent logical turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveRecord {
    /// Player who took the turn.
    pub player: Player,
    /// One coordinate for opening, two coordinates for a full normal turn.
    pub placements: Vec<HexCoord>,
}

/// Complete Hexo game state.
#[derive(Clone, Debug)]
pub struct HexoState {
    /// Sparse unlimited board.
    board: Board,
    /// Player who chooses the next placement.
    current_player: Player,
    /// Current point in the opening/first/second placement sequence.
    phase: TurnPhase,
    /// Total number of stones placed.
    placements_made: u32,
    /// Set once a player has six in a line.
    terminal: Option<GameOutcome>,
    /// Most recent logical turn progress.
    last_turn: Option<MoveRecord>,
    /// Full single-placement history for encoding recent stones.
    placement_history: Vec<PlacementRecord>,
}

/// Summary returned after applying one placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyResult {
    /// Coordinate that was placed.
    pub placed: HexCoord,
    /// Player who placed the stone.
    pub player: Player,
    /// Phase before applying the placement.
    pub phase_before: TurnPhase,
    /// Phase after applying the placement. Unchanged if the move ended game.
    pub phase_after: TurnPhase,
    /// Terminal outcome if this placement won immediately.
    pub outcome: Option<GameOutcome>,
    /// Windows changed by this placement plus any threat/win windows.
    pub window_update: WindowUpdate,
}

/// State and board changes made by one placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyDelta {
    board: BoardDelta,
    previous_current_player: Player,
    previous_phase: TurnPhase,
    previous_placements_made: u32,
    previous_terminal: Option<GameOutcome>,
    previous_last_turn: Option<MoveRecord>,
    previous_history_len: usize,
}

impl Default for HexoState {
    fn default() -> Self {
        Self::new()
    }
}

impl HexoState {
    /// Create the initial empty game state.
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            current_player: Player::Player0,
            phase: TurnPhase::Opening,
            placements_made: 0,
            terminal: None,
            last_turn: None,
            placement_history: Vec::new(),
        }
    }

    /// Read-only access to board occupancy.
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Player who must choose the next single placement.
    pub fn current_player(&self) -> Player {
        self.current_player
    }

    /// Current turn phase.
    pub fn phase(&self) -> TurnPhase {
        self.phase
    }

    /// Total stones placed so far.
    pub fn placements_made(&self) -> u32 {
        self.placements_made
    }

    /// Terminal result, if the game has ended.
    pub fn terminal(&self) -> Option<GameOutcome> {
        self.terminal
    }

    /// True once no more moves should be generated.
    pub fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    /// Most recent logical turn progress.
    pub fn last_turn(&self) -> Option<&MoveRecord> {
        self.last_turn.as_ref()
    }

    /// Complete single-placement history.
    pub fn placement_history(&self) -> &[PlacementRecord] {
        &self.placement_history
    }

    /// Number of legal single-stone moves in the current state.
    pub fn legal_move_count(&self) -> usize {
        if self.terminal.is_some() {
            return 0;
        }

        match self.phase {
            TurnPhase::Opening => usize::from(self.board.is_cell_empty(HexCoord::ZERO)),
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. } => self.board.legal_moves().len(),
        }
    }

    /// Fill `out` with deterministic legal single-stone move coordinates.
    pub fn write_legal_moves(&self, out: &mut Vec<HexCoord>) {
        out.clear();

        if self.terminal.is_some() {
            return;
        }

        match self.phase {
            TurnPhase::Opening => {
                if self.board.is_cell_empty(HexCoord::ZERO) {
                    out.push(HexCoord::ZERO);
                }
            }
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. } => {
                self.board.legal_moves().write_coords(out);
            }
        }
    }

    /// Fill `out` with deterministic compact legal action IDs.
    pub fn write_legal_action_ids(&self, out: &mut Vec<PackedCoord>) {
        out.clear();

        if self.terminal.is_some() {
            return;
        }

        match self.phase {
            TurnPhase::Opening => {
                if self.board.is_cell_empty(HexCoord::ZERO) {
                    out.push(pack_coord(HexCoord::ZERO));
                }
            }
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. } => {
                self.board.legal_moves().write_action_ids(out);
            }
        }
    }

    /// Export a compact snapshot that can be passed to `load_state`.
    pub fn snapshot(&self) -> StateSnapshot {
        StateSnapshot::new(
            self.placement_history
                .iter()
                .map(|record| record.coord)
                .collect(),
        )
    }

    /// Append a single-stone history entry after placement succeeds.
    fn push_history(&mut self, player: Player, coord: HexCoord, phase: TurnPhase) {
        self.placement_history.push(PlacementRecord {
            player,
            coord,
            phase,
            placement_index: self.placements_made,
        });
    }

    fn record_turn_progress(&mut self, player: Player, coord: HexCoord, phase: TurnPhase) {
        let placements = match phase {
            TurnPhase::Opening | TurnPhase::FirstStone => vec![coord],
            TurnPhase::SecondStone { first } => vec![first, coord],
        };
        self.last_turn = Some(MoveRecord { player, placements });
    }

    /// Apply one placement and return an explicit undo delta.
    ///
    /// This is the engine's MCTS hot path: the model crates
    /// (hexo_models/dense_cnn, hexo_models/hexgt, hexgnn) drive search via
    /// apply/undo on capsule-cloned states. Note `previous_last_turn` clones a
    /// heap Vec on every placement purely to support `undo`.
    pub fn apply_with_delta(
        &mut self,
        placement: Placement,
    ) -> Result<(ApplyResult, ApplyDelta), MoveError> {
        is_legal_placement(self, placement.coord)?;

        let previous_current_player = self.current_player;
        let previous_phase = self.phase;
        let previous_placements_made = self.placements_made;
        let previous_terminal = self.terminal;
        let previous_last_turn = self.last_turn.clone();
        let previous_history_len = self.placement_history.len();

        let player = self.current_player;
        let phase_before = self.phase;
        let (window_update, board_delta) = self.board.place_with_delta(placement.coord, player)?;
        self.placements_made += 1;
        self.push_history(player, placement.coord, phase_before);
        self.record_turn_progress(player, placement.coord, phase_before);

        let outcome = if window_update.has_win() {
            let outcome = GameOutcome {
                winner: player,
                placements: self.placements_made,
            };
            self.terminal = Some(outcome);
            Some(outcome)
        } else {
            match phase_before {
                TurnPhase::Opening => {
                    // Opening is a special one-stone turn by Player 0. After it,
                    // Player 1 starts the first normal two-stone turn.
                    self.current_player = Player::Player1;
                    self.phase = TurnPhase::FirstStone;
                }
                TurnPhase::FirstStone => {
                    // The same player remains to place the second stone.
                    self.phase = TurnPhase::SecondStone {
                        first: placement.coord,
                    };
                }
                TurnPhase::SecondStone { .. } => {
                    // A normal two-stone turn is complete, so control passes.
                    self.current_player = player.other();
                    self.phase = TurnPhase::FirstStone;
                }
            }
            None
        };

        let result = ApplyResult {
            placed: placement.coord,
            player,
            phase_before,
            phase_after: self.phase,
            outcome,
            window_update,
        };
        let delta = ApplyDelta {
            board: board_delta,
            previous_current_player,
            previous_phase,
            previous_placements_made,
            previous_terminal,
            previous_last_turn,
            previous_history_len,
        };

        Ok((result, delta))
    }

    /// Restore the exact state that existed before `apply_with_delta`.
    pub fn undo(&mut self, delta: ApplyDelta) {
        self.board.undo_place(delta.board);
        self.current_player = delta.previous_current_player;
        self.phase = delta.previous_phase;
        self.placements_made = delta.previous_placements_made;
        self.terminal = delta.previous_terminal;
        self.last_turn = delta.previous_last_turn;
        self.placement_history.truncate(delta.previous_history_len);
    }
}

/// Build authoritative state by replaying a validated startup/resume snapshot.
pub fn load_state(snapshot: &StateSnapshot) -> Result<HexoState, StateLoadError> {
    if snapshot.rules_version != HEXO_STATE_SNAPSHOT_VERSION {
        return Err(StateLoadError::UnsupportedSnapshotVersion {
            found: snapshot.rules_version,
            expected: HEXO_STATE_SNAPSHOT_VERSION,
        });
    }

    let mut state = HexoState::new();

    for (index, coord) in snapshot.placements.iter().copied().enumerate() {
        apply_placement(&mut state, Placement { coord })
            .map_err(|source| StateLoadError::IllegalPlacement { index, source })?;
    }

    Ok(state)
}

/// Apply one single-stone placement and advance the phase machine.
///
/// The function performs the full rule sequence:
/// 1. Validate the coordinate against the current phase.
/// 2. Place the stone for the current player.
/// 3. Record history.
/// 4. Check for an immediate six-in-line win.
/// 5. If not terminal, advance phase/current player.
pub fn apply_placement(
    state: &mut HexoState,
    placement: Placement,
) -> Result<ApplyResult, MoveError> {
    state
        .apply_with_delta(placement)
        .map(|(result, _delta)| result)
}

// --- invariant test suite: apply/undo round-trips, snapshot replay parity,
// --- incremental legal/window caches vs. slow recomputation, random games ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::coords_within_radius;
    use crate::legal::{pack_coord, unpack_coord, LEGAL_RADIUS};
    use crate::snapshot::HEXO_STATE_SNAPSHOT_VERSION;
    use crate::tactics::{Axis, WINDOWS_PER_PLACEMENT};
    use serde_json::json;
    use std::collections::HashSet;

    fn sample_state() -> HexoState {
        let mut state = HexoState::new();
        for coord in [
            HexCoord::ZERO,
            HexCoord::new(1, 0),
            HexCoord::new(2, 0),
            HexCoord::new(0, 1),
            HexCoord::new(0, 2),
        ] {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        state
    }

    fn assert_same_public_state(left: &HexoState, right: &HexoState) {
        assert_eq!(left.current_player(), right.current_player());
        assert_eq!(left.phase(), right.phase());
        assert_eq!(left.placements_made(), right.placements_made());
        assert_eq!(left.terminal(), right.terminal());
        assert_eq!(left.last_turn(), right.last_turn());
        assert_eq!(left.placement_history(), right.placement_history());
        assert_eq!(
            left.board().occupied_cells(),
            right.board().occupied_cells()
        );
        for coord in left.board().occupied_cells() {
            assert_eq!(left.board().get(*coord), right.board().get(*coord));
        }
        let mut left_legal = Vec::new();
        let mut right_legal = Vec::new();
        left.write_legal_action_ids(&mut left_legal);
        right.write_legal_action_ids(&mut right_legal);
        assert_eq!(left_legal, right_legal);
        assert_eq!(sorted_window_entries(left), sorted_window_entries(right));
    }

    fn sorted_window_entries(state: &HexoState) -> Vec<(Axis, i16, i16, u8, u8)> {
        let mut entries: Vec<_> = state
            .board()
            .windows()
            .entries()
            .map(|entry| {
                let key = entry.key();
                (
                    key.axis,
                    key.start.q,
                    key.start.r,
                    entry.mask(Player::Player0),
                    entry.mask(Player::Player1),
                )
            })
            .collect();
        entries.sort_by_key(|(axis, q, r, _, _)| (axis.index(), *q, *r));
        entries
    }

    fn assert_engine_invariants(state: &HexoState) {
        assert_board_occupancy_invariants(state);
        assert_legal_invariants(state);
        assert_window_masks_match_slow_scan(state);

        let decoded = load_state(&state.snapshot()).unwrap();
        assert_same_public_state(&decoded, state);
    }

    fn assert_board_occupancy_invariants(state: &HexoState) {
        let board = state.board();
        let occupied: HashSet<_> = board.occupied_cells().iter().copied().collect();

        assert_eq!(
            occupied.len(),
            board.occupied_cells().len(),
            "occupied list contains duplicates"
        );
        assert_eq!(
            board.debug_stones().len(),
            board.occupied_cells().len(),
            "stone map and occupied list length diverged"
        );

        for coord in board.occupied_cells() {
            assert!(
                board.get(*coord).is_some(),
                "occupied cell {:?} missing from stone map",
                coord
            );
            assert!(
                board.debug_stones().contains_key(coord),
                "occupied cell {:?} missing from debug stone map",
                coord
            );
        }

        for (coord, stone) in board.debug_stones() {
            assert!(
                occupied.contains(coord),
                "stone map coord {:?} missing from occupied list",
                coord
            );
            assert_eq!(board.get(*coord), Some(*stone));
        }
    }

    fn assert_legal_invariants(state: &HexoState) {
        let mut legal_ids = Vec::new();
        state.write_legal_action_ids(&mut legal_ids);

        if state.is_terminal() {
            assert!(
                legal_ids.is_empty(),
                "terminal states must expose no legal actions"
            );
            return;
        }

        for action_id in &legal_ids {
            let coord = unpack_coord(*action_id);
            assert!(
                state.board().is_cell_empty(coord),
                "legal action {:?} points at an occupied cell",
                coord
            );
            assert!(
                !state.board().occupied_cells().contains(&coord),
                "legal action {:?} appears in occupied list",
                coord
            );
        }

        match state.phase() {
            TurnPhase::Opening => {
                let expected = if state.board().is_cell_empty(HexCoord::ZERO) {
                    vec![pack_coord(HexCoord::ZERO)]
                } else {
                    Vec::new()
                };
                assert_eq!(legal_ids, expected);
            }
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. } => {
                let expected = recompute_non_opening_legal_ids(state);
                let actual: HashSet<_> = state.board().legal_moves().action_ids().collect();
                assert_eq!(actual, expected);
            }
        }
    }

    fn recompute_non_opening_legal_ids(state: &HexoState) -> HashSet<u32> {
        let mut expected = HashSet::new();
        for coord in state.board().occupied_cells() {
            for candidate in coords_within_radius(*coord, LEGAL_RADIUS) {
                if state.board().is_cell_empty(candidate) {
                    expected.insert(pack_coord(candidate));
                }
            }
        }
        expected
    }

    fn assert_window_masks_match_slow_scan(state: &HexoState) {
        for entry in state.board().windows().entries() {
            let key = entry.key();
            for player in [Player::Player0, Player::Player1] {
                let mut expected = 0u8;
                for (index, coord) in key.cells().into_iter().enumerate() {
                    if state.board().get(coord) == Some(player) {
                        expected |= 1u8 << index;
                    }
                }
                assert_eq!(
                    entry.mask(player),
                    expected,
                    "window mask mismatch for {:?} and {:?}",
                    key,
                    player
                );
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }

        fn index(&mut self, len: usize) -> usize {
            (self.next_u64() % len as u64) as usize
        }
    }

    fn first_legal_coord(state: &HexoState) -> HexCoord {
        let mut legal_ids = Vec::new();
        state.write_legal_action_ids(&mut legal_ids);
        unpack_coord(*legal_ids.first().expect("state should have a legal action"))
    }

    fn player0_win_sequence() -> [HexCoord; 12] {
        [
            HexCoord::new(0, 0),
            HexCoord::new(0, 1),
            HexCoord::new(0, 2),
            HexCoord::new(1, 0),
            HexCoord::new(2, 0),
            HexCoord::new(1, 1),
            HexCoord::new(1, 2),
            HexCoord::new(3, 0),
            HexCoord::new(4, 0),
            HexCoord::new(2, 1),
            HexCoord::new(2, 2),
            HexCoord::new(5, 0),
        ]
    }

    #[test]
    fn snapshot_uses_canonical_shape() {
        let state = sample_state();
        let snapshot = state.snapshot();
        let value = serde_json::to_value(&snapshot).unwrap();

        assert_eq!(value["rules_version"], json!(HEXO_STATE_SNAPSHOT_VERSION));
        assert_eq!(value["placements"].as_array().unwrap().len(), 5);
        assert!(value.get("board").is_none());
    }

    #[test]
    fn load_state_replays_snapshot() {
        let state = sample_state();
        let snapshot = state.snapshot();

        let decoded = load_state(&snapshot).unwrap();

        assert_same_public_state(&decoded, &state);
    }

    #[test]
    fn load_state_rejects_unsupported_snapshot_version() {
        let state = sample_state();
        let mut snapshot = state.snapshot();
        snapshot.rules_version = HEXO_STATE_SNAPSHOT_VERSION + 1;

        assert!(matches!(
            load_state(&snapshot),
            Err(StateLoadError::UnsupportedSnapshotVersion { .. })
        ));
    }

    #[test]
    fn load_state_rejects_illegal_snapshot_placement() {
        let state = sample_state();
        let mut snapshot = state.snapshot();
        snapshot.placements[1] = snapshot.placements[0];

        assert!(matches!(
            load_state(&snapshot),
            Err(StateLoadError::IllegalPlacement {
                index: 1,
                source: MoveError::Occupied(coord),
            }) if coord == HexCoord::ZERO
        ));
    }

    #[test]
    fn apply_then_undo_opening_restores_fresh_state() {
        let mut state = HexoState::new();
        let before = state.clone();

        let (_result, delta) = state
            .apply_with_delta(Placement {
                coord: HexCoord::ZERO,
            })
            .unwrap();
        assert_eq!(state.placements_made(), 1);

        state.undo(delta);

        assert_same_public_state(&state, &before);
        assert_engine_invariants(&state);
    }

    #[test]
    fn apply_then_undo_first_stone_restores_phase_and_legal_moves() {
        let mut state = HexoState::new();
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::ZERO,
            },
        )
        .unwrap();
        let before = state.clone();
        let coord = first_legal_coord(&state);

        let (_result, delta) = state.apply_with_delta(Placement { coord }).unwrap();
        assert!(matches!(state.phase(), TurnPhase::SecondStone { first } if first == coord));

        state.undo(delta);

        assert_same_public_state(&state, &before);
        assert_engine_invariants(&state);
    }

    #[test]
    fn apply_then_undo_second_stone_restores_turn_progress() {
        let mut state = HexoState::new();
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::ZERO,
            },
        )
        .unwrap();
        let first = first_legal_coord(&state);
        apply_placement(&mut state, Placement { coord: first }).unwrap();
        let before = state.clone();
        let second = first_legal_coord(&state);

        let (_result, delta) = state.apply_with_delta(Placement { coord: second }).unwrap();
        assert_eq!(state.current_player(), Player::Player0);
        assert_eq!(state.phase(), TurnPhase::FirstStone);

        state.undo(delta);

        assert_same_public_state(&state, &before);
        assert_engine_invariants(&state);
    }

    #[test]
    fn apply_then_undo_winning_placement_restores_non_terminal_state() {
        let mut state = HexoState::new();
        let sequence = player0_win_sequence();
        for coord in &sequence[..sequence.len() - 1] {
            apply_placement(&mut state, Placement { coord: *coord }).unwrap();
        }
        let before = state.clone();

        let (result, delta) = state
            .apply_with_delta(Placement {
                coord: *sequence.last().unwrap(),
            })
            .unwrap();
        assert!(result.outcome.is_some());
        assert!(state.is_terminal());

        state.undo(delta);

        assert!(!state.is_terminal());
        assert_same_public_state(&state, &before);
        assert_engine_invariants(&state);
    }

    #[test]
    fn random_apply_undo_round_trips_public_state() {
        let mut state = HexoState::new();
        let mut rng = Lcg::new(0x1234_5678_90ab_cdef);

        for _ in 0..96 {
            let mut legal_ids = Vec::new();
            state.write_legal_action_ids(&mut legal_ids);
            if legal_ids.is_empty() {
                break;
            }

            let action_id = legal_ids[rng.index(legal_ids.len())];
            let coord = unpack_coord(action_id);
            let before = state.clone();
            let (_result, delta) = state.apply_with_delta(Placement { coord }).unwrap();
            assert_engine_invariants(&state);

            state.undo(delta);
            assert_same_public_state(&state, &before);
            assert_engine_invariants(&state);

            apply_placement(&mut state, Placement { coord }).unwrap();
            assert_engine_invariants(&state);
            if state.is_terminal() {
                break;
            }
        }
    }

    #[test]
    fn random_legal_games_preserve_incremental_cache_invariants() {
        for game_index in 0..12u64 {
            let mut state = HexoState::new();
            let mut rng = Lcg::new(0x9e37_79b9_7f4a_7c15 ^ game_index);

            assert_engine_invariants(&state);

            for _ in 0..80 {
                let mut legal_ids = Vec::new();
                state.write_legal_action_ids(&mut legal_ids);
                if legal_ids.is_empty() {
                    assert!(state.is_terminal());
                    break;
                }

                let action_id = legal_ids[rng.index(legal_ids.len())];
                let result = apply_placement(
                    &mut state,
                    Placement {
                        coord: unpack_coord(action_id),
                    },
                )
                .unwrap();

                assert_eq!(
                    result.window_update.changed.len(),
                    WINDOWS_PER_PLACEMENT,
                    "every placement must touch exactly {WINDOWS_PER_PLACEMENT} windows"
                );
                assert_engine_invariants(&state);

                if state.is_terminal() {
                    break;
                }
            }
        }
    }

    #[test]
    fn terminal_snapshots_reject_additional_placements() {
        let mut state = HexoState::new();
        for coord in player0_win_sequence() {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }

        assert!(state.is_terminal());
        let mut snapshot = state.snapshot();
        let index = snapshot.placements.len();
        snapshot.placements.push(HexCoord::new(6, 0));

        assert!(matches!(
            load_state(&snapshot),
            Err(StateLoadError::IllegalPlacement {
                index: error_index,
                source: MoveError::TerminalState,
            }) if error_index == index
        ));
    }
}

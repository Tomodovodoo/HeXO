//! Deterministic state identity for model-side caches.
//!
//! The engine remains the source of truth for rules and mutation. This helper
//! derives a history-sensitive identity from the engine's public read-only
//! state, which is enough for dense-cnn evaluator caching without adding hash
//! fields to core state.
//!
//! Callers: the model-side MCTS evaluator caches (see
//! `packages/shrimp/rust/src/cache.rs`). No Python surface.
//!
//! Stability contract: the hash is process-internal (cache keys only) and is
//! never persisted, so the mixing constants may change between builds without
//! a migration. Determinism within one process is what matters: transposition
//! lookups during a single self-play/eval session.

use hexo_engine::{GameOutcome, HexCoord, HexoState, PlacementRecord, Player, TurnPhase};

/// Exact deterministic identity for model-visible state.
pub type StateHash = u64;

const HASH_EMPTY_HISTORY: StateHash = 0x8a31_7c91_3e25_ba77;
const HASH_FINAL_TAG: StateHash = 0xd6e8_feb8_6659_fd93;

/// Hash a full model-visible state through public engine accessors.
///
/// Placement order is included because current dense-cnn inputs use recency
/// planes. Board-equivalent states reached in a different order must not share
/// cached neural evaluations.
pub fn hash_state(state: &HexoState) -> StateHash {
    let history_hash = state
        .placement_history()
        .iter()
        .copied()
        .fold(HASH_EMPTY_HISTORY, append_history_hash);
    finalize_state_hash(
        history_hash,
        state.current_player(),
        state.phase(),
        state.terminal(),
        state.placements_made(),
    )
}

fn append_history_hash(history_hash: StateHash, record: PlacementRecord) -> StateHash {
    mix_hash(
        history_hash ^ record_hash(record) ^ mix_hash(record.placement_index as u64 ^ 0x9e37_79b9),
    )
}

fn finalize_state_hash(
    history_hash: StateHash,
    current_player: Player,
    phase: TurnPhase,
    terminal: Option<GameOutcome>,
    placements_made: u32,
) -> StateHash {
    let mut hash = history_hash ^ HASH_FINAL_TAG;
    hash = mix_hash(hash ^ player_hash(current_player).rotate_left(7));
    hash = mix_hash(hash ^ phase_hash(phase).rotate_left(17));
    hash = mix_hash(hash ^ (placements_made as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
    hash = mix_hash(hash ^ terminal_hash(terminal).rotate_left(29));
    hash
}

fn record_hash(record: PlacementRecord) -> StateHash {
    let mut hash = coord_hash(record.coord);
    hash = mix_hash(hash ^ player_hash(record.player).rotate_left(11));
    hash = mix_hash(hash ^ phase_hash(record.phase).rotate_left(23));
    mix_hash(hash ^ (record.placement_index as u64).rotate_left(31))
}

fn terminal_hash(terminal: Option<GameOutcome>) -> StateHash {
    match terminal {
        None => 0x5a17_2f31_c0de_0011,
        Some(outcome) => {
            let mut hash = 0xa11c_e5e5_7e1e_0001;
            hash = mix_hash(hash ^ player_hash(outcome.winner));
            mix_hash(hash ^ outcome.placements as u64)
        }
    }
}

fn phase_hash(phase: TurnPhase) -> StateHash {
    match phase {
        TurnPhase::Opening => 0x1020_3040_5060_7080,
        TurnPhase::FirstStone => 0x243f_6a88_85a3_08d3,
        TurnPhase::SecondStone { first } => mix_hash(0x1319_8a2e_0370_7344 ^ coord_hash(first)),
    }
}

fn player_hash(player: Player) -> StateHash {
    match player {
        Player::Player0 => 0x3c6e_f372_fe94_f82b,
        Player::Player1 => 0xa54f_f53a_5f1d_36f1,
    }
}

fn coord_hash(coord: HexCoord) -> StateHash {
    let q = (coord.q as i32 + 32_768) as u64;
    let r = (coord.r as i32 + 32_768) as u64;
    mix_hash((q << 16) | r)
}

fn mix_hash(mut value: StateHash) -> StateHash {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{apply_placement, Placement};
    use std::collections::HashSet;

    #[test]
    fn hash_changes_on_apply_and_restores_on_undo() {
        let mut state = HexoState::new();
        let initial = hash_state(&state);
        let (_result, delta) = state
            .apply_with_delta(Placement {
                coord: HexCoord::ZERO,
            })
            .unwrap();

        assert_ne!(hash_state(&state), initial);

        state.undo(delta);

        assert_eq!(hash_state(&state), initial);
    }

    #[test]
    fn hash_distinguishes_same_board_different_history_order() {
        let mut first_order = HexoState::new();
        let mut second_order = HexoState::new();
        for coord in [HexCoord::ZERO, HexCoord::new(1, 0), HexCoord::new(0, 1)] {
            apply_placement(&mut first_order, Placement { coord }).unwrap();
        }
        for coord in [HexCoord::ZERO, HexCoord::new(0, 1), HexCoord::new(1, 0)] {
            apply_placement(&mut second_order, Placement { coord }).unwrap();
        }

        let first_board: HashSet<_> = first_order
            .board()
            .occupied_cells()
            .iter()
            .copied()
            .collect();
        let second_board: HashSet<_> = second_order
            .board()
            .occupied_cells()
            .iter()
            .copied()
            .collect();

        assert_eq!(first_board, second_board);
        assert_ne!(
            first_order.placement_history(),
            second_order.placement_history()
        );
        assert_ne!(hash_state(&first_order), hash_state(&second_order));
    }

    #[test]
    fn hash_distinguishes_mirrored_second_stone_states() {
        let mut left = HexoState::new();
        let mut right = HexoState::new();
        for (state, coord) in [
            (&mut left, HexCoord::new(1, 7)),
            (&mut right, HexCoord::new(-1, -7)),
        ] {
            apply_placement(
                state,
                Placement {
                    coord: HexCoord::ZERO,
                },
            )
            .unwrap();
            apply_placement(state, Placement { coord }).unwrap();
        }

        assert_ne!(left.placement_history(), right.placement_history());
        assert_ne!(hash_state(&left), hash_state(&right));
    }
}

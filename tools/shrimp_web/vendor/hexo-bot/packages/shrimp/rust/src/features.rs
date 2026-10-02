//! Node features (F = NUM_FEATURES = 15) computed from the engine state.
//! Index 11 is distance-to-nearest-stone; indices 13-14 are the standing-win
//! planes. A matching implementation lives in python/shrimp/features.py.

use hexo_engine::{Axis, HexCoord, HexoState as RustHexoState, Player, TurnPhase};

use crate::constants::*;
use crate::support::Support;

pub fn axis_delta(axis: Axis) -> (i16, i16) {
    match axis {
        Axis::Q => (1, 0),
        Axis::R => (0, 1),
        Axis::QR => (1, -1),
    }
}

/// (N * NUM_FEATURES) node-major feature matrix in support node order.
pub fn build_features(state: &RustHexoState, sup: &Support) -> Vec<f32> {
    let n = sup.num_nodes();
    let mut feats = vec![0f32; n * NUM_FEATURES];
    let set = |feats: &mut Vec<f32>, row: usize, plane: usize, value: f32| {
        feats[row * NUM_FEATURES + plane] = value;
    };
    let current = state.current_player();
    let placements_made = state.placements_made();

    // Stones and recency. age = placements_made - placement_index (saturating);
    // recency weight = 1/(1+age), max-accumulated per cell.
    for record in state.placement_history().iter() {
        let row = sup.row(record.coord).expect("stone missing from support");
        let (stone_plane, recency_plane) = if record.player == current {
            (F_OWN_STONE, F_OWN_RECENCY)
        } else {
            (F_OPP_STONE, F_OPP_RECENCY)
        };
        set(&mut feats, row, stone_plane, 1.0);
        let age = placements_made.saturating_sub(record.placement_index);
        let weight = 1.0 / (1.0 + age as f32);
        let offset = row * NUM_FEATURES + recency_plane;
        feats[offset] = feats[offset].max(weight);
    }

    for row in 0..n {
        let own = feats[row * NUM_FEATURES + F_OWN_STONE];
        let opp = feats[row * NUM_FEATURES + F_OPP_STONE];
        feats[row * NUM_FEATURES + F_EMPTY] = 1.0 - own - opp;
        feats[row * NUM_FEATURES + F_DIST_TO_STONE] = sup.dist[row] as f32 / DIST_SCALE;
    }
    for row in 0..sup.legal_count {
        set(&mut feats, row, F_LEGAL, 1.0);
    }

    match state.phase() {
        TurnPhase::SecondStone { first } => {
            for row in 0..n {
                set(&mut feats, row, F_PHASE_SECOND, 1.0);
            }
            let row = sup.row(first).expect("first stone missing from support");
            set(&mut feats, row, F_FIRST_STONE, 1.0);
        }
        TurnPhase::Opening | TurnPhase::FirstStone => {}
    }

    if current == Player::Player0 {
        for row in 0..n {
            set(&mut feats, row, F_PLAYER_COLOUR, 1.0);
        }
    }

    fill_hot_and_win(state, current, sup, &mut feats);
    fill_opponent_last_turn(state, current, sup, &mut feats);

    feats
}

/// Hot (count >= 4, gated placements >= 7) and standing-win (count == 5,
/// ungated) planes over single-colour windows' EMPTY cells.
fn fill_hot_and_win(
    state: &RustHexoState,
    current: Player,
    sup: &Support,
    feats: &mut Vec<f32>,
) {
    let hot_enabled = state.placements_made() >= HOT_MIN_PLACEMENTS;
    for entry in state.board().windows().entries() {
        let p0 = entry.mask(Player::Player0);
        let p1 = entry.mask(Player::Player1);
        let c0 = (p0 as u32).count_ones();
        let c1 = (p1 as u32).count_ones();
        if (c0 > 0 && c1 > 0) || (c0 + c1) < HOT_MIN_COUNT {
            continue;
        }
        let count = c0 + c1;
        let owner = if c0 > 0 { Player::Player0 } else { Player::Player1 };
        let key = entry.key();
        let (dq, dr) = axis_delta(key.axis);
        let union = p0 | p1;
        for i in 0..WINDOW_LEN {
            if (union >> i) & 1 != 0 {
                continue;
            }
            let coord = HexCoord {
                q: key.start.q + dq * i as i16,
                r: key.start.r + dr * i as i16,
            };
            let row = sup
                .row(coord)
                .expect("window empty cell missing from support");
            if count == WIN_NOW_COUNT {
                let plane = if owner == current { F_OWN_WIN_NOW } else { F_OPP_WIN_NOW };
                feats[row * NUM_FEATURES + plane] = 1.0;
            }
            if hot_enabled {
                let plane = if owner == current { F_OWN_HOT } else { F_OPP_HOT };
                feats[row * NUM_FEATURES + plane] = 1.0;
            }
        }
    }
}

/// Cells of the opponent's most recent full turn.
fn fill_opponent_last_turn(
    state: &RustHexoState,
    current: Player,
    sup: &Support,
    feats: &mut Vec<f32>,
) {
    let opponent = current.other();
    for record in state.placement_history().iter().rev() {
        if record.player != opponent {
            continue;
        }
        match record.phase {
            TurnPhase::SecondStone { first } => {
                for coord in [first, record.coord] {
                    let row = sup.row(coord).expect("last-turn cell missing from support");
                    feats[row * NUM_FEATURES + F_OPP_LAST_TURN] = 1.0;
                }
                return;
            }
            TurnPhase::Opening => {
                let row = sup
                    .row(record.coord)
                    .expect("last-turn cell missing from support");
                feats[row * NUM_FEATURES + F_OPP_LAST_TURN] = 1.0;
                return;
            }
            TurnPhase::FirstStone => {}
        }
    }
}

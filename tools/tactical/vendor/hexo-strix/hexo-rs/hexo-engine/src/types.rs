use std::fmt;

use rustc_hash::{FxHashMap, FxHashSet};

/// Axial coordinate (q, r) on the hex grid.
pub type Coord = (i32, i32);

/// Sparse map from board coordinate to owning player.
///
/// Uses `rustc_hash`'s Fx hasher rather than the standard SipHash: coordinate
/// keys are small and non-adversarial, so the faster hash is a large win on
/// the win-check / legal-move / graph-build hot paths (see `benches/`).
pub type StoneMap = FxHashMap<Coord, Player>;

/// Set of board coordinates, Fx-hashed for the same reason as [`StoneMap`].
pub type CoordSet = FxHashSet<Coord>;

/// The 6 neighbor directions in axial coordinates.
pub const HEX_DIRS: [Coord; 6] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
];

/// The 3 win-detection axes (one direction per axis).
pub const WIN_AXES: [Coord; 3] = [
    (1, 0),
    (0, 1),
    (1, -1),
];

/// A player in the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    P1,
    P2,
}

impl Player {
    /// Returns the opposing player.
    pub fn opponent(self) -> Self {
        match self {
            Player::P1 => Player::P2,
            Player::P2 => Player::P1,
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Player::P1 => write!(f, "P1"),
            Player::P2 => write!(f, "P2"),
        }
    }
}

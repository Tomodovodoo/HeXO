//! Sparse board storage.
//!
//! Hexo has no fixed board bounds, so the board stores only occupied cells.
//! A hash map gives O(1)-ish lookup by coordinate, while `occupied` preserves a
//! compact list for legal-cell generation, encoding, and board summaries.

use super::coord::HexCoord;
use super::error::MoveError;
use super::legal::{LegalMoveDelta, LegalMoveStore};
use super::state::Player;
use super::tactics::{WindowStore, WindowStoreDelta, WindowUpdate};
use ahash::AHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// In the game engine, a stone is just the owning player.
pub type Stone = Player;

/// Sparse representation of all placed stones.
#[derive(Clone, Debug, Default)]
pub struct Board {
    /// Coordinate -> owner lookup for legality and window updates.
    stones: AHashMap<HexCoord, Stone>,
    /// Placement coordinates in insertion order.
    occupied: Vec<HexCoord>,
    /// Incrementally maintained six-cell window state.
    windows: WindowStore,
    /// Incrementally maintained legal non-opening placements.
    legal: LegalMoveStore,
}

/// Board-level changes made by one placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoardDelta {
    coord: HexCoord,
    previous_stone: Option<Stone>,
    previous_occupied_len: usize,
    legal: LegalMoveDelta,
    windows: WindowStoreDelta,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct BoardStone {
    coord: HexCoord,
    stone: Stone,
}

impl Board {
    /// Create an empty board.
    pub fn new() -> Self {
        Self::default()
    }

    /// True when the board has no stones.
    pub fn is_empty(&self) -> bool {
        self.occupied.is_empty()
    }

    /// True when no stone occupies `coord`.
    pub fn is_cell_empty(&self, coord: HexCoord) -> bool {
        !self.stones.contains_key(&coord)
    }

    /// Return the owner of a cell, if occupied.
    pub fn get(&self, coord: HexCoord) -> Option<Stone> {
        self.stones.get(&coord).copied()
    }

    /// Place one stone without checking higher-level turn rules.
    ///
    /// Callers should validate game legality before calling this method. This
    /// method only protects the board invariant that a cell cannot be occupied
    /// twice.
    pub(crate) fn place(
        &mut self,
        coord: HexCoord,
        stone: Stone,
    ) -> Result<WindowUpdate, MoveError> {
        let (update, _) = self.place_with_delta(coord, stone)?;
        Ok(update)
    }

    /// Place one stone and return the delta needed to undo the board mutation.
    pub(crate) fn place_with_delta(
        &mut self,
        coord: HexCoord,
        stone: Stone,
    ) -> Result<(WindowUpdate, BoardDelta), MoveError> {
        if !self.is_cell_empty(coord) {
            return Err(MoveError::Occupied(coord));
        }
        let previous_stone = self.stones.insert(coord, stone);
        let previous_occupied_len = self.occupied.len();
        self.occupied.push(coord);
        let legal = self.update_legal_for_placement_with_delta(coord);
        let (window_update, windows) = self.windows.update_for_placement_with_delta(coord, stone);
        Ok((
            window_update,
            BoardDelta {
                coord,
                previous_stone,
                previous_occupied_len,
                legal,
                windows,
            },
        ))
    }

    pub(crate) fn undo_place(&mut self, delta: BoardDelta) {
        self.occupied.truncate(delta.previous_occupied_len);
        if let Some(stone) = delta.previous_stone {
            self.stones.insert(delta.coord, stone);
        } else {
            self.stones.remove(&delta.coord);
        }
        self.legal.restore_delta(delta.legal);
        self.windows.restore_delta(delta.windows);
    }

    /// Incremental threat/win window state.
    pub fn windows(&self) -> &WindowStore {
        &self.windows
    }

    /// Incremental legal non-opening move store.
    pub fn legal_moves(&self) -> &LegalMoveStore {
        &self.legal
    }

    /// All occupied coordinates in placement order.
    pub fn occupied_cells(&self) -> &[HexCoord] {
        &self.occupied
    }

    /// Number of stones currently on the board.
    pub fn len(&self) -> usize {
        self.occupied.len()
    }

    /// Axis-aligned axial bounds around occupied cells.
    ///
    /// This is not a playable board boundary; it was meant as a convenience
    /// for encoding and diagnostics, but the actual encoders (e.g.
    /// hexo_models/dense_cnn rust/src/encoding.rs) compute their own crop from
    /// `occupied_cells()` instead.
    // UNUSED(2026-06-12): no references found in packages/tests/scripts —
    // grep for `.bounds()` across all crates returns zero callers.
    pub fn bounds(&self) -> Option<(HexCoord, HexCoord)> {
        let first = *self.occupied.first()?;
        let mut min_q = first.q;
        let mut max_q = first.q;
        let mut min_r = first.r;
        let mut max_r = first.r;

        for coord in &self.occupied {
            min_q = min_q.min(coord.q);
            max_q = max_q.max(coord.q);
            min_r = min_r.min(coord.r);
            max_r = max_r.max(coord.r);
        }

        Some((
            HexCoord { q: min_q, r: min_r },
            HexCoord { q: max_q, r: max_r },
        ))
    }

    fn update_legal_for_placement_with_delta(&mut self, coord: HexCoord) -> LegalMoveDelta {
        let stones = &self.stones;
        self.legal
            .update_for_placement_with_delta(coord, |candidate| !stones.contains_key(&candidate))
    }

    #[cfg(test)]
    pub(crate) fn debug_stones(&self) -> &AHashMap<HexCoord, Stone> {
        &self.stones
    }
}

impl Serialize for Board {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let placements: Vec<BoardStone> = self
            .occupied
            .iter()
            .filter_map(|coord| {
                self.get(*coord).map(|stone| BoardStone {
                    coord: *coord,
                    stone,
                })
            })
            .collect();
        placements.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Board {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let placements = Vec::<BoardStone>::deserialize(deserializer)?;
        let mut board = Self::new();
        for placement in placements {
            board
                .place(placement.coord, placement.stone)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(board)
    }
}

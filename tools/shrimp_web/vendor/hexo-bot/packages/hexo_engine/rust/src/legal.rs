//! Incremental legal move storage.
//!
//! The authoritative board updates this store as stones are placed. Membership
//! checks stay hash-based for validation, while enumeration uses a deterministic
//! ordered set of compact packed coordinates.
//!
//! `pack_coord`/`unpack_coord` here are the CANONICAL action-ID encoding. It is
//! deliberately duplicated in python/hexo_engine/types.py
//! (`pack_coord_id`/`unpack_coord_id`) and in the frontend JS
//! (hexo_frontend/static/app.js, offset 32768); the IDs are persisted in
//! training .npz shards and .hxr records, so none of the three may diverge.
//! tests/test_hexo_engine_rust_bridge.py is the cross-language check.

use super::coord::{coords_within_radius, HexCoord};
use ahash::AHashSet;

/// Maximum distance from any existing stone for non-opening placements.
pub const LEGAL_RADIUS: i16 = 8;

/// Compact action identifier preserving signed `(q, r)` sort order.
///
/// Coordinates are offset into unsigned 16-bit lanes, then packed as
/// `q_offset << 16 | r_offset`. Raw integer ordering therefore matches
/// deterministic `(q, r)` ordering.
pub type PackedCoord = u32;

const COORD_OFFSET: i32 = 1 << 15;
const COORD_MASK: i32 = 0xffff;

/// Convert a board coordinate into a compact legal-action ID.
pub const fn pack_coord(coord: HexCoord) -> PackedCoord {
    let q = (coord.q as i32 + COORD_OFFSET) as u32;
    let r = (coord.r as i32 + COORD_OFFSET) as u32;
    (q << 16) | r
}

/// Convert a compact legal-action ID back into a board coordinate.
pub const fn unpack_coord(action_id: PackedCoord) -> HexCoord {
    let q = ((action_id >> 16) as i32 - COORD_OFFSET) as i16;
    let r = ((action_id as i32 & COORD_MASK) - COORD_OFFSET) as i16;
    HexCoord { q, r }
}

/// Incrementally maintained legal non-opening moves.
#[derive(Clone, Debug, Default)]
pub struct LegalMoveStore {
    membership: AHashSet<PackedCoord>,
    ordered: Vec<PackedCoord>,
    version: u64,
}

/// Incremental legal-move changes made by one placement.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LegalMoveDelta {
    inserted: Vec<PackedCoord>,
    removed: Vec<PackedCoord>,
    previous_version: u64,
}

impl LegalMoveStore {
    /// Create an empty legal move store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Monotonic mutation version for cache users.
    // UNUSED(2026-06-12): no `.version()` caller exists anywhere in
    // packages/tests/scripts — the advertised "cache users" never materialized.
    // The underlying counter is still maintained (and restored by
    // `restore_delta`), so only this accessor is dead.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Number of legal non-opening moves.
    pub fn len(&self) -> usize {
        self.membership.len()
    }

    /// True when no non-opening moves are legal.
    pub fn is_empty(&self) -> bool {
        self.membership.is_empty()
    }

    /// True when `coord` is currently legal.
    pub fn contains(&self, coord: HexCoord) -> bool {
        self.membership.contains(&pack_coord(coord))
    }

    /// Iterate deterministic packed legal actions.
    pub fn action_ids(&self) -> impl Iterator<Item = PackedCoord> + '_ {
        self.ordered.iter().copied()
    }

    /// Iterate deterministic legal coordinates.
    pub fn coords(&self) -> impl Iterator<Item = HexCoord> + '_ {
        self.action_ids().map(unpack_coord)
    }

    /// Copy deterministic packed legal actions into `out`.
    pub fn write_action_ids(&self, out: &mut Vec<PackedCoord>) {
        out.clear();
        out.reserve(self.ordered.len());
        out.extend(self.action_ids());
    }

    /// Copy deterministic legal coordinates into `out`.
    pub fn write_coords(&self, out: &mut Vec<HexCoord>) {
        out.clear();
        out.reserve(self.ordered.len());
        out.extend(self.coords());
    }

    /// Update legal moves after a stone is placed.
    pub fn update_for_placement(
        &mut self,
        coord: HexCoord,
        mut is_cell_empty: impl FnMut(HexCoord) -> bool,
    ) {
        let _ = self.update_for_placement_with_delta(coord, &mut is_cell_empty);
    }

    /// Update legal moves after a stone is placed and return an undo delta.
    pub(crate) fn update_for_placement_with_delta(
        &mut self,
        coord: HexCoord,
        mut is_cell_empty: impl FnMut(HexCoord) -> bool,
    ) -> LegalMoveDelta {
        let mut delta = LegalMoveDelta {
            previous_version: self.version,
            ..LegalMoveDelta::default()
        };
        let mut changed = self.remove_recorded(coord, &mut delta.removed);

        for candidate in coords_within_radius(coord, LEGAL_RADIUS) {
            if is_cell_empty(candidate) {
                changed |= self.insert_recorded(candidate, &mut delta.inserted);
            }
        }

        if changed {
            self.version = self.version.wrapping_add(1);
        }

        delta
    }

    pub(crate) fn restore_delta(&mut self, delta: LegalMoveDelta) {
        for action_id in delta.inserted {
            self.remove(unpack_coord(action_id));
        }
        for action_id in delta.removed {
            self.insert(unpack_coord(action_id));
        }
        self.version = delta.previous_version;
    }

    fn insert(&mut self, coord: HexCoord) -> bool {
        let action_id = pack_coord(coord);
        if !self.membership.insert(action_id) {
            return false;
        }
        let index = self
            .ordered
            .binary_search(&action_id)
            .unwrap_or_else(|insertion_index| insertion_index);
        self.ordered.insert(index, action_id);
        true
    }

    fn insert_recorded(&mut self, coord: HexCoord, inserted: &mut Vec<PackedCoord>) -> bool {
        let action_id = pack_coord(coord);
        if self.insert(coord) {
            inserted.push(action_id);
            true
        } else {
            false
        }
    }

    fn remove(&mut self, coord: HexCoord) -> bool {
        let action_id = pack_coord(coord);
        if !self.membership.remove(&action_id) {
            return false;
        }
        if let Ok(index) = self.ordered.binary_search(&action_id) {
            self.ordered.remove(index);
        }
        true
    }

    fn remove_recorded(&mut self, coord: HexCoord, removed: &mut Vec<PackedCoord>) -> bool {
        let action_id = pack_coord(coord);
        if self.remove(coord) {
            removed.push(action_id);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_round_trips_and_sorts_like_signed_coords() {
        let coords = [
            HexCoord::new(0, 0),
            HexCoord::new(-2, 3),
            HexCoord::new(-2, -1),
            HexCoord::new(4, -5),
        ];
        let mut packed: Vec<_> = coords.into_iter().map(pack_coord).collect();
        packed.sort_unstable();
        let decoded: Vec<_> = packed.into_iter().map(unpack_coord).collect();

        assert_eq!(
            decoded,
            vec![
                HexCoord::new(-2, -1),
                HexCoord::new(-2, 3),
                HexCoord::new(0, 0),
                HexCoord::new(4, -5),
            ]
        );
    }

    #[test]
    fn store_updates_membership_and_order_incrementally() {
        let mut store = LegalMoveStore::new();
        store.update_for_placement(HexCoord::ZERO, |coord| coord != HexCoord::ZERO);

        assert_eq!(store.len(), 216);
        assert!(!store.contains(HexCoord::ZERO));
        assert!(store.contains(HexCoord::new(8, -8)));
        assert_eq!(store.coords().next(), Some(HexCoord::new(-8, 0)));
    }
}

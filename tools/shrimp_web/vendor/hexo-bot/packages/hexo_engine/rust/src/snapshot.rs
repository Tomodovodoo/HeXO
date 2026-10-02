//! Replayable engine snapshots.
//!
//! Status: public-but-dormant. No production code serializes snapshots today;
//! the machinery serves as (a) the `rules_version` source surfaced through
//! pybridge.rs `engine_metadata()` and (b) the replay oracle for the invariant
//! tests in state.rs. Treat it as reserved API rather than dead code — .hxr
//! record replay (hexo_utils) re-applies action IDs directly instead.

use crate::coord::HexCoord;
use serde::{Deserialize, Serialize};

pub(crate) const HEXO_STATE_SNAPSHOT_VERSION: u32 = 1;

/// Serializable startup/resume snapshot.
///
/// This is intentionally much smaller than `HexoState`: it records only the
/// move coordinates needed to rebuild authoritative state through
/// `apply_placement`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub(crate) rules_version: u32,
    pub(crate) placements: Vec<HexCoord>,
}

impl StateSnapshot {
    /// Create a snapshot from a sequence of single-stone placements.
    pub fn new(placements: Vec<HexCoord>) -> Self {
        Self {
            rules_version: HEXO_STATE_SNAPSHOT_VERSION,
            placements,
        }
    }

    /// Rules version expected by this engine.
    pub fn rules_version(&self) -> u32 {
        self.rules_version
    }

    /// Single-stone placements to replay from the initial state.
    pub fn placements(&self) -> &[HexCoord] {
        &self.placements
    }
}

//! Engine error types.

use crate::coord::HexCoord;
use thiserror::Error;

/// Errors produced when a placement violates the rules.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MoveError {
    #[error("cannot apply a move to a terminal state")]
    TerminalState,
    #[error("opening placement must be at (0, 0)")]
    IllegalOpening,
    #[error("cell {0:?} is already occupied")]
    Occupied(HexCoord),
    #[error("cell {0:?} is not a legal placement")]
    IllegalPlacement(HexCoord),
    #[error("second placement cannot reuse the first placement")]
    ReusedFirstStone,
}

/// Errors produced while constructing state from a startup/resume snapshot.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum StateLoadError {
    #[error("unsupported snapshot rules version {found}; expected {expected}")]
    UnsupportedSnapshotVersion { found: u32, expected: u32 },
    #[error("snapshot placement {index} is illegal: {source}")]
    IllegalPlacement { index: usize, source: MoveError },
}

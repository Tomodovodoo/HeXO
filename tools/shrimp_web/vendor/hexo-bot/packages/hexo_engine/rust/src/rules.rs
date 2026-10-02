//! Legal placement validation.
//!
//! The board owns incremental legal move storage. This module only validates a
//! submitted coordinate against the current phase and that store.

use super::coord::HexCoord;
use super::error::MoveError;
use super::state::{HexoState, TurnPhase};

/// Validate one coordinate against the current state and phase.
pub fn is_legal_placement(state: &HexoState, coord: HexCoord) -> Result<(), MoveError> {
    if state.terminal().is_some() {
        return Err(MoveError::TerminalState);
    }

    match state.phase() {
        TurnPhase::Opening => {
            if coord == HexCoord::ZERO && state.board().is_cell_empty(coord) {
                Ok(())
            } else {
                Err(MoveError::IllegalOpening)
            }
        }
        TurnPhase::FirstStone => legal_non_opening_placement(state, coord),
        TurnPhase::SecondStone { first } => {
            if coord == first {
                return Err(MoveError::ReusedFirstStone);
            }
            legal_non_opening_placement(state, coord)
        }
    }
}

/// Shared validation for all non-opening placements.
fn legal_non_opening_placement(state: &HexoState, coord: HexCoord) -> Result<(), MoveError> {
    if !state.board().is_cell_empty(coord) {
        return Err(MoveError::Occupied(coord));
    }

    if state.board().legal_moves().contains(coord) {
        Ok(())
    } else {
        Err(MoveError::IllegalPlacement(coord))
    }
}

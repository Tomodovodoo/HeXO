//! Hexo rule engine.
//!
//! This crate owns the authoritative game state and state transitions. Model,
//! search, and sample code live outside this crate so the rules layer stays
//! small, deterministic, and easy to audit.
//!
//! Consumed two ways:
//! - As an rlib by the sibling workspace crates `hexo_models` (dense_cnn +
//!   hexgt subcrates, threats_shared.rs, plus the #[path]-included hexgnn
//!   crate) and `hexo_utils` (state_hash.rs, records.rs).
//! - With the `python` feature, as the maturin-built extension
//!   `hexo_engine._rust` (pybridge.rs) behind python/hexo_engine/api.py.
//! See README.md in this package for the full contract map.

pub mod board;
pub mod coord;
pub mod error;
pub mod legal;
pub mod rules;
pub mod snapshot;
pub mod state;
pub mod tactics;

#[cfg(feature = "python")]
pub mod pybridge;

pub use board::{Board, BoardDelta, Stone};
pub use coord::{hex_distance, HexCoord};
pub use error::{MoveError, StateLoadError};
pub use legal::{
    pack_coord, unpack_coord, LegalMoveDelta, LegalMoveStore, PackedCoord, LEGAL_RADIUS,
};
pub use rules::is_legal_placement;
pub use snapshot::StateSnapshot;
pub use state::{
    apply_placement, load_state, ApplyDelta, ApplyResult, GameOutcome, HexoState, MoveRecord,
    Placement, PlacementRecord, Player, TurnPhase,
};
pub use tactics::{
    Axis, WindowEntry, WindowKey, WindowKeyList, WindowStore, WindowStoreDelta, WindowUpdate,
    WINDOWS_PER_PLACEMENT, WINDOW_LEN,
};

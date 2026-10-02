//! Shared record, sample-store, and state utility code for Hexo.
//!
//! Model-owned Rust crates own search, encoding, and sample generation. This
//! crate keeps only stable utilities that are intentionally shared across
//! training, runner, and model packages.
//!
//! Consumers:
//! - rlib: the `hexo_models` crate (dense_cnn + hexgt subcrates) and the
//!   `hexgnn` crate import `hash_state`/`StateHash` in their
//!   `mcts_eval.rs`/`mcts_tree.rs` as the neural-eval cache key. The active
//!   dense_cnn_restnet lineage reaches it through `hexo_models._rust.dense_cnn`.
//! - cdylib: maturin builds `pybridge.rs` into the `hexo_utils._rust` Python
//!   extension (see pyproject.toml); the `.hxr` codec classes are re-exported
//!   by `python/hexo_utils/records.py` and wrapped again by
//!   `packages/hexo_runner/python/hexo_runner/records/record.py`, which is the
//!   path every production record writer/reader uses.

pub mod records;
pub mod state_hash;

#[cfg(feature = "python")]
pub mod pybridge;

pub use records::{
    AbortRecord, HexoRecord, HexoRecordEngineMetadata, HexoRecordFile, HexoRecordFileMode,
    HexoRecordGameWriter, HexoRecordPlayer, HexoRecordRef, HexoRecordStatus,
    RecordError as HexoRecordError, HEXO_RECORD_MAGIC, HEXO_RECORD_SCHEMA_VERSION,
};
pub use state_hash::{hash_state, StateHash};

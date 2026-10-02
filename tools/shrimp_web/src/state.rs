//! The engine-error conversion the vendored tree imports from hexo-bot's PyO3 state bridge.

use hexo_engine::MoveError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn move_error(error: MoveError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

//! The two names the vendored shrimp search modules take from PyO3: `PyResult` and `PyValueError::new_err`.

use std::fmt;

/// An error raised by the search, carrying its message.
#[derive(Debug, Clone)]
pub struct PyErr(pub String);

impl fmt::Display for PyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub mod prelude {
    pub use crate::PyErr;
    pub type PyResult<T> = Result<T, PyErr>;
}

pub mod exceptions {
    use crate::PyErr;

    pub struct PyValueError;

    impl PyValueError {
        pub fn new_err(message: impl Into<String>) -> PyErr {
            PyErr(message.into())
        }
    }
}

//! Errors of `lumen-hist`.

use thiserror::Error;

pub type HistResult<T> = Result<T, HistError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum HistError {
    #[error(transparent)]
    Ops(#[from] lumen_ops::OpsError),
    #[error(transparent)]
    Core(#[from] lumen_core::CoreError),
    /// A setting or an input that cannot work.
    #[error("{0}")]
    Invalid(String),
}

pub(crate) fn invalid(msg: impl Into<String>) -> HistError {
    HistError::Invalid(msg.into())
}

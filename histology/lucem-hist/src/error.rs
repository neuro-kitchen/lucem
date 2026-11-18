//! Errors of `lucem-hist`.

use thiserror::Error;

pub type HistResult<T> = Result<T, HistError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum HistError {
    #[error(transparent)]
    Ops(#[from] lucem_ops::OpsError),
    #[error(transparent)]
    Core(#[from] lucem_core::CoreError),
    /// A setting or an input that cannot work.
    #[error("{0}")]
    Invalid(String),
}

pub(crate) fn invalid(msg: impl Into<String>) -> HistError {
    HistError::Invalid(msg.into())
}

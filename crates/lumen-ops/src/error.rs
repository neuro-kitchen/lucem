//! Errors of `lumen-ops`.

use lumen_core::CoreError;
use thiserror::Error;

pub type OpsResult<T> = Result<T, OpsError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum OpsError {
    /// Error from `lumen-core` (shapes, axes, element limits, device memory budget, runtime).
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Input and output arrays alias the same underlying device buffer.
    #[error("input and output arrays must not share the same device buffer")]
    AliasedBuffers,

    /// Two arrays that must have identical shapes disagree.
    #[error("{first_name} shape {first:?} does not match {second_name} shape {second:?}")]
    ShapeMismatch {
        first_name: &'static str,
        first: Vec<usize>,
        second_name: &'static str,
        second: Vec<usize>,
    },

    /// Unexpected array dimensionality (`ndim`).
    #[error("{what} expects {expected}-D array, got {actual}-D shape {shape:?}")]
    InvalidDimension {
        what: &'static str,
        expected: &'static str,
        actual: usize,
        shape: Vec<usize>,
    },

    /// Unknown boundary padding mode name.
    #[error("boundary pad mode '{0}' is not supported (expected reflect, constant, nearest, mirror, or wrap)")]
    UnknownPadMode(String),

    /// Invalid filter origin offset for a window of given size.
    #[error("origin {origin} is out of bounds for window of size {size} (must be in {min}..={max})")]
    InvalidOrigin { origin: i32, size: usize, min: i32, max: i32 },

    /// Empty or non-finite filter weights or structuring element.
    #[error("{what}: {reason}")]
    InvalidWeights { what: &'static str, reason: &'static str },

    /// Invalid parameter value (e.g. negative sigma, zero window size, percentile outside `0..=100`).
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter { name: &'static str, reason: String },

    /// Operation is not supported on an empty array (e.g. global min/max or threshold without elements).
    #[error("{0} cannot be computed on an empty array")]
    EmptyInput(&'static str),

    /// Error building an Apache Arrow `RecordBatch` or Polars `DataFrame` in analytics.
    #[error("analytics table error: {0}")]
    Analytics(String),
}

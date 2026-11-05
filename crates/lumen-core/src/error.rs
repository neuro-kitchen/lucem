//! Errors of lumen-core.

use thiserror::Error;

pub type CoreResult<T> = Result<T, CoreError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum CoreError {
    /// Two shapes that must agree do not.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    /// An axis outside `-ndim..ndim` (numpy's message).
    #[error("axis {axis} is out of bounds for array of dimension {ndim}")]
    AxisOutOfBounds { axis: isize, ndim: usize },

    /// The same axis given twice (numpy's message).
    #[error("repeated axis {axis}")]
    RepeatedAxis { axis: isize },

    /// More elements than one device buffer may hold (kernels index with 32-bit integers).
    #[error(
        "{what} {dims:?} needs more than {limit} elements in one device buffer (kernels use 32-bit indices): use smaller tiles or batches"
    )]
    TooManyElements { what: String, dims: Vec<usize>, limit: usize },

    /// An allocation larger than the device memory budget allows.
    #[error("{what} needs {requested} bytes of device memory; the budget leaves {available} (limit {limit})")]
    OutOfDeviceMemory { what: String, requested: u64, available: u64, limit: u64 },

    /// A buffer smaller than the data written into or read from it.
    #[error("{what}: {requested} bytes do not fit a device buffer of {capacity} bytes")]
    BufferTooSmall { what: String, requested: u64, capacity: u64 },

    /// A compute runtime could not be selected or opened.
    #[error(transparent)]
    Compute(#[from] crate::device::ComputeError),
}

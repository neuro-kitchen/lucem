//! Error types for `lumen-io`.

use lumen_core::CoreError;
use thiserror::Error;

pub type IoResult<T> = Result<T, IoError>;

#[derive(Debug, Error)]
pub enum IoError {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error("I/O error: {0}")]
    StdIo(#[from] std::io::Error),

    #[error("image decode/encode error: {0}")]
    Image(#[from] image::ImageError),

    #[error("project JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid image dimensions {width}x{height}: {reason}")]
    InvalidDimensions {
        width: usize,
        height: usize,
        reason: &'static str,
    },

    #[error("OME-Zarr decode error: {0}")]
    OmeZarr(String),
}

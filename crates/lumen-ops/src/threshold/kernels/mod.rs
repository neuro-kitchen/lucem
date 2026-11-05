//! `#[cube]` kernels for `lumen_ops::threshold`.

pub(crate) mod global;
pub(crate) mod local;

pub use global::otsu_from_hist_kernel;
pub use local::{niblack_threshold_kernel, sauvola_threshold_kernel, square_kernel};

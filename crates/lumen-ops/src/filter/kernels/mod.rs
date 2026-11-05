//! `#[cube]` kernels for `lumen_ops::filter`.

pub(crate) mod box_blur;
pub(crate) mod convolve;
pub(crate) mod gradient;
pub(crate) mod order;
pub(crate) mod ridge;
pub(crate) mod spectral;
pub(crate) mod windowed_hist;

pub use box_blur::box_blur_1d_kernel;
pub use convolve::{correlate_1d_kernel, correlate_2d_kernel};
pub use gradient::{add_sq_kernel, laplace_axis_add_kernel, sqrt_scale_kernel};
pub use order::{
    extrema_1d_kernel, extrema_2d_footprint_kernel, med9, median_3x3_kernel,
    rank_2d_footprint_kernel,
};
pub use ridge::{frangi_2d_scale_kernel, meijering_2d_scale_kernel, sato_2d_scale_kernel};
pub use spectral::{spectral_phase_axis_kernel, spectral_scale_axis_kernel};
pub use windowed_hist::{windowed_entropy_2d_kernel, windowed_stat_2d_kernel};

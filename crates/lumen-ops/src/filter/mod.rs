//! Spatial, derivative, order-statistic, ridge, spectral, and windowed-histogram filters.

pub mod box_blur;
pub mod convolve;
pub mod gaussian;
pub mod gradient;
pub mod kernels;
pub mod order;
pub mod ridge;
pub mod spectral;
pub mod windowed_hist;

pub use box_blur::{box_blur, box_blur_1d, box_blur_1d_into, box_blur_into, BoxBlur, BoxBlur1d};
pub use convolve::{
    convolve_1d, correlate_1d, correlate_1d_into, correlate_2d, correlate_2d_into, Correlate1d,
    Correlate2d,
};
pub use gaussian::{
    difference_of_gaussians, difference_of_gaussians_into, gaussian_1d, gaussian_1d_into,
    gaussian_blur, gaussian_blur_into, gaussian_grad_mag, gaussian_grad_mag_into,
    gaussian_kernel_1d, gaussian_laplace, gaussian_laplace_into, unsharp_mask, unsharp_mask_into,
    Gaussian1d, GaussianBlur,
};
pub use gradient::{
    farid, laplace, laplace_into, prewitt, roberts, roberts_into, scharr, sobel,
    stencil_filter_into, GradientFilter, StencilKind,
};
pub use order::{
    extrema_1d_into, extrema_filter_into, max_filter, max_filter_1d, max_filter_into,
    median_filter, median_filter_into, min_filter, min_filter_1d, min_filter_into,
    percentile_filter, rank_filter, rank_filter_into, rank_order, Extrema1d, OrderFilter,
};
pub use ridge::{
    frangi, frangi_into, meijering, meijering_into, sato, sato_into, RidgeFilter,
};
pub use spectral::{
    butterworth, fourier_box_into, fourier_gaussian_into, fourier_shift_into, gabor_2d,
    gabor_kernel_2d,
};
pub use windowed_hist::{
    windowed_entropy, windowed_entropy_into, windowed_hist_stat, windowed_hist_stat_into,
    WindowedHistStat,
};

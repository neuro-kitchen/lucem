//! `#[cube]` kernels for `lucem_ops::restore`.

pub(crate) mod background;
pub(crate) mod deconvolve;
pub(crate) mod denoise;
pub(crate) mod inpaint;

pub use background::{rolling_ball_2d_dilate_kernel, rolling_ball_2d_erode_kernel};
pub use deconvolve::{
    fft_shift_2d_kernel, rl_div_clamp_kernel, rl_mul_clamp_kernel, wiener_freq_filter_kernel,
};
pub use denoise::{
    bilateral_2d_kernel, immerkaer_laplacian_2d_kernel, nlm_2d_kernel, tv_bregman_2d_dual_kernel,
    tv_bregman_2d_gauss_seidel_kernel, tv_chambolle_2d_div_kernel, tv_chambolle_2d_dual_kernel,
};
pub use inpaint::{biharmonic_2d_step_kernel, unwrap_phase_2d_poisson_rhs_kernel};

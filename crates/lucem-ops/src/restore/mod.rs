//! Image restoration: total-variation (`tv_chambolle`, `tv_bregman`), bilateral (`bilateral`), and
//! non-local means (`non_local_means`) denoising, robust noise estimation (`estimate_noise`),
//! rolling-ball background estimation (`rolling_ball`), Richardson-Lucy & Wiener deconvolution
//! (`richardson_lucy`, `wiener_deconvolve`, `unsupervised_wiener`), biharmonic PDE inpainting
//! (`biharmonic_inpaint`), and reliability-guided phase unwrapping (`unwrap_phase`).

pub mod background;
pub mod deconvolve;
pub mod denoise;
pub mod inpaint;
pub mod kernels;

pub use background::{ball_kernel, ellipsoid_kernel, rolling_ball, rolling_ball_into, RollingBall};
pub use deconvolve::{
    richardson_lucy, richardson_lucy_into, unsupervised_wiener, wiener_deconvolve,
    wiener_deconvolve_into, RichardsonLucy, UnsupervisedWiener, WienerDeconvolve,
};
pub use denoise::{
    bilateral, bilateral_into, estimate_noise, non_local_means, non_local_means_into, tv_bregman,
    tv_bregman_into, tv_chambolle, tv_chambolle_into, Bilateral, NonLocalMeans, TvBregman,
    TvChambolle,
};
pub use inpaint::{
    biharmonic_inpaint, biharmonic_inpaint_into, unwrap_phase, unwrap_phase_into,
    unwrap_phase_poisson_rhs_2d, BiharmonicInpaint,
};

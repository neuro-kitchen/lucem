//! Image fidelity kernels: pointwise squared difference, pointwise product, and local SSIM map.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Computes `out[i] = (a[i] - b[i])^2`.
#[cube(launch)]
pub fn sq_diff_kernel<F: lumen_core::element::ImageFloat>(a: &[F], b: &[F], out: &mut [F], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        let d = a[i] - b[i];
        out[i] = d * d;
    }
}

/// Computes `out[i] = a[i] * b[i]`.
#[cube(launch)]
pub fn mul_kernel<F: lumen_core::element::ImageFloat>(a: &[F], b: &[F], out: &mut [F], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        out[i] = a[i] * b[i];
    }
}

/// Computes the per-pixel Structural Similarity Index (SSIM) from local smoothed moments
/// `(ux, uy, uxx, uyy, uxy)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn ssim_map_kernel<F: lumen_core::element::ImageFloat>(
    ux: &[F],
    uy: &[F],
    uxx: &[F],
    uyy: &[F],
    uxy: &[F],
    ssim_out: &mut [F],
    cov_norm: F,
    c1: F,
    c2: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let mx = ux[i];
        let my = uy[i];
        let mx2 = mx * mx;
        let my2 = my * my;
        let mxy = mx * my;

        let vx = cov_norm * (uxx[i] - mx2);
        let vy = cov_norm * (uyy[i] - my2);
        let vxy = cov_norm * (uxy[i] - mxy);

        let two = F::new(2.0f32);
        let a1 = two * mxy + c1;
        let a2 = two * vxy + c2;
        let b1 = mx2 + my2 + c1;
        let b2 = vx + vy + c2;
        ssim_out[i] = (a1 * a2) / (b1 * b2);
    }
}

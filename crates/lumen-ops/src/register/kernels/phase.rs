//! Phase cross-correlation kernels: cross-power spectrum normalization and complex magnitude squared.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Computes `R(k) = A(k) * conj(B(k))`, optionally normalizing by `max(|R(k)|, eps)` when
/// `normalize_phase = true`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cross_power_spectrum_kernel<F: Float>(
    a_re: &[F],
    a_im: &[F],
    b_re: &[F],
    b_im: &[F],
    out_re: &mut [F],
    out_im: &mut [F],
    eps: F,
    len: u32,
    #[comptime] normalize_phase: bool,
) {
    let i = element_position();
    if (i as u32) < len {
        let ar = a_re[i];
        let ai = a_im[i];
        let br = b_re[i];
        let bi = b_im[i];
        // A * conj(B) = (ar + i*ai) * (br - i*bi) = (ar*br + ai*bi) + i*(ai*br - ar*bi)
        let re = ar * br + ai * bi;
        let im = ai * br - ar * bi;
        if comptime!(normalize_phase) {
            let mag = F::sqrt(re * re + im * im);
            let scale = if mag > eps { F::new(1.0f32) / mag } else { F::new(0.0f32) };
            out_re[i] = re * scale;
            out_im[i] = im * scale;
        } else {
            out_re[i] = re;
            out_im[i] = im;
        }
    }
}

/// Computes `out[i] = re[i]^2 + im[i]^2`.
#[cube(launch)]
pub fn complex_mag_sq_kernel<F: Float>(re: &[F], im: &[F], out: &mut [F], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        let r = re[i];
        let m = im[i];
        out[i] = r * r + m * m;
    }
}

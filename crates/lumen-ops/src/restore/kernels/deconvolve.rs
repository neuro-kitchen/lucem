//! Deconvolution kernels: Richardson-Lucy ratio/update steps, Wiener spectral inversion, and
//! PSF origin centering.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Pointwise stabilized ratio `out[i] = obs[i] / max(est_conv[i], eps)` for Richardson-Lucy.
#[cube(launch)]
pub fn rl_div_clamp_kernel<F: lumen_core::element::ImageFloat>(
    obs: &[F],
    est_conv: &[F],
    out: &mut [F],
    eps: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let denom = est_conv[i];
        let safe_denom = if F::abs(denom) < eps { eps } else { denom };
        out[i] = obs[i] / safe_denom;
    }
}

/// Pointwise multiplicative update `est_out[i] = est_in[i] * corr[i]`, optionally clamping to
/// `[-1.0, 1.0]` when `clip` is enabled.
#[cube(launch)]
pub fn rl_mul_clamp_kernel<F: lumen_core::element::ImageFloat>(
    est_in: &[F],
    corr: &[F],
    est_out: &mut [F],
    len: u32,
    #[comptime] clip: bool,
) {
    let i = element_position();
    if (i as u32) < len {
        let val = est_in[i] * corr[i];
        est_out[i] = if comptime!(clip) {
            F::min(F::max(val, F::new(-1.0f32)), F::new(1.0f32))
        } else {
            val
        };
    }
}

/// Frequency-domain Wiener filter:
/// `X(f) = (conj(H(f)) * Y(f)) / (|H(f)|^2 + balance * |L(f)|^2)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn wiener_freq_filter_kernel<F: lumen_core::element::ImageFloat>(
    y_re: &[F],
    y_im: &[F],
    h_re: &[F],
    h_im: &[F],
    reg_mag_sq: &[F],
    out_re: &mut [F],
    out_im: &mut [F],
    balance: F,
    eps: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let yr = y_re[i];
        let yi = y_im[i];
        let hr = h_re[i];
        let hi = h_im[i];
        let h_norm2 = hr * hr + hi * hi;
        let denom = F::max(h_norm2 + balance * reg_mag_sq[i], eps);
        // conj(H) * Y = (hr - i*hi) * (yr + i*yi) = (hr*yr + hi*yi) + i*(hr*yi - hi*yr)
        out_re[i] = (hr * yr + hi * yi) / denom;
        out_im[i] = (hr * yi - hi * yr) / denom;
    }
}

/// Centers a small 2-D PSF `[k_h, k_w]` into a zero-padded `[height, width]` image buffer with the
/// PSF center `(k_h / 2, k_w / 2)` wrapped to `(0, 0)` for circular FFT convolution.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn fft_shift_2d_kernel<F: lumen_core::element::ImageFloat>(
    psf: &[F],
    padded: &mut [F],
    height: u32,
    width: u32,
    k_h: u32,
    k_w: u32,
    total_psf: u32,
) {
    let idx = element_position();
    if (idx as u32) < total_psf {
        let e = idx as u32;
        let ky = e / k_w;
        let kx = e % k_w;
        let cy = k_h / 2u32;
        let cx = k_w / 2u32;
        let dy = ((ky + height - cy) % height) * width;
        let dx = (kx + width - cx) % width;
        padded[(dy + dx) as usize] = psf[idx];
    }
}

//! Frequency-domain filter kernels (Fourier Gaussian, Fourier box/sinc, Fourier phase shift,
//! Butterworth transfer function).

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Multiplies a 1-D or N-D Fourier spectrum along an axis `[outer, len, inner]` by a precomputed
/// real transfer profile `transfer[freq]`.
#[cube(launch)]
pub fn spectral_scale_axis_kernel<F: lumen_core::element::ImageFloat>(
    re: &mut [F],
    im: &mut [F],
    transfer: &[F],
    len: u32,
    inner: u32,
    total: u32,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let e = idx as u32;
        let freq = (e / inner) % len;
        let w = transfer[freq as usize];
        re[idx] = re[idx] * w;
        im[idx] = im[idx] * w;
    }
}

/// Multiplies a Fourier spectrum along an axis `[outer, len, inner]` by the complex phase-shift
/// factor `(cos_phase[freq] + i * sin_phase[freq])`.
#[cube(launch)]
pub fn spectral_phase_axis_kernel<F: lumen_core::element::ImageFloat>(
    re: &mut [F],
    im: &mut [F],
    cos_phase: &[F],
    sin_phase: &[F],
    len: u32,
    inner: u32,
    total: u32,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let e = idx as u32;
        let freq = (e / inner) % len;
        let c = cos_phase[freq as usize];
        let s = sin_phase[freq as usize];
        let r = re[idx];
        let m = im[idx];
        re[idx] = r * c - m * s;
        im[idx] = r * s + m * c;
    }
}

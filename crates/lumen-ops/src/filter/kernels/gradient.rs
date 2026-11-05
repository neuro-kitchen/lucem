//! Kernels for spatial derivative and edge stencils (Laplacian, Roberts cross, Euclidean norm of
//! directional gradients).

use cubecl::prelude::*;
use lumen_core::compute::element_position;

use crate::core::kernels::read_padded;

/// Adds the second finite difference `input[o, i - 1, k] - 2 * input[o, i, k] + input[o, i + 1, k]`
/// along an axis `[outer, len, inner]` into `accum[o, i, k]`.
#[cube(launch)]
pub fn laplace_axis_add_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    accum: &mut [F],
    len: u32,
    inner: u32,
    cval: F,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    if idx < accum.len() {
        let e = idx as u32;
        let i = (e / inner) % len;
        let base = e - i * inner;
        let p = i as i32;
        let left = read_padded::<F>(input, base, inner, p - 1i32, len as i32, cval, mode);
        let center = input[idx];
        let right = read_padded::<F>(input, base, inner, p + 1i32, len as i32, cval, mode);
        let d2 = left - F::new(2.0f32) * center + right;
        accum[idx] = accum[idx] + d2;
    }
}

/// Adds `grad[i] * grad[i]` into `accum_sq[i]` for multi-axis gradient magnitude accumulation.
#[cube(launch)]
pub fn add_sq_kernel<F: lumen_core::element::ImageFloat>(grad: &[F], accum_sq: &mut [F], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        let g = grad[i];
        accum_sq[i] = accum_sq[i] + g * g;
    }
}

/// Scales the square root of `accum_sq[i]` by `scale` (`1 / sqrt(ndim)` for isotropic Sobel/Prewitt/Scharr
/// gradient magnitude, or `1.0` for Euclidean norm).
#[cube(launch)]
pub fn sqrt_scale_kernel<F: lumen_core::element::ImageFloat>(accum_sq: &[F], output: &mut [F], scale: F, len: u32) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = F::sqrt(F::max(accum_sq[i], F::new(0.0f32))) * scale;
    }
}

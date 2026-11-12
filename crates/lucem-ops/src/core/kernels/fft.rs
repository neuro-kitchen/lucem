//! Discrete Fourier transform kernels along an axis (`[outer, len, inner]`).

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// 1-D forward (`inverse = false`) or inverse (`inverse = true`) DFT along an axis seen as
/// `[outer, len, inner]`, operating on split real/imaginary buffers `(re_in, im_in)` -> `(re_out, im_out)`.
/// Works for arbitrary axis lengths `len` (including prime and non-power-of-two sizes).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn dft_axis_kernel<F: lucem_core::element::ImageFloat>(
    re_in: &[F],
    im_in: &[F],
    re_out: &mut [F],
    im_out: &mut [F],
    len: u32,
    inner: u32,
    total: u32,
    #[comptime] inverse: bool,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let e = idx as u32;
        let freq = (e / inner) % len;
        let base = e - freq * inner;

        let sign = if comptime!(inverse) { f32::new(1.0) } else { f32::new(-1.0) };
        let two_pi_over_n = sign * 6.283_185_5f32 / (len as f32);
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        let mut n = 0u32;
        while n < len {
            let at = (base + n * inner) as usize;
            let xr = f32::cast_from(re_in[at]);
            let xi = f32::cast_from(im_in[at]);
            let angle = two_pi_over_n * ((freq * n) as f32);
            let c = f32::cos(angle);
            let s = f32::sin(angle);
            sum_re += xr * c - xi * s;
            sum_im += xr * s + xi * c;
            n += 1u32;
        }
        let scale = if comptime!(inverse) { f32::new(1.0) / (len as f32) } else { f32::new(1.0) };
        re_out[idx] = F::cast_from(sum_re * scale);
        im_out[idx] = F::cast_from(sum_im * scale);
    }
}

/// Complex pointwise multiplication: `(ar + i ai) * (br + i bi)` (or conjugate of `b` when `conj_b`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn complex_mul_kernel<F: lucem_core::element::ImageFloat>(
    a_re: &[F],
    a_im: &[F],
    b_re: &[F],
    b_im: &[F],
    out_re: &mut [F],
    out_im: &mut [F],
    len: u32,
    #[comptime] conj_b: bool,
) {
    let i = element_position();
    if (i as u32) < len {
        let ar = a_re[i];
        let ai = a_im[i];
        let br = b_re[i];
        let bi = if comptime!(conj_b) { F::new(0.0f32) - b_im[i] } else { b_im[i] };
        out_re[i] = ar * br - ai * bi;
        out_im[i] = ar * bi + ai * br;
    }
}

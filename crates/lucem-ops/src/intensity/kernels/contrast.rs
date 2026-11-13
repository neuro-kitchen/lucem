//! GPU kernels for intensity conversion, contrast stretching, gamma, log, and sigmoid curves.

use cubecl::prelude::*;
use lucem_core::compute::element_position;
use crate::core::kernels::read_padded;

/// Rescale intensity using device-resident min/max values read from 1-element buffers.
#[cube(launch)]
pub(crate) fn rescale_range_device_kernel<F: lucem_core::element::ImageFloat>(
    input: &[F],
    in_min_buf: &[F],
    in_max_buf: &[F],
    output: &mut [F],
    out_min: F,
    out_max: F,
    total: u32,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let in_min = in_min_buf[0];
        let in_max = in_max_buf[0];
        let val = input[idx];
        let clamped = F::min(F::max(val, in_min), in_max);
        let span = in_max - in_min;
        let out_span = out_max - out_min;
        output[idx] = if span > F::new(1e-12) {
            out_min + (clamped - in_min) / span * out_span
        } else {
            out_min
        };
    }
}

/// Gamma correction: `gain * (x ^ gamma)`.
#[cube(launch)]
pub(crate) fn adjust_gamma_kernel<F: lucem_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    gamma: F,
    gain: F,
    total: u32,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let x = F::max(input[idx], F::new(0.0));
        output[idx] = gain * F::powf(x, gamma);
    }
}

/// Logarithmic adjustment: `gain * log2(1 + x)` or `gain * (2^x - 1)` when `inv != 0`.
#[cube(launch)]
pub(crate) fn adjust_log_kernel<F: lucem_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    gain: F,
    inv: u32,
    total: u32,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let x = F::max(input[idx], F::new(0.0));
        output[idx] = if inv != 0u32 {
            gain * (F::powf(F::new(2.0), x) - F::new(1.0))
        } else {
            gain * (F::ln(F::new(1.0) + x) / F::ln(F::new(2.0)))
        };
    }
}

/// Sigmoid contrast adjustment: `1 / (1 + exp(gain * (cutoff - x)))` (or `1 - ...` if `inv != 0`).
#[cube(launch)]
pub(crate) fn adjust_sigmoid_kernel<F: lucem_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    cutoff: F,
    gain: F,
    inv: u32,
    total: u32,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let x = input[idx];
        let s = F::new(1.0) / (F::new(1.0) + F::exp(gain * (cutoff - x)));
        output[idx] = if inv != 0u32 { F::new(1.0) - s } else { s };
    }
}

/// Sub-region crop in 1-D, 2-D, or 3-D.
#[cube(launch)]
pub(crate) fn crop_3d_kernel<E: Numeric>(
    input: &[E],
    output: &mut [E],
    in_d: u32,
    in_h: u32,
    in_w: u32,
    z0: u32,
    y0: u32,
    x0: u32,
    out_d: u32,
    out_h: u32,
    out_w: u32,
) {
    let idx = element_position() as u32;
    let total = out_d * out_h * out_w;
    if idx < total {
        let ox = idx % out_w;
        let oy = (idx / out_w) % out_h;
        let oz = idx / (out_w * out_h);
        let ix = u32::min(x0 + ox, in_w - 1u32);
        let iy = u32::min(y0 + oy, in_h - 1u32);
        let iz = u32::min(z0 + oz, in_d - 1u32);
        let in_idx = (iz * in_h + iy) * in_w + ix;
        output[idx as usize] = input[in_idx as usize];
    }
}

/// Pad along one axis `[outer, in_len, inner]` -> `[outer, out_len, inner]` using `PadMode`.
#[cube(launch)]
pub(crate) fn pad_axis_kernel<F: lucem_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    in_len: u32,
    out_len: u32,
    inner: u32,
    pad_before: i32,
    cval: F,
    total: u32,
    #[comptime] mode: u32,
) {
    let idx = element_position() as u32;
    if idx < total {
        let rem = idx % inner;
        let out_coord = (idx / inner) % out_len;
        let outer_idx = idx / (inner * out_len);
        let base = outer_idx * in_len * inner + rem;
        let src_coord = (out_coord as i32) - pad_before;
        output[idx as usize] = read_padded::<F>(input, base, inner, src_coord, in_len as i32, cval, mode);
    }
}

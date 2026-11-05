//! Elementwise mapping kernels: copy, affine (`scale * x + offset`), clip, LUT, threshold compare,
//! and binary arithmetic.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Copies `input[i]` into `output[i]` for `i < len`.
#[cube(launch)]
pub fn copy_kernel<E: Numeric>(input: &[E], output: &mut [E], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = input[i];
    }
}

/// `output[i] = input[i] * scale + offset`.
#[cube(launch)]
pub fn affine_kernel<F: lumen_core::element::ImageFloat>(input: &[F], output: &mut [F], scale: F, offset: F, len: u32) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = input[i] * scale + offset;
    }
}

/// Clamps `input[i]` to `[lo, hi]`.
#[cube(launch)]
pub fn clip_kernel<F: lumen_core::element::ImageFloat>(input: &[F], output: &mut [F], lo: F, hi: F, len: u32) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = F::min(F::max(input[i], lo), hi);
    }
}

/// Applies `scale * clamp(input[i], in_lo, in_hi) + offset` and clamps to `[out_lo, out_hi]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rescale_clip_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    in_lo: F,
    in_hi: F,
    scale: F,
    offset: F,
    out_lo: F,
    out_hi: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let clamped = F::min(F::max(input[i], in_lo), in_hi);
        let mapped = clamped * scale + offset;
        output[i] = F::min(F::max(mapped, out_lo), out_hi);
    }
}

/// Converts between numeric types `Src` and `Dst` (`output[i] = Dst::cast_from(input[i])`).
#[cube(launch)]
pub fn cast_kernel<Src: Numeric, Dst: Numeric>(input: &[Src], output: &mut [Dst], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = Dst::cast_from(input[i]);
    }
}

/// Scales a float in `[0.0, 1.0]` (clamped) by `max_val`, rounds to nearest integer, and casts to `Dst`.
#[cube(launch)]
pub fn float_to_uint_kernel<F: lumen_core::element::ImageFloat, Dst: lumen_core::element::ImageInt>(input: &[F], output: &mut [Dst], max_val: F, len: u32) {
    let i = element_position();
    if (i as u32) < len {
        let x = F::min(F::max(input[i], F::new(0.0f32)), F::new(1.0f32));
        let scaled = F::floor(x * max_val + F::new(0.5f32));
        output[i] = Dst::cast_from(scaled);
    }
}

/// Converts an unsigned integer `Src` in `0..=max_val` into `F` in `[0.0, 1.0]`.
#[cube(launch)]
pub fn uint_to_float_kernel<Src: lumen_core::element::ImageInt, F: lumen_core::element::ImageFloat>(input: &[Src], output: &mut [F], inv_max: F, len: u32) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = F::cast_from(input[i]) * inv_max;
    }
}

/// Look-up table mapping: `output[i] = lut[clamp(input[i], 0, lut_len - 1)]`.
#[cube(launch)]
pub fn lut_kernel<Idx: lumen_core::element::ImageInt, Val: Numeric>(input: &[Idx], lut: &[Val], output: &mut [Val], lut_len: u32, len: u32) {
    let i = element_position();
    if (i as u32) < len {
        let raw = i32::cast_from(input[i]);
        let clamped = i32::max(i32::min(raw, lut_len as i32 - 1i32), 0i32) as u32;
        output[i] = lut[clamped as usize];
    }
}

/// Compares `input[i]` against a device-resident scalar `threshold[0]`: writes `1` if `>` (or `>=` when
/// `inclusive`), else `0`. Keeps the threshold on the device with zero host synchronization.
#[cube(launch)]
pub fn threshold_device_kernel<F: lumen_core::element::ImageFloat, Out: Numeric>(
    input: &[F],
    threshold: &[F],
    output: &mut [Out],
    len: u32,
    #[comptime] inclusive: bool,
) {
    let i = element_position();
    if (i as u32) < len {
        let t = threshold[0];
        let v = input[i];
        let pass = if comptime!(inclusive) { v >= t } else { v > t };
        output[i] = if pass { Out::from_int(1) } else { Out::from_int(0) };
    }
}

/// Compares `input[i]` against a host scalar `threshold`: writes `1` if `>` (or `>=` when `inclusive`),
/// else `0`.
#[cube(launch)]
pub fn threshold_scalar_kernel<F: lumen_core::element::ImageFloat, Out: Numeric>(
    input: &[F],
    threshold: F,
    output: &mut [Out],
    len: u32,
    #[comptime] inclusive: bool,
) {
    let i = element_position();
    if (i as u32) < len {
        let v = input[i];
        let pass = if comptime!(inclusive) { v >= threshold } else { v > threshold };
        output[i] = if pass { Out::from_int(1) } else { Out::from_int(0) };
    }
}

/// Elementwise binary linear combination: `output[i] = alpha * a[i] + beta * b[i] + bias`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn linear_comb_kernel<F: lumen_core::element::ImageFloat>(
    a: &[F],
    b: &[F],
    output: &mut [F],
    alpha: F,
    beta: F,
    bias: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = alpha * a[i] + beta * b[i] + bias;
    }
}

/// Elementwise Euclidean magnitude: `output[i] = sqrt(a[i]^2 + b[i]^2)`.
#[cube(launch)]
pub fn hypot_kernel<F: lumen_core::element::ImageFloat>(a: &[F], b: &[F], output: &mut [F], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        let va = a[i];
        let vb = b[i];
        output[i] = F::sqrt(va * va + vb * vb);
    }
}

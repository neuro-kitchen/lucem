//! Binary morphological kernels: 1-D separable erosion/dilation, 2-D footprint erosion/dilation,
//! and hit-or-miss transform.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// 1-D binary erosion (`is_dilation = false`) or dilation (`is_dilation = true`) along an axis
/// seen as `[outer, len, inner]` with constant `border_value` (`0` or `1`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn binary_1d_kernel(
    input: &[u32],
    output: &mut [u32],
    len: u32,
    inner: u32,
    left: i32,
    border_val: u32,
    #[comptime] size: u32,
    #[comptime] is_dilation: bool,
) {
    let idx = element_position();
    if idx < output.len() {
        let e = idx as u32;
        let i = (e / inner) % len;
        let base = e - i * inner;
        let start = (i as i32) - left;
        let mut acc = 0u32;
        if comptime!(!is_dilation) {
            acc = 1u32;
        }
        let mut t = 0u32;
        while t < size {
            let p = start + (t as i32);
            let safe = i32::max(i32::min(p, (len as i32) - 1i32), 0i32) as u32;
            let raw = if input[(base + safe * inner) as usize] != 0u32 { 1u32 } else { 0u32 };
            let v = if p < 0i32 || p >= (len as i32) { border_val } else { raw };
            acc = if comptime!(is_dilation) {
                u32::max(acc, v)
            } else {
                u32::min(acc, v)
            };
            t += 1u32;
        }
        output[idx] = acc;
    }
}

/// 2-D binary erosion or dilation over active footprint `offsets` (`[n_active, 2]`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn binary_2d_footprint_kernel(
    input: &[u32],
    output: &mut [u32],
    offsets: &[i32],
    height: i32,
    width: i32,
    n_active: u32,
    border_val: u32,
    #[comptime] is_dilation: bool,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;
        let mut acc = 0u32;
        if comptime!(!is_dilation) {
            acc = 1u32;
        }
        let mut k = 0u32;
        while k < n_active {
            let dy = offsets[(k * 2u32) as usize];
            let dx = offsets[(k * 2u32 + 1u32) as usize];
            let py = y + dy;
            let px = x + dx;
            let sy = i32::max(i32::min(py, height - 1i32), 0i32) as u32;
            let sx = i32::max(i32::min(px, width - 1i32), 0i32) as u32;
            let raw = if input[(sy * w_u + sx) as usize] != 0u32 { 1u32 } else { 0u32 };
            let v = if py < 0i32 || py >= height || px < 0i32 || px >= width {
                border_val
            } else {
                raw
            };
            acc = if comptime!(is_dilation) {
                u32::max(acc, v)
            } else {
                u32::min(acc, v)
            };
            k += 1u32;
        }
        output[idx] = acc;
    }
}

/// Logical AND of `eroded_fg` with `(1 - dilated_bg)` for binary hit-or-miss.
#[cube(launch)]
pub fn hit_or_miss_combine_kernel(
    eroded_fg: &[u32],
    eroded_bg: &[u32],
    output: &mut [u32],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = if eroded_fg[i] != 0u32 && eroded_bg[i] != 0u32 { 1u32 } else { 0u32 };
    }
}

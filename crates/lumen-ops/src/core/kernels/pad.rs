//! Device-side boundary indexing and guarded reads.

use cubecl::prelude::*;

use crate::core::pad::{PAD_CONSTANT, PAD_NEAREST, PAD_REFLECT, PAD_WRAP};

/// Maps coordinate `p` along an axis of length `len` (`len >= 1`) under `mode` to an in-bounds index
/// in `0..len`, or `-1` when `mode == PAD_CONSTANT` and `p` lies outside `0..len`.
///
/// Works at arbitrary distances past the boundary (filters wider than the axis fold repeatedly).
#[cube]
pub fn pad_index(p: i32, len: i32, #[comptime] mode: u32) -> i32 {
    let mut q = p;
    if p < 0i32 || p >= len {
        if comptime!(mode == PAD_CONSTANT) {
            q = -1i32;
        } else if comptime!(mode == PAD_NEAREST) {
            q = i32::max(i32::min(p, len - 1i32), 0i32);
        } else if comptime!(mode == PAD_WRAP) {
            q = ((p % len) + len) % len;
        } else if comptime!(mode == PAD_REFLECT) {
            let period = 2i32 * len;
            let r = ((p % period) + period) % period;
            q = if r < len { r } else { period - 1i32 - r };
        } else {
            // Mirror mode: a length-1 axis has period 0, so clamp period to >= 1 to avoid division
            // by zero if the backend lowers `if` to an eager select instruction.
            let period = i32::max(2i32 * len - 2i32, 1i32);
            let r = ((p % period) + period) % period;
            q = if r < len { r } else { period - r };
        }
    }
    q
}

/// Reads element `p` along the 1-D line starting at `input[base]` with `len` elements spaced by
/// `stride`, applying `mode` (and returning `cval` when `mode == PAD_CONSTANT` out of bounds).
///
/// Always clamps the physical address before reading so that eager `select` lowering never reads
/// out of bounds.
#[cube]
pub fn read_padded<F: Float>(
    input: &[F],
    base: u32,
    stride: u32,
    p: i32,
    len: i32,
    cval: F,
    #[comptime] mode: u32,
) -> F {
    let q = pad_index(p, len, mode);
    let safe = i32::max(q, 0i32) as u32;
    let value = input[(base + safe * stride) as usize];
    if q < 0i32 { cval } else { value }
}

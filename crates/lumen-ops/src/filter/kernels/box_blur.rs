//! Kernels for 1-D uniform (box) filtering along an axis.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

use crate::core::kernels::read_padded;

/// 1-D uniform box filter of window `size` along an axis seen as `[outer, len, inner]`:
/// `output[o, i, k] = (1 / size) * sum_{t=0..size} input[o, i + t - left, k]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn box_blur_1d_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    len: u32,
    inner: u32,
    left: i32,
    inv_size: F,
    cval: F,
    #[comptime] size: u32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    if idx < output.len() {
        let e = idx as u32;
        let i = (e / inner) % len;
        let base = e - i * inner;
        let start = (i as i32) - left;
        let mut acc = F::new(0.0f32);
        let mut t = 0u32;
        if start >= 0i32 && start + comptime!(size as i32) <= (len as i32) {
            let mut at = base + (start as u32) * inner;
            while t < size {
                acc += input[at as usize];
                at += inner;
                t += 1u32;
            }
        } else {
            while t < size {
                acc += read_padded::<F>(input, base, inner, start + (t as i32), len as i32, cval, mode);
                t += 1u32;
            }
        }
        output[idx] = acc * inv_size;
    }
}

//! Grayscale morphological kernels (including non-flat additive structuring element weights).

use cubecl::prelude::*;
use lumen_core::compute::{element_position, negative_infinity, positive_infinity};

use crate::core::kernels::pad_index;

/// 2-D non-flat grayscale erosion (`is_dilation = false`) or dilation (`is_dilation = true`) with
/// additive `weights` over active footprint `offsets` (`[n_active, 2]`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn grey_weighted_2d_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    offsets: &[i32],
    weights: &[F],
    height: i32,
    width: i32,
    n_active: u32,
    cval: F,
    #[comptime] mode: u32,
    #[comptime] is_dilation: bool,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;
        let mut best = if comptime!(is_dilation) {
            negative_infinity::<F>()
        } else {
            positive_infinity::<F>()
        };
        let mut k = 0u32;
        while k < n_active {
            let dy = offsets[(k * 2u32) as usize];
            let dx = offsets[(k * 2u32 + 1u32) as usize];
            let py = pad_index(y + dy, height, mode);
            let px = pad_index(x + dx, width, mode);
            let sy = i32::max(py, 0i32) as u32;
            let sx = i32::max(px, 0i32) as u32;
            let raw = input[(sy * w_u + sx) as usize];
            let v = if py < 0i32 || px < 0i32 { cval } else { raw };
            let cand = if comptime!(is_dilation) {
                v + weights[k as usize]
            } else {
                v - weights[k as usize]
            };
            best = if comptime!(is_dilation) {
                F::max(best, cand)
            } else {
                F::min(best, cand)
            };
            k += 1u32;
        }
        output[idx] = best;
    }
}

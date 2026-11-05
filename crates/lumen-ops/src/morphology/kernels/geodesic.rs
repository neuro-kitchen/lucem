//! Geodesic morphological reconstruction and regional extrema kernels.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// One 2-D geodesic dilation (`is_erosion = false`) or erosion (`is_erosion = true`) step of `marker`
/// constrained by `mask`, setting `changed[0] = 1` when any pixel changes.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn geodesic_step_2d_kernel<F: lumen_core::element::ImageFloat>(
    src_marker: &[F],
    mask: &[F],
    dst_marker: &mut [F],
    changed: &mut [Atomic<u32>],
    height: u32,
    width: u32,
    #[comptime] eight_connected: bool,
    #[comptime] is_erosion: bool,
) {
    let idx = element_position();
    let total = height * width;
    if (idx as u32) < total {
        let e = idx as u32;
        let y = e / width;
        let x = e % width;
        let cur = src_marker[idx];
        let mut best = cur;

        if y > 0u32 {
            let v = src_marker[(e - width) as usize];
            best = if comptime!(is_erosion) { F::min(best, v) } else { F::max(best, v) };
        }
        if y + 1u32 < height {
            let v = src_marker[(e + width) as usize];
            best = if comptime!(is_erosion) { F::min(best, v) } else { F::max(best, v) };
        }
        if x > 0u32 {
            let v = src_marker[(e - 1u32) as usize];
            best = if comptime!(is_erosion) { F::min(best, v) } else { F::max(best, v) };
        }
        if x + 1u32 < width {
            let v = src_marker[(e + 1u32) as usize];
            best = if comptime!(is_erosion) { F::min(best, v) } else { F::max(best, v) };
        }
        if comptime!(eight_connected) {
            if y > 0u32 && x > 0u32 {
                let v = src_marker[(e - width - 1u32) as usize];
                best = if comptime!(is_erosion) { F::min(best, v) } else { F::max(best, v) };
            }
            if y > 0u32 && x + 1u32 < width {
                let v = src_marker[(e - width + 1u32) as usize];
                best = if comptime!(is_erosion) { F::min(best, v) } else { F::max(best, v) };
            }
            if y + 1u32 < height && x > 0u32 {
                let v = src_marker[(e + width - 1u32) as usize];
                best = if comptime!(is_erosion) { F::min(best, v) } else { F::max(best, v) };
            }
            if y + 1u32 < height && x + 1u32 < width {
                let v = src_marker[(e + width + 1u32) as usize];
                best = if comptime!(is_erosion) { F::min(best, v) } else { F::max(best, v) };
            }
        }
        let clamped = if comptime!(is_erosion) {
            F::max(best, mask[idx])
        } else {
            F::min(best, mask[idx])
        };
        dst_marker[idx] = clamped;
        if clamped != cur {
            Atomic::store(&changed[0], 1u32);
        }
    }
}

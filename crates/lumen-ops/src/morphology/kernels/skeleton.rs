//! 2-D Zhang-Suen parallel binary thinning sub-iteration kernels.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// One sub-iteration (`pass = 0` or `pass = 1`) of Zhang-Suen 2-D binary skeletonization,
/// setting `changed[0] = 1` whenever any foreground pixel is deleted.
#[cube(launch)]
pub fn zhang_suen_subpass_kernel(
    src: &[u32],
    dst: &mut [u32],
    changed: &mut [Atomic<u32>],
    height: u32,
    width: u32,
    #[comptime] second_subpass: bool,
) {
    let idx = element_position();
    let total = height * width;
    if (idx as u32) < total {
        let cur = if src[idx] != 0u32 { 1u32 } else { 0u32 };
        if cur == 0u32 {
            dst[idx] = 0u32;
        } else {
            let e = idx as u32;
            let y = e / width;
            let x = e % width;
            if y == 0u32 || y + 1u32 >= height || x == 0u32 || x + 1u32 >= width {
                dst[idx] = 1u32;
            } else {
                // 8-neighbors in clockwise order starting at North (P2..P9)
                let p2 = if src[(e - width) as usize] != 0u32 { 1u32 } else { 0u32 };
                let p3 = if src[(e - width + 1u32) as usize] != 0u32 { 1u32 } else { 0u32 };
                let p4 = if src[(e + 1u32) as usize] != 0u32 { 1u32 } else { 0u32 };
                let p5 = if src[(e + width + 1u32) as usize] != 0u32 { 1u32 } else { 0u32 };
                let p6 = if src[(e + width) as usize] != 0u32 { 1u32 } else { 0u32 };
                let p7 = if src[(e + width - 1u32) as usize] != 0u32 { 1u32 } else { 0u32 };
                let p8 = if src[(e - 1u32) as usize] != 0u32 { 1u32 } else { 0u32 };
                let p9 = if src[(e - width - 1u32) as usize] != 0u32 { 1u32 } else { 0u32 };

                let b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                let a = u32::cast_from(p2 == 0u32 && p3 == 1u32)
                    + u32::cast_from(p3 == 0u32 && p4 == 1u32)
                    + u32::cast_from(p4 == 0u32 && p5 == 1u32)
                    + u32::cast_from(p5 == 0u32 && p6 == 1u32)
                    + u32::cast_from(p6 == 0u32 && p7 == 1u32)
                    + u32::cast_from(p7 == 0u32 && p8 == 1u32)
                    + u32::cast_from(p8 == 0u32 && p9 == 1u32)
                    + u32::cast_from(p9 == 0u32 && p2 == 1u32);

                let cond_c = if comptime!(second_subpass) {
                    (p2 * p4 * p8 == 0u32) && (p2 * p6 * p8 == 0u32)
                } else {
                    (p2 * p4 * p6 == 0u32) && (p4 * p6 * p8 == 0u32)
                };

                if b >= 2u32 && b <= 6u32 && a == 1u32 && cond_c {
                    dst[idx] = 0u32;
                    Atomic::store(&changed[0], 1u32);
                } else {
                    dst[idx] = 1u32;
                }
            }
        }
    }
}

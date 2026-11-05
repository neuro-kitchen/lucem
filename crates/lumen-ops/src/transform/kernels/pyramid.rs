//! Block-mean downsampling, area-averaging resize, and summed-area box integration kernels.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

use crate::core::kernels::pad_index;

/// Downsamples a 2-D image `[in_h, in_w]` by integer block factors `(fy, fx)` into `[out_h, out_w]`,
/// averaging each `(fy, fx)` tile with boundary padding `mode`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn downsample_mean_2d_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    in_h: i32,
    in_w: i32,
    out_w: u32,
    fy: i32,
    fx: i32,
    cval: F,
    total: u32,
    #[comptime] mode: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let oy = (idx / out_w) as i32;
        let ox = (idx % out_w) as i32;
        let base_y = oy * fy;
        let base_x = ox * fx;
        let mut sum = F::new(0.0f32);
        let mut dy = 0i32;
        while dy < fy {
            let py = pad_index(base_y + dy, in_h, mode);
            let sy = i32::max(py, 0i32) as u32;
            let mut dx = 0i32;
            while dx < fx {
                let px = pad_index(base_x + dx, in_w, mode);
                let sx = i32::max(px, 0i32) as u32;
                let raw = input[(sy * (in_w as u32) + sx) as usize];
                let val = if py < 0i32 || px < 0i32 { cval } else { raw };
                sum += val;
                dx += 1i32;
            }
            dy += 1i32;
        }
        output[i] = sum / F::cast_from(fy * fx);
    }
}

/// Resizes `[in_h, in_w]` to `[out_h, out_w]` by exact fractional box-overlap area integration.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn resize_mean_2d_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    in_h: i32,
    in_w: i32,
    out_h: u32,
    out_w: u32,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let oy = idx / out_w;
        let ox = idx % out_w;
        let scale_y = (in_h as f32) / (out_h as f32);
        let scale_x = (in_w as f32) / (out_w as f32);
        let y0 = (oy as f32) * scale_y;
        let y1 = ((oy + 1u32) as f32) * scale_y;
        let x0 = (ox as f32) * scale_x;
        let x1 = ((ox + 1u32) as f32) * scale_x;

        let iy_start = i32::max(i32::cast_from(f32::floor(y0)), 0i32);
        let iy_end = i32::min(i32::cast_from(f32::ceil(y1)), in_h);
        let ix_start = i32::max(i32::cast_from(f32::floor(x0)), 0i32);
        let ix_end = i32::min(i32::cast_from(f32::ceil(x1)), in_w);

        let mut weighted_sum = F::new(0.0f32);
        let mut weight_total = F::new(0.0f32);

        let mut iy = iy_start;
        while iy < iy_end {
            let py0 = f32::max(y0, iy as f32);
            let py1 = f32::min(y1, (iy + 1i32) as f32);
            let wy = F::cast_from(f32::max(py1 - py0, 0.0f32));
            let mut ix = ix_start;
            while ix < ix_end {
                let px0 = f32::max(x0, ix as f32);
                let px1 = f32::min(x1, (ix + 1i32) as f32);
                let wx = F::cast_from(f32::max(px1 - px0, 0.0f32));
                let w = wy * wx;
                weighted_sum += w * input[(iy * in_w + ix) as usize];
                weight_total += w;
                ix += 1i32;
            }
            iy += 1i32;
        }
        output[i] = if weight_total > F::new(0.0f32) {
            weighted_sum / weight_total
        } else {
            F::new(0.0f32)
        };
    }
}

/// Evaluates $O(1)$ inclusive box sums over `[r0[i]..=r1[i], c0[i]..=c1[i]]` from a 2-D inclusive
/// integral image `sat` of shape `[height, width]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn integrate_box_2d_kernel<F: Float>(
    sat: &[F],
    r0: &[u32],
    c0: &[u32],
    r1: &[u32],
    c1: &[u32],
    output: &mut [F],
    width: u32,
    num_boxes: u32,
) {
    let i = element_position();
    if (i as u32) < num_boxes {
        let top = r0[i];
        let left = c0[i];
        let bot = r1[i];
        let right = c1[i];

        let br = sat[(bot * width + right) as usize];
        let tr = if top > 0u32 {
            sat[((top - 1u32) * width + right) as usize]
        } else {
            F::new(0.0f32)
        };
        let bl = if left > 0u32 {
            sat[(bot * width + (left - 1u32)) as usize]
        } else {
            F::new(0.0f32)
        };
        let tl = if top > 0u32 && left > 0u32 {
            sat[((top - 1u32) * width + (left - 1u32)) as usize]
        } else {
            F::new(0.0f32)
        };
        output[i] = br - tr - bl + tl;
    }
}

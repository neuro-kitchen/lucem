//! Rolling-ball non-flat morphological erosion and dilation kernels.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

use crate::core::kernels::pad_index;

/// Non-flat 2-D grayscale erosion for rolling-ball background estimation:
/// `eroded[y, x] = min_{ky, kx} (input[y + ky - ry, x + kx - rx] + kernel[ky, kx])`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rolling_ball_2d_erode_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    kernel: &[F],
    eroded: &mut [F],
    height: i32,
    width: i32,
    ry: i32,
    rx: i32,
    #[comptime] k_h: u32,
    #[comptime] k_w: u32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;
        let y0 = y - ry;
        let x0 = x - rx;

        let mut best = F::new(1.0e30f32);
        let mut ky = 0u32;
        while ky < k_h {
            let py = i32::max(pad_index(y0 + (ky as i32), height, mode), 0i32) as u32;
            let k_row = ky * k_w;
            let mut kx = 0u32;
            while kx < k_w {
                let kval = kernel[(k_row + kx) as usize];
                if kval < F::new(1.0e20f32) {
                    let px = i32::max(pad_index(x0 + (kx as i32), width, mode), 0i32) as u32;
                    let cand = input[(py * w_u + px) as usize] + kval;
                    best = F::min(best, cand);
                }
                kx += 1u32;
            }
            ky += 1u32;
        }
        eroded[idx] = best;
    }
}

/// Non-flat 2-D grayscale dilation for rolling-ball background estimation:
/// `background[y, x] = max_{ky, kx} (eroded[y - (ky - ry), x - (kx - rx)] - kernel[ky, kx])`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rolling_ball_2d_dilate_kernel<F: lumen_core::element::ImageFloat>(
    eroded: &[F],
    kernel: &[F],
    background: &mut [F],
    height: i32,
    width: i32,
    ry: i32,
    rx: i32,
    #[comptime] k_h: u32,
    #[comptime] k_w: u32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;

        let mut best = F::new(-1.0e30f32);
        let mut ky = 0u32;
        while ky < k_h {
            let dy = (ky as i32) - ry;
            let py = i32::max(pad_index(y - dy, height, mode), 0i32) as u32;
            let k_row = ky * k_w;
            let mut kx = 0u32;
            while kx < k_w {
                let kval = kernel[(k_row + kx) as usize];
                if kval < F::new(1.0e20f32) {
                    let dx = (kx as i32) - rx;
                    let px = i32::max(pad_index(x - dx, width, mode), 0i32) as u32;
                    let cand = eroded[(py * w_u + px) as usize] - kval;
                    best = F::max(best, cand);
                }
                kx += 1u32;
            }
            ky += 1u32;
        }
        background[idx] = best;
    }
}

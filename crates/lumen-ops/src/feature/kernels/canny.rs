//! Canny gradient magnitude, quantized orientation sector, and non-maximum suppression kernels.
#![allow(unused_assignments)]

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Computes gradient magnitude $\sqrt{g_y^2 + g_x^2}$ and quantizes gradient angle into 4 sectors:
/// - `0`: horizontal (`|gx| >= |gy|` with `|gy| <= tan(22.5°) |gx|`)
/// - `1`: main diagonal (`gx` and `gy` have same sign, `45°`)
/// - `2`: vertical (`|gy| > |gx|` with `|gx| <= tan(22.5°) |gy|`)
/// - `3`: anti-diagonal (`gx` and `gy` have opposite signs, `135°`)
#[cube(launch)]
pub fn canny_grad_dir_kernel<F: lumen_core::element::ImageFloat>(
    gy: &[F],
    gx: &[F],
    mag: &mut [F],
    sector: &mut [u32],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let vy = f32::cast_from(gy[i]);
        let vx = f32::cast_from(gx[i]);
        let m = f32::sqrt(vy * vy + vx * vx);
        mag[i] = F::cast_from(m);

        let ay = f32::abs(vy);
        let ax = f32::abs(vx);
        let tan22_5 = 0.414_213_57f32;
        let sec = if ay <= ax * tan22_5 {
            0u32
        } else if ax <= ay * tan22_5 {
            2u32
        } else if (vy * vx) >= 0.0f32 {
            1u32
        } else {
            3u32
        };
        sector[i] = sec;
    }
}

/// Non-maximum suppression along the quantized gradient direction in `[height, width]`.
#[cube(launch)]
#[allow(unused_assignments)]
pub fn canny_nms_kernel<F: lumen_core::element::ImageFloat>(
    mag: &[F],
    sector: &[u32],
    nms: &mut [F],
    height: u32,
    width: u32,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let y = idx / width;
        let x = idx % width;
        if y == 0u32 || x == 0u32 || (y + 1u32) >= height || (x + 1u32) >= width {
            nms[i] = F::new(0.0f32);
        } else {
            let m = mag[i];
            let s = sector[i];
            let mut n1 = F::new(0.0f32);
            let mut n2 = F::new(0.0f32);
            if s == 0u32 {
                n1 = mag[(idx - 1u32) as usize];
                n2 = mag[(idx + 1u32) as usize];
            } else if s == 2u32 {
                n1 = mag[(idx - width) as usize];
                n2 = mag[(idx + width) as usize];
            } else if s == 1u32 {
                n1 = mag[(idx - width - 1u32) as usize];
                n2 = mag[(idx + width + 1u32) as usize];
            } else {
                n1 = mag[(idx - width + 1u32) as usize];
                n2 = mag[(idx + width - 1u32) as usize];
            }
            if m >= n1 && m >= n2 {
                nms[i] = m;
            } else {
                nms[i] = F::new(0.0f32);
            }
        }
    }
}

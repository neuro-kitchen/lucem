//! B-spline prefiltering and N-D coordinate interpolation kernels (orders 0, 1, 3).

use cubecl::prelude::*;
use lucem_core::compute::{element_position, line_position};

use crate::core::kernels::pad_index;

/// Causal + anti-causal recursive IIR B-spline pole pass along one axis `[outer, len, inner]`:
/// scales the line by `gain = (1 - z) * (1 - 1/z)` and applies pole `z` forward and backward.
#[cube(launch)]
pub fn spline_pole_1d_kernel<F: lucem_core::element::ImageFloat>(
    data: &mut [F],
    len: u32,
    inner: u32,
    lines: u32,
    pole: F,
    gain: F,
) {
    let line = line_position();
    if line < lines && len > 1u32 {
        let k = line % inner;
        let o = line / inner;
        let base = o * len * inner + k;

        let mut i = 0u32;
        while i < len {
            let idx = (base + i * inner) as usize;
            data[idx] *= gain;
            i += 1u32;
        }

        // Initial causal boundary sum (truncated mirror expansion, 16 terms suffices for f32/f64)
        let mut c0 = data[base as usize];
        let mut zn = pole;
        let terms = u32::min(len, 16u32);
        let mut m = 1u32;
        while m < terms {
            c0 += zn * data[(base + m * inner) as usize];
            zn *= pole;
            m += 1u32;
        }
        data[base as usize] = c0;

        let mut f_idx = 1u32;
        while f_idx < len {
            let cur = (base + f_idx * inner) as usize;
            let prev = (base + (f_idx - 1u32) * inner) as usize;
            data[cur] += pole * data[prev];
            f_idx += 1u32;
        }

        // Anti-causal initialization at right boundary
        let last = (base + (len - 1u32) * inner) as usize;
        let denom = pole * pole - F::new(1.0f32);
        let prev_last = (base + (len - 2u32) * inner) as usize;
        data[last] = (pole / denom) * (data[last] + pole * data[prev_last]);

        let mut b_idx = len - 1u32;
        while b_idx > 0u32 {
            let idx_cur = (base + (b_idx - 1u32) * inner) as usize;
            let idx_next = (base + b_idx * inner) as usize;
            data[idx_cur] = pole * (data[idx_next] - data[idx_cur]);
            b_idx -= 1u32;
        }
    }
}

/// Evaluates cubic B-spline basis weight $B_3(t)$ for $|t| = |x - k|$.
#[cube]
pub fn cubic_bspline_weight<F: Float>(dist: F) -> F {
    let d = F::abs(dist);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let inv6 = F::new(1.0f32 / 6.0f32);
    let two_thirds = F::new(2.0f32 / 3.0f32);
    if d < one {
        two_thirds - d * d + F::new(0.5f32) * d * d * d
    } else if d < two {
        let t = two - d;
        t * t * t * inv6
    } else {
        F::new(0.0f32)
    }
}

/// 2-D coordinate interpolation at `(coords_y[i], coords_x[i])` for `order` (`0`, `1`, or `3` over
/// spline coefficients).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn sample_2d_kernel<F: lucem_core::element::ImageFloat>(
    coeffs: &[F],
    coords_y: &[F],
    coords_x: &[F],
    output: &mut [F],
    height: i32,
    width: i32,
    cval: F,
    num_points: u32,
    #[comptime] order: u32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    if (idx as u32) < num_points {
        let y = coords_y[idx];
        let x = coords_x[idx];
        if comptime!(order == 0u32) {
            let iy = i32::cast_from(F::floor(y + F::new(0.5f32)));
            let ix = i32::cast_from(F::floor(x + F::new(0.5f32)));
            let py = pad_index(iy, height, mode);
            let px = pad_index(ix, width, mode);
            let sy = i32::max(py, 0i32) as u32;
            let sx = i32::max(px, 0i32) as u32;
            let v = coeffs[(sy * (width as u32) + sx) as usize];
            output[idx] = if py < 0i32 || px < 0i32 { cval } else { v };
        } else if comptime!(order == 1u32) {
            let y0 = i32::cast_from(F::floor(y));
            let x0 = i32::cast_from(F::floor(x));
            let fy = y - F::floor(y);
            let fx = x - F::floor(x);
            let one = F::new(1.0f32);
            let mut acc = F::new(0.0f32);
            let mut dy = 0i32;
            while dy <= 1i32 {
                let wy = if dy == 0i32 { one - fy } else { fy };
                let py = pad_index(y0 + dy, height, mode);
                let sy = i32::max(py, 0i32) as u32;
                let mut dx = 0i32;
                while dx <= 1i32 {
                    let wx = if dx == 0i32 { one - fx } else { fx };
                    let px = pad_index(x0 + dx, width, mode);
                    let sx = i32::max(px, 0i32) as u32;
                    let raw = coeffs[(sy * (width as u32) + sx) as usize];
                    let sample = if py < 0i32 || px < 0i32 { cval } else { raw };
                    acc += wy * wx * sample;
                    dx += 1i32;
                }
                dy += 1i32;
            }
            output[idx] = acc;
        } else {
            let y_base = i32::cast_from(F::floor(y)) - 1i32;
            let x_base = i32::cast_from(F::floor(x)) - 1i32;
            let mut acc = F::new(0.0f32);
            let mut ky = 0i32;
            while ky < 4i32 {
                let iy = y_base + ky;
                let wy = cubic_bspline_weight::<F>(y - F::cast_from(iy));
                let py = pad_index(iy, height, mode);
                let sy = i32::max(py, 0i32) as u32;
                let mut kx = 0i32;
                while kx < 4i32 {
                    let ix = x_base + kx;
                    let wx = cubic_bspline_weight::<F>(x - F::cast_from(ix));
                    let px = pad_index(ix, width, mode);
                    let sx = i32::max(px, 0i32) as u32;
                    let raw = coeffs[(sy * (width as u32) + sx) as usize];
                    let sample = if py < 0i32 || px < 0i32 { cval } else { raw };
                    acc += wy * wx * sample;
                    kx += 1i32;
                }
                ky += 1i32;
            }
            output[idx] = acc;
        }
    }
}

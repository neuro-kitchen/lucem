//! Texture kernels: normalized cross-correlation (NCC) template matching and Local Binary Patterns (LBP).

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Zero-mean Normalized Cross-Correlation (ZNCC) template matching at valid or padded positions.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn ncc_template_2d_kernel<F: lumen_core::element::ImageFloat>(
    image: &[F],
    template_zm: &[F],
    out: &mut [F],
    img_h: i32,
    img_w: i32,
    tpl_h: i32,
    tpl_w: i32,
    out_h: u32,
    out_w: u32,
    origin_y: i32,
    origin_x: i32,
    tpl_norm: F,
    eps: F,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let oy = (idx / out_w) as i32;
        let ox = (idx % out_w) as i32;
        let top = oy - origin_y;
        let left = ox - origin_x;

        let n = F::cast_from(tpl_h * tpl_w);
        let mut sum_i = F::new(0.0f32);
        let mut ty = 0i32;
        while ty < tpl_h {
            let iy = i32::max(i32::min(top + ty, img_h - 1i32), 0i32);
            let mut tx = 0i32;
            while tx < tpl_w {
                let ix = i32::max(i32::min(left + tx, img_w - 1i32), 0i32);
                sum_i += image[(iy * img_w + ix) as usize];
                tx += 1i32;
            }
            ty += 1i32;
        }
        let mean_i = sum_i / n;

        let mut dot = F::new(0.0f32);
        let mut sq_i = F::new(0.0f32);
        let mut r = 0i32;
        while r < tpl_h {
            let iy = i32::max(i32::min(top + r, img_h - 1i32), 0i32);
            let mut c = 0i32;
            while c < tpl_w {
                let ix = i32::max(i32::min(left + c, img_w - 1i32), 0i32);
                let diff_i = image[(iy * img_w + ix) as usize] - mean_i;
                let t_val = template_zm[(r * tpl_w + c) as usize];
                dot += diff_i * t_val;
                sq_i += diff_i * diff_i;
                c += 1i32;
            }
            r += 1i32;
        }
        let denom = F::sqrt(sq_i) * tpl_norm;
        if denom <= eps {
            out[i] = F::new(0.0f32);
        } else {
            let _ = out_h;
            out[i] = dot / denom;
        }
    }
}

/// Circular Local Binary Pattern (LBP) with `num_points` (`<= 32`) on a circle of `radius` around
/// each pixel, using bilinear interpolation at precomputed `(dy[p], dx[p])` offsets.
///
/// - `method == 0`: default raw bit code $\sum_{p=0}^{P-1} s(g_p - g_c) 2^p$
/// - `method == 1`: rotation-invariant (`ror`) minimum cyclic bit shift
/// - `method == 2`: uniform (`uniform`) count of positive bits when bitwise transitions $\le 2$, else $P + 1$
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn lbp_2d_kernel<F: lumen_core::element::ImageFloat>(
    image: &[F],
    offsets_y: &[F],
    offsets_x: &[F],
    out: &mut [u32],
    height: i32,
    width: i32,
    num_points: u32,
    total: u32,
    #[comptime] method: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let y = (idx / (width as u32)) as i32;
        let x = (idx % (width as u32)) as i32;
        let center = image[i];
        let yf = F::cast_from(y);
        let xf = F::cast_from(x);
        let one = F::new(1.0f32);
        let eps = F::new(1e-5f32);

        let mut code = 0u32;
        let mut p = 0u32;
        while p < num_points {
            let py = yf + offsets_y[p as usize];
            let px = xf + offsets_x[p as usize];
            let y0 = i32::cast_from(F::floor(py));
            let x0 = i32::cast_from(F::floor(px));
            let fy = py - F::floor(py);
            let fx = px - F::floor(px);

            let y0c = i32::max(i32::min(y0, height - 1i32), 0i32);
            let x0c = i32::max(i32::min(x0, width - 1i32), 0i32);
            let y1c = i32::max(i32::min(y0 + 1i32, height - 1i32), 0i32);
            let x1c = i32::max(i32::min(x0 + 1i32, width - 1i32), 0i32);

            let v00 = image[(y0c * width + x0c) as usize];
            let v01 = image[(y0c * width + x1c) as usize];
            let v10 = image[(y1c * width + x0c) as usize];
            let v11 = image[(y1c * width + x1c) as usize];

            let sample = (one - fy) * ((one - fx) * v00 + fx * v01)
                + fy * ((one - fx) * v10 + fx * v11);
            if (sample + eps) >= center {
                code |= 1u32 << p;
            }
            p += 1u32;
        }

        if comptime!(method == 0u32) {
            out[i] = code;
        } else if comptime!(method == 1u32) {
            let mask = if num_points >= 32u32 { 0xFFFF_FFFFu32 } else { (1u32 << num_points) - 1u32 };
            let mut min_code = code;
            let mut cur = code;
            let mut s = 1u32;
            while s < num_points {
                let lsb = cur & 1u32;
                cur = ((cur >> 1u32) | (lsb << (num_points - 1u32))) & mask;
                if cur < min_code {
                    min_code = cur;
                }
                s += 1u32;
            }
            out[i] = min_code;
        } else {
            // Uniform LBP: count bitwise 0->1 and 1->0 transitions around the ring
            let mut transitions = 0u32;
            let mut ones = 0u32;
            let mut k = 0u32;
            while k < num_points {
                let b0 = (code >> k) & 1u32;
                let next_k = (k + 1u32) % num_points;
                let b1 = (code >> next_k) & 1u32;
                ones += b0;
                if b0 != b1 {
                    transitions += 1u32;
                }
                k += 1u32;
            }
            if transitions <= 2u32 {
                out[i] = ones;
            } else {
                out[i] = num_points + 1u32;
            }
        }
    }
}

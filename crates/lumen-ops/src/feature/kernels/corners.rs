//! Corner detector response kernels: Harris, Shi-Tomasi, Kitchen-Rosenfeld, Förstner, Moravec, and FAST-9.
#![allow(unused_assignments)]

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Harris corner response:
/// - `method == 0`: $\det(A) - k\,\operatorname{tr}(A)^2$
/// - `method == 1`: Noble $\frac{2\det(A)}{\operatorname{tr}(A) + \varepsilon}$
#[cube(launch)]
pub fn harris_2d_kernel<F: lumen_core::element::ImageFloat>(
    ayy: &[F],
    ayx: &[F],
    axx: &[F],
    out: &mut [F],
    k: F,
    eps: F,
    len: u32,
    #[comptime] method: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let a = ayy[i];
        let b = ayx[i];
        let c = axx[i];
        let det = a * c - b * b;
        let tr = a + c;
        if comptime!(method == 0u32) {
            out[i] = det - k * tr * tr;
        } else {
            out[i] = (F::new(2.0f32) * det) / (tr + eps);
        }
    }
}

/// Shi-Tomasi (Kanade-Tomasi) minimum eigenvalue $\lambda_{\min} = \frac{1}{2}\left((a+c) - \sqrt{(a-c)^2 + 4b^2}\right)$.
#[cube(launch)]
pub fn shi_tomasi_2d_kernel<F: lumen_core::element::ImageFloat>(
    ayy: &[F],
    ayx: &[F],
    axx: &[F],
    out: &mut [F],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let a = ayy[i];
        let b = ayx[i];
        let c = axx[i];
        let half = F::new(0.5f32);
        let mean = (a + c) * half;
        let diff = (a - c) * half;
        let radius = F::sqrt(diff * diff + b * b);
        out[i] = mean - radius;
    }
}

/// Kitchen-Rosenfeld corner measure:
/// $\frac{I_{xx} I_y^2 + I_{yy} I_x^2 - 2 I_{xy} I_x I_y}{I_x^2 + I_y^2 + \varepsilon}$.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn kitchen_rosenfeld_2d_kernel<F: lumen_core::element::ImageFloat>(
    iy: &[F],
    ix: &[F],
    iyy: &[F],
    iyx: &[F],
    ixx: &[F],
    out: &mut [F],
    eps: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let gy = iy[i];
        let gx = ix[i];
        let g2 = gx * gx + gy * gy;
        if g2 <= eps {
            out[i] = F::new(0.0f32);
        } else {
            let num = ixx[i] * gy * gy + iyy[i] * gx * gx - F::new(2.0f32) * iyx[i] * gx * gy;
            out[i] = F::abs(num) / g2;
        }
    }
}

/// Förstner interest operator producing error ellipse size $w = \frac{\det(A)}{\operatorname{tr}(A)}$
/// and roundness $q = \frac{4\det(A)}{\operatorname{tr}(A)^2}$.
#[cube(launch)]
pub fn foerstner_2d_kernel<F: lumen_core::element::ImageFloat>(
    ayy: &[F],
    ayx: &[F],
    axx: &[F],
    w_out: &mut [F],
    q_out: &mut [F],
    eps: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let a = ayy[i];
        let b = ayx[i];
        let c = axx[i];
        let det = a * c - b * b;
        let tr = a + c;
        if F::abs(tr) <= eps {
            w_out[i] = F::new(0.0f32);
            q_out[i] = F::new(0.0f32);
        } else {
            w_out[i] = det / tr;
            q_out[i] = (F::new(4.0f32) * det) / (tr * tr);
        }
    }
}

/// Moravec corner response: minimum directional SSD over 4 shifts `(1,0), (0,1), (1,1), (1,-1)` in a
/// `(2*window_radius + 1)^2` patch.
#[cube(launch)]
pub fn moravec_2d_kernel<F: lumen_core::element::ImageFloat>(
    image: &[F],
    out: &mut [F],
    height: i32,
    width: i32,
    window_radius: i32,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let y = (idx / (width as u32)) as i32;
        let x = (idx % (width as u32)) as i32;
        let margin = window_radius + 1i32;
        if y < margin || x < margin || y >= (height - margin) || x >= (width - margin) {
            out[i] = F::new(0.0f32);
        } else {
            let mut ssd0 = F::new(0.0f32);
            let mut ssd1 = F::new(0.0f32);
            let mut ssd2 = F::new(0.0f32);
            let mut ssd3 = F::new(0.0f32);

            let mut wy = -window_radius;
            while wy <= window_radius {
                let mut wx = -window_radius;
                while wx <= window_radius {
                    let cy = y + wy;
                    let cx = x + wx;
                    let c = image[(cy * width + cx) as usize];
                    let d0 = image[((cy + 1i32) * width + cx) as usize] - c;
                    let d1 = image[(cy * width + cx + 1i32) as usize] - c;
                    let d2 = image[((cy + 1i32) * width + cx + 1i32) as usize] - c;
                    let d3 = image[((cy + 1i32) * width + cx - 1i32) as usize] - c;
                    ssd0 += d0 * d0;
                    ssd1 += d1 * d1;
                    ssd2 += d2 * d2;
                    ssd3 += d3 * d3;
                    wx += 1i32;
                }
                wy += 1i32;
            }
            let m01 = F::min(ssd0, ssd1);
            let m23 = F::min(ssd2, ssd3);
            out[i] = F::min(m01, m23);
        }
    }
}

/// Rosten & Drummond FAST Bresenham circle of radius 3 (16 pixels): computes the corner score for
/// at least `min_contiguous` contiguous brighter (`p_k > p_0 + threshold`) or darker
/// (`p_k < p_0 - threshold`) ring pixels.
#[cube(launch)]
pub fn fast9_score_kernel<F: lumen_core::element::ImageFloat>(
    image: &[F],
    out: &mut [F],
    height: i32,
    width: i32,
    min_contiguous: u32,
    threshold: F,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let y = (idx / (width as u32)) as i32;
        let x = (idx % (width as u32)) as i32;
        if y < 3i32 || x < 3i32 || y >= (height - 3i32) || x >= (width - 3i32) {
            out[i] = F::new(0.0f32);
        } else {
            let center = image[i];
            let hi = center + threshold;
            let lo = center - threshold;

            // Unroll 31 steps around the 16-pixel Bresenham circle to find longest contiguous run
            let mut run_bright = 0u32;
            let mut run_dark = 0u32;
            let mut max_bright = 0u32;
            let mut max_dark = 0u32;
            let mut sum_bright = F::new(0.0f32);
            let mut sum_dark = F::new(0.0f32);

            let mut step = 0u32;
            while step < 31u32 {
                let k = step % 16u32;
                let mut dy = 0i32;
                let mut dx = 0i32;
                if k == 0u32 { dy = -3i32; dx = 0i32; }
                else if k == 1u32 { dy = -3i32; dx = 1i32; }
                else if k == 2u32 { dy = -2i32; dx = 2i32; }
                else if k == 3u32 { dy = -1i32; dx = 3i32; }
                else if k == 4u32 { dy = 0i32; dx = 3i32; }
                else if k == 5u32 { dy = 1i32; dx = 3i32; }
                else if k == 6u32 { dy = 2i32; dx = 2i32; }
                else if k == 7u32 { dy = 3i32; dx = 1i32; }
                else if k == 8u32 { dy = 3i32; dx = 0i32; }
                else if k == 9u32 { dy = 3i32; dx = -1i32; }
                else if k == 10u32 { dy = 2i32; dx = -2i32; }
                else if k == 11u32 { dy = 1i32; dx = -3i32; }
                else if k == 12u32 { dy = 0i32; dx = -3i32; }
                else if k == 13u32 { dy = -1i32; dx = -3i32; }
                else if k == 14u32 { dy = -2i32; dx = -2i32; }
                else { dy = -3i32; dx = -1i32; }

                let val = image[((y + dy) * width + (x + dx)) as usize];
                if val > hi {
                    run_bright += 1u32;
                    if step < 16u32 {
                        sum_bright += (val - center) - threshold;
                    }
                    if run_bright > max_bright {
                        max_bright = run_bright;
                    }
                } else {
                    run_bright = 0u32;
                }

                if val < lo {
                    run_dark += 1u32;
                    if step < 16u32 {
                        sum_dark += (center - val) - threshold;
                    }
                    if run_dark > max_dark {
                        max_dark = run_dark;
                    }
                } else {
                    run_dark = 0u32;
                }

                step += 1u32;
            }

            let mut score = F::new(0.0f32);
            if max_bright >= min_contiguous {
                score = F::max(score, sum_bright);
            }
            if max_dark >= min_contiguous {
                score = F::max(score, sum_dark);
            }
            out[i] = score;
        }
    }
}

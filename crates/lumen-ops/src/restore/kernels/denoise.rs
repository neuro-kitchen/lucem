//! Denoising kernels: Chambolle primal-dual TV, Bregman split TV, bilateral filter, non-local means,
//! and Immerkær 3x3 Laplacian noise estimation stencil.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

use crate::core::kernels::pad_index;

/// Chambolle 2-D divergence update: `u[y, x] = f[y, x] + weight * div(p)[y, x]`, where
/// `div(p)[y, x] = (py[y, x] - py[y-1, x]) + (px[y, x] - px[y, x-1])` with zero Dirichlet boundary
/// on `p` (`py[height-1, x] = 0`, `px[y, width-1] = 0`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn tv_chambolle_2d_div_kernel<F: lumen_core::element::ImageFloat>(
    f_img: &[F],
    py: &[F],
    px: &[F],
    u_out: &mut [F],
    weight: F,
    height: u32,
    width: u32,
    total: u32,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let e = idx as u32;
        let y = e / width;
        let x = e % width;

        let py_cur = if y + 1u32 < height { py[idx] } else { F::new(0.0f32) };
        let py_prev_idx = if y > 0u32 { (y - 1u32) * width + x } else { 0u32 };
        let py_prev = if y > 0u32 { py[py_prev_idx as usize] } else { F::new(0.0f32) };

        let px_cur = if x + 1u32 < width { px[idx] } else { F::new(0.0f32) };
        let px_prev_idx = if x > 0u32 { y * width + (x - 1u32) } else { 0u32 };
        let px_prev = if x > 0u32 { px[px_prev_idx as usize] } else { F::new(0.0f32) };

        let div = (py_cur - py_prev) + (px_cur - px_prev);
        u_out[idx] = f_img[idx] + weight * div;
    }
}

/// Chambolle 2-D dual projection step: computes forward differences `g = grad(u)` and updates
/// `p_new = (p + (tau / weight) * g) / (1 + (tau / weight) * |g|)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn tv_chambolle_2d_dual_kernel<F: lumen_core::element::ImageFloat>(
    u_img: &[F],
    py_in: &[F],
    px_in: &[F],
    py_out: &mut [F],
    px_out: &mut [F],
    step: F,
    height: u32,
    width: u32,
    total: u32,
    #[comptime] isotropic: bool,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let e = idx as u32;
        let y = e / width;
        let x = e % width;
        let u0 = u_img[idx];

        let ny_idx = if y + 1u32 < height { (y + 1u32) * width + x } else { e };
        let nx_idx = if x + 1u32 < width { y * width + (x + 1u32) } else { e };
        let gy = if y + 1u32 < height { u_img[ny_idx as usize] - u0 } else { F::new(0.0f32) };
        let gx = if x + 1u32 < width { u_img[nx_idx as usize] - u0 } else { F::new(0.0f32) };

        let tilde_py = py_in[idx] + step * gy;
        let tilde_px = px_in[idx] + step * gx;

        if comptime!(isotropic) {
            let norm = F::sqrt(gy * gy + gx * gx);
            let denom = F::new(1.0f32) + step * norm;
            py_out[idx] = tilde_py / denom;
            px_out[idx] = tilde_px / denom;
        } else {
            py_out[idx] = tilde_py / (F::new(1.0f32) + step * F::abs(gy));
            px_out[idx] = tilde_px / (F::new(1.0f32) + step * F::abs(gx));
        }
    }
}

/// Split-Bregman 2-D Jacobi/Gauss-Seidel primal update:
/// `u_new[y, x] = (mu * f[y, x] + lambda * (sum_neighbors(u) + div(b - d))) / (mu + 4 * lambda)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn tv_bregman_2d_gauss_seidel_kernel<F: lumen_core::element::ImageFloat>(
    f_img: &[F],
    u_in: &[F],
    dy: &[F],
    dx: &[F],
    by: &[F],
    bx: &[F],
    u_out: &mut [F],
    weight: F,
    lam: F,
    height: u32,
    width: u32,
    total: u32,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let e = idx as u32;
        let y = e / width;
        let x = e % width;

        let up_y = if y > 0u32 { y - 1u32 } else { 0u32 };
        let dn_y = if y + 1u32 < height { y + 1u32 } else { height - 1u32 };
        let lf_x = if x > 0u32 { x - 1u32 } else { 0u32 };
        let rt_x = if x + 1u32 < width { x + 1u32 } else { width - 1u32 };

        let u_up = u_in[(up_y * width + x) as usize];
        let u_dn = u_in[(dn_y * width + x) as usize];
        let u_lf = u_in[(y * width + lf_x) as usize];
        let u_rt = u_in[(y * width + rt_x) as usize];

        let up_idx = (up_y * width + x) as usize;
        let lf_idx = (y * width + lf_x) as usize;

        let d_by_cur = if y + 1u32 < height { dy[idx] - by[idx] } else { F::new(0.0f32) };
        let d_by_up = if y > 0u32 { dy[up_idx] - by[up_idx] } else { F::new(0.0f32) };
        let d_bx_cur = if x + 1u32 < width { dx[idx] - bx[idx] } else { F::new(0.0f32) };
        let d_bx_lf = if x > 0u32 { dx[lf_idx] - bx[lf_idx] } else { F::new(0.0f32) };

        let div_db = (d_by_up - d_by_cur) + (d_bx_lf - d_bx_cur);
        let rhs = weight * f_img[idx] + lam * (u_up + u_dn + u_lf + u_rt + div_db);
        let denom = weight + F::new(4.0f32) * lam;
        u_out[idx] = rhs / denom;
    }
}

/// Split-Bregman 2-D shrinkage and Bregman variable update (`d` and `b`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn tv_bregman_2d_dual_kernel<F: lumen_core::element::ImageFloat>(
    u_img: &[F],
    by_in: &[F],
    bx_in: &[F],
    dy_out: &mut [F],
    dx_out: &mut [F],
    by_out: &mut [F],
    bx_out: &mut [F],
    inv_lam: F,
    height: u32,
    width: u32,
    total: u32,
    #[comptime] isotropic: bool,
) {
    let idx = element_position();
    if (idx as u32) < total {
        let e = idx as u32;
        let y = e / width;
        let x = e % width;
        let u0 = u_img[idx];

        let ny_idx = if y + 1u32 < height { (y + 1u32) * width + x } else { e };
        let nx_idx = if x + 1u32 < width { y * width + (x + 1u32) } else { e };
        let gy = if y + 1u32 < height { u_img[ny_idx as usize] - u0 } else { F::new(0.0f32) };
        let gx = if x + 1u32 < width { u_img[nx_idx as usize] - u0 } else { F::new(0.0f32) };

        let sy = gy + by_in[idx];
        let sx = gx + bx_in[idx];

        if comptime!(isotropic) {
            let norm = F::sqrt(sy * sy + sx * sx);
            let scale = if norm > inv_lam { (norm - inv_lam) / norm } else { F::new(0.0f32) };
            let dy_new = scale * sy;
            let dx_new = scale * sx;
            dy_out[idx] = dy_new;
            dx_out[idx] = dx_new;
            by_out[idx] = sy - dy_new;
            bx_out[idx] = sx - dx_new;
        } else {
            let abs_y = F::abs(sy);
            let abs_x = F::abs(sx);
            let dy_new = if abs_y > inv_lam {
                (abs_y - inv_lam) * (sy / abs_y)
            } else {
                F::new(0.0f32)
            };
            let dx_new = if abs_x > inv_lam {
                (abs_x - inv_lam) * (sx / abs_x)
            } else {
                F::new(0.0f32)
            };
            dy_out[idx] = dy_new;
            dx_out[idx] = dx_new;
            by_out[idx] = sy - dy_new;
            bx_out[idx] = sx - dx_new;
        }
    }
}

/// 2-D edge-preserving bilateral filter with precomputed spatial Gaussian weights `[2*r+1, 2*r+1]`
/// and range Gaussian factor `inv_two_sigma_color_sq = -0.5 / sigma_color^2`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn bilateral_2d_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    spatial_weights: &[F],
    output: &mut [F],
    inv_two_sigma_color_sq: F,
    cval: F,
    height: i32,
    width: i32,
    #[comptime] radius: i32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;
        let center = input[idx];
        let win = 2i32 * radius + 1i32;

        let mut num = F::new(0.0f32);
        let mut den = F::new(0.0f32);
        // From a literal: a variable seeded from a #[comptime] value stays comptime (CPU MLIR rejects it)
        let mut dy_i = 0i32;
        while dy_i <= radius + radius {
            let dy = dy_i - radius;
            let py = pad_index(y + dy, height, mode);
            let sy = i32::max(py, 0i32) as u32;
            let wy_base = ((dy + radius) * win) as u32;
            // From a literal: a variable seeded from a #[comptime] value stays comptime (CPU MLIR rejects it)
        let mut dx_i = 0i32;
            while dx_i <= radius + radius {
                let dx = dx_i - radius;
                let px = pad_index(x + dx, width, mode);
                let sx = i32::max(px, 0i32) as u32;
                let raw = input[(sy * w_u + sx) as usize];
                let sample = if py < 0i32 || px < 0i32 { cval } else { raw };
                let diff = sample - center;
                let w_spatial = spatial_weights[(wy_base + ((dx + radius) as u32)) as usize];
                let w_range = F::exp(diff * diff * inv_two_sigma_color_sq);
                let w = w_spatial * w_range;
                num += w * sample;
                den += w;
                dx_i += 1i32;
            }
            dy_i += 1i32;
        }
        output[idx] = if den > F::new(0.0f32) { num / den } else { center };
    }
}

/// 2-D patch-based Non-Local Means (NLM) denoising kernel.
///
/// Compares `(2 * patch_radius + 1)^2` neighborhoods across a `(2 * search_radius + 1)^2` window.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn nlm_2d_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    inv_h2: F,
    sigma_sq: F,
    inv_patch_area: F,
    height: i32,
    width: i32,
    #[comptime] patch_radius: i32,
    #[comptime] search_radius: i32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;

        let mut num = F::new(0.0f32);
        let mut den = F::new(0.0f32);

        // From a literal: a variable seeded from a #[comptime] value stays comptime (CPU MLIR rejects it)
        let mut sy_i = 0i32;
        while sy_i <= search_radius + search_radius {
            let sy = sy_i - search_radius;
            let qy_center = y + sy;
            // From a literal: a variable seeded from a #[comptime] value stays comptime (CPU MLIR rejects it)
        let mut sx_i = 0i32;
            while sx_i <= search_radius + search_radius {
                let sx = sx_i - search_radius;
                let qx_center = x + sx;

                let mut dist_sq = F::new(0.0f32);
                // From a literal: a variable seeded from a #[comptime] value stays comptime (CPU MLIR rejects it)
        let mut py_i = 0i32;
                while py_i <= patch_radius + patch_radius {
                    let py = py_i - patch_radius;
                    let p_ry = i32::max(pad_index(y + py, height, mode), 0i32) as u32;
                    let q_ry = i32::max(pad_index(qy_center + py, height, mode), 0i32) as u32;
                    // From a literal: a variable seeded from a #[comptime] value stays comptime (CPU MLIR rejects it)
        let mut px_i = 0i32;
                    while px_i <= patch_radius + patch_radius {
                        let px = px_i - patch_radius;
                        let p_rx = i32::max(pad_index(x + px, width, mode), 0i32) as u32;
                        let q_rx = i32::max(pad_index(qx_center + px, width, mode), 0i32) as u32;
                        let vp = input[(p_ry * w_u + p_rx) as usize];
                        let vq = input[(q_ry * w_u + q_rx) as usize];
                        let d = vp - vq;
                        dist_sq += d * d;
                        px_i += 1i32;
                    }
                    py_i += 1i32;
                }

                let mean_dist = F::max(dist_sq * inv_patch_area - F::new(2.0f32) * sigma_sq, F::new(0.0f32));
                let w = F::exp(-mean_dist * inv_h2);
                let q_cy = i32::max(pad_index(qy_center, height, mode), 0i32) as u32;
                let q_cx = i32::max(pad_index(qx_center, width, mode), 0i32) as u32;
                let q_val = input[(q_cy * w_u + q_cx) as usize];

                num += w * q_val;
                den += w;
                sx_i += 1i32;
            }
            sy_i += 1i32;
        }
        output[idx] = if den > F::new(0.0f32) { num / den } else { input[idx] };
    }
}

/// Immerkær (1996) 3x3 zero-mean Laplacian stencil `[1, -2, 1; -2, 4, -2; 1, -2, 1]` on interior
/// pixels `(height - 2) * (width - 2)`, outputting `|L * I|` for fast Gaussian noise estimation.
#[cube(launch)]
pub fn immerkaer_laplacian_2d_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    out_abs: &mut [F],
    inner_h: u32,
    inner_w: u32,
    width: u32,
) {
    let idx = element_position();
    let total = inner_h * inner_w;
    if (idx as u32) < total {
        let e = idx as u32;
        let iy = e / inner_w;
        let ix = e % inner_w;
        let y = iy + 1u32;
        let x = ix + 1u32;

        let r0 = (y - 1u32) * width;
        let r1 = y * width;
        let r2 = (y + 1u32) * width;

        let c00 = input[(r0 + x - 1u32) as usize];
        let c01 = input[(r0 + x) as usize];
        let c02 = input[(r0 + x + 1u32) as usize];
        let c10 = input[(r1 + x - 1u32) as usize];
        let c11 = input[(r1 + x) as usize];
        let c12 = input[(r1 + x + 1u32) as usize];
        let c20 = input[(r2 + x - 1u32) as usize];
        let c21 = input[(r2 + x) as usize];
        let c22 = input[(r2 + x + 1u32) as usize];

        let corners = c00 + c02 + c20 + c22;
        let edges = c01 + c10 + c12 + c21;
        let resp = corners - F::new(2.0f32) * edges + F::new(4.0f32) * c11;
        out_abs[idx] = F::abs(resp);
    }
}

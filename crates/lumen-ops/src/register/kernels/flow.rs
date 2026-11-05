//! Dense optical flow kernels: bilinear warping with central spatial derivatives, Iterative
//! Lucas-Kanade (ILK) structure-tensor/RHS formation and $2 \times 2$ solver, and Zach-Pock-Bischof
//! Dual TV-$L_1$ primal-dual updates.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

use crate::core::kernels::pad_index;
use crate::core::pad::PAD_NEAREST;

/// Bilinearly samples `img` at `(y, x)` with nearest-edge clamping.
#[cube]
fn sample_bilinear_nearest<F: Float>(img: &[F], y: F, x: F, height: i32, width: i32) -> F {
    let y0 = F::floor(y);
    let x0 = F::floor(x);
    let fy = y - y0;
    let fx = x - x0;
    let iy0 = i32::cast_from(y0);
    let ix0 = i32::cast_from(x0);

    let r0 = i32::max(pad_index(iy0, height, PAD_NEAREST), 0i32) as u32;
    let r1 = i32::max(pad_index(iy0 + 1i32, height, PAD_NEAREST), 0i32) as u32;
    let c0 = i32::max(pad_index(ix0, width, PAD_NEAREST), 0i32) as u32;
    let c1 = i32::max(pad_index(ix0 + 1i32, width, PAD_NEAREST), 0i32) as u32;
    let w_u = width as u32;

    let v00 = img[(r0 * w_u + c0) as usize];
    let v01 = img[(r0 * w_u + c1) as usize];
    let v10 = img[(r1 * w_u + c0) as usize];
    let v11 = img[(r1 * w_u + c1) as usize];

    let one = F::new(1.0f32);
    let top = (one - fx) * v00 + fx * v01;
    let bot = (one - fx) * v10 + fx * v11;
    (one - fy) * top + fy * bot
}

/// Warps `img1` by displacement `(vy, vx)` and computes the temporal residual
/// `it = img1(y + vy, x + vx) - img0(y, x)` and blended central spatial derivatives `(iy, ix)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn warp_bilinear_grad_2d_kernel<F: Float>(
    img0: &[F],
    img1: &[F],
    vy: &[F],
    vx: &[F],
    iy_out: &mut [F],
    ix_out: &mut [F],
    it_out: &mut [F],
    height: i32,
    width: i32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y_i = (e / w_u) as i32;
        let x_i = (e % w_u) as i32;
        let y_f = F::cast_from(y_i);
        let x_f = F::cast_from(x_i);

        let dy = vy[idx];
        let dx = vx[idx];
        let wy = y_f + dy;
        let wx = x_f + dx;

        let w_val = sample_bilinear_nearest::<F>(img1, wy, wx, height, width);
        it_out[idx] = w_val - img0[idx];

        let one = F::new(1.0f32);
        let q_frac = F::new(0.25f32);

        // Blended central derivatives of warped img1 and reference img0
        let w_yp = sample_bilinear_nearest::<F>(img1, wy + one, wx, height, width);
        let w_ym = sample_bilinear_nearest::<F>(img1, wy - one, wx, height, width);
        let w_xp = sample_bilinear_nearest::<F>(img1, wy, wx + one, height, width);
        let w_xm = sample_bilinear_nearest::<F>(img1, wy, wx - one, height, width);

        let r_yp = i32::max(pad_index(y_i + 1i32, height, PAD_NEAREST), 0i32) as u32;
        let r_ym = i32::max(pad_index(y_i - 1i32, height, PAD_NEAREST), 0i32) as u32;
        let c_xp = i32::max(pad_index(x_i + 1i32, width, PAD_NEAREST), 0i32) as u32;
        let c_xm = i32::max(pad_index(x_i - 1i32, width, PAD_NEAREST), 0i32) as u32;
        let r_c = y_i as u32;
        let c_c = x_i as u32;

        let i0_yp = img0[(r_yp * w_u + c_c) as usize];
        let i0_ym = img0[(r_ym * w_u + c_c) as usize];
        let i0_xp = img0[(r_c * w_u + c_xp) as usize];
        let i0_xm = img0[(r_c * w_u + c_xm) as usize];

        iy_out[idx] = q_frac * ((w_yp - w_ym) + (i0_yp - i0_ym));
        ix_out[idx] = q_frac * ((w_xp - w_xm) + (i0_xp - i0_xm));
    }
}

/// Forms pointwise outer-product tensor terms `(jyy, jyx, jxx)` and negative temporal-gradient
/// products `(by, bx) = (-iy * it, -ix * it)` prior to spatial window smoothing in ILK.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn ilk_tensor_rhs_2d_kernel<F: Float>(
    iy: &[F],
    ix: &[F],
    it: &[F],
    jyy: &mut [F],
    jyx: &mut [F],
    jxx: &mut [F],
    by: &mut [F],
    bx: &mut [F],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let gy = iy[i];
        let gx = ix[i];
        let gt = it[i];
        jyy[i] = gy * gy;
        jyx[i] = gy * gx;
        jxx[i] = gx * gx;
        by[i] = -gy * gt;
        bx[i] = -gx * gt;
    }
}

/// Solves the $2 \times 2$ Lucas-Kanade normal equations at each pixel and increments `(vy, vx)`
/// by `(dvy, dvx)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn ilk_solve_2x2_kernel<F: Float>(
    jyy: &[F],
    jyx: &[F],
    jxx: &[F],
    by: &[F],
    bx: &[F],
    vy: &mut [F],
    vx: &mut [F],
    reg_eps: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let a = jyy[i] + reg_eps;
        let b = jyx[i];
        let c = jxx[i] + reg_eps;
        let rhs_y = by[i];
        let rhs_x = bx[i];
        let det = a * c - b * b;
        if det > reg_eps * reg_eps {
            let inv_det = F::new(1.0f32) / det;
            let dvy = (c * rhs_y - b * rhs_x) * inv_det;
            let dvx = (a * rhs_x - b * rhs_y) * inv_det;
            vy[i] += dvy;
            vx[i] += dvx;
        }
    }
}

/// Zach-Pock-Bischof Dual TV-$L_1$ primal update + soft-thresholding step:
/// 1. Updates `u = v + theta * div(p)`
/// 2. Evaluates linearized brightness constancy residual `rho = it + iy * (uy - u0y) + ix * (ux - u0x)`
/// 3. Soft-thresholds `v` toward `u` by `lambda * theta * grad(I)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn tvl1_primal_threshold_2d_kernel<F: Float>(
    u0y: &[F],
    u0x: &[F],
    iy: &[F],
    ix: &[F],
    it: &[F],
    py_y: &[F],
    py_x: &[F],
    px_y: &[F],
    px_x: &[F],
    vy_in: &[F],
    vx_in: &[F],
    uy_out: &mut [F],
    ux_out: &mut [F],
    vy_out: &mut [F],
    vx_out: &mut [F],
    theta: F,
    lam_theta: F,
    eps_grad: F,
    height: u32,
    width: u32,
) {
    let idx = element_position();
    let total = height * width;
    if (idx as u32) < total {
        let e = idx as u32;
        let y = e / width;
        let x = e % width;

        // div(py) and div(px)
        let py_y_cur = if y + 1u32 < height { py_y[idx] } else { F::new(0.0f32) };
        let py_y_prev = if y > 0u32 { py_y[((y - 1u32) * width + x) as usize] } else { F::new(0.0f32) };
        let py_x_cur = if x + 1u32 < width { py_x[idx] } else { F::new(0.0f32) };
        let py_x_prev = if x > 0u32 { py_x[(y * width + x - 1u32) as usize] } else { F::new(0.0f32) };
        let div_py = (py_y_cur - py_y_prev) + (py_x_cur - py_x_prev);

        let px_y_cur = if y + 1u32 < height { px_y[idx] } else { F::new(0.0f32) };
        let px_y_prev = if y > 0u32 { px_y[((y - 1u32) * width + x) as usize] } else { F::new(0.0f32) };
        let px_x_cur = if x + 1u32 < width { px_x[idx] } else { F::new(0.0f32) };
        let px_x_prev = if x > 0u32 { px_x[(y * width + x - 1u32) as usize] } else { F::new(0.0f32) };
        let div_px = (px_y_cur - px_y_prev) + (px_x_cur - px_x_prev);

        let uy = vy_in[idx] + theta * div_py;
        let ux = vx_in[idx] + theta * div_px;
        uy_out[idx] = uy;
        ux_out[idx] = ux;

        let gy = iy[idx];
        let gx = ix[idx];
        let grad_sq = gy * gy + gx * gx;
        let rho = it[idx] + gy * (uy - u0y[idx]) + gx * (ux - u0x[idx]);
        let thresh = lam_theta * grad_sq;

        let step_y = if rho < -thresh {
            lam_theta * gy
        } else if rho > thresh {
            -lam_theta * gy
        } else if grad_sq > eps_grad {
            -rho * gy / grad_sq
        } else {
            F::new(0.0f32)
        };

        let step_x = if rho < -thresh {
            lam_theta * gx
        } else if rho > thresh {
            -lam_theta * gx
        } else if grad_sq > eps_grad {
            -rho * gx / grad_sq
        } else {
            F::new(0.0f32)
        };

        vy_out[idx] = uy + step_y;
        vx_out[idx] = ux + step_x;
    }
}

/// Zach-Pock-Bischof Dual TV-$L_1$ dual variable projection update for both flow components
/// `(uy, ux)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn tvl1_dual_update_2d_kernel<F: Float>(
    uy: &[F],
    ux: &[F],
    py_y_in: &[F],
    py_x_in: &[F],
    px_y_in: &[F],
    px_x_in: &[F],
    py_y_out: &mut [F],
    py_x_out: &mut [F],
    px_y_out: &mut [F],
    px_x_out: &mut [F],
    tau_over_theta: F,
    height: u32,
    width: u32,
) {
    let idx = element_position();
    let total = height * width;
    if (idx as u32) < total {
        let e = idx as u32;
        let y = e / width;
        let x = e % width;

        let ny_idx = if y + 1u32 < height { (y + 1u32) * width + x } else { e };
        let nx_idx = if x + 1u32 < width { y * width + (x + 1u32) } else { e };

        let uy0 = uy[idx];
        let ux0 = ux[idx];
        let guy_y = if y + 1u32 < height { uy[ny_idx as usize] - uy0 } else { F::new(0.0f32) };
        let guy_x = if x + 1u32 < width { uy[nx_idx as usize] - uy0 } else { F::new(0.0f32) };
        let gux_y = if y + 1u32 < height { ux[ny_idx as usize] - ux0 } else { F::new(0.0f32) };
        let gux_x = if x + 1u32 < width { ux[nx_idx as usize] - ux0 } else { F::new(0.0f32) };

        let t_py_y = py_y_in[idx] + tau_over_theta * guy_y;
        let t_py_x = py_x_in[idx] + tau_over_theta * guy_x;
        let norm_y = F::max(F::new(1.0f32), F::sqrt(t_py_y * t_py_y + t_py_x * t_py_x));
        py_y_out[idx] = t_py_y / norm_y;
        py_x_out[idx] = t_py_x / norm_y;

        let t_px_y = px_y_in[idx] + tau_over_theta * gux_y;
        let t_px_x = px_x_in[idx] + tau_over_theta * gux_x;
        let norm_x = F::max(F::new(1.0f32), F::sqrt(t_px_y * t_px_y + t_px_x * t_px_x));
        px_y_out[idx] = t_px_y / norm_x;
        px_x_out[idx] = t_px_x / norm_x;
    }
}

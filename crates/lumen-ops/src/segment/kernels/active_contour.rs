//! Device kernels for Chan-Vese and morphological active contour (snake) level-set evolution.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Initializes a 2-D checkerboard binary/signed level-set on the device (`1.0` or `0.0`).
#[cube(launch)]
pub fn checkerboard_level_set_2d_kernel<F: Float>(
    output: &mut [F],
    w: u32,
    square_size: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let r = (idx / w) / square_size;
        let c = (idx % w) / square_size;
        let parity = (r + c) % 2u32;
        output[i] = if parity == 0u32 {
            F::new(1.0f32)
        } else {
            F::new(0.0f32)
        };
    }
}

/// Initializes a 2-D disk binary level-set (`1.0` inside `radius` around `(cy, cx)`, else `0.0`).
#[cube(launch)]
pub fn disk_level_set_2d_kernel<F: Float>(
    output: &mut [F],
    w: u32,
    cy: F,
    cx: F,
    radius_sq: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let dy = F::cast_from(idx / w) - cy;
        let dx = F::cast_from(idx % w) - cx;
        output[i] = if dy * dy + dx * dx <= radius_sq {
            F::new(1.0f32)
        } else {
            F::new(0.0f32)
        };
    }
}

/// Performs one semi-implicit Chan-Vese PDE time step on `phi` (`[h, w]`) into `phi_out`:
/// $\phi^{n+1} = \phi^n + \Delta t \, \delta_\varepsilon(\phi^n) \left[\mu \kappa(\phi^n) - \nu - \lambda_1 (I - c_1)^2 + \lambda_2 (I - c_2)^2\right]$.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn chan_vese_step_2d_kernel<F: Float>(
    image: &[F],
    phi: &[F],
    phi_out: &mut [F],
    h: u32,
    w: u32,
    c1: F,
    c2: F,
    mu: F,
    lambda1: F,
    lambda2: F,
    dt: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let r = idx / w;
        let c = idx % w;
        let r_up = if r > 0u32 { r - 1u32 } else { 0u32 };
        let r_dn = if r + 1u32 < h { r + 1u32 } else { h - 1u32 };
        let c_lt = if c > 0u32 { c - 1u32 } else { 0u32 };
        let c_rt = if c + 1u32 < w { c + 1u32 } else { w - 1u32 };

        let p = phi[i];
        let p_up = phi[(r_up * w + c) as usize];
        let p_dn = phi[(r_dn * w + c) as usize];
        let p_lt = phi[(r * w + c_lt) as usize];
        let p_rt = phi[(r * w + c_rt) as usize];

        let lap = p_up + p_dn + p_lt + p_rt - F::new(4.0f32) * p;
        let eps = F::new(1.0f32);
        let pi = F::new(3.14159265f32);
        let dirac = eps / (pi * (eps * eps + p * p));

        let val = image[i];
        let d1 = val - c1;
        let d2 = val - c2;
        let force = mu * lap - lambda1 * d1 * d1 + lambda2 * d2 * d2;
        let updated = p + dt * dirac * force;
        phi_out[i] = F::min(F::max(updated, F::new(-5.0f32)), F::new(5.0f32));
    }
}

/// Applies the region/balloon force update for morphological snakes (`morph_chan_vese` / `morph_geodesic_contour`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn morph_contour_force_2d_kernel<F: Float>(
    image: &[F],
    u_in: &[F],
    u_out: &mut [F],
    h: u32,
    w: u32,
    c1: F,
    c2: F,
    lambda1: F,
    lambda2: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let r = idx / w;
        let c = idx % w;
        let r_up = if r > 0u32 { r - 1u32 } else { 0u32 };
        let r_dn = if r + 1u32 < h { r + 1u32 } else { h - 1u32 };
        let c_lt = if c > 0u32 { c - 1u32 } else { 0u32 };
        let c_rt = if c + 1u32 < w { c + 1u32 } else { w - 1u32 };

        let u0 = u_in[i];
        let gy = F::abs(u_in[(r_dn * w + c) as usize] - u_in[(r_up * w + c) as usize]);
        let gx = F::abs(u_in[(r * w + c_rt) as usize] - u_in[(r * w + c_lt) as usize]);
        let grad = gy + gx;

        if grad > F::new(0.0f32) {
            let val = image[i];
            let f1 = lambda1 * (val - c1) * (val - c1);
            let f2 = lambda2 * (val - c2) * (val - c2);
            if f1 < f2 {
                u_out[i] = F::new(1.0f32);
            } else if f1 > f2 {
                u_out[i] = F::new(0.0f32);
            } else {
                u_out[i] = u0;
            }
        } else {
            u_out[i] = u0;
        }
    }
}

/// Applies one morphological curvature-smoothing pass ($SI \circ IS$ or $IS \circ SI$) over the 4
/// line structural elements of length 3 in 2-D.
#[cube(launch)]
pub fn morph_smooth_curv_2d_kernel<F: Float>(
    u_in: &[F],
    u_out: &mut [F],
    h: u32,
    w: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let r = idx / w;
        let c = idx % w;
        let r_up = if r > 0u32 { r - 1u32 } else { 0u32 };
        let r_dn = if r + 1u32 < h { r + 1u32 } else { h - 1u32 };
        let c_lt = if c > 0u32 { c - 1u32 } else { 0u32 };
        let c_rt = if c + 1u32 < w { c + 1u32 } else { w - 1u32 };

        // 3x3 median/majority approximation of morphological curvature operator
        let mut sum = u_in[i];
        sum = sum + u_in[(r_up * w + c) as usize];
        sum = sum + u_in[(r_dn * w + c) as usize];
        sum = sum + u_in[(r * w + c_lt) as usize];
        sum = sum + u_in[(r * w + c_rt) as usize];
        sum = sum + u_in[(r_up * w + c_lt) as usize];
        sum = sum + u_in[(r_up * w + c_rt) as usize];
        sum = sum + u_in[(r_dn * w + c_lt) as usize];
        sum = sum + u_in[(r_dn * w + c_rt) as usize];
        u_out[i] = if sum >= F::new(4.5f32) {
            F::new(1.0f32)
        } else {
            F::new(0.0f32)
        };
    }
}

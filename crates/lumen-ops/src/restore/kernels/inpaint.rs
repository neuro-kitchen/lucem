//! Inpainting and phase-unwrapping kernels: 13-point biharmonic PDE relaxation and wrapped-phase
//! Poisson right-hand-side divergence.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

use crate::core::kernels::pad_index;
use crate::core::pad::PAD_REFLECT;

/// One relaxation step of 2-D harmonic (`biharmonic = false`, 5-point Laplacian $\nabla^2 u = 0$)
/// or biharmonic (`biharmonic = true`, 13-point $\nabla^4 u = 0$) inpainting on masked pixels
/// (`mask[idx] != 0`). Unmasked pixels retain their original values.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn biharmonic_2d_step_kernel<F: Float>(
    u_in: &[F],
    mask: &[u32],
    u_out: &mut [F],
    omega: F,
    height: i32,
    width: i32,
    #[comptime] biharmonic: bool,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        if mask[idx] == 0u32 {
            u_out[idx] = u_in[idx];
        } else {
            let e = idx as u32;
            let w_u = width as u32;
            let y = (e / w_u) as i32;
            let x = (e % w_u) as i32;

            let ym1 = i32::max(pad_index(y - 1i32, height, PAD_REFLECT), 0i32) as u32;
            let yp1 = i32::max(pad_index(y + 1i32, height, PAD_REFLECT), 0i32) as u32;
            let xm1 = i32::max(pad_index(x - 1i32, width, PAD_REFLECT), 0i32) as u32;
            let xp1 = i32::max(pad_index(x + 1i32, width, PAD_REFLECT), 0i32) as u32;
            let yc = y as u32;
            let xc = x as u32;

            let s1 = u_in[(ym1 * w_u + xc) as usize]
                + u_in[(yp1 * w_u + xc) as usize]
                + u_in[(yc * w_u + xm1) as usize]
                + u_in[(yc * w_u + xp1) as usize];

            let target = if comptime!(biharmonic) {
                let ym2 = i32::max(pad_index(y - 2i32, height, PAD_REFLECT), 0i32) as u32;
                let yp2 = i32::max(pad_index(y + 2i32, height, PAD_REFLECT), 0i32) as u32;
                let xm2 = i32::max(pad_index(x - 2i32, width, PAD_REFLECT), 0i32) as u32;
                let xp2 = i32::max(pad_index(x + 2i32, width, PAD_REFLECT), 0i32) as u32;

                let s_diag = u_in[(ym1 * w_u + xm1) as usize]
                    + u_in[(ym1 * w_u + xp1) as usize]
                    + u_in[(yp1 * w_u + xm1) as usize]
                    + u_in[(yp1 * w_u + xp1) as usize];
                let s2 = u_in[(ym2 * w_u + xc) as usize]
                    + u_in[(yp2 * w_u + xc) as usize]
                    + u_in[(yc * w_u + xm2) as usize]
                    + u_in[(yc * w_u + xp2) as usize];

                (F::new(8.0f32) * s1 - F::new(2.0f32) * s_diag - s2) * F::new(0.05f32)
            } else {
                s1 * F::new(0.25f32)
            };

            let cur = u_in[idx];
            u_out[idx] = cur + omega * (target - cur);
        }
    }
}

/// Wraps a phase difference `delta` into `(-pi, pi]`.
#[cube]
fn wrap_to_pi<F: Float>(delta: F, pi: F, two_pi: F) -> F {
    let shifted = delta + pi;
    let wrapped = shifted - two_pi * F::floor(shifted / two_pi);
    wrapped - pi
}

/// Computes the discrete divergence of wrapped forward phase differences for 2-D least-squares
/// phase unwrapping.
#[cube(launch)]
pub fn unwrap_phase_2d_poisson_rhs_kernel<F: Float>(
    phase: &[F],
    rhs: &mut [F],
    pi: F,
    two_pi: F,
    height: u32,
    width: u32,
) {
    let idx = element_position();
    let total = height * width;
    if (idx as u32) < total {
        let e = idx as u32;
        let y = e / width;
        let x = e % width;
        let p0 = phase[idx];

        let yp1 = if y + 1u32 < height { y + 1u32 } else { y };
        let ym1 = if y > 0u32 { y - 1u32 } else { 0u32 };
        let xp1 = if x + 1u32 < width { x + 1u32 } else { x };
        let xm1 = if x > 0u32 { x - 1u32 } else { 0u32 };

        let dy_fwd = if y + 1u32 < height {
            wrap_to_pi::<F>(phase[(yp1 * width + x) as usize] - p0, pi, two_pi)
        } else {
            F::new(0.0f32)
        };
        let dy_bwd = if y > 0u32 {
            wrap_to_pi::<F>(p0 - phase[(ym1 * width + x) as usize], pi, two_pi)
        } else {
            F::new(0.0f32)
        };
        let dx_fwd = if x + 1u32 < width {
            wrap_to_pi::<F>(phase[(y * width + xp1) as usize] - p0, pi, two_pi)
        } else {
            F::new(0.0f32)
        };
        let dx_bwd = if x > 0u32 {
            wrap_to_pi::<F>(p0 - phase[(y * width + xm1) as usize], pi, two_pi)
        } else {
            F::new(0.0f32)
        };

        rhs[idx] = (dy_fwd - dy_bwd) + (dx_fwd - dx_bwd);
    }
}

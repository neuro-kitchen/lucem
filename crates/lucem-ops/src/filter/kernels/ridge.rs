//! Kernels for neurite and vessel ridge enhancement filters (Frangi, Sato, Meijering) from
//! per-pixel Hessian eigenvalues.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// 2-D Frangi vesselness response from Hessian eigenvalues $(\lambda_1, \lambda_2)$ at a single scale,
/// accumulated via elementwise `max` into `output`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn frangi_2d_scale_kernel<F: lucem_core::element::ImageFloat>(
    eig1: &[F],
    eig2: &[F],
    output: &mut [F],
    inv_2beta_sq: F,
    inv_2c_sq: F,
    len: u32,
    #[comptime] black_ridges: bool,
) {
    let i = element_position();
    if (i as u32) < len {
        let e1 = eig1[i];
        let e2 = eig2[i];
        // Sort by absolute magnitude: |l1| <= |l2|
        let a1 = F::abs(e1);
        let a2 = F::abs(e2);
        let l1 = if a1 <= a2 { e1 } else { e2 };
        let l2 = if a1 <= a2 { e2 } else { e1 };

        let valid = if comptime!(black_ridges) {
            l2 > F::new(0.0f32)
        } else {
            l2 < F::new(0.0f32)
        };
        if valid {
            let abs_l2 = F::max(F::abs(l2), F::new(1e-10f32));
            let rb = l1 / abs_l2;
            let s2 = l1 * l1 + l2 * l2;
            let resp = F::exp(F::new(0.0f32) - rb * rb * inv_2beta_sq)
                * (F::new(1.0f32) - F::exp(F::new(0.0f32) - s2 * inv_2c_sq));
            output[i] = F::max(output[i], resp);
        }
    }
}

/// 2-D Sato tubeness response from Hessian eigenvalues $(\lambda_1, \lambda_2)$ scaled by $\sigma^2$,
/// accumulated via elementwise `max` into `output`.
#[cube(launch)]
pub fn sato_2d_scale_kernel<F: lucem_core::element::ImageFloat>(
    eig1: &[F],
    eig2: &[F],
    output: &mut [F],
    sigma_sq: F,
    len: u32,
    #[comptime] black_ridges: bool,
) {
    let i = element_position();
    if (i as u32) < len {
        let e1 = eig1[i] * sigma_sq;
        let e2 = eig2[i] * sigma_sq;
        let a1 = F::abs(e1);
        let a2 = F::abs(e2);
        let l2 = if a1 <= a2 { e2 } else { e1 };
        let resp = if comptime!(black_ridges) {
            F::max(l2, F::new(0.0f32))
        } else {
            F::max(F::new(0.0f32) - l2, F::new(0.0f32))
        };
        output[i] = F::max(output[i], resp);
    }
}

/// 2-D Meijering neuriteness response from Hessian eigenvalues $(\lambda_1, \lambda_2)$ via modified
/// eigenvalues $\tilde{\lambda}_1 = \lambda_1 - \frac{1}{3}\lambda_2$, $\tilde{\lambda}_2 = \lambda_2 - \frac{1}{3}\lambda_1$.
#[cube(launch)]
pub fn meijering_2d_scale_kernel<F: lucem_core::element::ImageFloat>(
    eig1: &[F],
    eig2: &[F],
    output: &mut [F],
    len: u32,
    #[comptime] black_ridges: bool,
) {
    let i = element_position();
    if (i as u32) < len {
        let e1 = eig1[i];
        let e2 = eig2[i];
        let third = F::new(-1.0f32 / 3.0f32);
        let m1 = e1 + third * e2;
        let m2 = e2 + third * e1;
        let dominant = if F::abs(m1) >= F::abs(m2) { m1 } else { m2 };
        let resp = if comptime!(black_ridges) {
            F::max(dominant, F::new(0.0f32))
        } else {
            F::max(F::new(0.0f32) - dominant, F::new(0.0f32))
        };
        output[i] = F::max(output[i], resp);
    }
}

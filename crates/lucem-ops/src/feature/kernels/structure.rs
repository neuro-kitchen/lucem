//! Hessian determinant, structure tensor outer products, and Koenderink shape-index kernels.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// 2-D determinant $\det(H) = H_{rr} H_{cc} - H_{rc}^2$.
#[cube(launch)]
pub fn hessian_det_2d_kernel<F: lucem_core::element::ImageFloat>(
    hrr: &[F],
    hrc: &[F],
    hcc: &[F],
    out: &mut [F],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        out[i] = hrr[i] * hcc[i] - hrc[i] * hrc[i];
    }
}

/// Computes pointwise outer product `(gy * gy, gy * gx, gx * gx)` of 2-D gradients.
#[cube(launch)]
pub fn outer_product_2d_kernel<F: lucem_core::element::ImageFloat>(
    gy: &[F],
    gx: &[F],
    iyy: &mut [F],
    iyx: &mut [F],
    ixx: &mut [F],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let vy = gy[i];
        let vx = gx[i];
        iyy[i] = vy * vy;
        iyx[i] = vy * vx;
        ixx[i] = vx * vx;
    }
}

/// Koenderink & van Doorn shape index $s = \frac{2}{\pi} \arctan\!\left(\frac{\kappa_2 + \kappa_1}{\kappa_2 - \kappa_1}\right)$
/// from ordered Hessian eigenvalues `eig1 >= eig2`.
#[cube(launch)]
pub fn shape_index_2d_kernel<F: lucem_core::element::ImageFloat>(
    eig1: &[F],
    eig2: &[F],
    out: &mut [F],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let l1 = f32::cast_from(eig1[i]);
        let l2 = f32::cast_from(eig2[i]);
        let diff = l1 - l2;
        let sum = l1 + l2;
        let scale = f32::abs(l1) + f32::abs(l2);
        if f32::abs(diff) <= 1e-6f32 * scale || scale <= 1e-20f32 {
            // Equal curvatures (an umbilic point): the limit of the formula, +1 for a dome (both
            // negative), -1 for a cup, 0 for a flat surface (scikit-image reaches it through -0.0)
            out[i] = if sum < 0.0f32 && scale > 1e-20f32 { F::new(1.0f32) } else if sum > 0.0f32 && scale > 1e-20f32 { F::new(-1.0f32) } else { F::new(0.0f32) };
        } else {
            // Koenderink's shape index with l1 >= l2: (2/pi) * atan((-l1 - l2) / (l1 - l2))
            let ratio = (-sum) / diff;
            // Rational Padé / atan approximation accurate across R via atan(x) identity
            let ax = f32::abs(ratio);
            let inv = ax > 1.0f32;
            let z = if inv { 1.0f32 / ax } else { ax };
            let z2 = z * z;
            // Minimax polynomial for atan(z) on [0, 1]
            let atan_z = z * (0.999_866f32 + z2 * (-0.330_299_5f32 + z2 * (0.180_141f32 + z2 * (-0.085_133f32 + z2 * 0.020_835_1f32))));
            let half_pi = 1.570_796_4f32;
            let angle = if inv { half_pi - atan_z } else { atan_z };
            let signed = if ratio < 0.0f32 { -angle } else { angle };
            let two_over_pi = 0.636_619_75f32;
            out[i] = F::cast_from(signed * two_over_pi);
        }
    }
}

//! Per-pixel symmetric $2 \times 2$ and $3 \times 3$ eigenvalue kernels.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Computes the two real eigenvalues $(\lambda_1, \lambda_2)$ with $\lambda_1 \ge \lambda_2$ of each
/// symmetric $2 \times 2$ matrix $\begin{pmatrix} a_{xx} & a_{xy} \\ a_{xy} & a_{yy} \end{pmatrix}$.
#[cube(launch)]
pub fn sym_eigvals_2x2_kernel<F: lumen_core::element::ImageFloat>(
    axx: &[F],
    axy: &[F],
    ayy: &[F],
    eig1: &mut [F],
    eig2: &mut [F],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let a = axx[i];
        let b = axy[i];
        let c = ayy[i];
        let half = F::new(0.5f32);
        let mean = (a + c) * half;
        let diff = (a - c) * half;
        let radius = F::sqrt(diff * diff + b * b);
        eig1[i] = mean + radius;
        eig2[i] = mean - radius;
    }
}

/// Computes the three real eigenvalues $(\lambda_1 \ge \lambda_2 \ge \lambda_3)$ of each symmetric
/// $3 \times 3$ matrix using Cardano's trigonometric formula.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn sym_eigvals_3x3_kernel<F: lumen_core::element::ImageFloat>(
    a00: &[F],
    a01: &[F],
    a02: &[F],
    a11: &[F],
    a12: &[F],
    a22: &[F],
    eig1: &mut [F],
    eig2: &mut [F],
    eig3: &mut [F],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let m00 = f32::cast_from(a00[i]);
        let m01 = f32::cast_from(a01[i]);
        let m02 = f32::cast_from(a02[i]);
        let m11 = f32::cast_from(a11[i]);
        let m12 = f32::cast_from(a12[i]);
        let m22 = f32::cast_from(a22[i]);

        let p1 = m01 * m01 + m02 * m02 + m12 * m12;
        let q = (m00 + m11 + m22) / 3.0f32;
        let b00 = m00 - q;
        let b11 = m11 - q;
        let b22 = m22 - q;
        let p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0f32 * p1;
        let p = f32::sqrt(p2 / 6.0f32);

        if p <= 1e-12f32 {
            eig1[i] = F::cast_from(q);
            eig2[i] = F::cast_from(q);
            eig3[i] = F::cast_from(q);
        } else {
            let inv_p = 1.0f32 / p;
            let c00 = b00 * inv_p;
            let c01 = m01 * inv_p;
            let c02 = m02 * inv_p;
            let c11 = b11 * inv_p;
            let c12 = m12 * inv_p;
            let c22 = b22 * inv_p;
            let det_b = c00 * (c11 * c22 - c12 * c12)
                - c01 * (c01 * c22 - c12 * c02)
                + c02 * (c01 * c12 - c11 * c02);
            let r = f32::min(f32::max(0.5f32 * det_b, -1.0f32), 1.0f32);
            let phi = f32::acos(r) / 3.0f32;
            let two_pi_3 = 2.094_395_2f32;
            let l1 = q + 2.0f32 * p * f32::cos(phi);
            let l3 = q + 2.0f32 * p * f32::cos(phi + two_pi_3);
            let l2 = 3.0f32 * q - l1 - l3;
            eig1[i] = F::cast_from(l1);
            eig2[i] = F::cast_from(l2);
            eig3[i] = F::cast_from(l3);
        }
    }
}

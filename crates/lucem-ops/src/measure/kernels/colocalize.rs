//! Device kernels for fluorescence colocalization terms.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// Computes elementwise centered/raw products `a_b[i] = (a[i] - ma) * (b[i] - mb)`,
/// `a2[i] = (a[i] - ma)^2`, and `b2[i] = (b[i] - mb)^2`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn colocalize_terms_kernel<F: lucem_core::element::ImageFloat>(
    a: &[F],
    b: &[F],
    ab: &mut [F],
    a2: &mut [F],
    b2: &mut [F],
    ma: F,
    mb: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let da = a[i] - ma;
        let db = b[i] - mb;
        ab[i] = da * db;
        a2[i] = da * da;
        b2[i] = db * db;
    }
}

/// Computes gated Manders colocalization numerators:
/// `a_gated[i] = a[i]` if `b[i] > thresh_b` else `0`, and
/// `b_gated[i] = b[i]` if `a[i] > thresh_a` else `0`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn manders_gated_terms_kernel<F: lucem_core::element::ImageFloat>(
    a: &[F],
    b: &[F],
    a_gated: &mut [F],
    b_gated: &mut [F],
    thresh_a: F,
    thresh_b: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let va = a[i];
        let vb = b[i];
        a_gated[i] = if vb > thresh_b { va } else { F::new(0.0f32) };
        b_gated[i] = if va > thresh_a { vb } else { F::new(0.0f32) };
    }
}

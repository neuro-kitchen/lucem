//! Multi-scale blob response kernels.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// Scale-normalized Determinant of Hessian response $\sigma^4 (H_{yy} H_{xx} - H_{xy}^2)$ in 2-D.
#[cube(launch)]
pub fn doh_scale_kernel<F: lucem_core::element::ImageFloat>(
    hyy: &[F],
    hxy: &[F],
    hxx: &[F],
    out: &mut [F],
    sigma4: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let det = hyy[i] * hxx[i] - hxy[i] * hxy[i];
        out[i] = sigma4 * det;
    }
}

//! Device kernels for 2-D spatial moment monomials.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Computes `output[i] = input[i] * (y - cy)^p * (x - cx)^q` for every pixel `(y, x)` of a `[h, w]` image.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn moment_monomial_2d_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    w: u32,
    cy: F,
    cx: F,
    p: u32,
    q: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let y = F::cast_from(idx / w) - cy;
        let x = F::cast_from(idx % w) - cx;
        let mut yp = F::new(1.0f32);
        let mut k = 0u32;
        while k < p {
            yp = yp * y;
            k += 1u32;
        }
        let mut xq = F::new(1.0f32);
        let mut m = 0u32;
        while m < q {
            xq = xq * x;
            m += 1u32;
        }
        output[i] = input[i] * yp * xq;
    }
}

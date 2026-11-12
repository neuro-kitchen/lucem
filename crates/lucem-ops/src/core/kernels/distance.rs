//! Exact Euclidean distance transform (EDT) and chamfer distance kernels.

use cubecl::prelude::*;
use lucem_core::compute::{element_position, line_position};

/// Initializes squared distances: `0.0` where `input[i] == 0` (background) and `inf_sq` (`1e20`)
/// where `input[i] != 0` (foreground).
#[cube(launch)]
pub fn edt_init_kernel<E: Numeric, F: lucem_core::element::ImageFloat>(input: &[E], dist_sq: &mut [F], inf_sq: F, len: u32) {
    let i = element_position();
    if (i as u32) < len {
        dist_sq[i] = if input[i] == E::from_int(0) { F::new(0.0f32) } else { inf_sq };
    }
}

/// 1-D squared distance transform pass along an axis `[outer, len, inner]` with pixel spacing
/// `spacing_sq = spacing * spacing`.
#[cube(launch)]
pub fn edt_axis_1d_kernel<F: lucem_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    len: u32,
    inner: u32,
    lines: u32,
    spacing_sq: F,
) {
    let line = line_position();
    if line < lines {
        let k = line % inner;
        let o = line / inner;
        let base = o * len * inner + k;
        let mut i = 0u32;
        while i < len {
            let mut best = input[(base + i * inner) as usize];
            let mut q = 0u32;
            while q < len {
                let d = (i as i32) - (q as i32);
                let d2 = F::cast_from(d * d) * spacing_sq;
                let cand = input[(base + q * inner) as usize] + d2;
                best = F::min(best, cand);
                q += 1u32;
            }
            output[(base + i * inner) as usize] = best;
            i += 1u32;
        }
    }
}

/// Takes the square root of each squared distance in `input` into `output`.
#[cube(launch)]
pub fn sqrt_kernel<F: lucem_core::element::ImageFloat>(input: &[F], output: &mut [F], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = F::sqrt(F::max(input[i], F::new(0.0f32)));
    }
}

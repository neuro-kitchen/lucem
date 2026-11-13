//! Segmentation evaluation kernels: binary mask inversion for distance transforms and masked EDT
//! evaluation for Hausdorff distance.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// Writes `1u32` where `input[i] == 0` and `0u32` where `input[i] != 0` so that the Euclidean
/// distance transform computes the distance to the nearest foreground pixel of `input`.
#[cube(launch)]
pub fn invert_binary_mask_kernel<E: Numeric>(input: &[E], out: &mut [u32], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        out[i] = if input[i] == E::from_int(0) { 1u32 } else { 0u32 };
    }
}

/// Evaluates `edt[i]` at foreground pixels (`mask[i] != 0`) and writes `0.0` at background pixels.
#[cube(launch)]
pub fn mask_at_edt_kernel<M: Numeric, F: lucem_core::element::ImageFloat>(
    mask: &[M],
    edt: &[F],
    out: &mut [F],
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        out[i] = if mask[i] != M::from_int(0) {
            edt[i]
        } else {
            F::new(0.0f32)
        };
    }
}

//! Device kernels for region masking and per-label indicator extraction.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// Extracts a binary float mask (`1.0` where `labels[i] == target_label`, else `0.0`).
#[cube(launch)]
pub fn region_mask_kernel<F: lucem_core::element::ImageFloat>(
    labels: &[u32],
    output: &mut [F],
    target_label: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        output[i] = if labels[i] == target_label {
            F::new(1.0f32)
        } else {
            F::new(0.0f32)
        };
    }
}

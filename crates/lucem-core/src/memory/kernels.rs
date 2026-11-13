//! Kernels of the memory module.

use cubecl::prelude::*;

use crate::compute::element_position;
use crate::element::ImageFloat;

/// `out[i] = value` for `i < len`. One unit per element.
#[cube(launch)]
pub fn fill_kernel<F: ImageFloat>(out: &mut [F], value: F, len: u32) {
    let i = element_position() as u32;
    if i < len {
        out[i as usize] = value;
    }
}

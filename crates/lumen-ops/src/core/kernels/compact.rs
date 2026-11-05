//! Stream compaction kernels: flags non-zero elements and gathers their indices/coordinates.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Writes `1u32` where `input[i] != 0` and `0u32` otherwise.
#[cube(launch)]
pub fn flag_nonzero_kernel<E: Numeric>(input: &[E], flags: &mut [u32], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        flags[i] = if input[i] != E::from_int(0) { 1u32 } else { 0u32 };
    }
}

/// Scatters flat index `i` into `indices[offsets[i]]` for every element where `flags[i] != 0`.
#[cube(launch)]
pub fn scatter_indices_kernel(flags: &[u32], offsets: &[u32], indices: &mut [u32], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        if flags[i] != 0u32 {
            let dst = offsets[i] as usize;
            indices[dst] = i as u32;
        }
    }
}

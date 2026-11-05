//! Kernels for exact order statistics and percentile selection via IEEE-754 order-preserving
//! radix histograms.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Maps an IEEE-754 `f32` value's bit pattern to a monotone `u32` key and increments its 8-bit
/// digit bin for pass `shift` (`0`, `8`, `16`, `24`) when `(key & mask) == prefix`.
#[cube(launch)]
pub fn radix_pass_hist_kernel(
    input: &[f32],
    bins: &mut [Atomic<u32>],
    prefix: u32,
    mask: u32,
    shift: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let bits = u32::reinterpret(input[i]);
        let sign = bits >> 31u32;
        let key = if sign == 1u32 { !bits } else { bits ^ 0x8000_0000u32 };
        if (key & mask) == prefix {
            let bucket = (key >> shift) & 0xFFu32;
            Atomic::fetch_add(&bins[bucket as usize], 1u32);
        }
    }
}

/// Converts a monotone `u32` radix key back to `f32` and writes it to `output[0]`.
#[cube(launch)]
pub fn key_to_f32_kernel(key: u32, output: &mut [f32]) {
    let i = element_position();
    if i == 0usize {
        let high = key >> 31u32;
        let bits = if high == 1u32 { key ^ 0x8000_0000u32 } else { !key };
        output[0] = f32::reinterpret(bits);
    }
}

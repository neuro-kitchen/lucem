//! Prefix sum (scan) kernels along an axis or flat 1-D buffer.

use cubecl::prelude::*;
use lumen_core::compute::line_position;

/// Computes an inclusive or exclusive prefix sum along an axis seen as `[outer, len, inner]`.
/// One unit walks one line `(o, :, k)` of `len` elements; lines run in parallel across `(o, k)`.
#[cube(launch)]
pub fn scan_axis_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    len: u32,
    inner: u32,
    lines: u32,
    #[comptime] inclusive: bool,
) {
    let line = line_position();
    if line < lines {
        let k = line % inner;
        let o = line / inner;
        let base = o * len * inner + k;
        let mut acc = 0.0f32;
        let mut i = 0u32;
        while i < len {
            let idx = (base + i * inner) as usize;
            let v = f32::cast_from(input[idx]);
            if comptime!(inclusive) {
                acc += v;
                output[idx] = F::cast_from(acc);
            } else {
                output[idx] = F::cast_from(acc);
                acc += v;
            }
            i += 1u32;
        }
    }
}

/// Computes an inclusive or exclusive prefix sum of a `u32` buffer inside a single cube using
/// Hillis-Steele / Blelloch shared-memory steps over chunks, or sequential per-chunk blocks.
#[cube(launch)]
pub fn scan_u32_blocks_kernel(
    input: &[u32],
    output: &mut [u32],
    block_sums: &mut [u32],
    len: u32,
    #[comptime] block_size: u32,
    #[comptime] inclusive: bool,
) {
    let mut sh = Shared::<[u32]>::new_slice(block_size as usize);
    let block = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let u = UNIT_POS_X;
    let global_idx = block * block_size + u;
    let val = if global_idx < len { input[global_idx as usize] } else { 0u32 };
    sh[u as usize] = val;
    sync_cube();

    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < block_size {
        let step = offset.read();
        let prev = if u >= step { sh[(u - step) as usize] } else { 0u32 };
        sync_cube();
        if u >= step {
            sh[u as usize] = sh[u as usize] + prev;
        }
        sync_cube();
        offset.store(step * 2u32);
    }

    if global_idx < len {
        let inc = sh[u as usize];
        output[global_idx as usize] = if comptime!(inclusive) { inc } else { inc - val };
    }
    if u == block_size - 1u32 {
        block_sums[block as usize] = sh[(block_size - 1u32) as usize];
    }
}

/// Adds the scanned block prefix `block_offsets[block]` to every element in `block`.
#[cube(launch)]
pub fn scan_u32_add_offsets_kernel(
    output: &mut [u32],
    block_offsets: &[u32],
    len: u32,
    #[comptime] block_size: u32,
) {
    let block = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let u = UNIT_POS_X;
    let global_idx = block * block_size + u;
    if global_idx < len {
        output[global_idx as usize] = output[global_idx as usize] + block_offsets[block as usize];
    }
}

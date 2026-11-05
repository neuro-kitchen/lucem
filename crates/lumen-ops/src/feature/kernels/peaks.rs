//! Local peak candidate detection kernel.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Flags 2-D local maximum candidates where `image[idx] >= max_filtered[idx]`, `image[idx] > threshold`,
/// and the coordinate lies outside `border` pixels of the image boundary.
#[cube(launch)]
pub fn peak_candidates_2d_kernel<F: Float>(
    image: &[F],
    max_filtered: &[F],
    flags: &mut [u32],
    height: u32,
    width: u32,
    border: u32,
    threshold: F,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let y = idx / width;
        let x = idx % width;
        let in_interior = y >= border
            && x >= border
            && (y + border) < height
            && (x + border) < width;
        let val = image[i];
        let mx = max_filtered[i];
        if in_interior && val >= mx && val > threshold {
            flags[i] = 1u32;
        } else {
            flags[i] = 0u32;
        }
    }
}

//! Device kernels for watershed segmentation initialization.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// Initializes per-pixel steepest-descent neighbor pointers for a 2-D relief map (`[h, w]`) when a
/// pixel has no initial marker (`markers[i] == 0`).
#[cube(launch)]
pub fn watershed_steepest_init_2d_kernel<F: lucem_core::element::ImageFloat>(
    relief: &[F],
    markers: &[u32],
    next_idx: &mut [u32],
    h: u32,
    w: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        if markers[i] > 0u32 {
            next_idx[i] = idx;
        } else {
            let r = idx / w;
            let c = idx % w;
            let mut best_val = relief[i];
            let mut best_n = idx;

            if r > 0u32 {
                let n = (r - 1u32) * w + c;
                let v = relief[n as usize];
                if v < best_val {
                    best_val = v;
                    best_n = n;
                }
            }
            if r + 1u32 < h {
                let n = (r + 1u32) * w + c;
                let v = relief[n as usize];
                if v < best_val {
                    best_val = v;
                    best_n = n;
                }
            }
            if c > 0u32 {
                let n = r * w + (c - 1u32);
                let v = relief[n as usize];
                if v < best_val {
                    best_val = v;
                    best_n = n;
                }
            }
            if c + 1u32 < w {
                let n = r * w + (c + 1u32);
                let v = relief[n as usize];
                if v < best_val {
                    best_n = n;
                }
            }
            next_idx[i] = best_n;
        }
    }
}

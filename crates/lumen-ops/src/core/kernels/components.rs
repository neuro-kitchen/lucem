//! Parallel union-find connected-component labeling kernels on the device.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Initializes each foreground pixel (`mask[i] != 0`) to its own 1-based provisional label `i + 1`,
/// and background pixels to `0`.
#[cube(launch)]
pub fn cc_init_labels_kernel<E: Numeric>(mask: &[E], labels: &mut [u32], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        labels[i] = if mask[i] != E::from_int(0) { (i as u32) + 1u32 } else { 0u32 };
    }
}

/// Propagates minimum non-zero neighbor labels across a 2-D grid (`connectivity`: 1 = 4-connected,
/// 2 = 8-connected), setting `changed[0] = 1` whenever a pixel lowers its label.
#[cube(launch)]
pub fn cc_propagate_2d_kernel(
    src_labels: &[u32],
    dst_labels: &mut [u32],
    changed: &mut [Atomic<u32>],
    height: u32,
    width: u32,
    #[comptime] eight_connected: bool,
) {
    let idx = element_position();
    let total = height * width;
    if (idx as u32) < total {
        let cur = src_labels[idx];
        if cur == 0u32 {
            dst_labels[idx] = 0u32;
        } else {
            let e = idx as u32;
            let y = e / width;
            let x = e % width;
            let mut best = cur;

            if y > 0u32 {
                let n = src_labels[(e - width) as usize];
                if n > 0u32 && n < best {
                    best = n;
                }
            }
            if y + 1u32 < height {
                let s = src_labels[(e + width) as usize];
                if s > 0u32 && s < best {
                    best = s;
                }
            }
            if x > 0u32 {
                let w = src_labels[(e - 1u32) as usize];
                if w > 0u32 && w < best {
                    best = w;
                }
            }
            if x + 1u32 < width {
                let ea = src_labels[(e + 1u32) as usize];
                if ea > 0u32 && ea < best {
                    best = ea;
                }
            }
            if comptime!(eight_connected) {
                if y > 0u32 && x > 0u32 {
                    let nw = src_labels[(e - width - 1u32) as usize];
                    if nw > 0u32 && nw < best {
                        best = nw;
                    }
                }
                if y > 0u32 && x + 1u32 < width {
                    let ne = src_labels[(e - width + 1u32) as usize];
                    if ne > 0u32 && ne < best {
                        best = ne;
                    }
                }
                if y + 1u32 < height && x > 0u32 {
                    let sw = src_labels[(e + width - 1u32) as usize];
                    if sw > 0u32 && sw < best {
                        best = sw;
                    }
                }
                if y + 1u32 < height && x + 1u32 < width {
                    let se = src_labels[(e + width + 1u32) as usize];
                    if se > 0u32 && se < best {
                        best = se;
                    }
                }
            }
            // Pointer jumping through the root table (`src_labels[best - 1]`) accelerates convergence
            let jumped = src_labels[(best - 1u32) as usize];
            if jumped > 0u32 && jumped < best {
                best = jumped;
            }
            dst_labels[idx] = best;
            if best < cur {
                Atomic::store(&changed[0], 1u32);
            }
        }
    }
}

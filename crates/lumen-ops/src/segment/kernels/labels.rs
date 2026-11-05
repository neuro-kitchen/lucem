//! Device kernels for label expansion, boundary detection, boundary overlays, and segmentation joining.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Assigns each background (`labels[i] == 0`) pixel the label at `nearest_idx[i]` if its EDT
/// distance `edt[i] <= max_distance`.
#[cube(launch)]
pub fn expand_labels_gather_kernel<F: lumen_core::element::ImageFloat>(
    labels: &[u32],
    nearest_idx: &[u32],
    edt: &[F],
    output: &mut [u32],
    max_dist: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let lbl = labels[i];
        if lbl != 0u32 {
            output[i] = lbl;
        } else if edt[i] <= max_dist {
            let src = nearest_idx[i] as usize;
            output[i] = labels[src];
        } else {
            output[i] = 0u32;
        }
    }
}

/// Detects boundary pixels in a 2-D label map `[h, w]`.
/// `mode`: `0 = Thick` (any neighbor differs), `1 = Inner` (`labels[i] > 0` and any neighbor differs),
/// `2 = Outer` (`labels[i] == 0` or neighbor has larger label / background adjacent to foreground).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_boundaries_2d_kernel(
    labels: &[u32],
    output: &mut [u32],
    h: u32,
    w: u32,
    connectivity: u32,
    mode: u32,
    background: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let r = idx / w;
        let c = idx % w;
        let center = labels[i];
        let mut diff_any = false;
        let mut diff_fg = false;

        let r0 = if r > 0u32 { r - 1u32 } else { 0u32 };
        let r1 = if r + 1u32 < h { r + 1u32 } else { h - 1u32 };
        let c0 = if c > 0u32 { c - 1u32 } else { 0u32 };
        let c1 = if c + 1u32 < w { c + 1u32 } else { w - 1u32 };

        let mut nr = r0;
        while nr <= r1 {
            let mut nc = c0;
            while nc <= c1 {
                let dr = if nr >= r { nr - r } else { r - nr };
                let dc = if nc >= c { nc - c } else { c - nc };
                let dist = dr + dc;
                if dist > 0u32 && dist <= connectivity {
                    let n_lbl = labels[(nr * w + nc) as usize];
                    if n_lbl != center {
                        diff_any = true;
                        if n_lbl != background {
                            diff_fg = true;
                        }
                    }
                }
                nc += 1u32;
            }
            nr += 1u32;
        }

        let is_bound = if mode == 1u32 {
            // Inner boundary: foreground pixel with a different neighbor
            center != background && diff_any
        } else if mode == 2u32 {
            // Outer boundary: pixel adjacent to a different foreground label
            diff_fg && (center == background || diff_any)
        } else {
            // Thick boundary: any pixel whose neighborhood has >1 label
            diff_any
        };

        output[i] = if is_bound { 1u32 } else { 0u32 };
    }
}

/// Overlays a single-channel `[h, w]` image with `color` where `boundaries[i] != 0`.
#[cube(launch)]
pub fn mark_boundaries_gray_kernel<F: lumen_core::element::ImageFloat>(
    image: &[F],
    boundaries: &[u32],
    output: &mut [F],
    color_r: F,
    color_g: F,
    color_b: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let base = i * 3usize;
        if boundaries[i] != 0u32 {
            output[base] = color_r;
            output[base + 1usize] = color_g;
            output[base + 2usize] = color_b;
        } else {
            let v = image[i];
            output[base] = v;
            output[base + 1usize] = v;
            output[base + 2usize] = v;
        }
    }
}

/// Overlays an RGB `[h, w, 3]` image with `(color_r, color_g, color_b)` where `boundaries[i] != 0`.
#[cube(launch)]
pub fn mark_boundaries_rgb_kernel<F: lumen_core::element::ImageFloat>(
    image: &[F],
    boundaries: &[u32],
    output: &mut [F],
    color_r: F,
    color_g: F,
    color_b: F,
    num_pixels: u32,
) {
    let i = element_position();
    if (i as u32) < num_pixels {
        let base = i * 3usize;
        if boundaries[i] != 0u32 {
            output[base] = color_r;
            output[base + 1usize] = color_g;
            output[base + 2usize] = color_b;
        } else {
            output[base] = image[base];
            output[base + 1usize] = image[base + 1usize];
            output[base + 2usize] = image[base + 2usize];
        }
    }
}

/// Combines two label maps `s1` and `s2` via `s1[i] * stride + s2[i]` before sequential relabeling.
#[cube(launch)]
pub fn join_pair_key_kernel(
    s1: &[u32],
    s2: &[u32],
    out: &mut [u32],
    stride: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        out[i] = s1[i] * stride + s2[i];
    }
}

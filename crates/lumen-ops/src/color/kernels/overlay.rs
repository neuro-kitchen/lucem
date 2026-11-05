//! GPU kernels for rendering segmentation label maps into false-color RGB overlays.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Render integer labels `[H, W]` into RGB `[H, W, 3]` blended over a grayscale background `[H, W]`.
#[cube(launch)]
pub(crate) fn colorize_labels_gray_kernel<F: Float>(
    labels: &[u32],
    bg_gray: &[F],
    palette: &[F],
    rgb_out: &mut [F],
    num_colors: u32,
    bg_label: u32,
    alpha: F,
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let lbl = labels[idx as usize];
        let bg = bg_gray[idx as usize];
        let out_base = (idx * 3u32) as usize;
        if lbl == bg_label {
            rgb_out[out_base] = bg;
            rgb_out[out_base + 1usize] = bg;
            rgb_out[out_base + 2usize] = bg;
        } else {
            let c_idx = ((lbl - 1u32) % num_colors) * 3u32;
            let cr = palette[c_idx as usize];
            let cg = palette[(c_idx + 1u32) as usize];
            let cb = palette[(c_idx + 2u32) as usize];
            let inv_a = F::new(1.0) - alpha;
            rgb_out[out_base] = alpha * cr + inv_a * bg;
            rgb_out[out_base + 1usize] = alpha * cg + inv_a * bg;
            rgb_out[out_base + 2usize] = alpha * cb + inv_a * bg;
        }
    }
}

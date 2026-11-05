//! GPU kernels for Beer-Lambert histological stain deconvolution and remixing.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Beer-Lambert optical density stain unmixing: `stains = -log10(max(rgb, 1e-6)) * inv_stain_mat`.
#[cube(launch)]
pub(crate) fn unmix_stains_kernel<F: Float>(
    rgb: &[F],
    inv_mat: &[F],
    stains: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let eps = F::new(1e-6);
        let ln10 = F::new(2.302585092994046);
        let od_r = -F::ln(F::max(rgb[base], eps)) / ln10;
        let od_g = -F::ln(F::max(rgb[base + 1usize], eps)) / ln10;
        let od_b = -F::ln(F::max(rgb[base + 2usize], eps)) / ln10;

        stains[base] = od_r * inv_mat[0usize] + od_g * inv_mat[3usize] + od_b * inv_mat[6usize];
        stains[base + 1usize] =
            od_r * inv_mat[1usize] + od_g * inv_mat[4usize] + od_b * inv_mat[7usize];
        stains[base + 2usize] =
            od_r * inv_mat[2usize] + od_g * inv_mat[5usize] + od_b * inv_mat[8usize];
    }
}

/// Beer-Lambert optical density stain remixing: `rgb = clamp(10^(-stains * stain_mat), 0, 1)`.
#[cube(launch)]
pub(crate) fn mix_stains_kernel<F: Float>(
    stains: &[F],
    fwd_mat: &[F],
    rgb: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let s0 = stains[base];
        let s1 = stains[base + 1usize];
        let s2 = stains[base + 2usize];

        let od_r = -(s0 * fwd_mat[0usize] + s1 * fwd_mat[3usize] + s2 * fwd_mat[6usize]);
        let od_g = -(s0 * fwd_mat[1usize] + s1 * fwd_mat[4usize] + s2 * fwd_mat[7usize]);
        let od_b = -(s0 * fwd_mat[2usize] + s1 * fwd_mat[5usize] + s2 * fwd_mat[8usize]);

        let ten = F::new(10.0);
        rgb[base] = F::min(F::max(F::powf(ten, od_r), F::new(0.0)), F::new(1.0));
        rgb[base + 1usize] = F::min(F::max(F::powf(ten, od_g), F::new(0.0)), F::new(1.0));
        rgb[base + 2usize] = F::min(F::max(F::powf(ten, od_b), F::new(0.0)), F::new(1.0));
    }
}

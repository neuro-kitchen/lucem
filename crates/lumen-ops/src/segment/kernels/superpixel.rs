//! Device kernels for SLIC superpixel assignment.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Assigns each pixel `(y, x)` of a 2-D single-channel image to the nearest SLIC cluster center in
/// 3-D `(y, x, intensity)` feature space.
/// `centers` has shape `[num_clusters, 3]` stored as `[cy, cx, cval]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn slic_assign_2d_kernel<F: Float>(
    image: &[F],
    centers: &[F],
    labels: &mut [u32],
    w: u32,
    num_clusters: u32,
    spatial_weight_sq: F,
    max_search_dist_sq: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let y = F::cast_from(idx / w);
        let x = F::cast_from(idx % w);
        let val = image[i];

        let mut best_dist = F::new(1e30f32);
        let mut best_k = 1u32;
        let mut k = 0u32;
        while k < num_clusters {
            let base = (k * 3u32) as usize;
            let cy = centers[base];
            let cx = centers[base + 1usize];
            let dy = y - cy;
            let dx = x - cx;
            let d_spatial = dy * dy + dx * dx;
            if d_spatial <= max_search_dist_sq {
                let dv = val - centers[base + 2usize];
                let d_total = dv * dv + spatial_weight_sq * d_spatial;
                if d_total < best_dist {
                    best_dist = d_total;
                    best_k = k + 1u32;
                }
            }
            k += 1u32;
        }
        labels[i] = best_k;
    }
}

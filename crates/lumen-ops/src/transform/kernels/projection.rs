//! Forward Radon sinogram projection and filtered backprojection (`iradon`) kernels.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Computes the 2-D Radon sinogram of `image` `[height, width]` into `sinogram` `[n_bins, n_angles]`.
///
/// For each detector bin $t = r - r_{\text{center}}$ and angle $\theta$ (`cos_t`, `sin_t`), integrates
/// along the ray $(x(s), y(s)) = (c_x + t \cos\theta - s \sin\theta,\; c_y - t \sin\theta - s \cos\theta)$
/// using bilinear interpolation.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn radon_2d_kernel<F: lumen_core::element::ImageFloat>(
    image: &[F],
    cos_table: &[F],
    sin_table: &[F],
    sinogram: &mut [F],
    height: i32,
    width: i32,
    n_bins: u32,
    n_angles: u32,
    half_diag: i32,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let bin = (idx / n_angles) as i32;
        let ang = (idx % n_angles) as usize;
        let t = F::cast_from(bin - half_diag);
        let c = cos_table[ang];
        let s = sin_table[ang];

        let cy = F::cast_from(height - 1i32) * F::new(0.5f32);
        let cx = F::cast_from(width - 1i32) * F::new(0.5f32);
        let one = F::new(1.0f32);
        let mut sum = F::new(0.0f32);

        let mut step = -half_diag;
        while step <= half_diag {
            let tau = F::cast_from(step);
            let x = cx + t * c - tau * s;
            let y = cy + t * s + tau * c;
            let y0 = i32::cast_from(F::floor(y));
            let x0 = i32::cast_from(F::floor(x));
            if y0 >= 0i32 && (y0 + 1i32) < height && x0 >= 0i32 && (x0 + 1i32) < width {
                let fy = y - F::floor(y);
                let fx = x - F::floor(x);
                let v00 = image[(y0 * width + x0) as usize];
                let v01 = image[(y0 * width + x0 + 1i32) as usize];
                let v10 = image[((y0 + 1i32) * width + x0) as usize];
                let v11 = image[((y0 + 1i32) * width + x0 + 1i32) as usize];
                sum += (one - fy) * ((one - fx) * v00 + fx * v01)
                    + fy * ((one - fx) * v10 + fx * v11);
            }
            step += 1i32;
        }
        let _ = n_bins;
        sinogram[i] = sum;
    }
}

/// Backprojects a ramp-filtered sinogram `[n_bins, n_angles]` onto a reconstructed `[out_size, out_size]`
/// image grid using linear detector interpolation.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn iradon_backproject_kernel<F: lumen_core::element::ImageFloat>(
    filtered_sino: &[F],
    cos_table: &[F],
    sin_table: &[F],
    output: &mut [F],
    n_bins: i32,
    n_angles: u32,
    out_size: u32,
    half_diag: F,
    scale: F,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let row = idx / out_size;
        let col = idx % out_size;
        let center = F::cast_from(out_size - 1u32) * F::new(0.5f32);
        let dy = F::cast_from(row) - center;
        let dx = F::cast_from(col) - center;
        let one = F::new(1.0f32);

        let mut acc = F::new(0.0f32);
        let mut a = 0u32;
        while a < n_angles {
            let c = cos_table[a as usize];
            let s = sin_table[a as usize];
            let t = dx * c + dy * s + half_diag;
            let b0 = i32::cast_from(F::floor(t));
            if b0 >= 0i32 && (b0 + 1i32) < n_bins {
                let ft = t - F::floor(t);
                let v0 = filtered_sino[((b0 as u32) * n_angles + a) as usize];
                let v1 = filtered_sino[(((b0 + 1i32) as u32) * n_angles + a) as usize];
                acc += (one - ft) * v0 + ft * v1;
            }
            a += 1u32;
        }
        output[i] = acc * scale;
    }
}

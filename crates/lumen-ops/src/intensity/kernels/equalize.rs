//! GPU kernels for histogram equalization, CLAHE, and histogram matching.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Normalize a cumulative histogram (`f32`/`F`) into `[0, 1]` CDF.
#[cube(launch)]
pub(crate) fn normalize_cdf_kernel<F: lumen_core::element::ImageFloat>(
    cum_hist: &[F],
    cdf_out: &mut [F],
    num_bins: u32,
) {
    let idx = element_position() as u32;
    if idx < num_bins {
        let total = cum_hist[(num_bins - 1u32) as usize];
        cdf_out[idx as usize] = if total > F::new(0.0) {
            cum_hist[idx as usize] / total
        } else {
            F::new(0.0)
        };
    }
}

/// Apply a 1-D lookup table (`cdf` of length `num_bins`) to an image mapped over `[v_min, v_max]`.
#[cube(launch)]
pub(crate) fn apply_cdf_lut_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    cdf: &[F],
    v_min_buf: &[F],
    v_max_buf: &[F],
    output: &mut [F],
    num_bins: u32,
    total: u32,
) {
    let idx = element_position() as u32;
    if idx < total {
        let v_min = v_min_buf[0];
        let v_max = v_max_buf[0];
        let span = v_max - v_min;
        if span <= F::new(1e-12) {
            output[idx as usize] = F::new(0.0);
        } else {
            let x = input[idx as usize];
            let norm = F::min(F::max((x - v_min) / span, F::new(0.0)), F::new(1.0));
            let pos = norm * F::cast_from(num_bins - 1u32);
            let b0 = u32::cast_from(F::floor(pos));
            let b1 = u32::min(b0 + 1u32, num_bins - 1u32);
            let frac = pos - F::cast_from(b0);
            let v0 = cdf[b0 as usize];
            let v1 = cdf[b1 as usize];
            output[idx as usize] = v0 * (F::new(1.0) - frac) + v1 * frac;
        }
    }
}

/// Build per-tile clipped and redistributed CDFs for 2-D CLAHE.
#[cube(launch)]
pub(crate) fn clahe_tile_cdfs_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    v_min_buf: &[F],
    v_max_buf: &[F],
    tile_cdfs: &mut [F],
    height: u32,
    width: u32,
    tiles_y: u32,
    tiles_x: u32,
    num_bins: u32,
    clip_limit: F,
) {
    let tile_idx = element_position() as u32;
    let total_tiles = tiles_y * tiles_x;
    if tile_idx < total_tiles {
        let tx = tile_idx % tiles_x;
        let ty = tile_idx / tiles_x;

        let y0 = (ty * height) / tiles_y;
        let y1 = ((ty + 1u32) * height) / tiles_y;
        let x0 = (tx * width) / tiles_x;
        let x1 = ((tx + 1u32) * width) / tiles_x;

        let v_min = v_min_buf[0];
        let v_max = v_max_buf[0];
        let span = F::max(v_max - v_min, F::new(1e-12));
        let base = tile_idx * num_bins;

        for b in 0u32..num_bins {
            tile_cdfs[(base + b) as usize] = F::new(0.0);
        }

        let mut count = 0u32;
        let mut y = y0;
        while y < y1 {
            let mut x = x0;
            while x < x1 {
                let val = input[(y * width + x) as usize];
                let norm = F::min(F::max((val - v_min) / span, F::new(0.0)), F::new(1.0));
                let bin = u32::min(
                    u32::cast_from(F::floor(norm * F::cast_from(num_bins - 1u32) + F::new(0.5))),
                    num_bins - 1u32,
                );
                tile_cdfs[(base + bin) as usize] += F::new(1.0);
                count += 1u32;
                x += 1u32;
            }
            y += 1u32;
        }

        let total_f = F::max(F::cast_from(count), F::new(1.0));
        let actual_limit = F::max(clip_limit * total_f, F::new(1.0));
        let mut excess = F::new(0.0);
        for b in 0u32..num_bins {
            let c = tile_cdfs[(base + b) as usize];
            if c > actual_limit {
                excess += c - actual_limit;
                tile_cdfs[(base + b) as usize] = actual_limit;
            }
        }

        let redist = excess / F::cast_from(num_bins);
        let mut cum = F::new(0.0);
        for b in 0u32..num_bins {
            cum += tile_cdfs[(base + b) as usize] + redist;
            tile_cdfs[(base + b) as usize] = F::min(cum / total_f, F::new(1.0));
        }
    }
}

/// Bilinearly interpolate across neighboring tile CDFs for 2-D CLAHE.
#[cube(launch)]
pub(crate) fn clahe_interpolate_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    v_min_buf: &[F],
    v_max_buf: &[F],
    tile_cdfs: &[F],
    output: &mut [F],
    height: u32,
    width: u32,
    tiles_y: u32,
    tiles_x: u32,
    num_bins: u32,
) {
    let idx = element_position() as u32;
    let total = height * width;
    if idx < total {
        let x = idx % width;
        let y = idx / width;

        let v_min = v_min_buf[0];
        let v_max = v_max_buf[0];
        let span = F::max(v_max - v_min, F::new(1e-12));
        let val = input[idx as usize];
        let norm = F::min(F::max((val - v_min) / span, F::new(0.0)), F::new(1.0));
        let bin = u32::min(
            u32::cast_from(F::floor(norm * F::cast_from(num_bins - 1u32) + F::new(0.5))),
            num_bins - 1u32,
        );

        let gy = (F::cast_from(y) + F::new(0.5)) * F::cast_from(tiles_y) / F::cast_from(height)
            - F::new(0.5);
        let gx = (F::cast_from(x) + F::new(0.5)) * F::cast_from(tiles_x) / F::cast_from(width)
            - F::new(0.5);

        let ty0_i = i32::cast_from(F::floor(gy));
        let tx0_i = i32::cast_from(F::floor(gx));
        let fy = F::min(F::max(gy - F::cast_from(ty0_i), F::new(0.0)), F::new(1.0));
        let fx = F::min(F::max(gx - F::cast_from(tx0_i), F::new(0.0)), F::new(1.0));

        let ty0 = u32::cast_from(i32::min(i32::max(ty0_i, 0i32), (tiles_y - 1u32) as i32));
        let ty1 = u32::cast_from(i32::min(
            i32::max(ty0_i + 1i32, 0i32),
            (tiles_y - 1u32) as i32,
        ));
        let tx0 = u32::cast_from(i32::min(i32::max(tx0_i, 0i32), (tiles_x - 1u32) as i32));
        let tx1 = u32::cast_from(i32::min(
            i32::max(tx0_i + 1i32, 0i32),
            (tiles_x - 1u32) as i32,
        ));

        let c00 = tile_cdfs[((ty0 * tiles_x + tx0) * num_bins + bin) as usize];
        let c01 = tile_cdfs[((ty0 * tiles_x + tx1) * num_bins + bin) as usize];
        let c10 = tile_cdfs[((ty1 * tiles_x + tx0) * num_bins + bin) as usize];
        let c11 = tile_cdfs[((ty1 * tiles_x + tx1) * num_bins + bin) as usize];

        let top = c00 * (F::new(1.0) - fx) + c01 * fx;
        let bot = c10 * (F::new(1.0) - fx) + c11 * fx;
        output[idx as usize] = top * (F::new(1.0) - fy) + bot * fy;
    }
}

/// Build an inverse mapping LUT from source CDF to reference quantile values.
#[cube(launch)]
pub(crate) fn build_match_lut_kernel<F: lumen_core::element::ImageFloat>(
    src_cdf: &[F],
    ref_cdf: &[F],
    ref_min_buf: &[F],
    ref_max_buf: &[F],
    lut_out: &mut [F],
    num_bins: u32,
) {
    let b = element_position() as u32;
    if b < num_bins {
        let p = src_cdf[b as usize];
        let r_min = ref_min_buf[0];
        let r_max = ref_max_buf[0];
        let mut k = 0u32;
        while k + 1u32 < num_bins {
            let rk = ref_cdf[k as usize];
            if rk >= p {
                break;
            }
            k += 1u32;
        }
        let t = F::cast_from(k) / F::cast_from(u32::max(num_bins - 1u32, 1u32));
        lut_out[b as usize] = r_min + t * (r_max - r_min);
    }
}

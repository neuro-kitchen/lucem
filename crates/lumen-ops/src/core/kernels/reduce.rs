//! Reduction kernels: cube-wide row reductions (sum, min, max, Welford mean/variance, argmin, argmax)
//! and axis reductions.

use cubecl::prelude::*;
use lumen_core::compute::{
    cube_row, element_position, negative_infinity, positive_infinity, row_share,
};

/// Reduces each row of a `[rows, cols]` matrix to `(sum, min, max)` using one cube per row and
/// pairwise tree reduction in shared memory. Accumulates sums in `f32` for precision.
#[cube(launch)]
pub fn row_sum_min_max_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    out_sum: &mut [F],
    out_min: &mut [F],
    out_max: &mut [F],
    rows: u32,
    cols: u32,
    #[comptime] units: u32,
) {
    let mut sh_sum = Shared::<[f32]>::new_slice(units as usize);
    let mut sh_min = Shared::<[F]>::new_slice(units as usize);
    let mut sh_max = Shared::<[F]>::new_slice(units as usize);

    let row = cube_row();
    let u = UNIT_POS_X as usize;
    if row < rows {
        let share = row_share(cols, units);
        let base = row * cols;
        let mut s = 0.0f32;
        let mut mn = positive_infinity::<F>();
        let mut mx = negative_infinity::<F>();
        let mut c = share.start;
        while c < share.end {
            let v = input[(base + c) as usize];
            s += f32::cast_from(v);
            mn = F::min(mn, v);
            mx = F::max(mx, v);
            c += share.step;
        }
        sh_sum[u] = s;
        sh_min[u] = mn;
        sh_max[u] = mx;
        sync_cube();

        let stride = RuntimeCell::<u32>::new(CUBE_DIM_X / 2u32);
        while stride.read() > 0u32 {
            let half = stride.read() as usize;
            if u < half {
                sh_sum[u] += sh_sum[u + half];
                sh_min[u] = F::min(sh_min[u], sh_min[u + half]);
                sh_max[u] = F::max(sh_max[u], sh_max[u + half]);
            }
            sync_cube();
            stride.store(stride.read() / 2u32);
        }

        if u == 0usize {
            let r = row as usize;
            out_sum[r] = F::cast_from(sh_sum[0]);
            out_min[r] = sh_min[0];
            out_max[r] = sh_max[0];
        }
    }
}

/// Reduces each row of `[rows, cols]` to `(mean, variance)` using Welford's online algorithm per unit
/// and Chan's parallel pairwise variance merge across the cube in `f32`.
#[cube(launch)]
pub fn row_mean_var_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    out_mean: &mut [F],
    out_var: &mut [F],
    rows: u32,
    cols: u32,
    ddof: u32,
    #[comptime] units: u32,
) {
    let mut sh_count = Shared::<[u32]>::new_slice(units as usize);
    let mut sh_mean = Shared::<[f32]>::new_slice(units as usize);
    let mut sh_m2 = Shared::<[f32]>::new_slice(units as usize);

    let row = cube_row();
    let u = UNIT_POS_X as usize;
    if row < rows {
        let share = row_share(cols, units);
        let base = row * cols;
        let mut count = 0u32;
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        let mut c = share.start;
        while c < share.end {
            let x = f32::cast_from(input[(base + c) as usize]);
            count += 1u32;
            let delta = x - mean;
            mean += delta / (count as f32);
            let delta2 = x - mean;
            m2 += delta * delta2;
            c += share.step;
        }
        sh_count[u] = count;
        sh_mean[u] = mean;
        sh_m2[u] = m2;
        sync_cube();

        let stride = RuntimeCell::<u32>::new(CUBE_DIM_X / 2u32);
        while stride.read() > 0u32 {
            let half = stride.read() as usize;
            if u < half {
                let na = sh_count[u];
                let nb = sh_count[u + half];
                let n = na + nb;
                if nb > 0u32 {
                    let delta = sh_mean[u + half] - sh_mean[u];
                    let nf = n as f32;
                    sh_mean[u] += delta * (nb as f32) / nf;
                    sh_m2[u] = sh_m2[u] + sh_m2[u + half] + delta * delta * (na as f32) * (nb as f32) / nf;
                    sh_count[u] = n;
                }
            }
            sync_cube();
            stride.store(stride.read() / 2u32);
        }

        if u == 0usize {
            let r = row as usize;
            let denom = if cols > ddof { (cols - ddof) as f32 } else { 1.0f32 };
            out_mean[r] = F::cast_from(sh_mean[0]);
            out_var[r] = F::cast_from(sh_m2[0] / denom);
        }
    }
}

/// Reduces each row of `[rows, cols]` to `(argmin, argmax)` with first-index tie breaking.
#[cube(launch)]
pub fn row_argmin_argmax_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    out_argmin: &mut [u32],
    out_argmax: &mut [u32],
    rows: u32,
    cols: u32,
    #[comptime] units: u32,
) {
    let mut sh_min_val = Shared::<[F]>::new_slice(units as usize);
    let mut sh_min_idx = Shared::<[u32]>::new_slice(units as usize);
    let mut sh_max_val = Shared::<[F]>::new_slice(units as usize);
    let mut sh_max_idx = Shared::<[u32]>::new_slice(units as usize);

    let row = cube_row();
    let u = UNIT_POS_X as usize;
    if row < rows {
        let share = row_share(cols, units);
        let base = row * cols;
        let mut mn_val = positive_infinity::<F>();
        let mut mn_idx = 0xFFFF_FFFFu32;
        let mut mx_val = negative_infinity::<F>();
        let mut mx_idx = 0xFFFF_FFFFu32;
        let mut c = share.start;
        while c < share.end {
            let v = input[(base + c) as usize];
            if v < mn_val || (v == mn_val && c < mn_idx) {
                mn_val = v;
                mn_idx = c;
            }
            if v > mx_val || (v == mx_val && c < mx_idx) {
                mx_val = v;
                mx_idx = c;
            }
            c += share.step;
        }
        sh_min_val[u] = mn_val;
        sh_min_idx[u] = mn_idx;
        sh_max_val[u] = mx_val;
        sh_max_idx[u] = mx_idx;
        sync_cube();

        let stride = RuntimeCell::<u32>::new(CUBE_DIM_X / 2u32);
        while stride.read() > 0u32 {
            let half = stride.read() as usize;
            if u < half {
                let b_min_val = sh_min_val[u + half];
                let b_min_idx = sh_min_idx[u + half];
                if b_min_val < sh_min_val[u] || (b_min_val == sh_min_val[u] && b_min_idx < sh_min_idx[u]) {
                    sh_min_val[u] = b_min_val;
                    sh_min_idx[u] = b_min_idx;
                }
                let b_max_val = sh_max_val[u + half];
                let b_max_idx = sh_max_idx[u + half];
                if b_max_val > sh_max_val[u] || (b_max_val == sh_max_val[u] && b_max_idx < sh_max_idx[u]) {
                    sh_max_val[u] = b_max_val;
                    sh_max_idx[u] = b_max_idx;
                }
            }
            sync_cube();
            stride.store(stride.read() / 2u32);
        }

        if u == 0usize {
            let r = row as usize;
            out_argmin[r] = sh_min_idx[0];
            out_argmax[r] = sh_max_idx[0];
        }
    }
}

/// Reduces along an arbitrary axis seen as `[outer, len, inner]` (`mode`: 0 = sum, 1 = mean, 2 = min, 3 = max).
#[cube(launch)]
pub fn reduce_axis_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    len: u32,
    inner: u32,
    out_len: u32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    if (idx as u32) < out_len {
        let e = idx as u32;
        let k = e % inner;
        let o = e / inner;
        let base = o * len * inner + k;
        if comptime!(mode == 0u32 || mode == 1u32) {
            let mut sum = 0.0f32;
            let mut i = 0u32;
            while i < len {
                sum += f32::cast_from(input[(base + i * inner) as usize]);
                i += 1u32;
            }
            let res = if comptime!(mode == 1u32) { sum / (len as f32) } else { sum };
            output[idx] = F::cast_from(res);
        } else if comptime!(mode == 2u32) {
            let mut mn = positive_infinity::<F>();
            let mut i = 0u32;
            while i < len {
                mn = F::min(mn, input[(base + i * inner) as usize]);
                i += 1u32;
            }
            output[idx] = mn;
        } else {
            let mut mx = negative_infinity::<F>();
            let mut i = 0u32;
            while i < len {
                mx = F::max(mx, input[(base + i * inner) as usize]);
                i += 1u32;
            }
            output[idx] = mx;
        }
    }
}

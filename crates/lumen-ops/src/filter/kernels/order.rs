//! Order-statistic filter kernels: 1-D min/max, N-D footprint min/max, 9-element sorting network
//! (`med9`) for $3 \times 3$ median, and register rank-counting for arbitrary rank/median/percentile
//! windows.

use cubecl::prelude::*;
use lumen_core::compute::{element_position, negative_infinity, positive_infinity};

use crate::core::kernels::{pad_index, read_padded};

/// 1-D minimum (`is_max = false`) or maximum (`is_max = true`) filter of window `size` along an
/// axis seen as `[outer, len, inner]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn extrema_1d_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    len: u32,
    inner: u32,
    left: i32,
    cval: F,
    #[comptime] size: u32,
    #[comptime] mode: u32,
    #[comptime] is_max: bool,
) {
    let idx = element_position();
    if idx < output.len() {
        let e = idx as u32;
        let i = (e / inner) % len;
        let base = e - i * inner;
        let start = (i as i32) - left;
        let mut best = if comptime!(is_max) { negative_infinity::<F>() } else { positive_infinity::<F>() };
        let mut t = 0u32;
        if start >= 0i32 && start + comptime!(size as i32) <= (len as i32) {
            let mut at = base + (start as u32) * inner;
            while t < size {
                let v = input[at as usize];
                best = if comptime!(is_max) { F::max(best, v) } else { F::min(best, v) };
                at += inner;
                t += 1u32;
            }
        } else {
            while t < size {
                let v = read_padded::<F>(input, base, inner, start + (t as i32), len as i32, cval, mode);
                best = if comptime!(is_max) { F::max(best, v) } else { F::min(best, v) };
                t += 1u32;
            }
        }
        output[idx] = best;
    }
}

/// 2-D minimum or maximum filter over an arbitrary footprint given by `offsets` (`[n_active, 2]` flattened).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn extrema_2d_footprint_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    offsets: &[i32],
    height: i32,
    width: i32,
    n_active: u32,
    cval: F,
    #[comptime] mode: u32,
    #[comptime] is_max: bool,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;
        let mut best = if comptime!(is_max) { negative_infinity::<F>() } else { positive_infinity::<F>() };
        let mut k = 0u32;
        while k < n_active {
            let dy = offsets[(k * 2u32) as usize];
            let dx = offsets[(k * 2u32 + 1u32) as usize];
            let py = pad_index(y + dy, height, mode);
            let px = pad_index(x + dx, width, mode);
            let sy = i32::max(py, 0i32) as u32;
            let sx = i32::max(px, 0i32) as u32;
            let raw = input[(sy * w_u + sx) as usize];
            let v = if py < 0i32 || px < 0i32 { cval } else { raw };
            best = if comptime!(is_max) { F::max(best, v) } else { F::min(best, v) };
            k += 1u32;
        }
        output[idx] = best;
    }
}

/// Branch-free 19-comparator sorting network returning the exact median of 9 values (`med9`).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn med9<F: Float>(
    v0: F,
    v1: F,
    v2: F,
    v3: F,
    v4: F,
    v5: F,
    v6: F,
    v7: F,
    v8: F,
) -> F {
    let mut a0 = v0;
    let mut a1 = v1;
    let mut a2 = v2;
    let mut a3 = v3;
    let mut a4 = v4;
    let mut a5 = v5;
    let mut a6 = v6;
    let mut a7 = v7;
    let mut a8 = v8;

    let (t1, t2) = (F::min(a1, a2), F::max(a1, a2)); a1 = t1; a2 = t2;
    let (t4, t5) = (F::min(a4, a5), F::max(a4, a5)); a4 = t4; a5 = t5;
    let (t7, t8) = (F::min(a7, a8), F::max(a7, a8)); a7 = t7; a8 = t8;

    let (t0, t1b) = (F::min(a0, a1), F::max(a0, a1)); a0 = t0; a1 = t1b;
    let (t3, t4b) = (F::min(a3, a4), F::max(a3, a4)); a3 = t3; a4 = t4b;
    let (t6, t7b) = (F::min(a6, a7), F::max(a6, a7)); a6 = t6; a7 = t7b;

    let (t1c, t2b) = (F::min(a1, a2), F::max(a1, a2)); a1 = t1c; a2 = t2b;
    let (t4c, t5b) = (F::min(a4, a5), F::max(a4, a5)); a4 = t4c; a5 = t5b;
    let (t7c, t8b) = (F::min(a7, a8), F::max(a7, a8)); a7 = t7c; a8 = t8b;

    a3 = F::max(a0, a3);
    a5 = F::min(a5, a8);
    let (t4d, t7d) = (F::min(a4, a7), F::max(a4, a7)); a4 = t4d; a7 = t7d;
    a6 = F::max(a3, a6);
    a4 = F::max(a1, a4);
    a2 = F::min(a2, a5);
    a4 = F::min(a4, a7);
    let (t4e, t2c) = (F::min(a4, a2), F::max(a4, a2)); a4 = t4e; a2 = t2c;
    a4 = F::max(a6, a4);
    F::min(a4, a2)
}

/// Specialized branch-free $3 \times 3$ median filter using the 19-step `med9` network and an
/// interior fast path.
#[cube(launch)]
pub fn median_3x3_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    height: i32,
    width: i32,
    cval: F,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;
        if y >= 1i32 && y + 1i32 < height && x >= 1i32 && x + 1i32 < width {
            let r0 = ((y - 1i32) as u32) * w_u + ((x - 1i32) as u32);
            let r1 = r0 + w_u;
            let r2 = r1 + w_u;
            output[idx] = med9::<F>(
                input[r0 as usize],
                input[(r0 + 1u32) as usize],
                input[(r0 + 2u32) as usize],
                input[r1 as usize],
                input[(r1 + 1u32) as usize],
                input[(r1 + 2u32) as usize],
                input[r2 as usize],
                input[(r2 + 1u32) as usize],
                input[(r2 + 2u32) as usize],
            );
        } else {
            let mut vals = Array::<F>::new(9usize);
            let mut k = 0u32;
            let mut dy = -1i32;
            while dy <= 1i32 {
                let py = pad_index(y + dy, height, mode);
                let sy = i32::max(py, 0i32) as u32;
                let mut dx = -1i32;
                while dx <= 1i32 {
                    let px = pad_index(x + dx, width, mode);
                    let sx = i32::max(px, 0i32) as u32;
                    let raw = input[(sy * w_u + sx) as usize];
                    vals[k as usize] = if py < 0i32 || px < 0i32 { cval } else { raw };
                    k += 1u32;
                    dx += 1i32;
                }
                dy += 1i32;
            }
            output[idx] = med9::<F>(
                vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8],
            );
        }
    }
}

/// 2-D rank filter over an arbitrary footprint given by `offsets` (`[n_active, 2]`), selecting the
/// `target_rank`-th smallest element (`0 <= target_rank < n_active`) via exact rank counting.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rank_2d_footprint_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    offsets: &[i32],
    height: i32,
    width: i32,
    n_active: u32,
    target_rank: u32,
    cval: F,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;
        let mut chosen = input[idx];
        let mut i = 0u32;
        while i < n_active {
            let dy_i = offsets[(i * 2u32) as usize];
            let dx_i = offsets[(i * 2u32 + 1u32) as usize];
            let py_i = pad_index(y + dy_i, height, mode);
            let px_i = pad_index(x + dx_i, width, mode);
            let sy_i = i32::max(py_i, 0i32) as u32;
            let sx_i = i32::max(px_i, 0i32) as u32;
            let raw_i = input[(sy_i * w_u + sx_i) as usize];
            let vi = if py_i < 0i32 || px_i < 0i32 { cval } else { raw_i };

            let mut less = 0u32;
            let mut equal_before = 0u32;
            let mut j = 0u32;
            while j < n_active {
                let dy_j = offsets[(j * 2u32) as usize];
                let dx_j = offsets[(j * 2u32 + 1u32) as usize];
                let py_j = pad_index(y + dy_j, height, mode);
                let px_j = pad_index(x + dx_j, width, mode);
                let sy_j = i32::max(py_j, 0i32) as u32;
                let sx_j = i32::max(px_j, 0i32) as u32;
                let raw_j = input[(sy_j * w_u + sx_j) as usize];
                let vj = if py_j < 0i32 || px_j < 0i32 { cval } else { raw_j };
                if vj < vi {
                    less += 1u32;
                } else if vj == vi && j < i {
                    equal_before += 1u32;
                }
                j += 1u32;
            }
            if less + equal_before == target_rank {
                chosen = vi;
            }
            i += 1u32;
        }
        output[idx] = chosen;
    }
}

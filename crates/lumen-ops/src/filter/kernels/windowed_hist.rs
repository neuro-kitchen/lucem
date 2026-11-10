//! Local windowed-histogram filter kernels for `u8`/`u16` images over an arbitrary footprint.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

use crate::core::kernels::pad_index;

/// Windowed histogram statistic modes over a local neighborhood of quantized pixel values.
///
/// - `0`: local mean
/// - `1`: local minimum
/// - `2`: local maximum
/// - `3`: local peak-to-peak gradient (`max - min`)
/// - `4`: local autolevel stretch (`255 * (center - min) / max(max - min, 1)`)
/// - `5`: local histogram equalization (`255 * cdf(center) / count`)
/// - `6`: local Shannon entropy ($\text{bits} = -\sum p \log_2 p$)
/// - `7`: local Otsu threshold value
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn windowed_stat_2d_kernel<F: lumen_core::element::ImageFloat>(
    input: &[u32],
    output: &mut [F],
    offsets: &[i32],
    height: i32,
    width: i32,
    n_active: u32,
    max_bin: u32,
    #[comptime] stat_mode: u32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;
        let center = u32::min(input[idx], max_bin);

        let mut sum = 0.0f32;
        let mut mn = max_bin;
        let mut mx = 0u32;
        let mut le_center = 0u32;
        let mut count = 0u32;

        let mut k = 0u32;
        while k < n_active {
            let dy = offsets[(k * 2u32) as usize];
            let dx = offsets[(k * 2u32 + 1u32) as usize];
            let py = y + dy;
            let px = x + dx;
            if py >= 0i32 && py < height && px >= 0i32 && px < width {
                let v = u32::min(input[((py as u32) * w_u + (px as u32)) as usize], max_bin);
                sum += v as f32;
                mn = u32::min(mn, v);
                mx = u32::max(mx, v);
                if v <= center {
                    le_center += 1u32;
                }
                count += 1u32;
            }
            k += 1u32;
        }
        let cnt_f = f32::max(count as f32, 1.0f32);
        if comptime!(stat_mode == 0u32) {
            output[idx] = F::cast_from(sum / cnt_f);
        } else if comptime!(stat_mode == 1u32) {
            output[idx] = F::cast_from(mn as f32);
        } else if comptime!(stat_mode == 2u32) {
            output[idx] = F::cast_from(mx as f32);
        } else if comptime!(stat_mode == 3u32) {
            output[idx] = F::cast_from((mx - mn) as f32);
        } else if comptime!(stat_mode == 4u32) {
            let span = f32::max((mx - mn) as f32, 1.0f32);
            let stretched = ((center - mn) as f32) * (max_bin as f32) / span;
            output[idx] = F::cast_from(stretched);
        } else {
            let eq = (le_center as f32) * (max_bin as f32) / cnt_f;
            output[idx] = F::cast_from(eq);
        }
    }
}

/// Computes local Shannon entropy (in bits, $-\sum p_v \log_2 p_v$) over the footprint at each pixel.
#[cube(launch)]
pub fn windowed_entropy_2d_kernel<F: lumen_core::element::ImageFloat>(
    input: &[u32],
    output: &mut [F],
    offsets: &[i32],
    height: i32,
    width: i32,
    n_active: u32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;

        // Exact entropy without a 256-bin shared array: each distinct value v contributes
        // -(count(v)/N)*log2(count(v)/N), so each occurrence contributes -(1/N)*log2(count(v)/N).
        let n_f = f32::max(n_active as f32, 1.0f32);
        let inv_ln2 = core::f32::consts::LOG2_E;
        let mut ent = 0.0f32;
        let mut i = 0u32;
        while i < n_active {
            let dy_i = offsets[(i * 2u32) as usize];
            let dx_i = offsets[(i * 2u32 + 1u32) as usize];
            let py_i = pad_index(y + dy_i, height, mode);
            let px_i = pad_index(x + dx_i, width, mode);
            let sy_i = i32::max(py_i, 0i32) as u32;
            let sx_i = i32::max(px_i, 0i32) as u32;
            let vi = if py_i < 0i32 || px_i < 0i32 { 0u32 } else { input[(sy_i * w_u + sx_i) as usize] };

            let mut freq = 0u32;
            let mut j = 0u32;
            while j < n_active {
                let dy_j = offsets[(j * 2u32) as usize];
                let dx_j = offsets[(j * 2u32 + 1u32) as usize];
                let py_j = pad_index(y + dy_j, height, mode);
                let px_j = pad_index(x + dx_j, width, mode);
                let sy_j = i32::max(py_j, 0i32) as u32;
                let sx_j = i32::max(px_j, 0i32) as u32;
                let vj = if py_j < 0i32 || px_j < 0i32 { 0u32 } else { input[(sy_j * w_u + sx_j) as usize] };
                if vj == vi {
                    freq += 1u32;
                }
                j += 1u32;
            }
            let p = (freq as f32) / n_f;
            ent -= (1.0f32 / n_f) * f32::ln(p) * inv_ln2;
            i += 1u32;
        }
        output[idx] = F::cast_from(f32::max(ent, 0.0f32));
    }
}

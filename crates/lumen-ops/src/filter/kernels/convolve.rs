//! Kernels for 1-D and N-D spatial correlation and convolution.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

use crate::core::kernels::{pad_index, read_padded};

/// 1-D correlation along an axis seen as `[outer, len, inner]`:
/// `output[o, i, k] = sum_{t=0..size} weights[t] * input[o, i + t - left, k]`, where
/// `left = size / 2 + origin`.
///
/// Uses an interior fast path when the entire window `[i - left, i - left + size)` lies inside
/// `0..len`, avoiding boundary arithmetic on interior pixels.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn correlate_1d_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    weights: &[F],
    len: u32,
    inner: u32,
    left: i32,
    cval: F,
    #[comptime] size: u32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    if idx < output.len() {
        let e = idx as u32;
        let i = (e / inner) % len;
        let base = e - i * inner;
        let start = (i as i32) - left;
        let mut acc = F::new(0.0f32);
        let mut t = 0u32;
        if start >= 0i32 && start + comptime!(size as i32) <= (len as i32) {
            let mut at = base + (start as u32) * inner;
            while t < size {
                acc += weights[t as usize] * input[at as usize];
                at += inner;
                t += 1u32;
            }
        } else {
            while t < size {
                let sample = read_padded::<F>(
                    input,
                    base,
                    inner,
                    start + (t as i32),
                    len as i32,
                    cval,
                    mode,
                );
                acc += weights[t as usize] * sample;
                t += 1u32;
            }
        }
        output[idx] = acc;
    }
}

/// 2-D correlation with arbitrary dense weights `[k_h, k_w]` centered at `(left_y, left_x)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn correlate_2d_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    weights: &[F],
    height: i32,
    width: i32,
    left_y: i32,
    left_x: i32,
    cval: F,
    #[comptime] k_h: u32,
    #[comptime] k_w: u32,
    #[comptime] mode: u32,
) {
    let idx = element_position();
    let total = (height as u32) * (width as u32);
    if (idx as u32) < total {
        let e = idx as u32;
        let w_u = width as u32;
        let y = (e / w_u) as i32;
        let x = (e % w_u) as i32;
        let y0 = y - left_y;
        let x0 = x - left_x;
        let mut acc = F::new(0.0f32);
        let mut ky = 0u32;
        if y0 >= 0i32 && y0 + comptime!(k_h as i32) <= height && x0 >= 0i32 && x0 + comptime!(k_w as i32) <= width {
            while ky < k_h {
                let row_base = ((y0 + (ky as i32)) as u32) * w_u + (x0 as u32);
                let w_base = ky * k_w;
                let mut kx = 0u32;
                while kx < k_w {
                    acc += weights[(w_base + kx) as usize] * input[(row_base + kx) as usize];
                    kx += 1u32;
                }
                ky += 1u32;
            }
        } else {
            while ky < k_h {
                let py = pad_index(y0 + (ky as i32), height, mode);
                let sy = i32::max(py, 0i32) as u32;
                let w_base = ky * k_w;
                let mut kx = 0u32;
                while kx < k_w {
                    let px = pad_index(x0 + (kx as i32), width, mode);
                    let sx = i32::max(px, 0i32) as u32;
                    let raw = input[(sy * w_u + sx) as usize];
                    let sample = if py < 0i32 || px < 0i32 { cval } else { raw };
                    acc += weights[(w_base + kx) as usize] * sample;
                    kx += 1u32;
                }
                ky += 1u32;
            }
        }
        output[idx] = acc;
    }
}

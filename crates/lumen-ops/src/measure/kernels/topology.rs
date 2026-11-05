//! Device kernels for perimeter contributions, 2-D block reduction, and Laplacian sharpness energy.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Computes per-pixel 2-D contour configuration codes (`0..16`) for 2x2 cells.
#[cube(launch)]
pub fn perimeter_2d_contribs_kernel<E: Numeric, F: Float>(
    input: &[E],
    output: &mut [F],
    h: u32,
    w: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let r = idx / w;
        let c = idx % w;
        let v00 = input[i] != E::from_int(0);
        if !v00 {
            output[i] = F::new(0.0f32);
        } else {
            let n_up = if r > 0u32 {
                input[((r - 1u32) * w + c) as usize] != E::from_int(0)
            } else {
                false
            };
            let n_down = if r + 1u32 < h {
                input[((r + 1u32) * w + c) as usize] != E::from_int(0)
            } else {
                false
            };
            let n_left = if c > 0u32 {
                input[(r * w + (c - 1u32)) as usize] != E::from_int(0)
            } else {
                false
            };
            let n_right = if c + 1u32 < w {
                input[(r * w + (c + 1u32)) as usize] != E::from_int(0)
            } else {
                false
            };
            let mut exposed = 0u32;
            if !n_up {
                exposed += 1u32;
            }
            if !n_down {
                exposed += 1u32;
            }
            if !n_left {
                exposed += 1u32;
            }
            if !n_right {
                exposed += 1u32;
            }
            output[i] = F::cast_from(exposed);
        }
    }
}

/// Downsamples a 2-D `[in_h, in_w]` array into `[out_h, out_w]` blocks of size `(bh, bw)`.
/// `op`: `0 = Sum`, `1 = Mean`, `2 = Min`, `3 = Max`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn block_reduce_2d_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    in_h: u32,
    in_w: u32,
    out_w: u32,
    bh: u32,
    bw: u32,
    cval: F,
    op: u32,
    out_len: u32,
) {
    let i = element_position();
    if (i as u32) < out_len {
        let idx = i as u32;
        let br = idx / out_w;
        let bc = idx % out_w;
        let r0 = br * bh;
        let c0 = bc * bw;

        let mut acc = if op == 2u32 {
            F::new(1e30f32)
        } else if op == 3u32 {
            F::new(-1e30f32)
        } else {
            F::new(0.0f32)
        };

        let mut dr = 0u32;
        while dr < bh {
            let r = r0 + dr;
            let mut dc = 0u32;
            while dc < bw {
                let c = c0 + dc;
                let v = if r < in_h && c < in_w {
                    input[(r * in_w + c) as usize]
                } else {
                    cval
                };
                if op == 2u32 {
                    acc = F::min(acc, v);
                } else if op == 3u32 {
                    acc = F::max(acc, v);
                } else {
                    acc = acc + v;
                }
                dc += 1u32;
            }
            dr += 1u32;
        }

        if op == 1u32 {
            let count = F::cast_from(bh * bw);
            output[i] = acc / count;
        } else {
            output[i] = acc;
        }
    }
}

/// Discrete 2-D Laplacian response (`4*center - up - down - left - right`) with nearest clamping.
#[cube(launch)]
pub fn laplacian_energy_2d_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    h: u32,
    w: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let r = idx / w;
        let c = idx % w;
        let r_up = if r > 0u32 { r - 1u32 } else { 0u32 };
        let r_dn = if r + 1u32 < h { r + 1u32 } else { h - 1u32 };
        let c_lt = if c > 0u32 { c - 1u32 } else { 0u32 };
        let c_rt = if c + 1u32 < w { c + 1u32 } else { w - 1u32 };
        let center = input[i];
        let up = input[(r_up * w + c) as usize];
        let dn = input[(r_dn * w + c) as usize];
        let lt = input[(r * w + c_lt) as usize];
        let rt = input[(r * w + c_rt) as usize];
        output[i] = up + dn + lt + rt - F::new(4.0f32) * center;
    }
}

//! GPU kernels for CIE76, CIE94, CIEDE2000, and CMC perceptual color difference metrics.

// CubeCL kernels: `contains`, `clamp` and `abs_diff` do not lower to device code, so the
// comparisons stay written out
#![allow(clippy::manual_range_contains, clippy::manual_clamp, clippy::manual_abs_diff)]

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Euclidean CIE76 color difference in L*a*b* space.
#[cube(launch)]
pub(crate) fn cie76_kernel<F: lumen_core::element::ImageFloat>(
    lab1: &[F],
    lab2: &[F],
    out: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let dl = lab1[base] - lab2[base];
        let da = lab1[base + 1usize] - lab2[base + 1usize];
        let db = lab1[base + 2usize] - lab2[base + 2usize];
        out[idx as usize] = F::sqrt(dl * dl + da * da + db * db);
    }
}

/// CIE94 perceptual color difference in L*a*b* space.
#[cube(launch)]
pub(crate) fn ciede94_kernel<F: lumen_core::element::ImageFloat>(
    lab1: &[F],
    lab2: &[F],
    out: &mut [F],
    k_l: F,
    k_c: F,
    k_h: F,
    k_1: F,
    k_2: F,
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let l1 = lab1[base];
        let a1 = lab1[base + 1usize];
        let b1 = lab1[base + 2usize];
        let l2 = lab2[base];
        let a2 = lab2[base + 1usize];
        let b2 = lab2[base + 2usize];

        let dl = l1 - l2;
        let c1 = F::sqrt(a1 * a1 + b1 * b1);
        let c2 = F::sqrt(a2 * a2 + b2 * b2);
        let dc = c1 - c2;
        let da = a1 - a2;
        let db = b1 - b2;
        let dh_sq = F::max(da * da + db * db - dc * dc, F::new(0.0));

        let sl = F::new(1.0);
        let sc = F::new(1.0) + k_1 * c1;
        let sh = F::new(1.0) + k_2 * c1;

        let tl = dl / (k_l * sl);
        let tc = dc / (k_c * sc);
        let th_sq = dh_sq / (k_h * sh * k_h * sh);
        out[idx as usize] = F::sqrt(tl * tl + tc * tc + th_sq);
    }
}

/// CIEDE2000 perceptual color difference in L*a*b* space.
#[cube(launch)]
pub(crate) fn ciede2000_kernel<F: lumen_core::element::ImageFloat>(
    lab1: &[F],
    lab2: &[F],
    out: &mut [F],
    k_l: f32,
    k_c: f32,
    k_h: f32,
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let l1 = f32::cast_from(lab1[base]);
        let a1 = f32::cast_from(lab1[base + 1usize]);
        let b1 = f32::cast_from(lab1[base + 2usize]);
        let l2 = f32::cast_from(lab2[base]);
        let a2 = f32::cast_from(lab2[base + 1usize]);
        let b2 = f32::cast_from(lab2[base + 2usize]);

        let c1 = f32::sqrt(a1 * a1 + b1 * b1);
        let c2 = f32::sqrt(a2 * a2 + b2 * b2);
        let c_bar = 0.5f32 * (c1 + c2);
        let c_bar7 = f32::powf(c_bar, 7.0f32);
        let p25_7 = 6103515625.0f32;
        let g = 0.5f32 * (1.0f32 - f32::sqrt(c_bar7 / (c_bar7 + p25_7)));

        let a1p = (1.0f32 + g) * a1;
        let a2p = (1.0f32 + g) * a2;
        let c1p = f32::sqrt(a1p * a1p + b1 * b1);
        let c2p = f32::sqrt(a2p * a2p + b2 * b2);

        let deg_per_rad = 57.295_78_f32;
        let rad_per_deg = 0.017_453_292_f32;

        let mut h1p = f32::atan2(b1, a1p) * deg_per_rad;
        if h1p < 0.0f32 {
            h1p += 360.0f32;
        }
        let mut h2p = f32::atan2(b2, a2p) * deg_per_rad;
        if h2p < 0.0f32 {
            h2p += 360.0f32;
        }

        let dlp = l2 - l1;
        let dcp = c2p - c1p;
        let h_diff = h2p - h1p;
        let dhp_deg = if c1p * c2p == 0.0f32 {
            0.0f32
        } else if f32::abs(h_diff) <= 180.0f32 {
            h_diff
        } else if h_diff > 180.0f32 {
            h_diff - 360.0f32
        } else {
            h_diff + 360.0f32
        };
        let d_hp = 2.0f32 * f32::sqrt(c1p * c2p) * f32::sin(0.5f32 * dhp_deg * rad_per_deg);

        let l_bar = 0.5f32 * (l1 + l2);
        let cp_bar = 0.5f32 * (c1p + c2p);
        let hp_bar = if c1p * c2p == 0.0f32 {
            h1p + h2p
        } else if f32::abs(h1p - h2p) <= 180.0f32 {
            0.5f32 * (h1p + h2p)
        } else if h1p + h2p < 360.0f32 {
            0.5f32 * (h1p + h2p + 360.0f32)
        } else {
            0.5f32 * (h1p + h2p - 360.0f32)
        };

        let t = 1.0f32
            - 0.17f32 * f32::cos((hp_bar - 30.0f32) * rad_per_deg)
            + 0.24f32 * f32::cos(2.0f32 * hp_bar * rad_per_deg)
            + 0.32f32 * f32::cos((3.0f32 * hp_bar + 6.0f32) * rad_per_deg)
            - 0.20f32 * f32::cos((4.0f32 * hp_bar - 63.0f32) * rad_per_deg);

        let d_theta = 30.0f32 * f32::exp(-f32::powf((hp_bar - 275.0f32) / 25.0f32, 2.0f32));
        let cp_bar7 = f32::powf(cp_bar, 7.0f32);
        let r_c = 2.0f32 * f32::sqrt(cp_bar7 / (cp_bar7 + p25_7));
        let l_m50_sq = (l_bar - 50.0f32) * (l_bar - 50.0f32);
        let s_l = 1.0f32 + (0.015f32 * l_m50_sq) / f32::sqrt(20.0f32 + l_m50_sq);
        let s_c = 1.0f32 + 0.045f32 * cp_bar;
        let s_h = 1.0f32 + 0.015f32 * cp_bar * t;
        let r_t = -f32::sin(2.0f32 * d_theta * rad_per_deg) * r_c;

        let tl = dlp / (k_l * s_l);
        let tc = dcp / (k_c * s_c);
        let th = d_hp / (k_h * s_h);
        let de = f32::sqrt(tl * tl + tc * tc + th * th + r_t * tc * th);
        out[idx as usize] = F::cast_from(de);
    }
}

/// CMC(l:c) color difference in L*a*b* space.
#[cube(launch)]
pub(crate) fn cmc_kernel<F: lumen_core::element::ImageFloat>(
    lab1: &[F],
    lab2: &[F],
    out: &mut [F],
    k_l: f32,
    k_c: f32,
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let l1 = f32::cast_from(lab1[base]);
        let a1 = f32::cast_from(lab1[base + 1usize]);
        let b1 = f32::cast_from(lab1[base + 2usize]);
        let l2 = f32::cast_from(lab2[base]);
        let a2 = f32::cast_from(lab2[base + 1usize]);
        let b2 = f32::cast_from(lab2[base + 2usize]);

        let dl = l1 - l2;
        let c1 = f32::sqrt(a1 * a1 + b1 * b1);
        let c2 = f32::sqrt(a2 * a2 + b2 * b2);
        let dc = c1 - c2;
        let da = a1 - a2;
        let db = b1 - b2;
        let dh_sq = f32::max(da * da + db * db - dc * dc, 0.0f32);

        let mut h1 = f32::atan2(b1, a1) * 57.295_78_f32;
        if h1 < 0.0f32 {
            h1 += 360.0f32;
        }
        let rad_per_deg = 0.017_453_292_f32;
        let t = if h1 >= 164.0f32 && h1 <= 345.0f32 {
            0.56f32 + f32::abs(0.2f32 * f32::cos((h1 + 168.0f32) * rad_per_deg))
        } else {
            0.36f32 + f32::abs(0.4f32 * f32::cos((h1 + 35.0f32) * rad_per_deg))
        };
        let c1_4 = c1 * c1 * c1 * c1;
        let f = f32::sqrt(c1_4 / (c1_4 + 1900.0f32));
        let s_l = if l1 < 16.0f32 {
            0.511f32
        } else {
            (0.040975f32 * l1) / (1.0f32 + 0.01765f32 * l1)
        };
        let s_c = (0.0638f32 * c1) / (1.0f32 + 0.0131f32 * c1) + 0.638f32;
        let s_h = s_c * (f * t + 1.0f32 - f);

        let tl = dl / (k_l * s_l);
        let tc = dc / (k_c * s_c);
        let th_sq = dh_sq / (s_h * s_h);
        out[idx as usize] = F::cast_from(f32::sqrt(tl * tl + tc * tc + th_sq));
    }
}

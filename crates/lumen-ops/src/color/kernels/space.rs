//! GPU kernels for color space conversions (`[H, W, 3]` / `[H, W, 4]` interleaved channels).

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Convert sRGB `[H, W, 3]` to luminance `[H, W]` using ITU-R BT.709 weights (`0.2125 R + 0.7154 G + 0.0721 B`).
#[cube(launch)]
pub(crate) fn rgb_to_gray_kernel<F: lumen_core::element::ImageFloat>(
    rgb: &[F],
    gray: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let r = rgb[base];
        let g = rgb[base + 1usize];
        let b = rgb[base + 2usize];
        gray[idx as usize] = F::new(0.2125) * r + F::new(0.7154) * g + F::new(0.0721) * b;
    }
}

/// Replicate a grayscale image `[H, W]` into 3-channel RGB `[H, W, 3]`.
#[cube(launch)]
pub(crate) fn gray_to_rgb_kernel<F: lumen_core::element::ImageFloat>(
    gray: &[F],
    rgb: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let v = gray[idx as usize];
        let base = (idx * 3u32) as usize;
        rgb[base] = v;
        rgb[base + 1usize] = v;
        rgb[base + 2usize] = v;
    }
}

/// Composite RGBA `[H, W, 4]` over a uniform `[bg_r, bg_g, bg_b]` background into RGB `[H, W, 3]`.
#[cube(launch)]
pub(crate) fn rgba_to_rgb_kernel<F: lumen_core::element::ImageFloat>(
    rgba: &[F],
    rgb: &mut [F],
    bg_r: F,
    bg_g: F,
    bg_b: F,
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let in_base = (idx * 4u32) as usize;
        let out_base = (idx * 3u32) as usize;
        let r = rgba[in_base];
        let g = rgba[in_base + 1usize];
        let b = rgba[in_base + 2usize];
        let a = F::min(F::max(rgba[in_base + 3usize], F::new(0.0)), F::new(1.0));
        let inv_a = F::new(1.0) - a;
        rgb[out_base] = a * r + inv_a * bg_r;
        rgb[out_base + 1usize] = a * g + inv_a * bg_g;
        rgb[out_base + 2usize] = a * b + inv_a * bg_b;
    }
}

/// Apply a cached `3x3` color matrix (`mat` of length 9) per pixel.
#[cube(launch)]
pub(crate) fn apply_color_matrix_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    mat: &[F],
    output: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let c0 = input[base];
        let c1 = input[base + 1usize];
        let c2 = input[base + 2usize];
        output[base] = mat[0usize] * c0 + mat[1usize] * c1 + mat[2usize] * c2;
        output[base + 1usize] = mat[3usize] * c0 + mat[4usize] * c1 + mat[5usize] * c2;
        output[base + 2usize] = mat[6usize] * c0 + mat[7usize] * c1 + mat[8usize] * c2;
    }
}

/// Convert RGB `[0, 1]` to HSV (`H, S, V` in `[0, 1]`).
#[cube(launch)]
pub(crate) fn rgb_to_hsv_kernel<F: lumen_core::element::ImageFloat>(
    rgb: &[F],
    hsv: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let r = rgb[base];
        let g = rgb[base + 1usize];
        let b = rgb[base + 2usize];

        let c_max = F::max(r, F::max(g, b));
        let c_min = F::min(r, F::min(g, b));
        let delta = c_max - c_min;

        let s = if c_max > F::new(1e-10) {
            delta / c_max
        } else {
            F::new(0.0)
        };

        let mut h = if delta <= F::new(1e-10) {
            F::new(0.0)
        } else if c_max == r {
            (g - b) / (F::new(6.0) * delta)
        } else if c_max == g {
            (F::new(2.0) + (b - r) / delta) / F::new(6.0)
        } else {
            (F::new(4.0) + (r - g) / delta) / F::new(6.0)
        };
        if h < F::new(0.0) {
            h += F::new(1.0);
        }

        hsv[base] = h;
        hsv[base + 1usize] = s;
        hsv[base + 2usize] = c_max;
    }
}

/// Convert HSV (`H, S, V` in `[0, 1]`) to RGB `[0, 1]`.
#[cube(launch)]
pub(crate) fn hsv_to_rgb_kernel<F: lumen_core::element::ImageFloat>(
    hsv: &[F],
    rgb: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let h = hsv[base] * F::new(6.0);
        let s = hsv[base + 1usize];
        let v = hsv[base + 2usize];

        let hi = u32::cast_from(F::floor(h)) % 6u32;
        let f = h - F::floor(h);
        let p = v * (F::new(1.0) - s);
        let q = v * (F::new(1.0) - f * s);
        let t = v * (F::new(1.0) - (F::new(1.0) - f) * s);

        let mut r = v;
        let mut g = t;
        let mut b = p;
        if hi == 1u32 {
            r = q;
            g = v;
            b = p;
        } else if hi == 2u32 {
            r = p;
            g = v;
            b = t;
        } else if hi == 3u32 {
            r = p;
            g = q;
            b = v;
        } else if hi == 4u32 {
            r = t;
            g = p;
            b = v;
        } else if hi == 5u32 {
            r = v;
            g = p;
            b = q;
        }

        rgb[base] = r;
        rgb[base + 1usize] = g;
        rgb[base + 2usize] = b;
    }
}

#[cube]
fn srgb_to_linear<F: Float>(c: F) -> F {
    if c > F::new(0.04045) {
        F::powf((c + F::new(0.055)) / F::new(1.055), F::new(2.4))
    } else {
        c / F::new(12.92)
    }
}

#[cube]
fn linear_to_srgb<F: Float>(c: F) -> F {
    let v = F::max(c, F::new(0.0));
    if v > F::new(0.0031308) {
        F::new(1.055) * F::powf(v, F::new(1.0 / 2.4)) - F::new(0.055)
    } else {
        F::new(12.92) * v
    }
}

/// Convert sRGB `[0, 1]` to CIE 1931 XYZ (D65 illuminant, 2° observer).
#[cube(launch)]
pub(crate) fn rgb_to_xyz_kernel<F: lumen_core::element::ImageFloat>(
    rgb: &[F],
    mat: &[F],
    xyz: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let r = srgb_to_linear::<F>(rgb[base]);
        let g = srgb_to_linear::<F>(rgb[base + 1usize]);
        let b = srgb_to_linear::<F>(rgb[base + 2usize]);
        xyz[base] = mat[0usize] * r + mat[1usize] * g + mat[2usize] * b;
        xyz[base + 1usize] = mat[3usize] * r + mat[4usize] * g + mat[5usize] * b;
        xyz[base + 2usize] = mat[6usize] * r + mat[7usize] * g + mat[8usize] * b;
    }
}

/// Convert CIE 1931 XYZ (D65) to sRGB `[0, 1]`.
#[cube(launch)]
pub(crate) fn xyz_to_rgb_kernel<F: lumen_core::element::ImageFloat>(
    xyz: &[F],
    mat: &[F],
    rgb: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let x = xyz[base];
        let y = xyz[base + 1usize];
        let z = xyz[base + 2usize];
        let r_lin = mat[0usize] * x + mat[1usize] * y + mat[2usize] * z;
        let g_lin = mat[3usize] * x + mat[4usize] * y + mat[5usize] * z;
        let b_lin = mat[6usize] * x + mat[7usize] * y + mat[8usize] * z;
        rgb[base] = F::min(F::max(linear_to_srgb::<F>(r_lin), F::new(0.0)), F::new(1.0));
        rgb[base + 1usize] = F::min(F::max(linear_to_srgb::<F>(g_lin), F::new(0.0)), F::new(1.0));
        rgb[base + 2usize] = F::min(F::max(linear_to_srgb::<F>(b_lin), F::new(0.0)), F::new(1.0));
    }
}

#[cube]
fn lab_f<F: Float>(t: F) -> F {
    if t > F::new(0.008856451679035631) {
        F::powf(t, F::new(1.0 / 3.0))
    } else {
        F::new(7.787037037037037) * t + F::new(16.0 / 116.0)
    }
}

#[cube]
fn lab_f_inv<F: Float>(t: F) -> F {
    if t > F::new(0.20689655172413793) {
        t * t * t
    } else {
        (t - F::new(16.0 / 116.0)) / F::new(7.787037037037037)
    }
}

/// Convert CIE XYZ (D65: `Xn=0.95047, Yn=1.0, Zn=1.08883`) to CIE L*a*b*.
#[cube(launch)]
pub(crate) fn xyz_to_lab_kernel<F: lumen_core::element::ImageFloat>(
    xyz: &[F],
    lab: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let fx = lab_f::<F>(xyz[base] / F::new(0.95047));
        let fy = lab_f::<F>(xyz[base + 1usize]);
        let fz = lab_f::<F>(xyz[base + 2usize] / F::new(1.08883));
        lab[base] = F::new(116.0) * fy - F::new(16.0);
        lab[base + 1usize] = F::new(500.0) * (fx - fy);
        lab[base + 2usize] = F::new(200.0) * (fy - fz);
    }
}

/// Convert CIE L*a*b* to CIE XYZ (D65).
#[cube(launch)]
pub(crate) fn lab_to_xyz_kernel<F: lumen_core::element::ImageFloat>(
    lab: &[F],
    xyz: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let l = lab[base];
        let a = lab[base + 1usize];
        let b = lab[base + 2usize];
        let fy = (l + F::new(16.0)) / F::new(116.0);
        let fx = a / F::new(500.0) + fy;
        let fz = fy - b / F::new(200.0);
        xyz[base] = F::new(0.95047) * lab_f_inv::<F>(fx);
        xyz[base + 1usize] = lab_f_inv::<F>(fy);
        xyz[base + 2usize] = F::new(1.08883) * lab_f_inv::<F>(fz);
    }
}

/// Convert CIE XYZ (D65) to CIE L*u*v*.
#[cube(launch)]
pub(crate) fn xyz_to_luv_kernel<F: lumen_core::element::ImageFloat>(
    xyz: &[F],
    luv: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let x = xyz[base];
        let y = xyz[base + 1usize];
        let z = xyz[base + 2usize];

        let l = if y > F::new(0.008856451679035631) {
            F::new(116.0) * F::powf(y, F::new(1.0 / 3.0)) - F::new(16.0)
        } else {
            F::new(903.2962962962963) * y
        };

        let denom = x + F::new(15.0) * y + F::new(3.0) * z;
        let u_p = if denom > F::new(1e-12) {
            F::new(4.0) * x / denom
        } else {
            F::new(0.0)
        };
        let v_p = if denom > F::new(1e-12) {
            F::new(9.0) * y / denom
        } else {
            F::new(0.0)
        };

        let un = F::new(0.1978300066428368);
        let vn = F::new(0.468319994938791);
        luv[base] = l;
        luv[base + 1usize] = F::new(13.0) * l * (u_p - un);
        luv[base + 2usize] = F::new(13.0) * l * (v_p - vn);
    }
}

/// Convert CIE L*u*v* to CIE XYZ (D65).
#[cube(launch)]
pub(crate) fn luv_to_xyz_kernel<F: lumen_core::element::ImageFloat>(
    luv: &[F],
    xyz: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let l = luv[base];
        let u = luv[base + 1usize];
        let v = luv[base + 2usize];
        if l <= F::new(1e-8) {
            xyz[base] = F::new(0.0);
            xyz[base + 1usize] = F::new(0.0);
            xyz[base + 2usize] = F::new(0.0);
        } else {
            let un = F::new(0.1978300066428368);
            let vn = F::new(0.468319994938791);
            let up = u / (F::new(13.0) * l) + un;
            let vp = v / (F::new(13.0) * l) + vn;
            let y = if l > F::new(8.0) {
                let t = (l + F::new(16.0)) / F::new(116.0);
                t * t * t
            } else {
                l / F::new(903.2962962962963)
            };
            let x = y * F::new(9.0) * up / F::max(F::new(4.0) * vp, F::new(1e-12));
            let z = y * (F::new(12.0) - F::new(3.0) * up - F::new(20.0) * vp)
                / F::max(F::new(4.0) * vp, F::new(1e-12));
            xyz[base] = x;
            xyz[base + 1usize] = y;
            xyz[base + 2usize] = z;
        }
    }
}

/// Convert CIE L*a*b* to cylindrical CIE L*C*h° (`h` in radians `[0, 2π)`).
#[cube(launch)]
pub(crate) fn lab_to_lch_kernel<F: lumen_core::element::ImageFloat>(
    lab: &[F],
    lch: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let l = lab[base];
        let a = lab[base + 1usize];
        let b = lab[base + 2usize];
        let c = F::sqrt(a * a + b * b);
        let af = f32::cast_from(a);
        let bf = f32::cast_from(b);
        let mut h = f32::atan2(bf, af);
        if h < 0.0f32 {
            h += 6.283185307179586f32;
        }
        lch[base] = l;
        lch[base + 1usize] = c;
        lch[base + 2usize] = F::cast_from(h);
    }
}

/// Convert cylindrical CIE L*C*h° (`h` in radians) to CIE L*a*b*.
#[cube(launch)]
pub(crate) fn lch_to_lab_kernel<F: lumen_core::element::ImageFloat>(
    lch: &[F],
    lab: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let l = lch[base];
        let c = lch[base + 1usize];
        let h = lch[base + 2usize];
        lab[base] = l;
        lab[base + 1usize] = c * F::cos(h);
        lab[base + 2usize] = c * F::sin(h);
    }
}

/// Convert sRGB `[0, 1]` to ITU-R BT.601 YCbCr (`Y in [16, 235]`, `Cb, Cr in [16, 240]`).
#[cube(launch)]
pub(crate) fn rgb_to_ycbcr_kernel<F: lumen_core::element::ImageFloat>(
    rgb: &[F],
    ycbcr: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let r = rgb[base];
        let g = rgb[base + 1usize];
        let b = rgb[base + 2usize];
        ycbcr[base] = F::new(16.0) + F::new(65.481) * r + F::new(128.553) * g + F::new(24.966) * b;
        ycbcr[base + 1usize] =
            F::new(128.0) - F::new(37.797) * r - F::new(74.203) * g + F::new(112.0) * b;
        ycbcr[base + 2usize] =
            F::new(128.0) + F::new(112.0) * r - F::new(93.786) * g - F::new(18.214) * b;
    }
}

/// Convert ITU-R BT.601 YCbCr back to sRGB `[0, 1]`.
#[cube(launch)]
pub(crate) fn ycbcr_to_rgb_kernel<F: lumen_core::element::ImageFloat>(
    ycbcr: &[F],
    rgb: &mut [F],
    num_pixels: u32,
) {
    let idx = element_position() as u32;
    if idx < num_pixels {
        let base = (idx * 3u32) as usize;
        let y = (ycbcr[base] - F::new(16.0)) / F::new(219.0);
        let cb = (ycbcr[base + 1usize] - F::new(128.0)) / F::new(224.0);
        let cr = (ycbcr[base + 2usize] - F::new(128.0)) / F::new(224.0);
        rgb[base] = y + F::new(1.402) * cr;
        rgb[base + 1usize] = y - F::new(0.344136) * cb - F::new(0.714136) * cr;
        rgb[base + 2usize] = y + F::new(1.772) * cb;
    }
}

//! Color space transformations across sRGB, Grayscale, HSV, CIE XYZ, CIE L*a*b*, CIE L*u*v*,
//! CIE L*C*h°, YCbCr, YUV, and RGBA on the device.

use cubecl::prelude::*;
use lucem_core::{
    compute::LaunchGeometry,
    element::{cast, cast_all, ImageFloat},
    memory::{DeviceArray, DeviceCache},
    Shape,
};

use crate::{
    core::check,
    error::{OpsError, OpsResult},
};

use super::kernels::space::{
    apply_color_matrix_kernel, gray_to_rgb_kernel, hsv_to_rgb_kernel, lab_to_lch_kernel,
    lab_to_xyz_kernel, lch_to_lab_kernel, luv_to_xyz_kernel, rgb_to_gray_kernel,
    rgb_to_hsv_kernel, rgb_to_xyz_kernel, rgb_to_ycbcr_kernel, rgba_to_rgb_kernel,
    xyz_to_lab_kernel, xyz_to_luv_kernel, xyz_to_rgb_kernel, ycbcr_to_rgb_kernel,
};

static COLOR_MATRIX_CACHE: DeviceCache = DeviceCache::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixKind {
    RgbToXyzD65,
    XyzD65ToRgb,
    RgbToYuv,
    YuvToRgb,
}

fn get_matrix<F: ImageFloat>(client: &Client, kind: MatrixKind) -> DeviceArray<F> {
    let (key, vals): (&'static str, [f64; 9]) = match kind {
        MatrixKind::RgbToXyzD65 => (
            "rgb_to_xyz_d65",
            [
                0.4124564, 0.3575761, 0.1804375,
                0.2126729, 0.7151522, 0.0721750,
                0.0193339, 0.1191920, 0.9503041,
            ],
        ),
        MatrixKind::XyzD65ToRgb => (
            "xyz_d65_to_rgb",
            [
                3.2404542, -1.5371385, -0.4985314,
                -0.9692660, 1.8760108, 0.0415560,
                0.0556434, -0.2040259, 1.0572252,
            ],
        ),
        MatrixKind::RgbToYuv => (
            "rgb_to_yuv",
            [
                0.299, 0.587, 0.114,
                -0.14714119, -0.28886916, 0.43601035,
                0.61497538, -0.51496512, -0.10001026,
            ],
        ),
        MatrixKind::YuvToRgb => (
            "yuv_to_rgb",
            [
                1.0, 0.0, 1.13988303,
                1.0, -0.39464233, -0.58062185,
                1.0, 2.03206185, 0.0,
            ],
        ),
    };
    COLOR_MATRIX_CACHE.get_or_upload::<F>(client, key, Shape::new([3, 3]), || cast_all::<F>(&vals))
}

pub(crate) fn check_rgb_shape(shape: &Shape, op: &'static str) -> OpsResult<(usize, usize, usize)> {
    let dims = shape.dims();
    if dims.len() != 3 || dims[2] != 3 {
        return Err(OpsError::InvalidDimension {
            what: op,
            expected: "3 ([H, W, 3])",
            actual: dims.len(),
            shape: dims.to_vec(),
        });
    }
    if shape.is_empty() {
        return Err(OpsError::EmptyInput(op));
    }
    Ok((dims[0], dims[1], dims[0] * dims[1]))
}

/// Convert an sRGB image `[H, W, 3]` to grayscale luminance `[H, W]`.
pub fn rgb_to_gray<F: ImageFloat>(client: &Client, rgb: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (h, w, num_pixels) = check_rgb_shape(rgb.shape(), "rgb_to_gray")?;
    let out = DeviceArray::<F>::empty(client, Shape::new([h, w]))?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        rgb_to_gray_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            rgb.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Replicate a grayscale image `[H, W]` into 3-channel RGB `[H, W, 3]`.
pub fn gray_to_rgb<F: ImageFloat>(client: &Client, gray: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("gray_to_rgb", gray.shape(), &[2], "2")?;
    if gray.is_empty() {
        return Err(OpsError::EmptyInput("gray_to_rgb"));
    }
    let (h, w) = (gray.dims()[0], gray.dims()[1]);
    let num_pixels = h * w;
    let out = DeviceArray::<F>::empty(client, Shape::new([h, w, 3]))?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        gray_to_rgb_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            gray.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Composite an RGBA image `[H, W, 4]` onto a background RGB color `background` (default white `[1, 1, 1]`).
pub fn rgba_to_rgb<F: ImageFloat>(
    client: &Client,
    rgba: &DeviceArray<F>,
    background: [f64; 3],
) -> OpsResult<DeviceArray<F>> {
    let dims = rgba.dims();
    if dims.len() != 3 || dims[2] != 4 {
        return Err(OpsError::InvalidDimension {
            what: "rgba_to_rgb",
            expected: "3 ([H, W, 4])",
            actual: dims.len(),
            shape: dims.to_vec(),
        });
    }
    if rgba.is_empty() {
        return Err(OpsError::EmptyInput("rgba_to_rgb"));
    }
    let (h, w) = (dims[0], dims[1]);
    let num_pixels = h * w;
    let out = DeviceArray::<F>::empty(client, Shape::new([h, w, 3]))?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        rgba_to_rgb_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            rgba.binding(),
            out.binding(),
            cast::<F>(background[0]),
            cast::<F>(background[1]),
            cast::<F>(background[2]),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert RGB `[H, W, 3]` (`[0, 1]`) to HSV `[H, W, 3]` (`[0, 1]`).
pub fn rgb_to_hsv<F: ImageFloat>(client: &Client, rgb: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(rgb.shape(), "rgb_to_hsv")?;
    let out = DeviceArray::<F>::empty(client, rgb.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        rgb_to_hsv_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            rgb.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert HSV `[H, W, 3]` (`[0, 1]`) to RGB `[H, W, 3]` (`[0, 1]`).
pub fn hsv_to_rgb<F: ImageFloat>(client: &Client, hsv: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(hsv.shape(), "hsv_to_rgb")?;
    let out = DeviceArray::<F>::empty(client, hsv.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        hsv_to_rgb_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            hsv.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert sRGB `[H, W, 3]` to CIE 1931 XYZ (D65 illuminant) using a device-cached `3x3` matrix.
pub fn rgb_to_xyz<F: ImageFloat>(client: &Client, rgb: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(rgb.shape(), "rgb_to_xyz")?;
    let mat = get_matrix::<F>(client, MatrixKind::RgbToXyzD65);
    let out = DeviceArray::<F>::empty(client, rgb.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        rgb_to_xyz_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            rgb.binding(),
            mat.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert CIE 1931 XYZ (D65) `[H, W, 3]` to sRGB `[H, W, 3]`.
pub fn xyz_to_rgb<F: ImageFloat>(client: &Client, xyz: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(xyz.shape(), "xyz_to_rgb")?;
    let mat = get_matrix::<F>(client, MatrixKind::XyzD65ToRgb);
    let out = DeviceArray::<F>::empty(client, xyz.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        xyz_to_rgb_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            xyz.binding(),
            mat.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert CIE XYZ (D65) `[H, W, 3]` to CIE L*a*b* `[H, W, 3]`.
pub fn xyz_to_lab<F: ImageFloat>(client: &Client, xyz: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(xyz.shape(), "xyz_to_lab")?;
    let out = DeviceArray::<F>::empty(client, xyz.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        xyz_to_lab_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            xyz.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert CIE L*a*b* `[H, W, 3]` to CIE XYZ (D65) `[H, W, 3]`.
pub fn lab_to_xyz<F: ImageFloat>(client: &Client, lab: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(lab.shape(), "lab_to_xyz")?;
    let out = DeviceArray::<F>::empty(client, lab.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        lab_to_xyz_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            lab.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert sRGB `[H, W, 3]` to CIE L*a*b* `[H, W, 3]`.
pub fn rgb_to_lab<F: ImageFloat>(client: &Client, rgb: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let xyz = rgb_to_xyz(client, rgb)?;
    xyz_to_lab(client, &xyz)
}

/// Convert CIE L*a*b* `[H, W, 3]` to sRGB `[H, W, 3]`.
pub fn lab_to_rgb<F: ImageFloat>(client: &Client, lab: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let xyz = lab_to_xyz(client, lab)?;
    xyz_to_rgb(client, &xyz)
}

/// Convert sRGB `[H, W, 3]` to CIE L*u*v* `[H, W, 3]`.
pub fn rgb_to_luv<F: ImageFloat>(client: &Client, rgb: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let xyz = rgb_to_xyz(client, rgb)?;
    let (_, _, num_pixels) = check_rgb_shape(xyz.shape(), "rgb_to_luv")?;
    let out = DeviceArray::<F>::empty(client, xyz.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        xyz_to_luv_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            xyz.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert CIE L*u*v* `[H, W, 3]` to sRGB `[H, W, 3]`.
pub fn luv_to_rgb<F: ImageFloat>(client: &Client, luv: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(luv.shape(), "luv_to_rgb")?;
    let xyz = DeviceArray::<F>::empty(client, luv.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        luv_to_xyz_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            luv.binding(),
            xyz.binding(),
            num_pixels as u32,
        );
    }
    xyz_to_rgb(client, &xyz)
}

/// Convert CIE L*a*b* `[H, W, 3]` to cylindrical CIE L*C*h° `[H, W, 3]`.
pub fn lab_to_lch<F: ImageFloat>(client: &Client, lab: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(lab.shape(), "lab_to_lch")?;
    let out = DeviceArray::<F>::empty(client, lab.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        lab_to_lch_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            lab.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert cylindrical CIE L*C*h° `[H, W, 3]` to CIE L*a*b* `[H, W, 3]`.
pub fn lch_to_lab<F: ImageFloat>(client: &Client, lch: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(lch.shape(), "lch_to_lab")?;
    let out = DeviceArray::<F>::empty(client, lch.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        lch_to_lab_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            lch.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert sRGB `[H, W, 3]` to ITU-R BT.601 YCbCr `[H, W, 3]`.
pub fn rgb_to_ycbcr<F: ImageFloat>(client: &Client, rgb: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(rgb.shape(), "rgb_to_ycbcr")?;
    let out = DeviceArray::<F>::empty(client, rgb.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        rgb_to_ycbcr_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            rgb.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert ITU-R BT.601 YCbCr `[H, W, 3]` to sRGB `[H, W, 3]`.
pub fn ycbcr_to_rgb<F: ImageFloat>(
    client: &Client,
    ycbcr: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(ycbcr.shape(), "ycbcr_to_rgb")?;
    let out = DeviceArray::<F>::empty(client, ycbcr.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        ycbcr_to_rgb_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ycbcr.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert sRGB `[H, W, 3]` to PAL/SECAM YUV `[H, W, 3]` via a device-cached `3x3` matrix.
pub fn rgb_to_yuv<F: ImageFloat>(client: &Client, rgb: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(rgb.shape(), "rgb_to_yuv")?;
    let mat = get_matrix::<F>(client, MatrixKind::RgbToYuv);
    let out = DeviceArray::<F>::empty(client, rgb.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        apply_color_matrix_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            rgb.binding(),
            mat.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Convert YUV `[H, W, 3]` to sRGB `[H, W, 3]` via a device-cached `3x3` matrix.
pub fn yuv_to_rgb<F: ImageFloat>(client: &Client, yuv: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(yuv.shape(), "yuv_to_rgb")?;
    let mat = get_matrix::<F>(client, MatrixKind::YuvToRgb);
    let out = DeviceArray::<F>::empty(client, yuv.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        apply_color_matrix_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            yuv.binding(),
            mat.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

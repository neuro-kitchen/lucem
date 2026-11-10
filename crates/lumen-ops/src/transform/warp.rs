//! Geometric coordinate sampling, affine/projective warps, shift, zoom, rotate, resize, rescale,
//! polar/log-polar warp, and swirl warp built on [`crate::core::spline`] and [`crate::filter::gaussian`].

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, cast_all, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::kernels::{affine_coords_2d_kernel, polar_coords_2d_kernel, swirl_coords_2d_kernel};
use crate::core::spline::{sample_2d_into, spline_poles};
use crate::core::{check, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::filter::gaussian::{gaussian_blur_into, GaussianBlur};

/// Shared interpolation and boundary settings for geometric warps.
#[derive(Debug, Clone, PartialEq)]
pub struct WarpSettings {
    /// B-spline interpolation order (`0..=5`, where `0` is nearest-neighbor, `1` bilinear, `3` bicubic).
    pub order: usize,
    pub mode: PadMode,
    pub cval: f64,
    pub prefilter: bool,
}

impl Default for WarpSettings {
    fn default() -> Self {
        Self { order: 1, mode: PadMode::Constant, cval: 0.0, prefilter: true }
    }
}

/// Samples a 2-D image at device coordinate arrays `(coords_y, coords_x)`.
pub fn sample_coords<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    coords_y: &DeviceArray<F>,
    coords_x: &DeviceArray<F>,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, coords_y.shape().clone())?;
    let mut ws = Workspace::new(client);
    sample_2d_into(
        client,
        image,
        coords_y,
        coords_x,
        &output,
        settings.order,
        settings.mode,
        settings.cval,
        settings.prefilter,
        &mut ws,
    )?;
    Ok(output)
}

/// Warps a 2-D image into `output_shape` (`[out_h, out_w]`, defaulting to `image.shape()`) using a
/// $3 \times 3$ homogeneous inverse transform `inverse_matrix` mapping output `[y_out, x_out, 1]` to
/// source `[y_in, x_in, w_in]`.
pub fn affine_warp<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    inverse_matrix: &[f64; 9],
    output_shape: Option<[usize; 2]>,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("affine_warp", image.shape(), &[2], "2")?;
    let _ = spline_poles(settings.order)?;
    let [out_h, out_w] = output_shape.unwrap_or([image.dims()[0], image.dims()[1]]);
    let out_shape = Shape::new([out_h, out_w]);
    let output = DeviceArray::<F>::empty(client, out_shape.clone())?;
    if output.is_empty() || image.is_empty() {
        return Ok(output);
    }

    let m_dev = DeviceArray::from_host(client, Shape::vector(9), cast_all::<F>(inverse_matrix))?;
    let mut ws = Workspace::new(client);
    let coords_y = ws.array::<F>("warp_cy", &out_shape)?;
    let coords_x = ws.array::<F>("warp_cx", &out_shape)?;
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        affine_coords_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            m_dev.binding(),
            coords_y.binding(),
            coords_x.binding(),
            out_w as u32,
            output.len() as u32,
        );
    }
    sample_2d_into(
        client,
        image,
        &coords_y,
        &coords_x,
        &output,
        settings.order,
        settings.mode,
        settings.cval,
        settings.prefilter,
        &mut ws,
    )?;
    Ok(output)
}

/// Translates a 2-D image by `(shift_y, shift_x)` pixels (`output[y, x] = input[y - shift_y, x - shift_x]`).
pub fn shift<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    shift_yx: [f64; 2],
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>> {
    let inv = [
        1.0, 0.0, -shift_yx[0],
        0.0, 1.0, -shift_yx[1],
        0.0, 0.0, 1.0,
    ];
    affine_warp(client, image, &inv, None, settings)
}

/// Zooms a 2-D image by scale factors `[zoom_y, zoom_x]` (`> 0`).
pub fn zoom<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    factors: [f64; 2],
    grid_mode: bool,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("zoom", image.shape(), &[2], "2")?;
    if !factors[0].is_finite() || factors[0] <= 0.0 || !factors[1].is_finite() || factors[1] <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "factors",
            reason: format!("zoom factors must be positive and finite, got {factors:?}"),
        });
    }
    let (in_h, in_w) = (image.dims()[0], image.dims()[1]);
    let out_h = ((in_h as f64) * factors[0]).round().max(1.0) as usize;
    let out_w = ((in_w as f64) * factors[1]).round().max(1.0) as usize;

    let (sy, ty) = if grid_mode {
        let s = (in_h as f64) / (out_h as f64);
        (s, 0.5 * (s - 1.0))
    } else if out_h > 1 && in_h > 1 {
        (((in_h - 1) as f64) / ((out_h - 1) as f64), 0.0)
    } else {
        (1.0 / factors[0], 0.0)
    };
    let (sx, tx) = if grid_mode {
        let s = (in_w as f64) / (out_w as f64);
        (s, 0.5 * (s - 1.0))
    } else if out_w > 1 && in_w > 1 {
        (((in_w - 1) as f64) / ((out_w - 1) as f64), 0.0)
    } else {
        (1.0 / factors[1], 0.0)
    };

    let inv = [
        sy, 0.0, ty,
        0.0, sx, tx,
        0.0, 0.0, 1.0,
    ];
    affine_warp(client, image, &inv, Some([out_h, out_w]), settings)
}

/// Rotates a 2-D image counter-clockwise by `angle_deg` degrees around `center` (defaults to image center),
/// optionally expanding the output canvas when `resize_output = true`.
pub fn rotate<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    angle_deg: f64,
    resize_output: bool,
    center: Option<[f64; 2]>,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("rotate", image.shape(), &[2], "2")?;
    let (in_h, in_w) = (image.dims()[0], image.dims()[1]);
    let [cy, cx] = center.unwrap_or([(in_h as f64 - 1.0) * 0.5, (in_w as f64 - 1.0) * 0.5]);
    let theta = angle_deg.to_radians();
    let c = theta.cos();
    let s = theta.sin();

    let (out_h, out_w, out_cy, out_cx) = if resize_output {
        let nh = ((in_h as f64) * c.abs() + (in_w as f64) * s.abs()).ceil().max(1.0) as usize;
        let nw = ((in_h as f64) * s.abs() + (in_w as f64) * c.abs()).ceil().max(1.0) as usize;
        (nh, nw, (nh as f64 - 1.0) * 0.5, (nw as f64 - 1.0) * 0.5)
    } else {
        (in_h, in_w, cy, cx)
    };

    // Inverse mapping from output (oy, ox) to source (iy, ix):
    // iy = cy + c*(oy - out_cy) + s*(ox - out_cx)
    // ix = cx - s*(oy - out_cy) + c*(ox - out_cx)
    let ty = cy - c * out_cy - s * out_cx;
    let tx = cx + s * out_cy - c * out_cx;
    let inv = [
        c, s, ty,
        -s, c, tx,
        0.0, 0.0, 1.0,
    ];
    affine_warp(client, image, &inv, Some([out_h, out_w]), settings)
}

/// Resizes a 2-D image to `output_shape` (`[out_h, out_w]`), applying Gaussian anti-aliasing when
/// downsampling and `anti_aliasing = true`.
pub fn resize<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    output_shape: [usize; 2],
    anti_aliasing: bool,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("resize", image.shape(), &[2], "2")?;
    let [out_h, out_w] = output_shape;
    if out_h == 0 || out_w == 0 {
        return Err(OpsError::InvalidParameter {
            name: "output_shape",
            reason: "output_shape dimensions must be >= 1".to_string(),
        });
    }
    let (in_h, in_w) = (image.dims()[0], image.dims()[1]);
    let sy = (in_h as f64) / (out_h as f64);
    let sx = (in_w as f64) / (out_w as f64);

    let smoothed_buf;
    let src = if anti_aliasing && (sy > 1.0 || sx > 1.0) {
        let sig_y = ((sy - 1.0) * 0.5).max(0.0);
        let sig_x = ((sx - 1.0) * 0.5).max(0.0);
        smoothed_buf = DeviceArray::<F>::empty(client, image.shape().clone())?;
        let mut ws = Workspace::new(client);
        gaussian_blur_into(
            client,
            image,
            &[sig_y, sig_x],
            &smoothed_buf,
            &GaussianBlur { mode: settings.mode, cval: settings.cval, ..GaussianBlur::default() },
            &mut ws,
        )?;
        &smoothed_buf
    } else {
        image
    };

    let inv = [
        sy, 0.0, 0.5 * (sy - 1.0),
        0.0, sx, 0.5 * (sx - 1.0),
        0.0, 0.0, 1.0,
    ];
    affine_warp(client, src, &inv, Some([out_h, out_w]), settings)
}

/// Rescales a 2-D image by `scales` (`[scale_y, scale_x]`) with optional Gaussian anti-aliasing.
pub fn rescale<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    scales: [f64; 2],
    anti_aliasing: bool,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("rescale", image.shape(), &[2], "2")?;
    if !scales[0].is_finite() || scales[0] <= 0.0 || !scales[1].is_finite() || scales[1] <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "scales",
            reason: format!("scales must be positive and finite, got {scales:?}"),
        });
    }
    let out_h = ((image.dims()[0] as f64) * scales[0]).round().max(1.0) as usize;
    let out_w = ((image.dims()[1] as f64) * scales[1]).round().max(1.0) as usize;
    resize(client, image, [out_h, out_w], anti_aliasing, settings)
}

/// Warps a 2-D image into polar (`log_polar = false`) or log-polar (`log_polar = true`) coordinates
/// of `output_shape` (`[n_angles, n_radii]`, defaulting to `[360, ceil(hypot(h/2, w/2))]`).
pub fn polar_warp<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    center: Option<[f64; 2]>,
    max_radius: Option<f64>,
    log_polar: bool,
    output_shape: Option<[usize; 2]>,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("polar_warp", image.shape(), &[2], "2")?;
    let (h, w) = (image.dims()[0], image.dims()[1]);
    let [cy, cx] = center.unwrap_or([(h as f64 - 1.0) * 0.5, (w as f64 - 1.0) * 0.5]);
    let radius = max_radius.unwrap_or_else(|| (cy.hypot(cx)).max(1.0));
    if !radius.is_finite() || radius <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "max_radius",
            reason: format!("max_radius must be positive and finite, got {radius}"),
        });
    }
    let [n_angles, n_radii] = output_shape.unwrap_or([360, radius.ceil().max(1.0) as usize]);
    if n_angles == 0 || n_radii == 0 {
        return Err(OpsError::InvalidParameter {
            name: "output_shape",
            reason: "polar_warp output dimensions must be >= 1".to_string(),
        });
    }

    let out_shape = Shape::new([n_angles, n_radii]);
    let output = DeviceArray::<F>::empty(client, out_shape.clone())?;
    let mut ws = Workspace::new(client);
    let coords_y = ws.array::<F>("polar_cy", &out_shape)?;
    let coords_x = ws.array::<F>("polar_cx", &out_shape)?;
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        polar_coords_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            coords_y.binding(),
            coords_x.binding(),
            cast::<F>(cy),
            cast::<F>(cx),
            cast::<F>(radius),
            n_angles as u32,
            n_radii as u32,
            output.len() as u32,
            log_polar,
        );
    }
    sample_2d_into(
        client,
        image,
        &coords_y,
        &coords_x,
        &output,
        settings.order,
        settings.mode,
        settings.cval,
        settings.prefilter,
        &mut ws,
    )?;
    Ok(output)
}

/// Applies a non-linear 2-D swirl warp around `center` with `strength` and radial extent `radius` (`> 0`).
pub fn swirl<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    center: Option<[f64; 2]>,
    strength: f64,
    radius: f64,
    rotation: f64,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("swirl", image.shape(), &[2], "2")?;
    if !radius.is_finite() || radius <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "radius",
            reason: format!("swirl radius must be positive and finite, got {radius}"),
        });
    }
    let (h, w) = (image.dims()[0], image.dims()[1]);
    let [cy, cx] = center.unwrap_or([(h as f64 - 1.0) * 0.5, (w as f64 - 1.0) * 0.5]);
    let out_shape = image.shape().clone();
    let output = DeviceArray::<F>::empty(client, out_shape.clone())?;
    if output.is_empty() {
        return Ok(output);
    }

    // Decay constant r_0 = radius / (5 * ln(2)) so swirl decays to 1/32 at `radius`
    let radius_decay = radius / (5.0 * std::f64::consts::LN_2);
    let mut ws = Workspace::new(client);
    let coords_y = ws.array::<F>("swirl_cy", &out_shape)?;
    let coords_x = ws.array::<F>("swirl_cx", &out_shape)?;
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        swirl_coords_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            coords_y.binding(),
            coords_x.binding(),
            cast::<F>(cy),
            cast::<F>(cx),
            cast::<F>(strength),
            cast::<F>(radius_decay),
            cast::<F>(rotation),
            w as u32,
            output.len() as u32,
        );
    }
    sample_2d_into(
        client,
        image,
        &coords_y,
        &coords_x,
        &output,
        settings.order,
        settings.mode,
        settings.cval,
        settings.prefilter,
        &mut ws,
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};

    fn shift_rotate_and_resize_preserve_known_structures(client: &Client) {
        let mut data = vec![0.0f32; 25];
        data[2 * 5 + 2] = 10.0;
        let img = DeviceArray::from_host(client, Shape::new([5, 5]), data).unwrap();
        let cfg = WarpSettings { order: 1, mode: PadMode::Constant, cval: 0.0, prefilter: false };

        // Integer shift by (+1, +2) moves impulse from (2, 2) to (3, 4)
        let shifted = shift(client, &img, [1.0, 2.0], &cfg).unwrap().to_host(client);
        assert_close(&[shifted[3 * 5 + 4] as f64], &[10.0], 1e-5, client.name());
        assert_close(&[shifted[2 * 5 + 2] as f64], &[0.0], 1e-5, client.name());

        // 180-degree rotation around center (2, 2) maps (1, 2) to (3, 2)
        let mut asym = vec![0.0f32; 25];
        asym[5 + 2] = 8.0;
        let asym_img = DeviceArray::from_host(client, Shape::new([5, 5]), asym).unwrap();
        let rot = rotate(client, &asym_img, 180.0, false, None, &cfg).unwrap().to_host(client);
        assert_close(&[rot[3 * 5 + 2] as f64], &[8.0], 1e-4, client.name());

        // Resize constant image preserves constant value
        let ones = DeviceArray::<f32>::filled(client, Shape::new([8, 10]), 4.5).unwrap();
        let res = resize(
            client,
            &ones,
            [4, 5],
            true,
            &WarpSettings { order: 1, mode: PadMode::Nearest, cval: 0.0, prefilter: false },
        )
        .unwrap()
        .to_host(client);
        assert_close(&res, &[4.5; 20], 1e-4, client.name());
    }
    runtime_test!(test_geometric_warps, shift_rotate_and_resize_preserve_known_structures);
}

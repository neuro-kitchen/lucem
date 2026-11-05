//! Multi-resolution pyramids and block/area downsampling: `downsample_mean`, `resize_mean`,
//! `pyramid_reduce`, `pyramid_expand`, `gaussian_pyramid`, and `laplacian_pyramid`.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::kernels::{downsample_mean_2d_kernel, resize_mean_2d_kernel};
use super::warp::{resize, WarpSettings};
use crate::core::{check, elementwise, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::filter::gaussian::{gaussian_blur_into, GaussianBlur};

/// Downsamples a 2-D image by integer block factors `[factor_y, factor_x]` by averaging each tile,
/// padding partial border tiles with `mode` and `cval`.
pub fn downsample_mean<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    factors: [usize; 2],
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("downsample_mean", image.shape(), &[2], "2")?;
    let [fy, fx] = factors;
    if fy == 0 || fx == 0 {
        return Err(OpsError::InvalidParameter {
            name: "factors",
            reason: "downsample_mean factors must be >= 1".to_string(),
        });
    }
    let (in_h, in_w) = (image.dims()[0], image.dims()[1]);
    let out_h = in_h.div_ceil(fy);
    let out_w = in_w.div_ceil(fx);
    let output = DeviceArray::<F>::empty(client, Shape::new([out_h, out_w]))?;
    if output.is_empty() || image.is_empty() {
        return Ok(output);
    }

    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        downsample_mean_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            image.binding(),
            output.binding(),
            in_h as i32,
            in_w as i32,
            out_w as u32,
            fy as i32,
            fx as i32,
            cast::<F>(cval),
            output.len() as u32,
            mode.id(),
        );
    }
    Ok(output)
}

/// Resizes a 2-D image to `output_shape` (`[out_h, out_w]`) by exact local box-overlap area averaging.
pub fn resize_mean<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    output_shape: [usize; 2],
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("resize_mean", image.shape(), &[2], "2")?;
    let [out_h, out_w] = output_shape;
    if out_h == 0 || out_w == 0 {
        return Err(OpsError::InvalidParameter {
            name: "output_shape",
            reason: "resize_mean output dimensions must be >= 1".to_string(),
        });
    }
    let output = DeviceArray::<F>::empty(client, Shape::new([out_h, out_w]))?;
    if image.is_empty() {
        return Ok(output);
    }
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        resize_mean_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            image.binding(),
            output.binding(),
            image.dims()[0] as i32,
            image.dims()[1] as i32,
            out_h as u32,
            out_w as u32,
            output.len() as u32,
        );
    }
    Ok(output)
}

/// Settings for [`pyramid_reduce`], [`pyramid_expand`], [`gaussian_pyramid`], and [`laplacian_pyramid`].
#[derive(Debug, Clone, PartialEq)]
pub struct Pyramid {
    pub downscale: f64,
    /// Optional Gaussian smoothing $\sigma$; defaults to `2 * downscale / 6.0`.
    pub sigma: Option<f64>,
    pub order: usize,
    pub mode: PadMode,
    pub cval: f64,
}

impl Default for Pyramid {
    fn default() -> Self {
        Self { downscale: 2.0, sigma: None, order: 1, mode: PadMode::Reflect, cval: 0.0 }
    }
}

impl Pyramid {
    fn resolved_sigma(&self) -> OpsResult<f64> {
        if !self.downscale.is_finite() || self.downscale <= 1.0 {
            return Err(OpsError::InvalidParameter {
                name: "downscale",
                reason: format!("pyramid scale factor must be > 1.0, got {}", self.downscale),
            });
        }
        let s = self.sigma.unwrap_or(2.0 * self.downscale / 6.0);
        if !s.is_finite() || s < 0.0 {
            return Err(OpsError::InvalidParameter {
                name: "sigma",
                reason: format!("sigma must be non-negative and finite, got {s}"),
            });
        }
        Ok(s)
    }
}

/// Smooths a 2-D image with a Gaussian filter and downsamples it by `settings.downscale`.
pub fn pyramid_reduce<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &Pyramid,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("pyramid_reduce", image.shape(), &[2], "2")?;
    let sigma = settings.resolved_sigma()?;
    let out_h = ((image.dims()[0] as f64) / settings.downscale).ceil().max(1.0) as usize;
    let out_w = ((image.dims()[1] as f64) / settings.downscale).ceil().max(1.0) as usize;

    let smoothed = DeviceArray::<F>::empty(client, image.shape().clone())?;
    let mut ws = Workspace::new(client);
    gaussian_blur_into(
        client,
        image,
        &[sigma, sigma],
        &smoothed,
        &GaussianBlur { mode: settings.mode, cval: settings.cval, ..GaussianBlur::default() },
        &mut ws,
    )?;
    resize(
        client,
        &smoothed,
        [out_h, out_w],
        false,
        &WarpSettings {
            order: settings.order,
            mode: settings.mode,
            cval: settings.cval,
            prefilter: settings.order >= 2,
        },
    )
}

/// Upsamples a 2-D image by `settings.downscale` (or into `output_shape` when specified) and smooths
/// with a Gaussian filter.
pub fn pyramid_expand<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    output_shape: Option<[usize; 2]>,
    settings: &Pyramid,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("pyramid_expand", image.shape(), &[2], "2")?;
    let sigma = settings.resolved_sigma()?;
    let [out_h, out_w] = output_shape.unwrap_or_else(|| [
        ((image.dims()[0] as f64) * settings.downscale).ceil().max(1.0) as usize,
        ((image.dims()[1] as f64) * settings.downscale).ceil().max(1.0) as usize,
    ]);

    let upsampled = resize(
        client,
        image,
        [out_h, out_w],
        false,
        &WarpSettings {
            order: settings.order,
            mode: settings.mode,
            cval: settings.cval,
            prefilter: settings.order >= 2,
        },
    )?;
    let output = DeviceArray::<F>::empty(client, upsampled.shape().clone())?;
    let mut ws = Workspace::new(client);
    gaussian_blur_into(
        client,
        &upsampled,
        &[sigma, sigma],
        &output,
        &GaussianBlur { mode: settings.mode, cval: settings.cval, ..GaussianBlur::default() },
        &mut ws,
    )?;
    Ok(output)
}

/// Builds a multi-scale Gaussian pyramid `[level_0, level_1, ...]` where `level_0` is a copy of `image`
/// and at most `max_layer` reductions are performed (stopping early when any dimension reaches `1`).
pub fn gaussian_pyramid<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    max_layer: usize,
    settings: &Pyramid,
) -> OpsResult<Vec<DeviceArray<F>>> {
    check::ndim_in("gaussian_pyramid", image.shape(), &[2], "2")?;
    let _ = settings.resolved_sigma()?;
    let mut pyramid = Vec::with_capacity(max_layer + 1);
    pyramid.push(elementwise::copy(client, image)?);

    for _ in 0..max_layer {
        let cur = pyramid.last().unwrap();
        if cur.dims()[0] <= 1 && cur.dims()[1] <= 1 {
            break;
        }
        let next = pyramid_reduce(client, cur, settings)?;
        let done = next.dims() == cur.dims();
        pyramid.push(next);
        if done {
            break;
        }
    }
    Ok(pyramid)
}

/// Builds a multi-scale Laplacian pyramid `[L_0, L_1, ..., G_N]` where each bandpass layer
/// $L_k = G_k - \operatorname{expand}(G_{k+1})$ and the final layer is the coarsest Gaussian residual.
pub fn laplacian_pyramid<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    max_layer: usize,
    settings: &Pyramid,
) -> OpsResult<Vec<DeviceArray<F>>> {
    let g_pyr = gaussian_pyramid(client, image, max_layer, settings)?;
    if g_pyr.len() <= 1 {
        return Ok(g_pyr);
    }
    let mut l_pyr = Vec::with_capacity(g_pyr.len());
    for i in 0..(g_pyr.len() - 1) {
        let fine = &g_pyr[i];
        let coarse = &g_pyr[i + 1];
        let expanded = pyramid_expand(
            client,
            coarse,
            Some([fine.dims()[0], fine.dims()[1]]),
            settings,
        )?;
        let diff = DeviceArray::<F>::empty(client, fine.shape().clone())?;
        elementwise::linear_combination_into(client, fine, &expanded, &diff, 1.0, -1.0, 0.0)?;
        l_pyr.push(diff);
    }
    l_pyr.push(g_pyr.into_iter().last().unwrap());
    Ok(l_pyr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};

    fn downsample_mean_and_pyramids_match_expectations(client: &Client) {
        let host: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let img = DeviceArray::from_host(client, Shape::new([4, 4]), host).unwrap();
        let ds = downsample_mean(client, &img, [2, 2], PadMode::Nearest, 0.0).unwrap().to_host(client);
        // 2x2 tile averages: top-left (1+2+5+6)/4 = 3.5, top-right (3+4+7+8)/4 = 5.5,
        // bottom-left (9+10+13+14)/4 = 11.5, bottom-right (11+12+15+16)/4 = 13.5
        assert_close(&ds, &[3.5, 5.5, 11.5, 13.5], 1e-6, client.name());

        let rm = resize_mean(client, &img, [2, 2]).unwrap().to_host(client);
        assert_close(&rm, &[3.5, 5.5, 11.5, 13.5], 1e-6, client.name());

        let g_pyr = gaussian_pyramid(client, &img, 2, &Pyramid::default()).unwrap();
        assert_eq!(g_pyr.len(), 3);
        assert_eq!(g_pyr[0].dims(), &[4, 4]);
        assert_eq!(g_pyr[1].dims(), &[2, 2]);
        assert_eq!(g_pyr[2].dims(), &[1, 1]);

        let l_pyr = laplacian_pyramid(client, &img, 2, &Pyramid::default()).unwrap();
        assert_eq!(l_pyr.len(), 3);
        assert_eq!(l_pyr[0].dims(), &[4, 4]);
    }
    runtime_test!(test_pyramids, downsample_mean_and_pyramids_match_expectations);
}

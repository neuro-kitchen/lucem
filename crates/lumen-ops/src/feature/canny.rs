//! Canny edge detector: Gaussian smoothing, Sobel gradient magnitude & orientation, directional
//! non-maximum suppression, and dual-threshold hysteresis tracking.

use std::collections::VecDeque;

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{to_f64, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::kernels::{canny_grad_dir_kernel, canny_nms_kernel};
use crate::core::{check, reduce, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::filter::gaussian::{gaussian_blur_into, GaussianBlur};
use crate::filter::gradient::{directional_stencil_into, StencilKind};

/// Settings for the [`canny`] edge detector.
#[derive(Debug, Clone, PartialEq)]
pub struct Canny {
    /// Standard deviation of the Gaussian pre-smoothing filter (`>= 0.0`).
    pub sigma: f64,
    /// Lower hysteresis threshold (`None` defaults to `0.1 * max_grad`).
    pub low_threshold: Option<f64>,
    /// Upper hysteresis threshold (`None` defaults to `0.2 * max_grad`).
    pub high_threshold: Option<f64>,
    /// When `true`, `low_threshold` and `high_threshold` are interpreted as fractions in `0.0..=1.0`
    /// of the maximum gradient magnitude.
    pub use_quantiles: bool,
    pub mode: PadMode,
    pub cval: f64,
}

impl Default for Canny {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            low_threshold: None,
            high_threshold: None,
            use_quantiles: false,
            mode: PadMode::Reflect,
            cval: 0.0,
        }
    }
}

/// Runs the 2-D Canny edge detector on `image`, returning a binary `u32` edge map (`1` on edges, `0` elsewhere).
pub fn canny<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &Canny,
) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("canny", image.shape(), &[2], "2")?;
    if !settings.sigma.is_finite() || settings.sigma < 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "sigma",
            reason: format!("sigma must be non-negative and finite, got {}", settings.sigma),
        });
    }
    if image.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, image.shape().clone())?);
    }

    let shape = image.shape().clone();
    let mut ws = Workspace::new(client);
    let smoothed = DeviceArray::<F>::empty(client, shape.clone())?;
    gaussian_blur_into(
        client,
        image,
        &[settings.sigma, settings.sigma],
        &smoothed,
        &GaussianBlur { mode: settings.mode, cval: settings.cval, ..GaussianBlur::default() },
        &mut ws,
    )?;

    let gy = DeviceArray::<F>::empty(client, shape.clone())?;
    let gx = DeviceArray::<F>::empty(client, shape.clone())?;
    directional_stencil_into(client, &smoothed, &gy, 0, StencilKind::Sobel, settings.mode, settings.cval, &mut ws)?;
    directional_stencil_into(client, &smoothed, &gx, 1, StencilKind::Sobel, settings.mode, settings.cval, &mut ws)?;

    let mag = DeviceArray::<F>::empty(client, shape.clone())?;
    let sector = DeviceArray::<u32>::empty(client, shape.clone())?;
    let nms = DeviceArray::<F>::empty(client, shape.clone())?;
    let geom = LaunchGeometry::elementwise(client, image.len());

    unsafe {
        canny_grad_dir_kernel::launch::<F>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim,
            gy.binding(),
            gx.binding(),
            mag.binding(),
            sector.binding(),
            image.len() as u32,
        );
        canny_nms_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            mag.binding(),
            sector.binding(),
            nms.binding(),
            shape.dims()[0] as u32,
            shape.dims()[1] as u32,
            image.len() as u32,
        );
    }

    let max_mag = reduce::sum_min_max(client, &mag)?.max.max(0.0);
    let low_raw = settings.low_threshold.unwrap_or(0.1 * max_mag);
    let high_raw = settings.high_threshold.unwrap_or(0.2 * max_mag);
    let (low, high) = if settings.use_quantiles {
        (low_raw * max_mag, high_raw * max_mag)
    } else {
        (low_raw, high_raw)
    };
    if high < low {
        return Err(OpsError::InvalidParameter {
            name: "high_threshold",
            reason: format!("high_threshold ({high}) must be >= low_threshold ({low})"),
        });
    }

    // 8-connected hysteresis tracking from strong NMS pixels (> high) through weak NMS pixels (>= low)
    let host_nms: Vec<f64> = nms.to_host(client).into_iter().map(to_f64).collect();
    let (h, w) = (shape.dims()[0], shape.dims()[1]);
    let mut edges = vec![0u32; h * w];
    let mut queue = VecDeque::new();

    for idx in 0..(h * w) {
        if host_nms[idx] >= high && high > 0.0 {
            edges[idx] = 1;
            queue.push_back(idx);
        }
    }

    while let Some(idx) = queue.pop_front() {
        let y = (idx / w) as isize;
        let x = (idx % w) as isize;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let ny = y + dy;
                let nx = x + dx;
                if ny >= 0 && ny < (h as isize) && nx >= 0 && nx < (w as isize) {
                    let nidx = (ny as usize) * w + (nx as usize);
                    if edges[nidx] == 0 && host_nms[nidx] >= low && low > 0.0 {
                        edges[nidx] = 1;
                        queue.push_back(nidx);
                    }
                }
            }
        }
    }

    Ok(DeviceArray::from_host(client, shape, edges)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lumen_core::Shape;

    fn canny_detects_square_boundary(client: &Client) {
        let (h, w) = (20usize, 20usize);
        let mut data = vec![0.0f32; h * w];
        for y in 5..15 {
            for x in 5..15 {
                data[y * w + x] = 1.0;
            }
        }
        let img = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();
        let edges = canny(client, &img, &Canny { sigma: 1.0, ..Canny::default() })
            .unwrap()
            .to_host(client);
        let count: u32 = edges.iter().sum();
        assert!((28..=60).contains(&count), "unexpected edge count {count} ({})", client.name());
        assert_eq!(edges[10 * w + 10], 0, "interior of square should not be an edge");
        assert_eq!(edges[0], 0, "exterior corner should not be an edge");
    }
    runtime_test!(test_canny_edges, canny_detects_square_boundary);
}

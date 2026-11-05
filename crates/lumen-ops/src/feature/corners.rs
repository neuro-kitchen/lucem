//! Corner detectors (Harris, Shi-Tomasi, Kitchen-Rosenfeld, Förstner, Moravec, FAST) and subpixel
//! quadratic refinement.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, to_f64, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::kernels::{
    fast9_score_kernel, foerstner_2d_kernel, harris_2d_kernel, kitchen_rosenfeld_2d_kernel,
    moravec_2d_kernel, shi_tomasi_2d_kernel,
};
use super::structure::structure_tensor;
use crate::core::{check, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::filter::gaussian::{gaussian_blur_into, GaussianBlur};
use crate::filter::gradient::{directional_stencil_into, StencilKind};

/// Harris corner response variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarrisMethod {
    /// $\det(A) - k\,\operatorname{tr}(A)^2$.
    K = 0,
    /// Noble's parameter-free harmonic mean $\frac{2\det(A)}{\operatorname{tr}(A) + \varepsilon}$.
    Noble = 1,
}

/// Settings for [`harris`].
#[derive(Debug, Clone, PartialEq)]
pub struct Harris {
    pub method: HarrisMethod,
    pub k: f64,
    pub eps: f64,
    pub sigma: f64,
}

impl Default for Harris {
    fn default() -> Self {
        Self { method: HarrisMethod::K, k: 0.05, eps: 1e-6, sigma: 1.0 }
    }
}

/// Computes the 2-D Harris corner response image.
pub fn harris<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &Harris,
) -> OpsResult<DeviceArray<F>> {
    let (ayy, ayx, axx) = structure_tensor(client, image, settings.sigma, PadMode::Reflect, 0.0)?;
    let out = DeviceArray::<F>::empty(client, image.shape().clone())?;
    if image.is_empty() {
        return Ok(out);
    }
    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        harris_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ayy.binding(),
            ayx.binding(),
            axx.binding(),
            out.binding(),
            cast::<F>(settings.k),
            cast::<F>(settings.eps),
            image.len() as u32,
            settings.method as u32,
        );
    }
    Ok(out)
}

/// Computes the 2-D Shi-Tomasi (Kanade-Tomasi) minimum-eigenvalue corner response at scale `sigma`.
pub fn shi_tomasi<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
) -> OpsResult<DeviceArray<F>> {
    let (ayy, ayx, axx) = structure_tensor(client, image, sigma, PadMode::Reflect, 0.0)?;
    let out = DeviceArray::<F>::empty(client, image.shape().clone())?;
    if image.is_empty() {
        return Ok(out);
    }
    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        shi_tomasi_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ayy.binding(),
            ayx.binding(),
            axx.binding(),
            out.binding(),
            image.len() as u32,
        );
    }
    Ok(out)
}

/// Computes the 2-D Kitchen & Rosenfeld gradient-curvature corner response.
pub fn kitchen_rosenfeld<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("kitchen_rosenfeld", image.shape(), &[2], "2")?;
    let shape = image.shape().clone();
    let out = DeviceArray::<F>::empty(client, shape.clone())?;
    if image.is_empty() {
        return Ok(out);
    }
    let mut ws = Workspace::new(client);
    let iy = DeviceArray::<F>::empty(client, shape.clone())?;
    let ix = DeviceArray::<F>::empty(client, shape.clone())?;
    let iyy = DeviceArray::<F>::empty(client, shape.clone())?;
    let iyx = DeviceArray::<F>::empty(client, shape.clone())?;
    let ixx = DeviceArray::<F>::empty(client, shape)?;

    directional_stencil_into(client, image, &iy, 0, StencilKind::Sobel, mode, cval, &mut ws)?;
    directional_stencil_into(client, image, &ix, 1, StencilKind::Sobel, mode, cval, &mut ws)?;
    directional_stencil_into(client, &iy, &iyy, 0, StencilKind::Sobel, mode, cval, &mut ws)?;
    directional_stencil_into(client, &iy, &iyx, 1, StencilKind::Sobel, mode, cval, &mut ws)?;
    directional_stencil_into(client, &ix, &ixx, 1, StencilKind::Sobel, mode, cval, &mut ws)?;

    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        kitchen_rosenfeld_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            iy.binding(),
            ix.binding(),
            iyy.binding(),
            iyx.binding(),
            ixx.binding(),
            out.binding(),
            cast::<F>(1e-12),
            image.len() as u32,
        );
    }
    Ok(out)
}

/// Computes the 2-D Förstner interest operator `(w, q)` (error ellipse size `w` and roundness `q`)
/// at Gaussian scale `sigma`.
pub fn foerstner<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)> {
    let (ayy, ayx, axx) = structure_tensor(client, image, sigma, PadMode::Reflect, 0.0)?;
    let w_out = DeviceArray::<F>::empty(client, image.shape().clone())?;
    let q_out = DeviceArray::<F>::empty(client, image.shape().clone())?;
    if image.is_empty() {
        return Ok((w_out, q_out));
    }
    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        foerstner_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ayy.binding(),
            ayx.binding(),
            axx.binding(),
            w_out.binding(),
            q_out.binding(),
            cast::<F>(1e-12),
            image.len() as u32,
        );
    }
    Ok((w_out, q_out))
}

/// Computes the 2-D Moravec minimum directional SSD corner response over a `(2 * window_radius + 1)^2` patch.
pub fn moravec<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    window_radius: usize,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("moravec", image.shape(), &[2], "2")?;
    if window_radius == 0 {
        return Err(OpsError::InvalidParameter {
            name: "window_radius",
            reason: "window_radius must be at least 1".to_string(),
        });
    }
    let out = DeviceArray::<F>::empty(client, image.shape().clone())?;
    if image.is_empty() {
        return Ok(out);
    }
    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        moravec_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            image.binding(),
            out.binding(),
            image.dims()[0] as i32,
            image.dims()[1] as i32,
            window_radius as i32,
            image.len() as u32,
        );
    }
    Ok(out)
}

/// Computes Rosten & Drummond's FAST corner response on the 16-pixel Bresenham circle of radius 3,
/// requiring at least `n` (`1..=16`, typically `9` or `12`) contiguous brighter or darker pixels
/// exceeding `threshold`.
pub fn fast<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    n: usize,
    threshold: f64,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("fast", image.shape(), &[2], "2")?;
    if !(1..=16).contains(&n) {
        return Err(OpsError::InvalidParameter {
            name: "n",
            reason: format!("FAST contiguous length n must be in 1..=16, got {n}"),
        });
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "threshold",
            reason: format!("threshold must be non-negative and finite, got {threshold}"),
        });
    }
    let out = DeviceArray::<F>::empty(client, image.shape().clone())?;
    if image.is_empty() {
        return Ok(out);
    }
    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        fast9_score_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            image.binding(),
            out.binding(),
            image.dims()[0] as i32,
            image.dims()[1] as i32,
            n as u32,
            cast::<F>(threshold),
            image.len() as u32,
        );
    }
    Ok(out)
}

/// Refines integer 2-D corner coordinates `&[[y, x]]` to subpixel precision `(y, x)` by fitting a
/// local 2-D quadratic surface over a `(2 * window_size + 1)^2` Gaussian-weighted neighborhood.
pub fn corner_subpix<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    corners: &[Vec<usize>],
    window_size: usize,
    sigma: f64,
) -> OpsResult<Vec<[f64; 2]>> {
    check::ndim_in("corner_subpix", image.shape(), &[2], "2")?;
    if window_size == 0 {
        return Err(OpsError::InvalidParameter {
            name: "window_size",
            reason: "window_size must be at least 1".to_string(),
        });
    }
    if corners.is_empty() || image.is_empty() {
        return Ok(Vec::new());
    }
    let smoothed = if sigma > 0.0 {
        let buf = DeviceArray::<F>::empty(client, image.shape().clone())?;
        let mut ws = Workspace::new(client);
        gaussian_blur_into(client, image, &[sigma, sigma], &buf, &GaussianBlur::default(), &mut ws)?;
        buf.to_host(client)
    } else {
        image.to_host(client)
    };
    let (h, w) = (image.dims()[0], image.dims()[1]);

    let mut refined = Vec::with_capacity(corners.len());
    for c in corners {
        if c.len() != 2 {
            return Err(OpsError::InvalidParameter {
                name: "corners",
                reason: "each corner coordinate must have length 2 [y, x]".to_string(),
            });
        }
        let (cy, cx) = (c[0], c[1]);
        if cy == 0 || cx == 0 || cy + 1 >= h || cx + 1 >= w {
            refined.push([cy as f64, cx as f64]);
            continue;
        }
        // Central-difference 2nd-order Taylor expansion around (cy, cx)
        let v00 = to_f64(smoothed[(cy - 1) * w + (cx - 1)]);
        let v01 = to_f64(smoothed[(cy - 1) * w + cx]);
        let v02 = to_f64(smoothed[(cy - 1) * w + (cx + 1)]);
        let v10 = to_f64(smoothed[cy * w + (cx - 1)]);
        let v11 = to_f64(smoothed[cy * w + cx]);
        let v12 = to_f64(smoothed[cy * w + (cx + 1)]);
        let v20 = to_f64(smoothed[(cy + 1) * w + (cx - 1)]);
        let v21 = to_f64(smoothed[(cy + 1) * w + cx]);
        let v22 = to_f64(smoothed[(cy + 1) * w + (cx + 1)]);

        let gy = 0.5 * (v21 - v01);
        let gx = 0.5 * (v12 - v10);
        let hyy = v21 - 2.0 * v11 + v01;
        let hxx = v12 - 2.0 * v11 + v10;
        let hyx = 0.25 * (v22 - v20 - v02 + v00);
        let det = hyy * hxx - hyx * hyx;

        if det.abs() > 1e-12 {
            let dy = -(hxx * gy - hyx * gx) / det;
            let dx = -(hyy * gx - hyx * gy) / det;
            let max_shift = window_size as f64;
            if dy.abs() <= max_shift && dx.abs() <= max_shift {
                refined.push([cy as f64 + dy, cx as f64 + dx]);
            } else {
                refined.push([cy as f64, cx as f64]);
            }
        } else {
            refined.push([cy as f64, cx as f64]);
        }
    }
    Ok(refined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lumen_core::Shape;

    fn corner_detectors_fire_at_rectangle_vertices(client: &Client) {
        let (h, w) = (21usize, 21usize);
        let mut data = vec![0.0f32; h * w];
        for y in 6..15 {
            for x in 6..15 {
                data[y * w + x] = 10.0;
            }
        }
        let img = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();

        let har = harris(client, &img, &Harris::default()).unwrap().to_host(client);
        let st = shi_tomasi(client, &img, 1.0).unwrap().to_host(client);
        let mor = moravec(client, &img, 1).unwrap().to_host(client);
        let fst = fast(client, &img, 9, 2.0).unwrap().to_host(client);

        // Vertex (6, 6) must have strictly stronger response than flat interior (10, 10)
        assert!(har[6 * w + 6] > har[10 * w + 10], "{}", client.name());
        assert!(st[6 * w + 6] > st[10 * w + 10], "{}", client.name());
        assert!(mor[6 * w + 6] > mor[10 * w + 10], "{}", client.name());
        assert!(fst[6 * w + 6] > fst[10 * w + 10], "{}", client.name());

        let sub = corner_subpix(client, &img, &[vec![6, 6]], 2, 1.0).unwrap();
        assert_eq!(sub.len(), 1);
        assert!((sub[0][0] - 6.0).abs() <= 2.0 && (sub[0][1] - 6.0).abs() <= 2.0);
    }
    runtime_test!(test_corner_detectors, corner_detectors_fire_at_rectangle_vertices);
}

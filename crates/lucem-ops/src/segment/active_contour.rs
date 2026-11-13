//! Variational and morphological active contours (level-set snakes): Chan-Vese (`chan_vese`),
//! Morphological Active Contours Without Edges (`morph_chan_vese`), Morphological Geodesic Active
//! Contours (`morph_geodesic_contour`), and level-set initializers (`checkerboard_level_set`,
//! `disk_level_set`).

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast, to_f64, ImageFloat};
use lucem_core::memory::{DeviceArray, Workspace};
use lucem_core::Shape;

use super::kernels::{
    chan_vese_step_2d_kernel, checkerboard_level_set_2d_kernel, disk_level_set_2d_kernel,
    morph_contour_force_2d_kernel, morph_smooth_curv_2d_kernel,
};
use crate::core::{check, elementwise};
use crate::error::{OpsError, OpsResult};

/// Parameters for the Chan-Vese active contour model ([`chan_vese`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ChanVeseOptions {
    pub mu: f64,
    pub lambda1: f64,
    pub lambda2: f64,
    pub tol: f64,
    pub max_iter: usize,
    pub dt: f64,
}

impl Default for ChanVeseOptions {
    fn default() -> Self {
        Self {
            mu: 0.25,
            lambda1: 1.0,
            lambda2: 1.0,
            tol: 1e-3,
            max_iter: 100,
            dt: 0.5,
        }
    }
}

/// Generates a 2-D checkerboard binary level-set (`1.0` / `0.0`) of `shape = [h, w]` with tile
/// size `square_size` on the device.
pub fn checkerboard_level_set<F: ImageFloat>(
    client: &Client,
    shape: Shape,
    square_size: usize,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("checkerboard_level_set", &shape, &[2], "2")?;
    if square_size == 0 {
        return Err(OpsError::InvalidParameter {
            name: "square_size",
            reason: "square_size must be >= 1".to_string(),
        });
    }
    let output = DeviceArray::<F>::empty(client, shape)?;
    if output.is_empty() {
        return Ok(output);
    }
    let w = output.dims()[1];
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        checkerboard_level_set_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            output.binding(),
            w as u32,
            square_size as u32,
            output.len() as u32,
        );
    }
    Ok(output)
}

/// Generates a 2-D binary disk level-set (`1.0` inside `radius` around `center`, `0.0` outside) on
/// the device. Defaults to the image center and `0.375 * min(h, w)` when `None`.
pub fn disk_level_set<F: ImageFloat>(
    client: &Client,
    shape: Shape,
    center: Option<(f64, f64)>,
    radius: Option<f64>,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("disk_level_set", &shape, &[2], "2")?;
    let (h, w) = (shape.dims()[0], shape.dims()[1]);
    let (cy, cx) = center.unwrap_or(((h as f64 - 1.0) * 0.5, (w as f64 - 1.0) * 0.5));
    let rad = radius.unwrap_or_else(|| 0.375 * (h.min(w) as f64));
    if rad < 0.0 || !rad.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "radius",
            reason: "radius must be finite and >= 0".to_string(),
        });
    }
    let output = DeviceArray::<F>::empty(client, shape)?;
    if output.is_empty() {
        return Ok(output);
    }
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        disk_level_set_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            output.binding(),
            w as u32,
            cast::<F>(cy),
            cast::<F>(cx),
            cast::<F>(rad * rad),
            output.len() as u32,
        );
    }
    Ok(output)
}

/// Computes the region means `(c1, c2)` where `phi > threshold` (`c1`) and `phi <= threshold` (`c2`).
fn region_means<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    phi: &DeviceArray<F>,
    threshold: f64,
) -> (f64, f64) {
    let img_h = image.to_host(client);
    let phi_h = phi.to_host(client);
    let mut sum1 = 0.0;
    let mut cnt1 = 0usize;
    let mut sum2 = 0.0;
    let mut cnt2 = 0usize;
    for (&v, &p) in img_h.iter().zip(phi_h.iter()) {
        let vf = to_f64(v);
        if to_f64(p) > threshold {
            sum1 += vf;
            cnt1 += 1;
        } else {
            sum2 += vf;
            cnt2 += 1;
        }
    }
    let c1 = if cnt1 > 0 { sum1 / (cnt1 as f64) } else { 0.0 };
    let c2 = if cnt2 > 0 { sum2 / (cnt2 as f64) } else { 0.0 };
    (c1, c2)
}

/// Segments a 2-D image using the Chan-Vese active contour without edges, returning the binary
/// mask (`1` where $\phi > 0$, `0` elsewhere) and the evolved level-set $\phi$.
pub fn chan_vese<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    init_level_set: Option<&DeviceArray<F>>,
    options: &ChanVeseOptions,
) -> OpsResult<(DeviceArray<u32>, DeviceArray<F>)> {
    check::ndim_in("chan_vese", image.shape(), &[2], "2")?;
    if image.is_empty() {
        return Ok((
            DeviceArray::<u32>::zeros(client, image.shape().clone())?,
            DeviceArray::<F>::zeros(client, image.shape().clone())?,
        ));
    }

    let phi = DeviceArray::<F>::empty(client, image.shape().clone())?;
    match init_level_set {
        Some(init) => {
            check::same_shape("image", image, "init_level_set", init)?;
            // Shift `[0, 1]` binary initializers to `[-0.5, 0.5]` if non-negative
            let host = init.to_host(client);
            let min_v = host.iter().map(|&x| to_f64(x)).fold(f64::INFINITY, f64::min);
            if min_v >= 0.0 {
                elementwise::affine_into(client, init, &phi, 1.0, -0.5)?;
            } else {
                elementwise::copy_into(client, init, &phi)?;
            }
        }
        None => {
            let chk = checkerboard_level_set::<F>(client, image.shape().clone(), 5)?;
            elementwise::affine_into(client, &chk, &phi, 1.0, -0.5)?;
        }
    }

    let mut ws = Workspace::new(client);
    let phi_next = ws.array::<F>("cv_phi_next", image.shape())?;
    let (h, w) = (image.dims()[0], image.dims()[1]);
    let geom = LaunchGeometry::elementwise(client, image.len());

    let iters = options.max_iter.max(1);
    for _ in 0..iters {
        let (c1, c2) = region_means(client, image, &phi, 0.0);
        unsafe {
            chan_vese_step_2d_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                image.binding(),
                phi.binding(),
                phi_next.binding(),
                h as u32,
                w as u32,
                cast::<F>(c1),
                cast::<F>(c2),
                cast::<F>(options.mu),
                cast::<F>(options.lambda1),
                cast::<F>(options.lambda2),
                cast::<F>(options.dt),
                image.len() as u32,
            );
        }
        elementwise::copy_into(client, &phi_next, &phi)?;
    }

    let mask = DeviceArray::<u32>::empty(client, image.shape().clone())?;
    elementwise::threshold_scalar_into(client, &phi, 0.0, false, &mask)?;
    Ok((mask, phi))
}

/// Morphological Active Contours Without Edges (MorphACWE / `morph_chan_vese`) on a 2-D image.
pub fn morph_chan_vese<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    iterations: usize,
    init_level_set: Option<&DeviceArray<F>>,
    smoothing: usize,
    lambda1: f64,
    lambda2: f64,
) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("morph_chan_vese", image.shape(), &[2], "2")?;
    if image.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, image.shape().clone())?);
    }

    let u = match init_level_set {
        Some(init) => {
            check::same_shape("image", image, "init_level_set", init)?;
            let out = DeviceArray::<F>::empty(client, image.shape().clone())?;
            elementwise::threshold_scalar_into(client, init, 0.5, true, &out)?;
            out
        }
        None => checkerboard_level_set::<F>(client, image.shape().clone(), 5)?,
    };

    let mut ws = Workspace::new(client);
    let tmp = ws.array::<F>("morph_cv_tmp", image.shape())?;
    let (h, w) = (image.dims()[0], image.dims()[1]);
    let geom = LaunchGeometry::elementwise(client, image.len());

    for _ in 0..iterations {
        let (c1, c2) = region_means(client, image, &u, 0.5);
        unsafe {
            morph_contour_force_2d_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                image.binding(),
                u.binding(),
                tmp.binding(),
                h as u32,
                w as u32,
                cast::<F>(c1),
                cast::<F>(c2),
                cast::<F>(lambda1),
                cast::<F>(lambda2),
                image.len() as u32,
            );
        }
        elementwise::copy_into(client, &tmp, &u)?;

        for _ in 0..smoothing {
            unsafe {
                morph_smooth_curv_2d_kernel::launch::<F>(
                    client,
                    geom.cube_count.clone(),
                    geom.cube_dim,
                    u.binding(),
                    tmp.binding(),
                    h as u32,
                    w as u32,
                    image.len() as u32,
                );
            }
            elementwise::copy_into(client, &tmp, &u)?;
        }
    }

    let mask = DeviceArray::<u32>::empty(client, image.shape().clone())?;
    elementwise::threshold_scalar_into(client, &u, 0.5, true, &mask)?;
    Ok(mask)
}

/// Morphological Geodesic Active Contours (MorphGAC / `morph_geodesic_contour`) on a 2-D inverse
/// edge-indicator map `gimage`.
pub fn morph_geodesic_contour<F: ImageFloat>(
    client: &Client,
    gimage: &DeviceArray<F>,
    iterations: usize,
    init_level_set: Option<&DeviceArray<F>>,
    smoothing: usize,
    threshold: f64,
    balloon: f64,
) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("morph_geodesic_contour", gimage.shape(), &[2], "2")?;
    if gimage.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, gimage.shape().clone())?);
    }

    let u = match init_level_set {
        Some(init) => {
            check::same_shape("gimage", gimage, "init_level_set", init)?;
            let out = DeviceArray::<F>::empty(client, gimage.shape().clone())?;
            elementwise::threshold_scalar_into(client, init, 0.5, true, &out)?;
            out
        }
        None => disk_level_set::<F>(client, gimage.shape().clone(), None, None)?,
    };

    let mut ws = Workspace::new(client);
    let tmp = ws.array::<F>("morph_gac_tmp", gimage.shape())?;
    let (h, w) = (gimage.dims()[0], gimage.dims()[1]);
    let geom = LaunchGeometry::elementwise(client, gimage.len());
    let (l1, l2) = if balloon >= 0.0 {
        (1.0, 1.0 + balloon.abs())
    } else {
        (1.0 + balloon.abs(), 1.0)
    };

    for _ in 0..iterations {
        unsafe {
            morph_contour_force_2d_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                gimage.binding(),
                u.binding(),
                tmp.binding(),
                h as u32,
                w as u32,
                cast::<F>(threshold * 1.5),
                cast::<F>(threshold * 0.5),
                cast::<F>(l1),
                cast::<F>(l2),
                gimage.len() as u32,
            );
        }
        elementwise::copy_into(client, &tmp, &u)?;

        for _ in 0..smoothing {
            unsafe {
                morph_smooth_curv_2d_kernel::launch::<F>(
                    client,
                    geom.cube_count.clone(),
                    geom.cube_dim,
                    u.binding(),
                    tmp.binding(),
                    h as u32,
                    w as u32,
                    gimage.len() as u32,
                );
            }
            elementwise::copy_into(client, &tmp, &u)?;
        }
    }

    let mask = DeviceArray::<u32>::empty(client, gimage.shape().clone())?;
    elementwise::threshold_scalar_into(client, &u, 0.5, true, &mask)?;
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn active_contours_segment_bright_square(client: &Client) {
        let mut img = vec![0.0f32; 64];
        for r in 2..6 {
            for c in 2..6 {
                img[r * 8 + c] = 10.0;
            }
        }
        let arr = DeviceArray::from_host(client, Shape::new([8, 8]), img).unwrap();
        let disk = disk_level_set::<f32>(client, Shape::new([8, 8]), Some((3.5, 3.5)), Some(2.5)).unwrap();

        let (cv_mask, _) = chan_vese(
            client,
            &arr,
            Some(&disk),
            &ChanVeseOptions {
                mu: 0.05,
                max_iter: 40,
                dt: 0.5,
                ..ChanVeseOptions::default()
            },
        )
        .unwrap();
        let cv_host = cv_mask.to_host(client);
        assert_eq!(cv_host[3 * 8 + 3], 1, "{}", client.name());
        assert_eq!(cv_host[0], 0);

        let mcv = morph_chan_vese(client, &arr, 10, Some(&disk), 1, 1.0, 1.0)
            .unwrap()
            .to_host(client);
        assert_eq!(mcv[3 * 8 + 3], 1);
        assert_eq!(mcv[0], 0);

        let mgac = morph_geodesic_contour(client, &arr, 5, Some(&disk), 1, 5.0, 0.5)
            .unwrap()
            .to_host(client);
        assert_eq!(mgac[3 * 8 + 3], 1);
    }
    runtime_test!(test_active_contour, active_contours_segment_bright_square);
}

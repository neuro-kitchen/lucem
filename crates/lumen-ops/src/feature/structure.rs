//! Differential geometric structure: Hessian matrix, Hessian determinant & eigenvalues, structure
//! tensor & eigenvalues, and Koenderink shape index.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::ImageFloat;
use lumen_core::memory::{DeviceArray, Workspace};

use super::kernels::{hessian_det_2d_kernel, outer_product_2d_kernel, shape_index_2d_kernel};
use crate::core::{check, eigen, PadMode};
use crate::error::OpsResult;
use crate::filter::gaussian::{gaussian_blur_into, GaussianBlur};
use crate::filter::gradient::{directional_stencil_into, StencilKind};

/// Computes the 2-D Gaussian Hessian components `(H_rr, H_rc, H_cc)` at scale `sigma`.
pub fn hessian_matrix<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>, DeviceArray<F>)> {
    check::ndim_in("hessian_matrix", image.shape(), &[2], "2")?;
    let shape = image.shape().clone();
    let hrr = DeviceArray::<F>::empty(client, shape.clone())?;
    let hrc = DeviceArray::<F>::empty(client, shape.clone())?;
    let hcc = DeviceArray::<F>::empty(client, shape)?;
    let mut ws = Workspace::new(client);
    gaussian_blur_into(
        client,
        image,
        &[sigma, sigma],
        &hrr,
        &GaussianBlur { orders: vec![2, 0], mode, cval, ..GaussianBlur::default() },
        &mut ws,
    )?;
    gaussian_blur_into(
        client,
        image,
        &[sigma, sigma],
        &hrc,
        &GaussianBlur { orders: vec![1, 1], mode, cval, ..GaussianBlur::default() },
        &mut ws,
    )?;
    gaussian_blur_into(
        client,
        image,
        &[sigma, sigma],
        &hcc,
        &GaussianBlur { orders: vec![0, 2], mode, cval, ..GaussianBlur::default() },
        &mut ws,
    )?;
    Ok((hrr, hrc, hcc))
}

/// Computes the determinant $\det(H_\sigma) = H_{rr} H_{cc} - H_{rc}^2$ of the 2-D Gaussian Hessian.
pub fn hessian_det<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>> {
    let (hrr, hrc, hcc) = hessian_matrix(client, image, sigma, mode, cval)?;
    let out = DeviceArray::<F>::empty(client, image.shape().clone())?;
    if image.is_empty() {
        return Ok(out);
    }
    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        hessian_det_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            hrr.binding(),
            hrc.binding(),
            hcc.binding(),
            out.binding(),
            image.len() as u32,
        );
    }
    Ok(out)
}

/// Computes ordered eigenvalues $(\lambda_1 \ge \lambda_2)$ of the 2-D Gaussian Hessian at scale `sigma`.
pub fn hessian_eigvals<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)> {
    let (hrr, hrc, hcc) = hessian_matrix(client, image, sigma, mode, cval)?;
    let eig1 = DeviceArray::<F>::empty(client, image.shape().clone())?;
    let eig2 = DeviceArray::<F>::empty(client, image.shape().clone())?;
    eigen::symmetric_eigenvalues_2x2_into(client, &hrr, &hrc, &hcc, &eig1, &eig2)?;
    Ok((eig1, eig2))
}

/// Computes the 2-D structure tensor `(A_rr, A_rc, A_cc)` by smoothing outer products of Sobel gradients
/// with a Gaussian window of scale `sigma`.
pub fn structure_tensor<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>, DeviceArray<F>)> {
    check::ndim_in("structure_tensor", image.shape(), &[2], "2")?;
    let shape = image.shape().clone();
    let arr = DeviceArray::<F>::empty(client, shape.clone())?;
    let arc = DeviceArray::<F>::empty(client, shape.clone())?;
    let acc = DeviceArray::<F>::empty(client, shape.clone())?;
    if image.is_empty() {
        return Ok((arr, arc, acc));
    }

    let mut ws = Workspace::new(client);
    let gy = DeviceArray::<F>::empty(client, shape.clone())?;
    let gx = DeviceArray::<F>::empty(client, shape.clone())?;
    directional_stencil_into(client, image, &gy, 0, StencilKind::Sobel, mode, cval, &mut ws)?;
    directional_stencil_into(client, image, &gx, 1, StencilKind::Sobel, mode, cval, &mut ws)?;

    let iyy = DeviceArray::<F>::empty(client, shape.clone())?;
    let iyx = DeviceArray::<F>::empty(client, shape.clone())?;
    let ixx = DeviceArray::<F>::empty(client, shape)?;
    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        outer_product_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            gy.binding(),
            gx.binding(),
            iyy.binding(),
            iyx.binding(),
            ixx.binding(),
            image.len() as u32,
        );
    }

    let blur = GaussianBlur { mode, cval, ..GaussianBlur::default() };
    gaussian_blur_into(client, &iyy, &[sigma, sigma], &arr, &blur, &mut ws)?;
    gaussian_blur_into(client, &iyx, &[sigma, sigma], &arc, &blur, &mut ws)?;
    gaussian_blur_into(client, &ixx, &[sigma, sigma], &acc, &blur, &mut ws)?;
    Ok((arr, arc, acc))
}

/// Computes ordered eigenvalues $(\lambda_1 \ge \lambda_2)$ of the 2-D structure tensor `(arr, arc, acc)`.
pub fn structure_tensor_eigvals<F: ImageFloat>(
    client: &Client,
    arr: &DeviceArray<F>,
    arc: &DeviceArray<F>,
    acc: &DeviceArray<F>,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)> {
    let eig1 = DeviceArray::<F>::empty(client, arr.shape().clone())?;
    let eig2 = DeviceArray::<F>::empty(client, arr.shape().clone())?;
    eigen::symmetric_eigenvalues_2x2_into(client, arr, arc, acc, &eig1, &eig2)?;
    Ok((eig1, eig2))
}

/// Computes Koenderink & van Doorn's local 2-D shape index $s \in [-1, 1]$ at scale `sigma`:
/// cup (`-1.0`), rut (`-0.5`), saddle (`0.0`), ridge (`+0.5`), dome/cap (`+1.0`).
pub fn shape_index<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>> {
    let (eig1, eig2) = hessian_eigvals(client, image, sigma, mode, cval)?;
    let out = DeviceArray::<F>::empty(client, image.shape().clone())?;
    if image.is_empty() {
        return Ok(out);
    }
    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        shape_index_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            eig1.binding(),
            eig2.binding(),
            out.binding(),
            image.len() as u32,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lumen_core::Shape;

    fn shape_index_distinguishes_dome_and_cup(client: &Client) {
        let (h, w) = (17usize, 17usize);
        let mut dome = vec![0.0f32; h * w];
        let mut cup = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let r2 = ((y as f64 - 8.0).powi(2) + (x as f64 - 8.0).powi(2)) as f32;
                dome[y * w + x] = -r2;
                cup[y * w + x] = r2;
            }
        }
        let dome_arr = DeviceArray::from_host(client, Shape::new([h, w]), dome).unwrap();
        let cup_arr = DeviceArray::from_host(client, Shape::new([h, w]), cup).unwrap();

        let s_dome = shape_index(client, &dome_arr, 1.5, PadMode::Nearest, 0.0).unwrap().to_host(client);
        let s_cup = shape_index(client, &cup_arr, 1.5, PadMode::Nearest, 0.0).unwrap().to_host(client);
        assert_close(&[s_dome[8 * w + 8] as f64], &[1.0], 2e-2, client.name());
        assert_close(&[s_cup[8 * w + 8] as f64], &[-1.0], 2e-2, client.name());
    }
    runtime_test!(test_structure_and_shape_index, shape_index_distinguishes_dome_and_cup);
}

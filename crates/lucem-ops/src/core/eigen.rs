//! Per-pixel analytic eigenvalues of symmetric $2 \times 2$ and $3 \times 3$ matrices (used by
//! Hessian, structure tensor, ridge filters, shape index, and inertia tensors).

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::ImageFloat;
use lucem_core::memory::DeviceArray;

use super::check;
use super::kernels::{sym_eigvals_2x2_kernel, sym_eigvals_3x3_kernel};
use crate::error::OpsResult;

/// Computes ordered eigenvalues $\lambda_1 \ge \lambda_2$ for each pixel of a symmetric $2 \times 2$
/// field `(axx, axy, ayy)` into `(eig1, eig2)`.
pub fn symmetric_eigenvalues_2x2_into<F: ImageFloat>(
    client: &Client,
    axx: &DeviceArray<F>,
    axy: &DeviceArray<F>,
    ayy: &DeviceArray<F>,
    eig1: &DeviceArray<F>,
    eig2: &DeviceArray<F>,
) -> OpsResult<()> {
    check::same_shape("axx", axx, "axy", axy)?;
    check::same_shape("axx", axx, "ayy", ayy)?;
    check::same_shape("axx", axx, "eig1", eig1)?;
    check::same_shape("axx", axx, "eig2", eig2)?;
    if axx.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, axx.len());
    unsafe {
        sym_eigvals_2x2_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            axx.binding(),
            axy.binding(),
            ayy.binding(),
            eig1.binding(),
            eig2.binding(),
            axx.len() as u32,
        );
    }
    Ok(())
}

/// Computes ordered eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$ for each pixel of a symmetric
/// $3 \times 3$ field `(a00, a01, a02, a11, a12, a22)` into `(eig1, eig2, eig3)`.
#[allow(clippy::too_many_arguments)]
pub fn symmetric_eigenvalues_3x3_into<F: ImageFloat>(
    client: &Client,
    a00: &DeviceArray<F>,
    a01: &DeviceArray<F>,
    a02: &DeviceArray<F>,
    a11: &DeviceArray<F>,
    a12: &DeviceArray<F>,
    a22: &DeviceArray<F>,
    eig1: &DeviceArray<F>,
    eig2: &DeviceArray<F>,
    eig3: &DeviceArray<F>,
) -> OpsResult<()> {
    check::same_shape("a00", a00, "a01", a01)?;
    check::same_shape("a00", a00, "a02", a02)?;
    check::same_shape("a00", a00, "a11", a11)?;
    check::same_shape("a00", a00, "a12", a12)?;
    check::same_shape("a00", a00, "a22", a22)?;
    check::same_shape("a00", a00, "eig1", eig1)?;
    check::same_shape("a00", a00, "eig2", eig2)?;
    check::same_shape("a00", a00, "eig3", eig3)?;
    if a00.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, a00.len());
    unsafe {
        sym_eigvals_3x3_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            a00.binding(),
            a01.binding(),
            a02.binding(),
            a11.binding(),
            a12.binding(),
            a22.binding(),
            eig1.binding(),
            eig2.binding(),
            eig3.binding(),
            a00.len() as u32,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lucem_core::Shape;

    fn eigenvalues_2x2_match_known(client: &Client) {
        let shape = Shape::vector(2);
        let axx = DeviceArray::from_host(client, shape.clone(), vec![3.0f32, 5.0]).unwrap();
        let axy = DeviceArray::from_host(client, shape.clone(), vec![1.0f32, 0.0]).unwrap();
        let ayy = DeviceArray::from_host(client, shape.clone(), vec![3.0f32, 2.0]).unwrap();
        let e1 = DeviceArray::<f32>::empty(client, shape.clone()).unwrap();
        let e2 = DeviceArray::<f32>::empty(client, shape).unwrap();
        symmetric_eigenvalues_2x2_into(client, &axx, &axy, &ayy, &e1, &e2).unwrap();
        assert_close(&e1.to_host(client), &[4.0, 5.0], 1e-6, client.name());
        assert_close(&e2.to_host(client), &[2.0, 2.0], 1e-6, client.name());
    }
    runtime_test!(test_eigen_2x2, eigenvalues_2x2_match_known);
}

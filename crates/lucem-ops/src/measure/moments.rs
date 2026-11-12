//! Spatial raw, central, scale-normalized, and Hu rotation-invariant moments, plus inertia tensors.

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast, ImageFloat};
use lucem_core::memory::{DeviceArray, Workspace};

use super::kernels::moment_monomial_2d_kernel;
use crate::core::{check, reduce};
use crate::error::{OpsError, OpsResult};

/// Evaluates all 2-D spatial moments $M_{p, q} = \sum_{y, x} I(y, x) (y - c_y)^p (x - c_x)^q$
/// for $0 \le p, q \le \text{order}$ into an `(order + 1) x (order + 1)` row-major vector.
fn moments_about_center<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    center: (f64, f64),
    order: usize,
    ws: &mut Workspace,
) -> OpsResult<Vec<Vec<f64>>> {
    check::ndim_in("moments", input.shape(), &[2], "2")?;
    if input.is_empty() {
        return Err(OpsError::EmptyInput("moments"));
    }
    let w = input.dims()[1];
    let scratch = ws.array::<F>("moment_monomial", input.shape())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    let cy = cast::<F>(center.0);
    let cx = cast::<F>(center.1);

    let mut mat = vec![vec![0.0; order + 1]; order + 1];
    for (p, row) in mat.iter_mut().enumerate() {
        for (q, cell) in row.iter_mut().enumerate() {
            unsafe {
                moment_monomial_2d_kernel::launch::<F>(
                    client,
                    geom.cube_count.clone(),
                    geom.cube_dim,
                    input.binding(),
                    scratch.binding(),
                    w as u32,
                    cy,
                    cx,
                    p as u32,
                    q as u32,
                    input.len() as u32,
                );
            }
            *cell = reduce::sum_min_max(client, &scratch)?.sum;
        }
    }
    Ok(mat)
}

/// Computes raw 2-D spatial moments $M_{p,q}$ up to `order` around `center` (defaults to `(0.0, 0.0)`).
pub fn moments<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    order: usize,
    center: Option<(f64, f64)>,
) -> OpsResult<Vec<Vec<f64>>> {
    let mut ws = Workspace::new(client);
    moments_about_center(client, input, center.unwrap_or((0.0, 0.0)), order, &mut ws)
}

/// Computes central 2-D moments $\mu_{p,q}$ up to `order` about the image centroid $(\bar{y}, \bar{x})$.
pub fn central_moments<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    order: usize,
) -> OpsResult<Vec<Vec<f64>>> {
    let mut ws = Workspace::new(client);
    let raw1 = moments_about_center(client, input, (0.0, 0.0), 1, &mut ws)?;
    let m00 = raw1[0][0];
    let center = if m00.abs() > f64::EPSILON {
        (raw1[1][0] / m00, raw1[0][1] / m00)
    } else {
        (0.0, 0.0)
    };
    moments_about_center(client, input, center, order, &mut ws)
}

/// Computes scale-normalized central 2-D moments $\eta_{p,q} = \mu_{p,q} / \mu_{0,0}^{1 + (p + q)/2}$
/// for $p + q \ge 2$ (`NaN` for $p + q < 2$ when `mu00 == 0`, `0.0` at $(0,1),(1,0)$, `1.0` at $(0,0)$).
pub fn normalized_moments<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    order: usize,
) -> OpsResult<Vec<Vec<f64>>> {
    let mu = central_moments(client, input, order)?;
    let m00 = mu[0][0];
    let mut nu = vec![vec![0.0; order + 1]; order + 1];
    if m00.abs() <= f64::EPSILON {
        return Ok(nu);
    }
    for p in 0..=order {
        for q in 0..=order {
            if p + q >= 2 {
                let gamma = 1.0 + 0.5 * ((p + q) as f64);
                nu[p][q] = mu[p][q] / m00.powf(gamma);
            } else if p == 0 && q == 0 {
                nu[p][q] = 1.0;
            } else {
                nu[p][q] = 0.0;
            }
        }
    }
    Ok(nu)
}

/// Computes the 7 classic Hu rotation-, scale-, and translation-invariant moments of a 2-D image.
pub fn hu_moments<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<[f64; 7]> {
    let nu = normalized_moments(client, input, 3)?;
    let n20 = nu[2][0];
    let n02 = nu[0][2];
    let n11 = nu[1][1];
    let n30 = nu[3][0];
    let n12 = nu[1][2];
    let n21 = nu[2][1];
    let n03 = nu[0][3];

    let t1 = n30 + n12;
    let t2 = n21 + n03;
    let q1 = t1 * t1 - t2 * t2;
    let q2 = 2.0 * t1 * t2;
    let s1 = n30 - 3.0 * n12;
    let s2 = 3.0 * n21 - n03;

    let h1 = n20 + n02;
    let h2 = (n20 - n02).powi(2) + 4.0 * n11 * n11;
    let h3 = s1 * s1 + s2 * s2;
    let h4 = t1 * t1 + t2 * t2;
    let h5 = s1 * t1 * (t1 * t1 - 3.0 * t2 * t2) - s2 * t2 * (3.0 * t1 * t1 - t2 * t2);
    let h6 = (n20 - n02) * q1 + 2.0 * n11 * q2;
    let h7 = s2 * t1 * (t1 * t1 - 3.0 * t2 * t2) + s1 * t2 * (3.0 * t1 * t1 - t2 * t2);

    Ok([h1, h2, h3, h4, h5, h6, h7])
}

/// Computes the 2-D central inertia tensor `[[I_yy, I_yx], [I_xy, I_xx]]` normalized by total mass:
/// `[[mu02 / mu00, -mu11 / mu00], [-mu11 / mu00, mu20 / mu00]]`.
pub fn inertia_tensor<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<[[f64; 2]; 2]> {
    let mu = central_moments(client, input, 2)?;
    let m00 = mu[0][0];
    if m00.abs() <= f64::EPSILON {
        return Ok([[0.0, 0.0], [0.0, 0.0]]);
    }
    let i_yy = mu[0][2] / m00;
    let i_xx = mu[2][0] / m00;
    let i_xy = -mu[1][1] / m00;
    Ok([[i_yy, i_xy], [i_xy, i_xx]])
}

/// Computes the eigenvalues `[lambda_1, lambda_2]` (`lambda_1 >= lambda_2`) of the 2-D inertia tensor.
pub fn inertia_tensor_eigvals<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<[f64; 2]> {
    let t = inertia_tensor(client, input)?;
    let a = t[0][0];
    let b = t[0][1];
    let c = t[1][1];
    let tr = a + c;
    let disc = ((a - c).powi(2) + 4.0 * b * b).sqrt();
    Ok([0.5 * (tr + disc), 0.5 * (tr - disc)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lucem_core::Shape;

    fn moments_and_hu_invariants_match_symmetric_patch(client: &Client) {
        #[rustfmt::skip]
        let img = vec![
            0.0f32, 0.0, 0.0, 0.0, 0.0,
            0.0,    1.0, 2.0, 1.0, 0.0,
            0.0,    2.0, 4.0, 2.0, 0.0,
            0.0,    1.0, 2.0, 1.0, 0.0,
            0.0,    0.0, 0.0, 0.0, 0.0,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([5, 5]), img).unwrap();
        let raw = moments(client, &arr, 2, None).unwrap();
        assert_close(&[raw[0][0]], &[16.0], 1e-5, client.name());
        assert_close(&[raw[1][0] / raw[0][0], raw[0][1] / raw[0][0]], &[2.0, 2.0], 1e-5, client.name());

        let mu = central_moments(client, &arr, 2).unwrap();
        assert_close(&[mu[1][0], mu[0][1], mu[1][1]], &[0.0, 0.0, 0.0], 1e-5, client.name());
        assert_close(&[mu[2][0], mu[0][2]], &[8.0, 8.0], 1e-5, client.name());

        let eig = inertia_tensor_eigvals(client, &arr).unwrap();
        assert_close(&eig, &[0.5, 0.5], 1e-5, client.name());

        let hu = hu_moments(client, &arr).unwrap();
        assert!(hu[0] > 0.0);
        assert_close(&hu[1..], &[0.0; 6], 1e-6, client.name());
    }
    runtime_test!(test_moments, moments_and_hu_invariants_match_symmetric_patch);
}

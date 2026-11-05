//! Multi-scale neurite and vessel ridge enhancement filters: Frangi, Sato, and Meijering.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::gaussian::{gaussian_blur_into, GaussianBlur};
use super::kernels::{frangi_2d_scale_kernel, meijering_2d_scale_kernel, sato_2d_scale_kernel};
use crate::core::{check, eigen, PadMode};
use crate::error::{OpsError, OpsResult};

/// Settings for multi-scale ridge filters ([`frangi`], [`sato`], [`meijering`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RidgeFilter {
    /// Gaussian scales $\sigma$ evaluated and combined via elementwise maximum.
    pub sigmas: Vec<f64>,
    /// `true` to detect dark ridges on a bright background; `false` for bright fluorescence neurites/vessels.
    pub black_ridges: bool,
    pub mode: PadMode,
    pub cval: f64,
    /// Frangi blobness sensitivity parameter $\beta$.
    pub beta: f64,
    /// Frangi structureness sensitivity parameter $\gamma$ (`c`).
    pub gamma: f64,
}

impl Default for RidgeFilter {
    fn default() -> Self {
        Self {
            sigmas: vec![1.0, 3.0, 5.0, 7.0, 9.0],
            black_ridges: true,
            mode: PadMode::Reflect,
            cval: 0.0,
            beta: 0.5,
            gamma: 15.0,
        }
    }
}

/// Computes the 2-D Hessian components `(hxx, hxy, hyy)` at scale `sigma` into workspace arrays and
/// extracts their ordered eigenvalues `(eig1, eig2)`.
#[allow(clippy::too_many_arguments)]
fn hessian_eigvals_2d_at_scale<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
    eig1: &DeviceArray<F>,
    eig2: &DeviceArray<F>,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let shape = input.shape().clone();
    let hxx = ws.array::<F>("ridge_hxx", &shape)?;
    let hxy = ws.array::<F>("ridge_hxy", &shape)?;
    let hyy = ws.array::<F>("ridge_hyy", &shape)?;

    gaussian_blur_into(
        client,
        input,
        &[sigma, sigma],
        &hxx,
        &GaussianBlur { orders: vec![2, 0], mode, cval, ..GaussianBlur::default() },
        ws,
    )?;
    gaussian_blur_into(
        client,
        input,
        &[sigma, sigma],
        &hxy,
        &GaussianBlur { orders: vec![1, 1], mode, cval, ..GaussianBlur::default() },
        ws,
    )?;
    gaussian_blur_into(
        client,
        input,
        &[sigma, sigma],
        &hyy,
        &GaussianBlur { orders: vec![0, 2], mode, cval, ..GaussianBlur::default() },
        ws,
    )?;
    eigen::symmetric_eigenvalues_2x2_into(client, &hxx, &hxy, &hyy, eig1, eig2)
}

/// Multi-scale 2-D Frangi vesselness filter into `output`.
pub fn frangi_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &RidgeFilter,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::ndim_in("frangi", input.shape(), &[2], "2")?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if settings.sigmas.is_empty() {
        return Err(OpsError::InvalidParameter {
            name: "sigmas",
            reason: "at least one sigma scale is required".to_string(),
        });
    }
    if input.is_empty() {
        return Ok(());
    }
    let zero = DeviceArray::<F>::zeros(client, output.shape().clone())?;
    crate::core::elementwise::copy_into(client, &zero, output)?;

    let eig1 = ws.array::<F>("ridge_e1", input.shape())?;
    let eig2 = ws.array::<F>("ridge_e2", input.shape())?;
    let inv_2beta_sq = 1.0 / (2.0 * settings.beta.powi(2).max(1e-12));
    let inv_2c_sq = 1.0 / (2.0 * settings.gamma.powi(2).max(1e-12));
    let geom = LaunchGeometry::elementwise(client, input.len());

    for &sigma in &settings.sigmas {
        hessian_eigvals_2d_at_scale(client, input, sigma, settings.mode, settings.cval, &eig1, &eig2, ws)?;
        unsafe {
            frangi_2d_scale_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                eig1.binding(),
                eig2.binding(),
                output.binding(),
                cast::<F>(inv_2beta_sq),
                cast::<F>(inv_2c_sq),
                input.len() as u32,
                settings.black_ridges,
            );
        }
    }
    Ok(())
}

/// Multi-scale 2-D Frangi vesselness filter allocating `output`.
pub fn frangi<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &RidgeFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    frangi_into(client, input, &output, settings, &mut ws)?;
    Ok(output)
}

/// Multi-scale 2-D Sato tubeness filter into `output`.
pub fn sato_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &RidgeFilter,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::ndim_in("sato", input.shape(), &[2], "2")?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let zero = DeviceArray::<F>::zeros(client, output.shape().clone())?;
    crate::core::elementwise::copy_into(client, &zero, output)?;
    let eig1 = ws.array::<F>("ridge_e1", input.shape())?;
    let eig2 = ws.array::<F>("ridge_e2", input.shape())?;
    let geom = LaunchGeometry::elementwise(client, input.len());

    for &sigma in &settings.sigmas {
        hessian_eigvals_2d_at_scale(client, input, sigma, settings.mode, settings.cval, &eig1, &eig2, ws)?;
        unsafe {
            sato_2d_scale_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                eig1.binding(),
                eig2.binding(),
                output.binding(),
                cast::<F>(sigma * sigma),
                input.len() as u32,
                settings.black_ridges,
            );
        }
    }
    Ok(())
}

/// Multi-scale 2-D Sato tubeness filter allocating `output`.
pub fn sato<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &RidgeFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    sato_into(client, input, &output, settings, &mut ws)?;
    Ok(output)
}

/// Multi-scale 2-D Meijering neuriteness filter into `output`.
pub fn meijering_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &RidgeFilter,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::ndim_in("meijering", input.shape(), &[2], "2")?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let zero = DeviceArray::<F>::zeros(client, output.shape().clone())?;
    crate::core::elementwise::copy_into(client, &zero, output)?;
    let eig1 = ws.array::<F>("ridge_e1", input.shape())?;
    let eig2 = ws.array::<F>("ridge_e2", input.shape())?;
    let geom = LaunchGeometry::elementwise(client, input.len());

    for &sigma in &settings.sigmas {
        hessian_eigvals_2d_at_scale(client, input, sigma, settings.mode, settings.cval, &eig1, &eig2, ws)?;
        unsafe {
            meijering_2d_scale_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                eig1.binding(),
                eig2.binding(),
                output.binding(),
                input.len() as u32,
                settings.black_ridges,
            );
        }
    }
    Ok(())
}

/// Multi-scale 2-D Meijering neuriteness filter allocating `output`.
pub fn meijering<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &RidgeFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    meijering_into(client, input, &output, settings, &mut ws)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lumen_core::Shape;

    fn ridge_filters_detect_bright_horizontal_neurite(client: &Client) {
        let (h, w) = (15usize, 15usize);
        let mut img = vec![0.0f32; h * w];
        for x in 0..w {
            img[7 * w + x] = 10.0;
        }
        let arr = DeviceArray::from_host(client, Shape::new([h, w]), img).unwrap();
        let settings = RidgeFilter { sigmas: vec![1.0, 2.0], black_ridges: false, ..RidgeFilter::default() };
        let s = sato(client, &arr, &settings).unwrap().to_host(client);
        let m = meijering(client, &arr, &settings).unwrap().to_host(client);
        assert!(s[7 * w + 7] > s[0], "{}", client.name());
        assert!(m[7 * w + 7] > m[0], "{}", client.name());
    }
    runtime_test!(test_ridge_filters, ridge_filters_detect_bright_horizontal_neurite);
}

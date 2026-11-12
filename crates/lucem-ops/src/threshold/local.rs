//! Adaptive spatial thresholding: Gaussian/mean/median local threshold, Niblack, and Sauvola.

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast, ImageFloat};
use lucem_core::memory::{DeviceArray, Workspace};

use super::kernels::{niblack_threshold_kernel, sauvola_threshold_kernel, square_kernel};
use crate::core::{check, elementwise, PadMode, StructuringElement};
use crate::filter::{
    box_blur_into, gaussian_blur_into, median_filter_into, BoxBlur, GaussianBlur, OrderFilter,
};
use crate::error::OpsResult;

/// Method for computing the local reference surface in [`local_threshold`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LocalMethod {
    #[default]
    Gaussian,
    Mean,
    Median,
}

/// Settings for [`local_threshold`].
#[derive(Debug, Clone, PartialEq)]
pub struct LocalThreshold {
    pub method: LocalMethod,
    /// Constant subtracted from the local weighted average (`T(x) = local_stat(x) - offset`).
    pub offset: f64,
    pub mode: PadMode,
    pub cval: f64,
}

impl Default for LocalThreshold {
    fn default() -> Self {
        Self { method: LocalMethod::Gaussian, offset: 0.0, mode: PadMode::Reflect, cval: 0.0 }
    }
}

/// Computes the local adaptive threshold surface $T(x) = \text{local\_stat}(x) - \text{offset}$
/// over windows of side `block_size` into `output`.
pub fn local_threshold_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    block_size: usize,
    output: &DeviceArray<F>,
    settings: &LocalThreshold,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let base = ws.array::<F>("local_base", input.shape())?;
    match settings.method {
        LocalMethod::Gaussian => {
            let sigma = (block_size.saturating_sub(1) as f64) / 6.0;
            gaussian_blur_into(
                client,
                input,
                &[sigma.max(0.5)],
                &base,
                &GaussianBlur { mode: settings.mode, cval: settings.cval, ..GaussianBlur::default() },
                ws,
            )?;
        }
        LocalMethod::Mean => {
            box_blur_into(
                client,
                input,
                &[block_size.max(1)],
                &base,
                &BoxBlur { mode: settings.mode, cval: settings.cval, origins: None, axes: None },
                ws,
            )?;
        }
        LocalMethod::Median => {
            let se = StructuringElement::hypercube(input.ndim(), block_size.max(1))?;
            median_filter_into(
                client,
                input,
                &se,
                &base,
                &OrderFilter { mode: settings.mode, cval: settings.cval, origins: None },
                ws,
            )?;
        }
    }
    elementwise::scale_offset_into(client, &base, output, 1.0, -settings.offset)
}

/// Allocates and returns the local adaptive threshold surface.
pub fn local_threshold<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    block_size: usize,
    settings: &LocalThreshold,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    local_threshold_into(client, input, block_size, &output, settings, &mut ws)?;
    Ok(output)
}

/// Settings for [`niblack`] and [`sauvola`] local thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowThreshold {
    pub window_size: usize,
    pub k: f64,
    /// Dynamic range of standard deviation (used by Sauvola; defaults to `0.5` for float images).
    pub r: f64,
    pub mode: PadMode,
    pub cval: f64,
}

impl Default for WindowThreshold {
    fn default() -> Self {
        Self { window_size: 15, k: 0.2, r: 0.5, mode: PadMode::Reflect, cval: 0.0 }
    }
}

/// Computes local mean and mean-of-squares over `window_size` into workspace slots.
fn local_mean_and_mean_sq<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &WindowThreshold,
    mean: &DeviceArray<F>,
    mean_sq: &DeviceArray<F>,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let sq = ws.array::<F>("win_sq", input.shape())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        square_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            sq.binding(),
            input.len() as u32,
        );
    }
    let box_cfg = BoxBlur { mode: settings.mode, cval: settings.cval, origins: None, axes: None };
    box_blur_into(client, input, &[settings.window_size], mean, &box_cfg, ws)?;
    box_blur_into(client, &sq, &[settings.window_size], mean_sq, &box_cfg, ws)
}

/// Niblack local threshold surface $T = \mu - k \sigma$ into `output`.
pub fn niblack_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &WindowThreshold,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let mean = ws.array::<F>("nib_mean", input.shape())?;
    let mean_sq = ws.array::<F>("nib_msq", input.shape())?;
    local_mean_and_mean_sq(client, input, settings, &mean, &mean_sq, ws)?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        niblack_threshold_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            mean.binding(),
            mean_sq.binding(),
            output.binding(),
            cast::<F>(settings.k),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Niblack local threshold surface allocating `output`.
pub fn niblack<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &WindowThreshold,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    niblack_into(client, input, &output, settings, &mut ws)?;
    Ok(output)
}

/// Sauvola local threshold surface $T = \mu \cdot \left(1 + k \left(\frac{\sigma}{R} - 1\right)\right)$
/// into `output`.
pub fn sauvola_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &WindowThreshold,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let mean = ws.array::<F>("sau_mean", input.shape())?;
    let mean_sq = ws.array::<F>("sau_msq", input.shape())?;
    local_mean_and_mean_sq(client, input, settings, &mean, &mean_sq, ws)?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        sauvola_threshold_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            mean.binding(),
            mean_sq.binding(),
            output.binding(),
            cast::<F>(settings.k),
            cast::<F>(1.0 / settings.r.max(1e-12)),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Sauvola local threshold surface allocating `output`.
pub fn sauvola<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &WindowThreshold,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    sauvola_into(client, input, &output, settings, &mut ws)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lucem_core::Shape;

    fn niblack_and_sauvola_on_uniform_image(client: &Client) {
        let shape = Shape::new([9, 9]);
        let img = DeviceArray::<f32>::filled(client, shape.clone(), 10.0).unwrap();
        let cfg = WindowThreshold { window_size: 3, k: 0.2, r: 0.5, mode: PadMode::Reflect, cval: 0.0 };
        let nib = niblack(client, &img, &cfg).unwrap().to_host(client);
        let sau = sauvola(client, &img, &cfg).unwrap().to_host(client);
        assert_close(&nib, &vec![10.0; shape.len()], 1e-5, client.name());
        assert_close(&sau, &vec![10.0 * (1.0 - 0.2); shape.len()], 1e-5, client.name());
    }
    runtime_test!(test_local_thresholds, niblack_and_sauvola_on_uniform_image);
}

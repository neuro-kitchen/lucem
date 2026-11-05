//! 1-D and N-D box (uniform mean) filters.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::kernels::box_blur_1d_kernel;
use crate::core::{check, separable, PadMode};
use crate::error::{OpsError, OpsResult};

/// Settings for [`box_blur_1d`].
#[derive(Debug, Clone, PartialEq)]
pub struct BoxBlur1d {
    pub axis: isize,
    pub mode: PadMode,
    pub cval: f64,
    pub origin: i32,
}

impl Default for BoxBlur1d {
    fn default() -> Self {
        Self { axis: -1, mode: PadMode::Reflect, cval: 0.0, origin: 0 }
    }
}

/// 1-D uniform box filter of window `size` along `settings.axis` into `output`.
pub fn box_blur_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    output: &DeviceArray<F>,
    settings: &BoxBlur1d,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::origin(settings.origin, size)?;
    let axis = input.shape().axis(settings.axis)?;
    if input.is_empty() {
        return Ok(());
    }
    let split = input.shape().axis_split(axis);
    let left = (size / 2) as i32 + settings.origin;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        box_blur_1d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            split.len as u32,
            split.inner as u32,
            left,
            cast::<F>(1.0 / (size as f64)),
            cast::<F>(settings.cval),
            size as u32,
            settings.mode.id(),
        );
    }
    Ok(())
}

/// 1-D box filter allocating `output`.
pub fn box_blur_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    settings: &BoxBlur1d,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    box_blur_1d_into(client, input, size, &output, settings)?;
    Ok(output)
}

/// Settings for N-D separable [`box_blur`].
#[derive(Debug, Clone, PartialEq)]
pub struct BoxBlur {
    pub mode: PadMode,
    pub cval: f64,
    pub origins: Option<Vec<i32>>,
    pub axes: Option<Vec<isize>>,
}

impl Default for BoxBlur {
    fn default() -> Self {
        Self { mode: PadMode::Reflect, cval: 0.0, origins: None, axes: None }
    }
}

/// N-D separable uniform box filter across axes using `ws` ping-pong buffers.
pub fn box_blur_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sizes: &[usize],
    output: &DeviceArray<F>,
    settings: &BoxBlur,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let axes = input.shape().axes(settings.axes.as_deref())?;
    let per_axis_sizes = check::broadcast_per_axis("sizes", sizes, axes.len())?;
    for &s in &per_axis_sizes {
        if s == 0 {
            return Err(OpsError::InvalidParameter {
                name: "sizes",
                reason: "box filter size must be at least 1".to_string(),
            });
        }
    }
    let origins = match &settings.origins {
        Some(o) => check::broadcast_per_axis("origins", o, axes.len())?,
        None => vec![0; axes.len()],
    };
    let passes: Vec<(usize, usize, i32)> = axes
        .into_iter()
        .zip(per_axis_sizes)
        .zip(origins)
        .filter(|&((_, sz), _)| sz > 1)
        .map(|((ax, sz), orig)| (ax, sz, orig))
        .collect();

    separable::run(client, input, output, ws, &passes, |client, src, dst, &(ax, sz, orig)| {
        box_blur_1d_into(
            client,
            src,
            sz,
            dst,
            &BoxBlur1d { axis: ax as isize, mode: settings.mode, cval: settings.cval, origin: orig },
        )
    })
}

/// N-D separable uniform box filter allocating `output`.
pub fn box_blur<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sizes: &[usize],
    settings: &BoxBlur,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    box_blur_into(client, input, sizes, &output, settings, &mut ws)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::convolve::{correlate_1d, Correlate1d};
    use crate::test_support::{assert_close, runtime_test, sample_f64};
    use lumen_core::element::cast_all;
    use lumen_core::Shape;

    fn box_blur_matches_uniform_correlation(client: &Client) {
        let shape = Shape::new([19, 23]);
        let x = sample_f64(shape.len());
        let input = DeviceArray::from_host(client, shape, cast_all::<f32>(&x)).unwrap();
        let got = box_blur_1d(client, &input, 5, &BoxBlur1d::default()).unwrap().to_host(client);
        let want: Vec<f64> = correlate_1d(client, &input, &[0.2; 5], &Correlate1d::default())
            .unwrap()
            .to_host(client)
            .into_iter()
            .map(f64::from)
            .collect();
        assert_close(&got, &want, 1e-5, client.name());
    }
    runtime_test!(test_box_blur, box_blur_matches_uniform_correlation);
}

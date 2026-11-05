//! B-spline prefiltering and N-D coordinate sampling (orders 0–5).

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::check;
use super::elementwise;
use super::kernels::{sample_2d_kernel, spline_pole_1d_kernel};
use super::PadMode;
use crate::error::{OpsError, OpsResult};

/// Returns the B-spline filter poles for interpolation `order` (`2..=5`).
pub fn spline_poles(order: usize) -> OpsResult<&'static [f64]> {
    match order {
        0 | 1 => Ok(&[]),
        2 => Ok(&[-0.171_572_875_253_809_9]), // sqrt(8) - 3
        3 => Ok(&[-0.267_949_192_431_122_7]), // sqrt(3) - 2
        4 => Ok(&[-0.361_341_225_900_220_17, -0.013_725_429_297_339_12]),
        5 => Ok(&[-0.430_575_347_099_973_8, -0.043_096_288_203_264_65]),
        _ => Err(OpsError::InvalidParameter {
            name: "order",
            reason: format!("spline order must be in 0..=5, got {order}"),
        }),
    }
}

/// Applies 1-D B-spline prefiltering of `order` (`2..=5`) along `axis` from `input` into `output`.
pub fn spline_prefilter_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    order: usize,
    axis: isize,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let poles = spline_poles(order)?;
    let ax = input.shape().axis(axis)?;
    elementwise::copy_into(client, input, output)?;
    if input.is_empty() || poles.is_empty() {
        return Ok(());
    }
    let split = output.shape().axis_split(ax);
    if split.len <= 1 {
        return Ok(());
    }
    let lines = split.outer * split.inner;
    let geom = LaunchGeometry::per_line(client, lines);
    for &z in poles {
        let gain = (1.0 - z) * (1.0 - 1.0 / z);
        unsafe {
            spline_pole_1d_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                output.binding(),
                split.len as u32,
                split.inner as u32,
                lines as u32,
                cast::<F>(z),
                cast::<F>(gain),
            );
        }
    }
    Ok(())
}

/// Applies N-D B-spline prefiltering across all axes of `input` into `output`.
pub fn spline_prefilter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    order: usize,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let poles = spline_poles(order)?;
    if poles.is_empty() {
        return elementwise::copy_into(client, input, output);
    }
    let axes: Vec<usize> = (0..input.ndim()).collect();
    super::separable::run(client, input, output, ws, &axes, |client, src, dst, &ax| {
        spline_prefilter_1d_into(client, src, dst, order, ax as isize)
    })
}

/// Samples a 2-D image at device coordinate arrays `(coords_y, coords_x)` into `output`.
#[allow(clippy::too_many_arguments)]
pub fn sample_2d_into<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    coords_y: &DeviceArray<F>,
    coords_x: &DeviceArray<F>,
    output: &DeviceArray<F>,
    order: usize,
    mode: PadMode,
    cval: f64,
    prefilter: bool,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::ndim_in("sample_2d", image.shape(), &[2], "2")?;
    check::same_shape("coords_y", coords_y, "coords_x", coords_x)?;
    check::same_shape("coords_y", coords_y, "output", output)?;
    check::not_aliased(image, output)?;
    let _ = spline_poles(order)?;
    if output.is_empty() {
        return Ok(());
    }

    let coeffs_buf;
    let coeffs = if prefilter && order >= 2 {
        coeffs_buf = ws.array::<F>("spline_coeffs", image.shape())?;
        spline_prefilter_into(client, image, &coeffs_buf, order, ws)?;
        &coeffs_buf
    } else {
        image
    };

    let kernel_order = match order {
        0 => 0u32,
        1 => 1u32,
        _ => 3u32,
    };
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        sample_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            coeffs.binding(),
            coords_y.binding(),
            coords_x.binding(),
            output.binding(),
            image.dims()[0] as i32,
            image.dims()[1] as i32,
            cast::<F>(cval),
            output.len() as u32,
            kernel_order,
            mode.id(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lumen_core::Shape;

    fn bilinear_sampling_matches_analytic(client: &Client) {
        let img = DeviceArray::from_host(client, Shape::new([2, 2]), vec![0.0f32, 10.0, 20.0, 30.0]).unwrap();
        let cy = DeviceArray::from_host(client, Shape::vector(3), vec![0.0f32, 0.5, 1.0]).unwrap();
        let cx = DeviceArray::from_host(client, Shape::vector(3), vec![0.0f32, 0.5, 1.0]).unwrap();
        let out = DeviceArray::<f32>::empty(client, Shape::vector(3)).unwrap();
        let mut ws = Workspace::new(client);
        sample_2d_into(client, &img, &cy, &cx, &out, 1, PadMode::Nearest, 0.0, false, &mut ws).unwrap();
        assert_close(&out.to_host(client), &[0.0, 15.0, 30.0], 1e-6, client.name());
    }
    runtime_test!(test_spline_sampling, bilinear_sampling_matches_analytic);
}

//! Host dispatchers for shared elementwise operations on [`DeviceArray`]s.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat, ImageInt};
use lumen_core::memory::DeviceArray;

use super::check;
use super::kernels::{
    affine_kernel, cast_kernel, clip_kernel, copy_kernel, float_to_uint_kernel, hypot_kernel,
    linear_comb_kernel, lut_kernel, rescale_clip_kernel, threshold_device_kernel,
    threshold_scalar_kernel, uint_to_float_kernel,
};
use crate::error::OpsResult;

/// Copies `input` into `output` on the device.
pub fn copy_into<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    output: &DeviceArray<E>,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        copy_kernel::launch::<E>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Allocates and returns a device copy of `input`.
pub fn copy<E: Numeric + CubeElement>(client: &Client, input: &DeviceArray<E>) -> OpsResult<DeviceArray<E>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    copy_into(client, input, &output)?;
    Ok(output)
}

/// Computes `output = input * scale + offset` on the device.
pub fn scale_offset_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    scale: f64,
    offset: f64,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        affine_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            cast::<F>(scale),
            cast::<F>(offset),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Clamps `input` into `[lo, hi]` and writes into `output`.
pub fn clip_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    lo: f64,
    hi: f64,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        clip_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            cast::<F>(lo),
            cast::<F>(hi),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Linearly maps `[in_lo, in_hi]` to `[out_lo, out_hi]`, clamping values outside the range.
pub fn rescale_clip_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    in_range: (f64, f64),
    out_range: (f64, f64),
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let span = in_range.1 - in_range.0;
    let (scale, offset) = if span.abs() <= f64::EPSILON {
        (0.0, out_range.0)
    } else {
        let s = (out_range.1 - out_range.0) / span;
        (s, out_range.0 - in_range.0 * s)
    };
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        rescale_clip_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            cast::<F>(in_range.0),
            cast::<F>(in_range.1),
            cast::<F>(scale),
            cast::<F>(offset),
            cast::<F>(out_range.0.min(out_range.1)),
            cast::<F>(out_range.0.max(out_range.1)),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Casts elements from `Src` to `Dst` on the device.
pub fn cast_into<Src: Numeric + CubeElement, Dst: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<Src>,
    output: &DeviceArray<Dst>,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        cast_kernel::launch::<Src, Dst>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Allocates and casts elements from `Src` to `Dst` on the device.
pub fn cast<Src: Numeric + CubeElement, Dst: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<Src>,
) -> OpsResult<DeviceArray<Dst>> {
    let output = DeviceArray::<Dst>::empty(client, input.shape().clone())?;
    cast_into(client, input, &output)?;
    Ok(output)
}

/// Alias for [`scale_offset_into`]: `output = input * scale + offset`.
pub fn affine_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    scale: f64,
    offset: f64,
) -> OpsResult<()> {
    scale_offset_into(client, input, output, scale, offset)
}

/// Convenience wrapper for [`compare_threshold_scalar_into`] with `(threshold, inclusive, output)` argument order.
pub fn threshold_scalar_into<F: ImageFloat, Out: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    threshold: f64,
    inclusive: bool,
    output: &DeviceArray<Out>,
) -> OpsResult<()> {
    compare_threshold_scalar_into(client, input, threshold, output, inclusive)
}

/// Converts an unsigned integer image in `0..=max_val` to float in `[0.0, 1.0]`.
pub fn uint_to_unit_float_into<Src: ImageInt, F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<Src>,
    output: &DeviceArray<F>,
    max_val: f64,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        uint_to_float_kernel::launch::<Src, F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            cast::<F>(1.0 / max_val),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Converts a unit float image (`[0.0, 1.0]`, clamped) to unsigned integer `0..=max_val` with nearest
/// rounding.
pub fn unit_float_to_uint_into<F: ImageFloat, Dst: ImageInt>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<Dst>,
    max_val: f64,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        float_to_uint_kernel::launch::<F, Dst>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            cast::<F>(max_val),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Maps integer indices through a device-resident look-up table `lut`.
pub fn apply_lut_into<Idx: ImageInt, Val: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<Idx>,
    lut: &DeviceArray<Val>,
    output: &DeviceArray<Val>,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() || lut.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        lut_kernel::launch::<Idx, Val>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            lut.binding(),
            output.binding(),
            lut.len() as u32,
            input.len() as u32,
        );
    }
    Ok(())
}

/// Compares `input` against a 1-element device array `threshold` (`>` or `>=` when `inclusive`) without
/// synchronizing with the host.
pub fn compare_threshold_device_into<F: ImageFloat, Out: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    threshold: &DeviceArray<F>,
    output: &DeviceArray<Out>,
    inclusive: bool,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        threshold_device_kernel::launch::<F, Out>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            threshold.binding(),
            output.binding(),
            input.len() as u32,
            inclusive,
        );
    }
    Ok(())
}

/// Compares `input` against a host scalar `threshold` (`>` or `>=` when `inclusive`).
pub fn compare_threshold_scalar_into<F: ImageFloat, Out: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    threshold: f64,
    output: &DeviceArray<Out>,
    inclusive: bool,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        threshold_scalar_kernel::launch::<F, Out>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            cast::<F>(threshold),
            output.binding(),
            input.len() as u32,
            inclusive,
        );
    }
    Ok(())
}

/// Computes `output = alpha * a + beta * b + bias` on the device.
pub fn linear_combination_into<F: ImageFloat>(
    client: &Client,
    a: &DeviceArray<F>,
    b: &DeviceArray<F>,
    output: &DeviceArray<F>,
    alpha: f64,
    beta: f64,
    bias: f64,
) -> OpsResult<()> {
    check::same_shape("a", a, "b", b)?;
    check::same_shape("a", a, "output", output)?;
    check::not_aliased(a, output)?;
    check::not_aliased(b, output)?;
    if a.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, a.len());
    unsafe {
        linear_comb_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            a.binding(),
            b.binding(),
            output.binding(),
            cast::<F>(alpha),
            cast::<F>(beta),
            cast::<F>(bias),
            a.len() as u32,
        );
    }
    Ok(())
}

/// Computes `output = sqrt(a^2 + b^2)` on the device.
pub fn hypot_into<F: ImageFloat>(
    client: &Client,
    a: &DeviceArray<F>,
    b: &DeviceArray<F>,
    output: &DeviceArray<F>,
) -> OpsResult<()> {
    check::same_shape("a", a, "b", b)?;
    check::same_shape("a", a, "output", output)?;
    check::not_aliased(a, output)?;
    check::not_aliased(b, output)?;
    if a.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, a.len());
    unsafe {
        hypot_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            a.binding(),
            b.binding(),
            output.binding(),
            a.len() as u32,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test, sample_f64};
    use lumen_core::element::cast_all;
    use lumen_core::Shape;

    fn elementwise_ops_match_host(client: &Client) {
        let shape = Shape::new([17, 29]);
        let x = sample_f64(shape.len());
        let input = DeviceArray::from_host(client, shape.clone(), cast_all::<f32>(&x)).unwrap();
        let output = DeviceArray::<f32>::empty(client, shape.clone()).unwrap();

        scale_offset_into(client, &input, &output, 2.5, -3.0).unwrap();
        let want: Vec<f64> = x.iter().map(|&v| v * 2.5 - 3.0).collect();
        assert_close(&output.to_host(client), &want, 1e-6, client.name());

        clip_into(client, &input, &output, -10.0, 15.0).unwrap();
        let want_clip: Vec<f64> = x.iter().map(|&v| v.clamp(-10.0, 15.0)).collect();
        assert_close(&output.to_host(client), &want_clip, 1e-6, client.name());

        let mask = DeviceArray::<u32>::empty(client, shape).unwrap();
        compare_threshold_scalar_into(client, &input, 0.0, &mask, false).unwrap();
        let want_mask: Vec<u32> = x.iter().map(|&v| u32::from((v as f32) > 0.0)).collect();
        assert_eq!(mask.to_host(client), want_mask, "{}", client.name());
    }
    runtime_test!(test_elementwise_ops, elementwise_ops_match_host);
}

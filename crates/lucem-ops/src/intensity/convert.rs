//! Image dtype conversions, inversion, cropping, and padding on the device.

use cubecl::prelude::*;
use lucem_core::{
    compute::LaunchGeometry,
    element::{cast, ImageFloat, ImageInt},
    memory::DeviceArray,
    Shape,
};

use crate::{
    core::{check, elementwise, PadMode},
    error::{OpsError, OpsResult},
};

use super::kernels::contrast::{crop_3d_kernel, pad_axis_kernel};

/// Convert an unsigned integer `DeviceArray` in `0..=max_val` to a unit floating-point `DeviceArray<F>` in `[0, 1]`.
pub fn to_float<Src: ImageInt, F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<Src>,
    max_val: f64,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    elementwise::uint_to_unit_float_into(client, input, &output, max_val)?;
    Ok(output)
}

/// Rescale a unit floating-point array in `[0, 1]` to `u8` (`[0, 255]`) on the device.
pub fn to_u8<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<DeviceArray<u8>> {
    let output = DeviceArray::<u8>::empty(client, input.shape().clone())?;
    elementwise::unit_float_to_uint_into(client, input, &output, 255.0)?;
    Ok(output)
}

/// Rescale a unit floating-point array in `[0, 1]` to `u16` (`[0, 65535]`) on the device.
pub fn to_u16<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<DeviceArray<u16>> {
    let output = DeviceArray::<u16>::empty(client, input.shape().clone())?;
    elementwise::unit_float_to_uint_into(client, input, &output, 65535.0)?;
    Ok(output)
}

/// Rescale a unit floating-point array in `[0, 1]` to `u32` (`[0, max_val]`) on the device.
pub fn to_u32<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    max_val: f64,
) -> OpsResult<DeviceArray<u32>> {
    let output = DeviceArray::<u32>::empty(client, input.shape().clone())?;
    elementwise::unit_float_to_uint_into(client, input, &output, max_val)?;
    Ok(output)
}

/// Convert a floating-point array to a binary `u8` mask (`1` where `x > 0`, `0` otherwise).
pub fn to_bool<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<DeviceArray<u8>> {
    let output = DeviceArray::<u8>::empty(client, input.shape().clone())?;
    elementwise::compare_threshold_scalar_into(client, input, 0.0, &output, false)?;
    Ok(output)
}

/// Invert a floating-point image around `[0, max_val]` (`max_val - x`).
pub fn invert<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    max_val: f64,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    elementwise::scale_offset_into(client, input, &output, -1.0, max_val)?;
    Ok(output)
}

/// Crop `[(before_0, after_0), ...]` elements from each spatial axis of an N-D array (1-D, 2-D, or 3-D).
pub fn crop<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    crop_widths: &[(usize, usize)],
) -> OpsResult<DeviceArray<E>> {
    let rank = input.ndim();
    check::ndim_in("crop", input.shape(), &[1, 2, 3], "1, 2, or 3")?;
    if crop_widths.len() != rank {
        return Err(OpsError::InvalidParameter {
            name: "crop_widths",
            reason: format!("expected {rank} (before, after) pairs, got {}", crop_widths.len()),
        });
    }
    let dims = input.dims();
    let mut out_dims = Vec::with_capacity(rank);
    for (ax, &(before, after)) in crop_widths.iter().enumerate() {
        if before + after >= dims[ax] {
            return Err(OpsError::InvalidParameter {
                name: "crop_widths",
                reason: format!("crop ({before}, {after}) exceeds axis {ax} length {}", dims[ax]),
            });
        }
        out_dims.push(dims[ax] - before - after);
    }

    let (in_d, in_h, in_w, z0, y0, x0, out_d, out_h, out_w) = match rank {
        1 => (1, 1, dims[0], 0, 0, crop_widths[0].0, 1, 1, out_dims[0]),
        2 => (
            1,
            dims[0],
            dims[1],
            0,
            crop_widths[0].0,
            crop_widths[1].0,
            1,
            out_dims[0],
            out_dims[1],
        ),
        3 => (
            dims[0],
            dims[1],
            dims[2],
            crop_widths[0].0,
            crop_widths[1].0,
            crop_widths[2].0,
            out_dims[0],
            out_dims[1],
            out_dims[2],
        ),
        _ => unreachable!(),
    };

    let out_shape = Shape::new(out_dims);
    let output = DeviceArray::<E>::empty(client, out_shape)?;
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        crop_3d_kernel::launch::<E>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            in_d as u32,
            in_h as u32,
            in_w as u32,
            z0 as u32,
            y0 as u32,
            x0 as u32,
            out_d as u32,
            out_h as u32,
            out_w as u32,
        );
    }
    Ok(output)
}

/// Pad a floating-point array on each spatial axis according to `pad_widths` and `mode`.
pub fn pad<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    pad_widths: &[(usize, usize)],
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>> {
    let rank = input.ndim();
    if pad_widths.len() != rank {
        return Err(OpsError::InvalidParameter {
            name: "pad_widths",
            reason: format!("expected {rank} (before, after) pairs, got {}", pad_widths.len()),
        });
    }
    if input.is_empty() {
        return Err(OpsError::EmptyInput("pad"));
    }

    let mut current = elementwise::copy(client, input)?;
    for (ax, &(before, after)) in pad_widths.iter().enumerate() {
        if before == 0 && after == 0 {
            continue;
        }
        let split = current.shape().axis_split(ax);
        let out_len = split.len + before + after;
        let mut next_dims = current.dims().to_vec();
        next_dims[ax] = out_len;
        let next = DeviceArray::<F>::empty(client, Shape::new(next_dims))?;
        let geom = LaunchGeometry::elementwise(client, next.len());
        unsafe {
            pad_axis_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                current.binding(),
                next.binding(),
                split.len as u32,
                out_len as u32,
                split.inner as u32,
                before as i32,
                cast::<F>(cval),
                next.len() as u32,
                mode.id(),
            );
        }
        current = next;
    }
    Ok(current)
}

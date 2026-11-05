//! Binary mathematical morphology: erosion, dilation, opening, closing, hit-or-miss, and hole filling.

use std::collections::BTreeSet;

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::memory::{DeviceArray, Workspace};

use super::kernels::{binary_1d_kernel, binary_2d_footprint_kernel, hit_or_miss_combine_kernel};
use crate::core::{check, components, elementwise, StructuringElement};
use crate::error::{OpsError, OpsResult};

/// Settings for binary morphology operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryMorphology {
    pub iterations: usize,
    pub border_value: bool,
    pub origins: Option<Vec<i32>>,
}

impl Default for BinaryMorphology {
    fn default() -> Self {
        Self { iterations: 1, border_value: false, origins: None }
    }
}

fn single_binary_step(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    output: &DeviceArray<u32>,
    border_value: bool,
    origins: &[i32],
    is_dilation: bool,
) -> OpsResult<()> {
    let border_val = u32::from(border_value);
    if footprint.is_rectangle() && footprint.ndim() == 1 {
        let size = footprint.dims()[0];
        let left = (size / 2) as i32 + origins[0];
        let geom = LaunchGeometry::elementwise(client, input.len());
        unsafe {
            binary_1d_kernel::launch(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                input.len() as u32,
                1,
                left,
                border_val,
                size as u32,
                is_dilation,
            );
        }
        return Ok(());
    }
    check::ndim_in("binary_morphology", input.shape(), &[2], "2")?;
    let se = if is_dilation { footprint.mirrored() } else { footprint.clone() };
    let eff_origins: Vec<i32> = if is_dilation {
        origins
            .iter()
            .zip(footprint.dims())
            .map(|(&o, &sz)| -o - i32::from(sz.is_multiple_of(2)))
            .collect()
    } else {
        origins.to_vec()
    };
    let offsets = se.device_offsets(client, Some(&eff_origins));
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        binary_2d_footprint_kernel::launch(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            offsets.binding(),
            input.dims()[0] as i32,
            input.dims()[1] as i32,
            se.count_active() as u32,
            border_val,
            is_dilation,
        );
    }
    Ok(())
}

/// Binary erosion (`is_dilation = false`) or dilation (`is_dilation = true`) for `settings.iterations`
/// steps using `ws` ping-pong slots.
pub fn binary_step_into(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    output: &DeviceArray<u32>,
    settings: &BinaryMorphology,
    is_dilation: bool,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if footprint.ndim() != input.ndim() {
        return Err(OpsError::InvalidDimension {
            what: "footprint",
            expected: "matching input ndim",
            actual: footprint.ndim(),
            shape: footprint.dims().to_vec(),
        });
    }
    if input.is_empty() || settings.iterations == 0 {
        return elementwise::copy_into(client, input, output);
    }
    let origins = match &settings.origins {
        Some(o) => check::broadcast_per_axis("origins", o, input.ndim())?,
        None => vec![0; input.ndim()],
    };
    if settings.iterations == 1 {
        return single_binary_step(
            client,
            input,
            footprint,
            output,
            settings.border_value,
            &origins,
            is_dilation,
        );
    }
    let ping = ws.array::<u32>("bin_ping", input.shape())?;
    single_binary_step(client, input, footprint, &ping, settings.border_value, &origins, is_dilation)?;
    if settings.iterations == 2 {
        return single_binary_step(
            client,
            &ping,
            footprint,
            output,
            settings.border_value,
            &origins,
            is_dilation,
        );
    }
    let pong = ws.array::<u32>("bin_pong", input.shape())?;
    let mut src_is_ping = true;
    for _ in 1..settings.iterations - 1 {
        if src_is_ping {
            single_binary_step(client, &ping, footprint, &pong, settings.border_value, &origins, is_dilation)?;
        } else {
            single_binary_step(client, &pong, footprint, &ping, settings.border_value, &origins, is_dilation)?;
        }
        src_is_ping = !src_is_ping;
    }
    let last_src = if src_is_ping { &ping } else { &pong };
    single_binary_step(client, last_src, footprint, output, settings.border_value, &origins, is_dilation)
}

/// Binary erosion allocating `output`.
pub fn binary_erode(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    settings: &BinaryMorphology,
) -> OpsResult<DeviceArray<u32>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    binary_step_into(client, input, footprint, &output, settings, false, &mut ws)?;
    Ok(output)
}

/// Binary dilation allocating `output`.
pub fn binary_dilate(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    settings: &BinaryMorphology,
) -> OpsResult<DeviceArray<u32>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    binary_step_into(client, input, footprint, &output, settings, true, &mut ws)?;
    Ok(output)
}

/// Binary opening (erosion followed by dilation).
pub fn binary_open(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    settings: &BinaryMorphology,
) -> OpsResult<DeviceArray<u32>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    let tmp = ws.array::<u32>("bopen_tmp", input.shape())?;
    binary_step_into(client, input, footprint, &tmp, settings, false, &mut ws)?;
    binary_step_into(client, &tmp, footprint, &output, settings, true, &mut ws)?;
    Ok(output)
}

/// Binary closing (dilation followed by erosion).
pub fn binary_close(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    settings: &BinaryMorphology,
) -> OpsResult<DeviceArray<u32>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    let tmp = ws.array::<u32>("bclose_tmp", input.shape())?;
    binary_step_into(client, input, footprint, &tmp, settings, true, &mut ws)?;
    binary_step_into(client, &tmp, footprint, &output, settings, false, &mut ws)?;
    Ok(output)
}

/// Binary hit-or-miss transform matching foreground `fg_structure` and background `bg_structure`.
pub fn hit_or_miss(
    client: &Client,
    input: &DeviceArray<u32>,
    fg_structure: &StructuringElement,
    bg_structure: &StructuringElement,
) -> OpsResult<DeviceArray<u32>> {
    let eroded_fg = binary_erode(client, input, fg_structure, &BinaryMorphology::default())?;
    let host_in = input.to_host(client);
    let comp_in: Vec<u32> = host_in.into_iter().map(|v| u32::from(v == 0)).collect();
    let comp_dev = DeviceArray::from_host(client, input.shape().clone(), comp_in)?;
    let eroded_bg = binary_erode(
        client,
        &comp_dev,
        bg_structure,
        &BinaryMorphology { border_value: true, ..BinaryMorphology::default() },
    )?;
    let output = DeviceArray::<u32>::empty(client, input.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        hit_or_miss_combine_kernel::launch(
            client,
            geom.cube_count,
            geom.cube_dim,
            eroded_fg.binding(),
            eroded_bg.binding(),
            output.binding(),
            input.len() as u32,
        );
    }
    Ok(output)
}

/// Fills internal background (`0`) holes enclosed by foreground (`!= 0`) pixels: labels background
/// components and fills every background component that does not touch the array border.
pub fn fill_holes(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<u32>> {
    if input.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, input.shape().clone())?);
    }
    let host = input.to_host(client);
    let inv: Vec<u32> = host.iter().map(|&v| u32::from(v == 0)).collect();
    let inv_dev = DeviceArray::from_host(client, input.shape().clone(), inv)?;
    let cc = components::label_components(client, &inv_dev, footprint)?;
    let labels = cc.labels.to_host(client);
    let dims = input.dims();
    let strides = input.shape().strides();

    let mut border_labels = BTreeSet::new();
    for (idx, &lbl) in labels.iter().enumerate() {
        if lbl == 0 {
            continue;
        }
        let on_border = (0..dims.len()).any(|d| {
            let c = (idx / strides[d]) % dims[d];
            c == 0 || c + 1 == dims[d]
        });
        if on_border {
            border_labels.insert(lbl);
        }
    }

    let filled: Vec<u32> = labels
        .into_iter()
        .map(|l| u32::from(l == 0 || !border_labels.contains(&l)))
        .collect();
    Ok(DeviceArray::from_host(client, input.shape().clone(), filled)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lumen_core::Shape;

    fn fills_interior_hole_and_erodes(client: &Client) {
        #[rustfmt::skip]
        let ring = vec![
            0u32, 0, 0, 0, 0,
            0,    1, 1, 1, 0,
            0,    1, 0, 1, 0,
            0,    1, 1, 1, 0,
            0,    0, 0, 0, 0,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([5, 5]), ring).unwrap();
        let filled = fill_holes(client, &arr, None).unwrap();
        assert_eq!(filled.to_host(client)[12], 1, "center hole filled ({})", client.name());

        let se = StructuringElement::connectivity(2, 1).unwrap();
        let eroded = binary_erode(client, &filled, &se, &BinaryMorphology::default()).unwrap().to_host(client);
        assert_eq!(eroded[12], 1, "center survives 3x3 cross erosion ({})", client.name());
        assert_eq!(eroded[6], 0, "corner eroded ({})", client.name());
    }
    runtime_test!(test_binary_morphology, fills_interior_hole_and_erodes);
}

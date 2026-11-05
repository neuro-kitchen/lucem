//! Local windowed-histogram filters over a [`StructuringElement`] for quantized (`u8`/`u16`/`u32`)
//! images.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::ImageFloat;
use lumen_core::memory::DeviceArray;

use super::kernels::{windowed_entropy_2d_kernel, windowed_stat_2d_kernel};
use crate::core::{check, PadMode, StructuringElement};
use crate::error::OpsResult;

/// Local statistic computed over the quantized sliding neighborhood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowedHistStat {
    Mean = 0,
    Min = 1,
    Max = 2,
    Gradient = 3,
    Autolevel = 4,
    Equalize = 5,
}

/// Applies a local windowed-histogram statistic (`Mean`, `Min`, `Max`, `Gradient`, `Autolevel`, or
/// `Equalize`) over `footprint` for a 2-D `u32` quantized image (`0..=max_bin`).
pub fn windowed_hist_stat_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    stat: WindowedHistStat,
    max_bin: u32,
) -> OpsResult<()> {
    check::ndim_in("windowed_hist_stat", input.shape(), &[2], "2")?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let offsets = footprint.device_offsets(client, None);
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        windowed_stat_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            offsets.binding(),
            input.dims()[0] as i32,
            input.dims()[1] as i32,
            footprint.count_active() as u32,
            max_bin,
            stat as u32,
        );
    }
    Ok(())
}

/// Allocates and computes a local windowed-histogram statistic over `footprint`.
pub fn windowed_hist_stat<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    stat: WindowedHistStat,
    max_bin: u32,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    windowed_hist_stat_into(client, input, footprint, &output, stat, max_bin)?;
    Ok(output)
}

/// Computes local Shannon entropy (in bits, base 2) over `footprint` into `output`.
pub fn windowed_entropy_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    mode: PadMode,
) -> OpsResult<()> {
    check::ndim_in("windowed_entropy", input.shape(), &[2], "2")?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let offsets = footprint.device_offsets(client, None);
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        windowed_entropy_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            offsets.binding(),
            input.dims()[0] as i32,
            input.dims()[1] as i32,
            footprint.count_active() as u32,
            mode.id(),
        );
    }
    Ok(())
}

/// Allocates and computes local Shannon entropy (in bits) over `footprint`.
pub fn windowed_entropy<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    mode: PadMode,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    windowed_entropy_into(client, input, footprint, &output, mode)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lumen_core::Shape;

    fn entropy_zero_on_uniform_and_one_on_binary_checkerboard(client: &Client) {
        let se = StructuringElement::rectangle([2, 2]).unwrap();
        let flat = DeviceArray::from_host(client, Shape::new([4, 4]), vec![5u32; 16]).unwrap();
        let ent0: Vec<f32> = windowed_entropy(client, &flat, &se, PadMode::Nearest).unwrap().to_host(client);
        assert_close(&ent0, &vec![0.0; 16], 1e-6, client.name());
    }
    runtime_test!(test_windowed_entropy, entropy_zero_on_uniform_and_one_on_binary_checkerboard);
}

//! Prefix sums (scans) and N-D integral scans on [`DeviceArray`]s.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::ImageFloat;
use lumen_core::memory::{DeviceArray, Workspace};

use super::check;
use super::kernels::scan_axis_kernel;
use super::separable;
use crate::error::OpsResult;

/// Computes an inclusive (`inclusive = true`) or exclusive (`inclusive = false`) prefix sum along
/// `axis` into `output`.
pub fn scan_axis_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    axis: isize,
    inclusive: bool,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let ax = input.shape().axis(axis)?;
    if input.is_empty() {
        return Ok(());
    }
    let split = input.shape().axis_split(ax);
    let lines = split.outer * split.inner;
    let geom = LaunchGeometry::per_line(client, lines);
    unsafe {
        scan_axis_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            split.len as u32,
            split.inner as u32,
            lines as u32,
            inclusive,
        );
    }
    Ok(())
}

/// Computes an N-D inclusive prefix sum (integral image / summed-area table) across every axis of
/// `input` into `output` using `ws` ping-pong buffers.
pub fn integral_scan_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let axes: Vec<usize> = (0..input.ndim()).collect();
    separable::run(client, input, output, ws, &axes, |client, src, dst, &ax| {
        scan_axis_into(client, src, dst, ax as isize, true)
    })
}

/// Allocates and returns the N-D integral image of `input`.
pub fn integral_scan<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    integral_scan_into(client, input, &output, &mut ws)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lumen_core::Shape;

    fn integral_scan_matches_host(client: &Client) {
        let shape = Shape::new([4, 5]);
        let host: Vec<f32> = (1..=20).map(|i| i as f32).collect();
        let input = DeviceArray::from_host(client, shape, host).unwrap();
        let got = integral_scan(client, &input).unwrap().to_host(client);
        let mut want = vec![0.0f64; 20];
        for r in 0..4 {
            for c in 0..5 {
                let mut sum = 0.0;
                for rr in 0..=r {
                    for cc in 0..=c {
                        sum += (rr * 5 + cc + 1) as f64;
                    }
                }
                want[r * 5 + c] = sum;
            }
        }
        assert_close(&got, &want, 1e-6, client.name());
    }
    runtime_test!(test_integral_scan, integral_scan_matches_host);
}

//! Global and per-axis reductions on [`DeviceArray`]s.
//!
//! Device-resident reductions (`*_device`) return 1-element [`DeviceArray`]s so subsequent kernels
//! consume the reduced scalar without synchronizing with the host.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{to_f64, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::check;
use super::kernels::{
    reduce_axis_kernel, row_argmin_argmax_kernel, row_mean_var_kernel, row_sum_min_max_kernel,
};
use crate::error::{OpsError, OpsResult};

/// Summary statistics of an array (`sum`, `min`, `max`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SummaryStats {
    pub sum: f64,
    pub min: f64,
    pub max: f64,
}

/// Computes `(sum, min, max)` per row of a `[rows, cols]` view into 1-D device arrays of length `rows`.
pub fn row_sum_min_max_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    rows: usize,
    cols: usize,
    out_sum: &DeviceArray<F>,
    out_min: &DeviceArray<F>,
    out_max: &DeviceArray<F>,
) -> OpsResult<()> {
    if rows == 0 || cols == 0 {
        return Err(OpsError::EmptyInput("row_sum_min_max"));
    }
    let geom = LaunchGeometry::per_row(client, rows, cols);
    let units = geom.cube_dim.x;
    unsafe {
        row_sum_min_max_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            out_sum.binding(),
            out_min.binding(),
            out_max.binding(),
            rows as u32,
            cols as u32,
            units,
        );
    }
    Ok(())
}

/// Computes global `(sum, min, max)` as three 1-element [`DeviceArray`]s (no host read).
pub fn sum_min_max_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>, DeviceArray<F>)> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("sum_min_max"));
    }
    let scalar = Shape::vector(1);
    let out_sum = DeviceArray::<F>::empty(client, scalar.clone())?;
    let out_min = DeviceArray::<F>::empty(client, scalar.clone())?;
    let out_max = DeviceArray::<F>::empty(client, scalar)?;
    row_sum_min_max_into(client, input, 1, input.len(), &out_sum, &out_min, &out_max)?;
    Ok((out_sum, out_min, out_max))
}

/// Host summary `(sum, min, max)` of `input`.
pub fn sum_min_max<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<SummaryStats> {
    let (s, mn, mx) = sum_min_max_device(client, input)?;
    Ok(SummaryStats {
        sum: to_f64(s.to_host(client)[0]),
        min: to_f64(mn.to_host(client)[0]),
        max: to_f64(mx.to_host(client)[0]),
    })
}

/// Computes `(mean, variance)` per row of `[rows, cols]` into 1-D device arrays of length `rows`.
pub fn row_mean_var_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    rows: usize,
    cols: usize,
    ddof: usize,
    out_mean: &DeviceArray<F>,
    out_var: &DeviceArray<F>,
) -> OpsResult<()> {
    if rows == 0 || cols == 0 {
        return Err(OpsError::EmptyInput("row_mean_var"));
    }
    let geom = LaunchGeometry::per_row(client, rows, cols);
    let units = geom.cube_dim.x;
    unsafe {
        row_mean_var_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            out_mean.binding(),
            out_var.binding(),
            rows as u32,
            cols as u32,
            ddof as u32,
            units,
        );
    }
    Ok(())
}

/// Computes global `(mean, variance)` as two 1-element [`DeviceArray`]s (no host read).
pub fn mean_var_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    ddof: usize,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("mean_var"));
    }
    let scalar = Shape::vector(1);
    let out_mean = DeviceArray::<F>::empty(client, scalar.clone())?;
    let out_var = DeviceArray::<F>::empty(client, scalar)?;
    row_mean_var_into(client, input, 1, input.len(), ddof, &out_mean, &out_var)?;
    Ok((out_mean, out_var))
}

/// Host `(mean, variance)` of `input`.
pub fn mean_var<F: ImageFloat>(client: &Client, input: &DeviceArray<F>, ddof: usize) -> OpsResult<(f64, f64)> {
    let (m, v) = mean_var_device(client, input, ddof)?;
    Ok((to_f64(m.to_host(client)[0]), to_f64(v.to_host(client)[0])))
}

/// Computes global `(argmin, argmax)` flat indices as 1-element `u32` device arrays.
pub fn argmin_argmax_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<(DeviceArray<u32>, DeviceArray<u32>)> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("argmin_argmax"));
    }
    let scalar = Shape::vector(1);
    let out_min = DeviceArray::<u32>::empty(client, scalar.clone())?;
    let out_max = DeviceArray::<u32>::empty(client, scalar)?;
    let geom = LaunchGeometry::per_row(client, 1, input.len());
    let units = geom.cube_dim.x;
    unsafe {
        row_argmin_argmax_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            out_min.binding(),
            out_max.binding(),
            1,
            input.len() as u32,
            units,
        );
    }
    Ok((out_min, out_max))
}

/// Host `(argmin, argmax)` flat indices of `input` (first-index tie rule).
pub fn argmin_argmax<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<(usize, usize)> {
    let (mn, mx) = argmin_argmax_device(client, input)?;
    Ok((mn.to_host(client)[0] as usize, mx.to_host(client)[0] as usize))
}

/// Axis reduction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum = 0,
    Mean = 1,
    Min = 2,
    Max = 3,
}

/// Reduces `input` along `axis` into `output` (whose shape removes `axis`).
pub fn reduce_axis_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    axis: isize,
    op: ReduceOp,
) -> OpsResult<()> {
    let ax = input.shape().axis(axis)?;
    let split = input.shape().axis_split(ax);
    if split.len == 0 {
        return Err(OpsError::EmptyInput("reduce_axis"));
    }
    check::not_aliased(input, output)?;
    let out_len = split.outer * split.inner;
    if output.len() != out_len {
        return Err(OpsError::ShapeMismatch {
            first_name: "reduced shape",
            first: vec![out_len],
            second_name: "output",
            second: output.dims().to_vec(),
        });
    }
    if out_len == 0 {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, out_len);
    unsafe {
        reduce_axis_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            split.len as u32,
            split.inner as u32,
            out_len as u32,
            op as u32,
        );
    }
    Ok(())
}

/// Reduces `input` along `axis`, allocating the reduced output array.
pub fn reduce_axis<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    axis: isize,
    op: ReduceOp,
    _ws: &mut Workspace,
) -> OpsResult<DeviceArray<F>> {
    let ax = input.shape().axis(axis)?;
    let mut out_dims: Vec<usize> = input
        .dims()
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| (i != ax).then_some(d))
        .collect();
    if out_dims.is_empty() {
        out_dims.push(1);
    }
    let output = DeviceArray::<F>::empty(client, Shape::new(out_dims))?;
    reduce_axis_into(client, input, &output, axis, op)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test, sample_f64};
    use lumen_core::element::cast_all;

    fn reductions_match_f64_reference(client: &Client) {
        for n in [1usize, 13, 1024, 10_007] {
            let x = sample_f64(n);
            let arr = DeviceArray::from_host(client, Shape::vector(n), cast_all::<f32>(&x)).unwrap();
            let stats = sum_min_max(client, &arr).unwrap();
            let want_sum: f64 = x.iter().map(|&v| v as f32 as f64).sum();
            let want_min = x.iter().map(|&v| v as f32 as f64).fold(f64::INFINITY, f64::min);
            let want_max = x.iter().map(|&v| v as f32 as f64).fold(f64::NEG_INFINITY, f64::max);
            assert_close(&[stats.sum], &[want_sum], 1e-5, client.name());
            assert_close(&[stats.min], &[want_min], 1e-6, client.name());
            assert_close(&[stats.max], &[want_max], 1e-6, client.name());

            let (mean, var) = mean_var(client, &arr, 0).unwrap();
            let want_mean = want_sum / (n as f64);
            let want_var: f64 = x.iter().map(|&v| (v as f32 as f64 - want_mean).powi(2)).sum::<f64>() / (n as f64);
            assert_close(&[mean], &[want_mean], 1e-5, client.name());
            assert_close(&[var], &[want_var], 1e-4, client.name());
        }
    }
    runtime_test!(test_reductions_match_host, reductions_match_f64_reference);
}

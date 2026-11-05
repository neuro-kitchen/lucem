//! Contrast stretching, gamma, logarithmic, and sigmoid intensity adjustments, and low-contrast detection.

use cubecl::prelude::*;
use lumen_core::{
    compute::LaunchGeometry,
    element::{cast, ImageFloat},
    memory::DeviceArray,
};

use crate::{
    core::{elementwise, histogram, reduce},
    error::{OpsError, OpsResult},
};

use super::kernels::contrast::{
    adjust_gamma_kernel, adjust_log_kernel, adjust_sigmoid_kernel, rescale_range_device_kernel,
};

/// Input range specification for [`rescale_range`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntensityRange {
    /// Automatically stretch from the image's minimum to maximum intensity on the device.
    Image,
    /// Stretch between explicit `(min, max)` bounds.
    Explicit(f64, f64),
}

/// Rescale image intensities from `in_range` into `(out_min, out_max)` on the device.
pub fn rescale_range<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    in_range: IntensityRange,
    out_min: f64,
    out_max: f64,
) -> OpsResult<DeviceArray<F>> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("rescale_range"));
    }
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;

    match in_range {
        IntensityRange::Explicit(in_min, in_max) => {
            elementwise::rescale_clip_into(client, input, &output, (in_min, in_max), (out_min, out_max))?;
        }
        IntensityRange::Image => {
            let (_sum, in_min_dev, in_max_dev) = reduce::sum_min_max_device(client, input)?;
            let geom = LaunchGeometry::elementwise(client, input.len());
            unsafe {
                rescale_range_device_kernel::launch::<F>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    input.binding(),
                    in_min_dev.binding(),
                    in_max_dev.binding(),
                    output.binding(),
                    cast::<F>(out_min),
                    cast::<F>(out_max),
                    input.len() as u32,
                );
            }
        }
    }
    Ok(output)
}

/// Power-law (gamma) transformation: `gain * (x ^ gamma)`.
pub fn adjust_gamma<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    gamma: f64,
    gain: f64,
) -> OpsResult<DeviceArray<F>> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("adjust_gamma"));
    }
    if gamma < 0.0 || !gamma.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "gamma",
            reason: format!("gamma must be non-negative and finite, got {gamma}"),
        });
    }
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        adjust_gamma_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            cast::<F>(gamma),
            cast::<F>(gain),
            input.len() as u32,
        );
    }
    Ok(output)
}

/// Logarithmic intensity adjustment: `gain * log2(1 + x)` (or `gain * (2^x - 1)` when `inv` is true).
pub fn adjust_log<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    gain: f64,
    inv: bool,
) -> OpsResult<DeviceArray<F>> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("adjust_log"));
    }
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        adjust_log_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            cast::<F>(gain),
            u32::from(inv),
            input.len() as u32,
        );
    }
    Ok(output)
}

/// Sigmoid contrast adjustment: `1 / (1 + exp(gain * (cutoff - x)))`.
pub fn adjust_sigmoid<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    cutoff: f64,
    gain: f64,
    inv: bool,
) -> OpsResult<DeviceArray<F>> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("adjust_sigmoid"));
    }
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        adjust_sigmoid_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            cast::<F>(cutoff),
            cast::<F>(gain),
            u32::from(inv),
            input.len() as u32,
        );
    }
    Ok(output)
}

/// Determine whether an image has low contrast by comparing its `[lower_percentile, upper_percentile]`
/// intensity span to `fraction_threshold` of the dynamic range.
pub fn is_low_contrast<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    fraction_threshold: f64,
    lower_percentile: f64,
    upper_percentile: f64,
) -> OpsResult<bool> {
    let stats = reduce::sum_min_max(client, input)?;
    let (v_min, v_max) = (stats.min, stats.max);
    if (v_max - v_min).abs() <= 1e-12 {
        return Ok(true);
    }
    let num_bins = 256usize;
    let hist = histogram::histogram_ranged_device(client, input, num_bins, (v_min, v_max))?;
    let counts = hist.to_host(client);
    let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    if total == 0 {
        return Ok(true);
    }
    let lo_target = ((lower_percentile / 100.0).clamp(0.0, 1.0) * total as f64).ceil() as u64;
    let hi_target = ((upper_percentile / 100.0).clamp(0.0, 1.0) * total as f64).ceil() as u64;
    let mut cum = 0u64;
    let mut lo_bin = 0usize;
    let mut hi_bin = num_bins - 1;
    let mut found_lo = false;
    for (i, &c) in counts.iter().enumerate() {
        cum += u64::from(c);
        if !found_lo && cum >= lo_target {
            lo_bin = i;
            found_lo = true;
        }
        if cum >= hi_target {
            hi_bin = i;
            break;
        }
    }
    let lo_val = v_min + (lo_bin as f64 / (num_bins - 1) as f64) * (v_max - v_min);
    let hi_val = v_min + (hi_bin as f64 / (num_bins - 1) as f64) * (v_max - v_min);
    let full_range = v_max.abs().max(1.0);
    Ok((hi_val - lo_val) / full_range < fraction_threshold)
}

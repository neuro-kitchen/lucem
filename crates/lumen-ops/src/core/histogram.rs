//! Global and ranged histograms on the device.
//!
//! Uses `u32` atomic increments (exact and order-independent) and returns counts in a
//! [`DeviceArray<u32>`] along with bin centers.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat, ImageInt};
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;

use super::kernels::{histogram_float_kernel, histogram_uint_kernel};
use super::reduce;
use crate::error::{OpsError, OpsResult};

/// Computes a `nbins`-bin histogram of `input` over `[min_val, max_val]` on the device.
pub fn histogram_ranged_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
    range: (f64, f64),
) -> OpsResult<DeviceArray<u32>> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("histogram"));
    }
    if nbins == 0 {
        return Err(OpsError::InvalidParameter {
            name: "nbins",
            reason: "nbins must be at least 1".to_string(),
        });
    }
    let bins = DeviceArray::<u32>::zeros(client, Shape::vector(nbins))?;
    let span = (range.1 - range.0).max(f64::EPSILON);
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        histogram_float_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            bins.binding(),
            cast::<F>(range.0),
            cast::<F>(1.0 / span),
            nbins as u32,
            input.len() as u32,
        );
    }
    Ok(bins)
}

/// Computes a histogram and bin centers over `range` (or `(min, max)` of `input` when `None`).
pub fn histogram<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
    range: Option<(f64, f64)>,
) -> OpsResult<(DeviceArray<u32>, Vec<f64>)> {
    let (lo, hi) = match range {
        Some(r) => r,
        None => {
            let stats = reduce::sum_min_max(client, input)?;
            (stats.min, stats.max)
        }
    };
    let counts = histogram_ranged_device(client, input, nbins, (lo, hi))?;
    let step = if hi > lo { (hi - lo) / (nbins as f64) } else { 1.0 };
    let centers: Vec<f64> = (0..nbins).map(|b| lo + (b as f64 + 0.5) * step).collect();
    Ok((counts, centers))
}

/// Direct integer histogram of unsigned values in `0..nbins`.
pub fn histogram_uint_device<I: ImageInt>(
    client: &Client,
    input: &DeviceArray<I>,
    nbins: usize,
) -> OpsResult<DeviceArray<u32>> {
    if nbins == 0 {
        return Err(OpsError::InvalidParameter {
            name: "nbins",
            reason: "nbins must be at least 1".to_string(),
        });
    }
    let bins = DeviceArray::<u32>::zeros(client, Shape::vector(nbins))?;
    if input.is_empty() {
        return Ok(bins);
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        histogram_uint_kernel::launch::<I>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            bins.binding(),
            nbins as u32,
            input.len() as u32,
        );
    }
    Ok(bins)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{runtime_test, sample_f64};
    use lumen_core::element::cast_all;

    fn histogram_counts_sum_to_length(client: &Client) {
        let x = sample_f64(4099);
        let arr = DeviceArray::from_host(client, Shape::vector(x.len()), cast_all::<f32>(&x)).unwrap();
        let (counts, centers) = histogram(client, &arr, 64, Some((-60.0, 60.0))).unwrap();
        let host_counts = counts.to_host(client);
        assert_eq!(centers.len(), 64);
        assert_eq!(host_counts.iter().sum::<u32>() as usize, x.len(), "{}", client.name());
    }
    runtime_test!(test_histogram, histogram_counts_sum_to_length);
}

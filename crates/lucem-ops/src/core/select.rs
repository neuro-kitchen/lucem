//! Exact $k$-th smallest element and percentile selection.
//!
//! Uses a 4-pass 256-bin IEEE-754 radix selection on the device for `f32` (or exact host selection
//! for small arrays / `f64`), matching linear percentile interpolation.

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast_all, to_f64, ImageFloat};
use lucem_core::memory::DeviceArray;
use lucem_core::Shape;

use super::kernels::radix_pass_hist_kernel;
use crate::error::{OpsError, OpsResult};

/// Converts an `f32` bit pattern to a monotone `u32` key and back.
fn key_to_f32(key: u32) -> f32 {
    let bits = if (key >> 31) == 1 { key ^ 0x8000_0000 } else { !key };
    f32::from_bits(bits)
}

/// Selects the exact `k`-th smallest `f32` element (`0 <= k < input.len()`) using 4-pass 8-bit
/// radix histograms on the device.
pub fn kth_f32(client: &Client, input: &DeviceArray<f32>, k: usize) -> OpsResult<f32> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("kth_f32"));
    }
    let k = k.min(input.len() - 1) as u32;
    let geom = LaunchGeometry::elementwise(client, input.len());
    let mut prefix = 0u32;
    let mut mask = 0u32;
    let mut remaining = k;

    for pass in (0..4u32).rev() {
        let shift = pass * 8;
        let bins = DeviceArray::<u32>::zeros(client, Shape::vector(256))?;
        unsafe {
            radix_pass_hist_kernel::launch(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                input.binding(),
                bins.binding(),
                prefix,
                mask,
                shift,
                input.len() as u32,
            );
        }
        let counts = bins.to_host(client);
        let mut chosen = 0u32;
        for (b, &c) in counts.iter().enumerate() {
            if remaining < c {
                chosen = b as u32;
                break;
            }
            remaining -= c;
        }
        prefix |= chosen << shift;
        mask |= 0xFFu32 << shift;
    }
    Ok(key_to_f32(prefix))
}

/// Computes exact percentiles (`q` in `0.0..=100.0`) with linear interpolation.
pub fn percentiles<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    qs: &[f64],
) -> OpsResult<Vec<f64>> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("percentiles"));
    }
    for &q in qs {
        if !(0.0..=100.0).contains(&q) || !q.is_finite() {
            return Err(OpsError::InvalidParameter {
                name: "q",
                reason: format!("percentile {q} must be in 0.0..=100.0"),
            });
        }
    }
    let mut vals: Vec<f64> = input.to_host(client).into_iter().map(to_f64).collect();
    vals.sort_by(f64::total_cmp);
    let n = vals.len();
    Ok(qs
        .iter()
        .map(|&q| {
            if n == 1 {
                return vals[0];
            }
            let pos = (q / 100.0) * ((n - 1) as f64);
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            let frac = pos - (lo as f64);
            vals[lo] + frac * (vals[hi] - vals[lo])
        })
        .collect())
}

/// Computes exact percentiles and uploads them as a 1-D [`DeviceArray`].
pub fn percentiles_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    qs: &[f64],
) -> OpsResult<DeviceArray<F>> {
    let host = percentiles(client, input, qs)?;
    Ok(DeviceArray::from_host(client, Shape::vector(host.len()), cast_all::<F>(&host))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test, sample_f64};

    fn radix_and_percentiles_match_sorted(client: &Client) {
        let x = sample_f64(2053);
        let arr = DeviceArray::from_host(client, Shape::vector(x.len()), cast_all::<f32>(&x)).unwrap();
        let mut sorted: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        sorted.sort_by(f32::total_cmp);

        for k in [0usize, 1, 500, 1026, 2052] {
            let got = kth_f32(client, &arr, k).unwrap();
            assert_eq!(got, sorted[k], "{}", client.name());
        }

        let p = percentiles(client, &arr, &[0.0, 25.0, 50.0, 75.0, 100.0]).unwrap();
        assert_close(&[p[0] as f32], &[sorted[0] as f64], 1e-6, client.name());
        assert_close(&[p[2] as f32], &[sorted[1026] as f64], 1e-6, client.name());
        assert_close(&[p[4] as f32], &[sorted[2052] as f64], 1e-6, client.name());
    }
    runtime_test!(test_select_and_percentiles, radix_and_percentiles_match_sorted);
}

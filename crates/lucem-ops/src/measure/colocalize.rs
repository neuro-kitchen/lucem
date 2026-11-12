//! Fluorescence colocalization coefficients: Pearson's $r$, Manders' split colocalization
//! coefficients $(M_1, M_2)$, Manders' overlap coefficient $R$, and binary intersection coefficient.

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast, ImageFloat};
use lucem_core::memory::{DeviceArray, Workspace};

use super::kernels::{colocalize_terms_kernel, manders_gated_terms_kernel};
use crate::core::{check, compact, reduce};
use crate::error::{OpsError, OpsResult};

/// Computes Pearson's linear correlation coefficient $r \in [-1, 1]$ between two identically
/// shaped intensity channels `channel_a` and `channel_b`.
pub fn pearson_r<F: ImageFloat>(
    client: &Client,
    channel_a: &DeviceArray<F>,
    channel_b: &DeviceArray<F>,
) -> OpsResult<f64> {
    check::same_shape("channel_a", channel_a, "channel_b", channel_b)?;
    if channel_a.is_empty() {
        return Err(OpsError::EmptyInput("pearson_r"));
    }
    let (ma, _) = reduce::mean_var(client, channel_a, 0)?;
    let (mb, _) = reduce::mean_var(client, channel_b, 0)?;

    let mut ws = Workspace::new(client);
    let ab = ws.array::<F>("coloc_ab", channel_a.shape())?;
    let a2 = ws.array::<F>("coloc_a2", channel_a.shape())?;
    let b2 = ws.array::<F>("coloc_b2", channel_a.shape())?;
    let geom = LaunchGeometry::elementwise(client, channel_a.len());
    unsafe {
        colocalize_terms_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            channel_a.binding(),
            channel_b.binding(),
            ab.binding(),
            a2.binding(),
            b2.binding(),
            cast::<F>(ma),
            cast::<F>(mb),
            channel_a.len() as u32,
        );
    }
    let cov = reduce::sum_min_max(client, &ab)?.sum;
    let va = reduce::sum_min_max(client, &a2)?.sum;
    let vb = reduce::sum_min_max(client, &b2)?.sum;
    let denom = (va * vb).sqrt();
    if denom <= f64::EPSILON {
        Ok(0.0)
    } else {
        Ok((cov / denom).clamp(-1.0, 1.0))
    }
}

/// Computes Manders' split colocalization coefficients `(M1, M2)`:
/// - `M1`: fraction of `channel_a` intensity in pixels where `channel_b > threshold_b`
/// - `M2`: fraction of `channel_b` intensity in pixels where `channel_a > threshold_a`
pub fn manders_coeff<F: ImageFloat>(
    client: &Client,
    channel_a: &DeviceArray<F>,
    channel_b: &DeviceArray<F>,
    threshold_a: f64,
    threshold_b: f64,
) -> OpsResult<(f64, f64)> {
    check::same_shape("channel_a", channel_a, "channel_b", channel_b)?;
    if channel_a.is_empty() {
        return Err(OpsError::EmptyInput("manders_coeff"));
    }
    let mut ws = Workspace::new(client);
    let a_gated = ws.array::<F>("manders_a", channel_a.shape())?;
    let b_gated = ws.array::<F>("manders_b", channel_a.shape())?;
    let geom = LaunchGeometry::elementwise(client, channel_a.len());
    unsafe {
        manders_gated_terms_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            channel_a.binding(),
            channel_b.binding(),
            a_gated.binding(),
            b_gated.binding(),
            cast::<F>(threshold_a),
            cast::<F>(threshold_b),
            channel_a.len() as u32,
        );
    }
    let sum_a = reduce::sum_min_max(client, channel_a)?.sum;
    let sum_b = reduce::sum_min_max(client, channel_b)?.sum;
    let num_a = reduce::sum_min_max(client, &a_gated)?.sum;
    let num_b = reduce::sum_min_max(client, &b_gated)?.sum;

    let m1 = if sum_a.abs() > f64::EPSILON { num_a / sum_a } else { 0.0 };
    let m2 = if sum_b.abs() > f64::EPSILON { num_b / sum_b } else { 0.0 };
    Ok((m1, m2))
}

/// Computes Manders' uncentered overlap coefficient
/// $R = \frac{\sum I_A I_B}{\sqrt{\sum I_A^2 \sum I_B^2}}$.
pub fn manders_overlap<F: ImageFloat>(
    client: &Client,
    channel_a: &DeviceArray<F>,
    channel_b: &DeviceArray<F>,
) -> OpsResult<f64> {
    check::same_shape("channel_a", channel_a, "channel_b", channel_b)?;
    if channel_a.is_empty() {
        return Err(OpsError::EmptyInput("manders_overlap"));
    }
    let mut ws = Workspace::new(client);
    let ab = ws.array::<F>("overlap_ab", channel_a.shape())?;
    let a2 = ws.array::<F>("overlap_a2", channel_a.shape())?;
    let b2 = ws.array::<F>("overlap_b2", channel_a.shape())?;
    let geom = LaunchGeometry::elementwise(client, channel_a.len());
    unsafe {
        colocalize_terms_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            channel_a.binding(),
            channel_b.binding(),
            ab.binding(),
            a2.binding(),
            b2.binding(),
            cast::<F>(0.0),
            cast::<F>(0.0),
            channel_a.len() as u32,
        );
    }
    let sum_ab = reduce::sum_min_max(client, &ab)?.sum;
    let sum_a2 = reduce::sum_min_max(client, &a2)?.sum;
    let sum_b2 = reduce::sum_min_max(client, &b2)?.sum;
    let denom = (sum_a2 * sum_b2).sqrt();
    if denom <= f64::EPSILON {
        Ok(0.0)
    } else {
        Ok((sum_ab / denom).clamp(-1.0, 1.0))
    }
}

/// Computes the binary intersection (pixel-overlap) fraction $|A \cap B| / |A|$ between two binary
/// masks `mask_a` and `mask_b` (non-zero treated as foreground).
pub fn intersection_coeff<E: Numeric + CubeElement>(
    client: &Client,
    mask_a: &DeviceArray<E>,
    mask_b: &DeviceArray<E>,
) -> OpsResult<f64> {
    check::same_shape("mask_a", mask_a, "mask_b", mask_b)?;
    if mask_a.is_empty() {
        return Err(OpsError::EmptyInput("intersection_coeff"));
    }
    let fa = compact::flag_nonzero(client, mask_a)?.to_host(client);
    let fb = compact::flag_nonzero(client, mask_b)?.to_host(client);
    let mut count_a = 0usize;
    let mut both = 0usize;
    for (&a, &b) in fa.iter().zip(fb.iter()) {
        if a != 0 {
            count_a += 1;
            if b != 0 {
                both += 1;
            }
        }
    }
    if count_a == 0 {
        Ok(0.0)
    } else {
        Ok((both as f64) / (count_a as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lucem_core::Shape;

    fn colocalization_metrics_match_analytic(client: &Client) {
        let a = DeviceArray::from_host(client, Shape::vector(4), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b = DeviceArray::from_host(client, Shape::vector(4), vec![2.0f32, 4.0, 6.0, 8.0]).unwrap();
        assert_close(&[pearson_r(client, &a, &b).unwrap()], &[1.0], 1e-6, client.name());
        assert_close(&[manders_overlap(client, &a, &b).unwrap()], &[1.0], 1e-6, client.name());

        let (m1, m2) = manders_coeff(client, &a, &b, 2.5, 5.0).unwrap();
        // b > 5.0 at indices 2, 3 -> a sum = 3 + 4 = 7 / 10 = 0.7
        // a > 2.5 at indices 2, 3 -> b sum = 6 + 8 = 14 / 20 = 0.7
        assert_close(&[m1, m2], &[0.7, 0.7], 1e-6, client.name());

        let ma = DeviceArray::from_host(client, Shape::vector(4), vec![1u32, 1, 0, 0]).unwrap();
        let mb = DeviceArray::from_host(client, Shape::vector(4), vec![0u32, 1, 1, 0]).unwrap();
        assert_close(&[intersection_coeff(client, &ma, &mb).unwrap()], &[0.5], 1e-6, client.name());
    }
    runtime_test!(test_colocalize, colocalization_metrics_match_analytic);
}

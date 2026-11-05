//! Seeded counter-based random number generation on the device.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat};
use lumen_core::memory::DeviceArray;

use super::kernels::{random_normal_kernel, random_uniform_kernel};
use crate::error::OpsResult;

/// Fills `output` with deterministic uniform random values in `[0.0, 1.0)` seeded by `seed`.
pub fn fill_uniform<F: ImageFloat>(
    client: &Client,
    output: &DeviceArray<F>,
    seed: u64,
) -> OpsResult<()> {
    if output.is_empty() {
        return Ok(());
    }
    let s = (seed as u32) ^ ((seed >> 32) as u32).wrapping_mul(0x9E37_79B9);
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        random_uniform_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            output.binding(),
            s,
            output.len() as u32,
        );
    }
    Ok(())
}

/// Fills `output` with deterministic Gaussian $\mathcal{N}(\text{mean}, \text{std\_dev}^2)$ values
/// seeded by `seed`.
pub fn fill_normal<F: ImageFloat>(
    client: &Client,
    output: &DeviceArray<F>,
    mean: f64,
    std_dev: f64,
    seed: u64,
) -> OpsResult<()> {
    if output.is_empty() {
        return Ok(());
    }
    let s = (seed as u32) ^ ((seed >> 32) as u32).wrapping_mul(0x9E37_79B9);
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        random_normal_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            output.binding(),
            cast::<F>(mean),
            cast::<F>(std_dev),
            s,
            output.len() as u32,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::reduce;
    use crate::test_support::{assert_close, runtime_test};
    use lumen_core::Shape;

    fn normal_rng_matches_moments(client: &Client) {
        let out = DeviceArray::<f32>::empty(client, Shape::vector(16_384)).unwrap();
        fill_normal(client, &out, 2.0, 0.5, 42).unwrap();
        let (mean, var) = reduce::mean_var(client, &out, 0).unwrap();
        assert_close(&[mean as f32], &[2.0], 0.05, client.name());
        assert_close(&[var as f32], &[0.25], 0.1, client.name());
    }
    runtime_test!(test_random_normal, normal_rng_matches_moments);
}

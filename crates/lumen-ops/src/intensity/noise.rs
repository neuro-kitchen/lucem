//! Stochastic synthetic noise generation on the device via counter-based PCG/Box-Muller PRNG.

use cubecl::prelude::*;
use lumen_core::{
    compute::LaunchGeometry,
    element::{cast, ImageFloat},
    memory::DeviceArray,
};

use crate::error::{OpsError, OpsResult};

use super::kernels::noise::{
    add_gaussian_noise_kernel, add_impulse_noise_kernel, add_poisson_noise_kernel,
    add_speckle_noise_kernel,
};

/// Noise distribution model for [`add_noise`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseModel {
    /// Additive Gaussian white noise with `(mean, std)`.
    Gaussian { mean: f64, std: f64 },
    /// Multiplicative speckle noise `x + x * N(mean, std^2)`.
    Speckle { mean: f64, std: f64 },
    /// Impulse salt noise replacing `amount` fraction of pixels with `1.0`.
    Salt { amount: f64 },
    /// Impulse pepper noise replacing `amount` fraction of pixels with `0.0`.
    Pepper { amount: f64 },
    /// Impulse salt-and-pepper noise replacing `amount` fraction of pixels with `1.0` (`salt_vs_pepper`) or `0.0`.
    SaltAndPepper { amount: f64, salt_vs_pepper: f64 },
    /// Shot / Poisson noise scaled by `scale` photons per unit intensity.
    Poisson { scale: f64 },
}

/// Add synthetic random noise to an image on the device.
pub fn add_noise<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    model: NoiseModel,
    seed: u64,
    clip: bool,
) -> OpsResult<DeviceArray<F>> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("add_noise"));
    }
    let total = input.len();
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, total);
    let seed_u32 = (seed as u32) ^ ((seed >> 32) as u32).wrapping_mul(0x9E37_79B9);
    let clip_u32 = u32::from(clip);

    match model {
        NoiseModel::Gaussian { mean, std } => unsafe {
            add_gaussian_noise_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                seed_u32,
                cast::<F>(mean),
                cast::<F>(std),
                clip_u32,
                total as u32,
            );
        },
        NoiseModel::Speckle { mean, std } => unsafe {
            add_speckle_noise_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                seed_u32,
                cast::<F>(mean),
                cast::<F>(std),
                clip_u32,
                total as u32,
            );
        },
        NoiseModel::Salt { amount } => unsafe {
            add_impulse_noise_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                seed_u32,
                amount as f32,
                1.0f32,
                cast::<F>(0.0),
                cast::<F>(1.0),
                total as u32,
            );
        },
        NoiseModel::Pepper { amount } => unsafe {
            add_impulse_noise_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                seed_u32,
                amount as f32,
                0.0f32,
                cast::<F>(0.0),
                cast::<F>(1.0),
                total as u32,
            );
        },
        NoiseModel::SaltAndPepper {
            amount,
            salt_vs_pepper,
        } => unsafe {
            add_impulse_noise_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                seed_u32,
                amount as f32,
                salt_vs_pepper as f32,
                cast::<F>(0.0),
                cast::<F>(1.0),
                total as u32,
            );
        },
        NoiseModel::Poisson { scale } => unsafe {
            add_poisson_noise_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                seed_u32,
                cast::<F>(scale.max(1.0)),
                clip_u32,
                total as u32,
            );
        },
    }
    Ok(output)
}

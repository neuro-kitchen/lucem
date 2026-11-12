//! Counter-based pseudo-random number generator kernels (PCG32 / Box-Muller) on the device.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// Hash function mapping `(index, seed)` to a well-mixed `u32`.
#[cube]
pub fn pcg_hash(input: u32) -> u32 {
    let state = input * 747_796_405u32 + 2_891_336_453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277_803_737u32;
    (word >> 22u32) ^ word
}

/// Generates uniform `F` values in `[0.0, 1.0)` from a deterministic counter hash of `(i, seed)`.
#[cube(launch)]
pub fn random_uniform_kernel<F: lucem_core::element::ImageFloat>(output: &mut [F], seed: u32, len: u32) {
    let i = element_position();
    if (i as u32) < len {
        let h = pcg_hash((i as u32) ^ seed);
        let u = (h as f32) * (1.0f32 / 4_294_967_296.0f32);
        output[i] = F::cast_from(u);
    }
}

/// Generates standard normal $\mathcal{N}(\mu, \sigma^2)$ values using counter-based Box-Muller.
#[cube(launch)]
pub fn random_normal_kernel<F: lucem_core::element::ImageFloat>(output: &mut [F], mean: F, std_dev: F, seed: u32, len: u32) {
    let i = element_position();
    if (i as u32) < len {
        let idx = i as u32;
        let h1 = pcg_hash((idx * 2u32) ^ seed);
        let h2 = pcg_hash((idx * 2u32 + 1u32) ^ seed);
        let u1 = f32::max((h1 as f32) * (1.0f32 / 4_294_967_296.0f32), 1e-7f32);
        let u2 = (h2 as f32) * (1.0f32 / 4_294_967_296.0f32);
        let z = f32::sqrt(-2.0f32 * f32::ln(u1)) * f32::cos(6.283_185_5f32 * u2);
        output[i] = mean + std_dev * F::cast_from(z);
    }
}

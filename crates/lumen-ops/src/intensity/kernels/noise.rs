//! GPU kernels for stochastic image noise synthesis (Gaussian, Poisson, Salt & Pepper, Speckle).

use cubecl::prelude::*;
use lumen_core::compute::element_position;
use crate::core::kernels::random::pcg_hash;

#[cube]
fn uniform_f32(seed: u32, idx: u32) -> f32 {
    let h = pcg_hash(idx ^ seed);
    (h as f32) * (1.0f32 / 4_294_967_296.0f32)
}

#[cube]
fn normal_f32(seed: u32, idx: u32) -> f32 {
    let u1 = f32::max(uniform_f32(seed, idx * 2u32), 1e-7f32);
    let u2 = uniform_f32(seed, idx * 2u32 + 1u32);
    f32::sqrt(-2.0f32 * f32::ln(u1)) * f32::cos(6.283_185_5f32 * u2)
}

/// Add additive Gaussian noise `x + mean + std * N(0, 1)`.
#[cube(launch)]
pub(crate) fn add_gaussian_noise_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    seed: u32,
    mean: F,
    std: F,
    clip: u32,
    total: u32,
) {
    let idx = element_position() as u32;
    if idx < total {
        let z = normal_f32(seed, idx);
        let mut val = input[idx as usize] + mean + std * F::cast_from(z);
        if clip != 0u32 {
            val = F::min(F::max(val, F::new(0.0)), F::new(1.0));
        }
        output[idx as usize] = val;
    }
}

/// Add multiplicative speckle noise `x + x * (mean + std * N(0, 1))`.
#[cube(launch)]
pub(crate) fn add_speckle_noise_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    seed: u32,
    mean: F,
    std: F,
    clip: u32,
    total: u32,
) {
    let idx = element_position() as u32;
    if idx < total {
        let z = normal_f32(seed, idx);
        let x = input[idx as usize];
        let mut val = x + x * (mean + std * F::cast_from(z));
        if clip != 0u32 {
            val = F::min(F::max(val, F::new(0.0)), F::new(1.0));
        }
        output[idx as usize] = val;
    }
}

/// Add impulse (salt, pepper, or salt-and-pepper) noise.
#[cube(launch)]
pub(crate) fn add_impulse_noise_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    seed: u32,
    amount: f32,
    salt_vs_pepper: f32,
    low_val: F,
    high_val: F,
    total: u32,
) {
    let idx = element_position() as u32;
    if idx < total {
        let u = uniform_f32(seed, idx * 2u32);
        let v = uniform_f32(seed, idx * 2u32 + 1u32);
        let mut val = input[idx as usize];
        if u < amount {
            if v < salt_vs_pepper {
                val = high_val;
            } else {
                val = low_val;
            }
        }
        output[idx as usize] = val;
    }
}

/// Add Poisson shot noise parameterized by `scale` (photons at intensity 1.0).
#[cube(launch)]
pub(crate) fn add_poisson_noise_kernel<F: lumen_core::element::ImageFloat>(
    input: &[F],
    output: &mut [F],
    seed: u32,
    scale: F,
    clip: u32,
    total: u32,
) {
    let idx = element_position() as u32;
    if idx < total {
        let x = F::max(input[idx as usize], F::new(0.0));
        let lambda = x * scale;
        let mut val = if lambda <= F::new(0.0) {
            F::new(0.0)
        } else if lambda < F::new(20.0) {
            let l = f32::exp(-f32::cast_from(lambda));
            let mut k = 0u32;
            let mut p = 1.0f32;
            while k < 128u32 {
                k += 1u32;
                let u = f32::max(uniform_f32(seed, idx * 131u32 + k), 1e-7f32);
                p = p * u;
                if p <= l {
                    break;
                }
            }
            F::cast_from(k - 1u32) / scale
        } else {
            let z = normal_f32(seed, idx);
            let sample = F::max(
                F::floor(lambda + F::sqrt(lambda) * F::cast_from(z) + F::new(0.5)),
                F::new(0.0),
            );
            sample / scale
        };
        if clip != 0u32 {
            val = F::min(F::max(val, F::new(0.0)), F::new(1.0));
        }
        output[idx as usize] = val;
    }
}

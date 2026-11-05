//! Device kernels for adaptive local thresholds (Niblack, Sauvola, and local offset thresholding).

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Squares each element: `output[i] = input[i] * input[i]`.
#[cube(launch)]
pub fn square_kernel<F: Float>(input: &[F], output: &mut [F], len: u32) {
    let i = element_position();
    if (i as u32) < len {
        let v = input[i];
        output[i] = v * v;
    }
}

/// Computes local Niblack threshold surface `T[i] = mean[i] - k * sqrt(max(mean_sq[i] - mean[i]^2, 0))`.
#[cube(launch)]
pub fn niblack_threshold_kernel<F: Float>(
    mean: &[F],
    mean_sq: &[F],
    output: &mut [F],
    k: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let m = mean[i];
        let var = F::max(mean_sq[i] - m * m, F::new(0.0f32));
        let std = F::sqrt(var);
        output[i] = m - k * std;
    }
}

/// Computes local Sauvola threshold surface `T[i] = mean[i] * (1 + k * (std[i] / r - 1))`.
#[cube(launch)]
pub fn sauvola_threshold_kernel<F: Float>(
    mean: &[F],
    mean_sq: &[F],
    output: &mut [F],
    k: F,
    inv_r: F,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let m = mean[i];
        let var = F::max(mean_sq[i] - m * m, F::new(0.0f32));
        let std = F::sqrt(var);
        let one = F::new(1.0f32);
        output[i] = m * (one + k * (std * inv_r - one));
    }
}

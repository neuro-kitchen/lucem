//! Device kernels for computing global thresholds from a histogram without host round-trips,
//! and for applying device-resident thresholds.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Computes the exact Otsu threshold on the device from `counts[0..nbins]` over `[min_val, max_val]`
/// and writes the scalar threshold into `out_thresh[0]`.
#[cube(launch)]
pub fn otsu_from_hist_kernel<F: lumen_core::element::ImageFloat>(
    counts: &[u32],
    out_thresh: &mut [F],
    min_val: F,
    step: F,
    nbins: u32,
) {
    let idx = element_position();
    if idx == 0usize {
        let mut total_w = 0.0f32;
        let mut total_mu = 0.0f32;
        let min_f = f32::cast_from(min_val);
        let step_f = f32::cast_from(step);
        let mut b = 0u32;
        while b < nbins {
            let c = counts[b as usize] as f32;
            let center = min_f + ((b as f32) + 0.5f32) * step_f;
            total_w += c;
            total_mu += c * center;
            b += 1u32;
        }

        let mut w0 = 0.0f32;
        let mut mu0_sum = 0.0f32;
        let mut best_var = f32::new(-1.0);
        let mut best_bin = 0u32;
        let mut t = 0u32;
        while t + 1u32 < nbins {
            let c = counts[t as usize] as f32;
            let center = min_f + ((t as f32) + 0.5f32) * step_f;
            w0 += c;
            mu0_sum += c * center;
            let w1 = total_w - w0;
            if w0 > 0.0f32 && w1 > 0.0f32 {
                let mu0 = mu0_sum / w0;
                let mu1 = (total_mu - mu0_sum) / w1;
                let diff = mu0 - mu1;
                let between = w0 * w1 * diff * diff;
                if between > best_var {
                    best_var = between;
                    best_bin = t;
                }
            }
            t += 1u32;
        }
        let chosen = min_f + ((best_bin as f32) + 0.5f32) * step_f;
        out_thresh[0] = F::cast_from(chosen);
    }
}

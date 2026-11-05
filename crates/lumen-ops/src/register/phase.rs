//! Subpixel N-D Fourier phase cross-correlation image registration (`phase_correlation`).
//!
//! Implements Guizar-Sicairos, Thurman & Fienup's (2008) efficient subpixel image registration by
//! combining whole-array N-D FFT cross-correlation with local frequency-domain DFT upsampling around
//! the initial integer correlation peak.

use std::f64::consts::PI;

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, to_f64, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::kernels::{complex_mag_sq_kernel, cross_power_spectrum_kernel};
use crate::core::{check, fft, reduce};
use crate::error::{OpsError, OpsResult};

/// Settings for [`phase_correlation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseCorrelation {
    /// Subpixel upsampling factor (`1` for integer-pixel registration; `> 1` for subpixel accuracy
    /// of `1.0 / upsample_factor` pixels).
    pub upsample_factor: usize,
    /// When `true`, normalizes the cross-power spectrum by its pointwise magnitude (`R / |R|`, phase
    /// correlation); when `false`, uses unnormalized cross-correlation (`R = F_ref * conj(F_mov)`).
    pub normalize_phase: bool,
}

impl Default for PhaseCorrelation {
    fn default() -> Self {
        Self {
            upsample_factor: 1,
            normalize_phase: true,
        }
    }
}

/// Output of [`phase_correlation`].
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseCorrelationResult {
    /// Shift vector (in pixels along each axis) required to register `moving` with `reference`.
    pub shift: Vec<f64>,
    /// Translation-invariant normalized RMS registration error in `[0.0, 1.0]`.
    pub error: f64,
    /// Global phase difference (in radians, `[-pi, pi]`) between the two images at the peak.
    pub phasediff: f64,
}

/// Estimates the N-D translation shift (with optional `1 / upsample_factor` subpixel precision) that
/// registers `moving` onto `reference`.
pub fn phase_correlation<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    settings: &PhaseCorrelation,
) -> OpsResult<PhaseCorrelationResult> {
    check::same_shape("reference", reference, "moving", moving)?;
    if reference.is_empty() {
        return Err(OpsError::EmptyInput("phase_correlation"));
    }
    if settings.upsample_factor == 0 {
        return Err(OpsError::InvalidParameter {
            name: "upsample_factor",
            reason: "upsample_factor must be >= 1".to_string(),
        });
    }

    let shape = reference.shape();
    let dims = shape.dims();
    let strides = shape.strides();
    let ndim = dims.len();
    let n_total = shape.len();
    let mut ws = Workspace::new(client);

    let zero_im = DeviceArray::<F>::zeros(client, shape.clone())?;
    let ref_re = DeviceArray::<F>::empty(client, shape.clone())?;
    let ref_im = DeviceArray::<F>::empty(client, shape.clone())?;
    let mov_re = DeviceArray::<F>::empty(client, shape.clone())?;
    let mov_im = DeviceArray::<F>::empty(client, shape.clone())?;

    fft::fft_nd_into(client, reference, &zero_im, &ref_re, &ref_im, false, &mut ws)?;
    fft::fft_nd_into(client, moving, &zero_im, &mov_re, &mov_im, false, &mut ws)?;

    let prod_re = DeviceArray::<F>::empty(client, shape.clone())?;
    let prod_im = DeviceArray::<F>::empty(client, shape.clone())?;
    let geom = LaunchGeometry::elementwise(client, n_total);
    unsafe {
        cross_power_spectrum_kernel::launch::<F>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim,
            ref_re.binding(),
            ref_im.binding(),
            mov_re.binding(),
            mov_im.binding(),
            prod_re.binding(),
            prod_im.binding(),
            cast::<F>(1.0e-12),
            n_total as u32,
            settings.normalize_phase,
        );
    }

    let cc_re = DeviceArray::<F>::empty(client, shape.clone())?;
    let cc_im = DeviceArray::<F>::empty(client, shape.clone())?;
    fft::fft_nd_into(client, &prod_re, &prod_im, &cc_re, &cc_im, true, &mut ws)?;

    let cc_mag_sq = DeviceArray::<F>::empty(client, shape.clone())?;
    unsafe {
        complex_mag_sq_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            cc_re.binding(),
            cc_im.binding(),
            cc_mag_sq.binding(),
            n_total as u32,
        );
    }

    let (_, max_idx) = reduce::argmin_argmax(client, &cc_mag_sq)?;
    let mut shift: Vec<f64> = (0..ndim)
        .map(|d| {
            let n = dims[d];
            let k = (max_idx / strides[d]) % n;
            if k <= n / 2 {
                k as f64
            } else {
                (k as isize - n as isize) as f64
            }
        })
        .collect();

    let mut peak_re = to_f64(cc_re.to_host(client)[max_idx]);
    let mut peak_im = to_f64(cc_im.to_host(client)[max_idx]);

    // Subpixel refinement around the integer peak via local inverse DFT evaluation
    if settings.upsample_factor > 1 {
        let u = settings.upsample_factor as f64;
        let half_span = (1.5 * u).ceil() as isize;
        let pr_host: Vec<f64> = prod_re.to_host(client).into_iter().map(to_f64).collect();
        let pi_host: Vec<f64> = prod_im.to_host(client).into_iter().map(to_f64).collect();

        // Precompute signed normalized frequencies f_d[k] in [-0.5, 0.5) per axis
        let freqs: Vec<Vec<f64>> = (0..ndim)
            .map(|d| {
                let n = dims[d];
                (0..n)
                    .map(|k| {
                        if k <= n / 2 {
                            (k as f64) / (n as f64)
                        } else {
                            ((k as isize - n as isize) as f64) / (n as f64)
                        }
                    })
                    .collect()
            })
            .collect();

        // Coordinate-wise + joint refinement around current shift
        for pass in 0..2 {
            let radius = if pass == 0 { half_span } else { 2 };
            let step = if pass == 0 { 1.0 / u } else { 0.25 / u };
            for ax in 0..ndim {
                let center_s = shift[ax];
                let mut best_mag2 = -1.0f64;
                let mut best_s = center_s;
                let mut best_re = peak_re;
                let mut best_im = peak_im;

                // Precompute base phase contribution from all axes != ax
                let mut other_cos = vec![1.0f64; n_total];
                let mut other_sin = vec![0.0f64; n_total];
                for (idx, (oc, os)) in other_cos.iter_mut().zip(other_sin.iter_mut()).enumerate() {
                    let mut angle = 0.0;
                    for d in 0..ndim {
                        if d != ax {
                            let k = (idx / strides[d]) % dims[d];
                            angle += 2.0 * PI * freqs[d][k] * shift[d];
                        }
                    }
                    *oc = angle.cos();
                    *os = angle.sin();
                }

                for m in -radius..=radius {
                    let cand_s = center_s + (m as f64) * step;
                    let n_ax = dims[ax];
                    let ax_cos: Vec<f64> = (0..n_ax)
                        .map(|k| (2.0 * PI * freqs[ax][k] * cand_s).cos())
                        .collect();
                    let ax_sin: Vec<f64> = (0..n_ax)
                        .map(|k| (2.0 * PI * freqs[ax][k] * cand_s).sin())
                        .collect();

                    let mut sum_re = 0.0;
                    let mut sum_im = 0.0;
                    for idx in 0..n_total {
                        let k_ax = (idx / strides[ax]) % n_ax;
                        // cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
                        // sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
                        let c = other_cos[idx] * ax_cos[k_ax] - other_sin[idx] * ax_sin[k_ax];
                        let s = other_sin[idx] * ax_cos[k_ax] + other_cos[idx] * ax_sin[k_ax];
                        let r = pr_host[idx];
                        let im = pi_host[idx];
                        // (r + i*im) * (c + i*s)
                        sum_re += r * c - im * s;
                        sum_im += r * s + im * c;
                    }
                    sum_re /= n_total as f64;
                    sum_im /= n_total as f64;
                    let mag2 = sum_re * sum_re + sum_im * sum_im;
                    if mag2 > best_mag2 {
                        best_mag2 = mag2;
                        best_s = cand_s;
                        best_re = sum_re;
                        best_im = sum_im;
                    }
                }
                shift[ax] = best_s;
                peak_re = best_re;
                peak_im = best_im;
            }
        }
        // Snap to exact upsampled grid increment `1.0 / upsample_factor`
        for s in &mut shift {
            *s = (*s * u).round() / u;
        }
    }

    let phasediff = peak_im.atan2(peak_re);

    // Compute unnormalized cross-correlation magnitude at the estimated shift for RMS error
    let (ref_mean, ref_var) = reduce::mean_var(client, reference, 0)?;
    let (mov_mean, mov_var) = reduce::mean_var(client, moving, 0)?;
    let ref_power = (ref_var + ref_mean * ref_mean) * (n_total as f64);
    let mov_power = (mov_var + mov_mean * mov_mean) * (n_total as f64);
    let denom = ref_power * mov_power;
    let error = if denom <= f64::EPSILON {
        0.0
    } else if !settings.normalize_phase {
        let cc_amp2 = (peak_re * peak_re + peak_im * peak_im) * (n_total as f64) * (n_total as f64);
        (1.0 - (cc_amp2 / denom).clamp(0.0, 1.0)).sqrt()
    } else {
        // Evaluate unnormalized peak amplitude from ref & mov spectra
        let unnorm_re = DeviceArray::<F>::empty(client, shape.clone())?;
        let unnorm_im = DeviceArray::<F>::empty(client, shape.clone())?;
        fft::complex_mul_into(client, &ref_re, &ref_im, &mov_re, &mov_im, &unnorm_re, &unnorm_im, true)?;
        let unnorm_cc_re = DeviceArray::<F>::empty(client, shape.clone())?;
        let unnorm_cc_im = DeviceArray::<F>::empty(client, shape.clone())?;
        fft::fft_nd_into(client, &unnorm_re, &unnorm_im, &unnorm_cc_re, &unnorm_cc_im, true, &mut ws)?;
        let ur = to_f64(unnorm_cc_re.to_host(client)[max_idx]);
        let ui = to_f64(unnorm_cc_im.to_host(client)[max_idx]);
        let cc_amp2 = (ur * ur + ui * ui) * (n_total as f64) * (n_total as f64);
        (1.0 - (cc_amp2 / denom).clamp(0.0, 1.0)).sqrt()
    };

    Ok(PhaseCorrelationResult {
        shift,
        error,
        phasediff,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::spectral::fourier_shift_into;
    use crate::test_support::{assert_close, runtime_test, sample_positive_f64};
    use lumen_core::element::cast_all;
    use lumen_core::Shape;

    fn phase_correlation_recovers_integer_and_subpixel_shifts(client: &Client) {
        let shape = Shape::new([16, 16]);
        let base = sample_positive_f64(shape.len());
        let ref_img = DeviceArray::from_host(client, shape.clone(), cast_all::<f32>(&base)).unwrap();

        // Construct shifted image in Fourier domain with subpixel shift (dy = 2.25, dx = -1.5)
        let mut ws = Workspace::new(client);
        let zero = DeviceArray::<f32>::zeros(client, shape.clone()).unwrap();
        let f_re = DeviceArray::<f32>::empty(client, shape.clone()).unwrap();
        let f_im = DeviceArray::<f32>::empty(client, shape.clone()).unwrap();
        fft::fft_nd_into(client, &ref_img, &zero, &f_re, &f_im, false, &mut ws).unwrap();
        fourier_shift_into(client, &f_re, &f_im, &[2.25, -1.5]).unwrap();
        let mov_re = DeviceArray::<f32>::empty(client, shape.clone()).unwrap();
        let mov_im = DeviceArray::<f32>::empty(client, shape).unwrap();
        fft::fft_nd_into(client, &f_re, &f_im, &mov_re, &mov_im, true, &mut ws).unwrap();

        let res = phase_correlation(
            client,
            &ref_img,
            &mov_re,
            &PhaseCorrelation {
                upsample_factor: 20,
                normalize_phase: true,
            },
        )
        .unwrap();
        // Shift to align `moving` (shifted by +2.25, -1.5) back onto `reference` is (-2.25, +1.5)
        assert_close(&res.shift, &[-2.25, 1.5], 0.1, client.name());
    }
    runtime_test!(test_phase_correlation, phase_correlation_recovers_integer_and_subpixel_shifts);
}

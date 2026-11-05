//! Point-spread-function (PSF) image deconvolution: iterative Richardson-Lucy maximum-likelihood
//! deconvolution (`richardson_lucy`), regularized frequency-domain Wiener deconvolution
//! (`wiener_deconvolve`), and Bayesian Gibbs-sampled unsupervised Wiener deconvolution
//! (`unsupervised_wiener`).

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, cast_all, to_f64, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::kernels::{
    fft_shift_2d_kernel, rl_div_clamp_kernel, rl_mul_clamp_kernel, wiener_freq_filter_kernel,
};
use crate::core::{check, elementwise, fft, reduce, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::filter::convolve::{correlate_2d_into, Correlate2d};

/// Settings for iterative Richardson-Lucy deconvolution ([`richardson_lucy`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RichardsonLucy {
    /// Number of multiplicative expectation-maximization iterations.
    pub num_iter: usize,
    /// Whether to clamp the restored image to `[-1.0, 1.0]`.
    pub clip: bool,
    /// Minimum denominator threshold below which ratios are stabilized (defaults to `1e-12`).
    pub filter_epsilon: Option<f64>,
    /// Spatial boundary padding mode.
    pub mode: PadMode,
}

impl Default for RichardsonLucy {
    fn default() -> Self {
        Self {
            num_iter: 30,
            clip: true,
            filter_epsilon: None,
            mode: PadMode::Reflect,
        }
    }
}

/// Restores `input` blurred by 2-D `psf` into `output` using the Richardson-Lucy (1972/1974)
/// multiplicative update algorithm.
pub fn richardson_lucy_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &RichardsonLucy,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::ndim_in("richardson_lucy input", input.shape(), &[2], "2")?;
    check::ndim_in("richardson_lucy psf", psf.shape(), &[2], "2")?;
    if psf.is_empty() {
        return Err(OpsError::InvalidWeights {
            what: "psf",
            reason: "PSF cannot be empty",
        });
    }
    if settings.num_iter == 0 {
        return Err(OpsError::InvalidParameter {
            name: "num_iter",
            reason: "num_iter must be >= 1".to_string(),
        });
    }
    if input.is_empty() {
        return Ok(());
    }

    // Normalize PSF to sum to 1 (if positive sum) and build spatially reversed PSF for convolution
    let kdims = psf.dims();
    let kh = kdims[0];
    let kw = kdims[1];
    let host_psf: Vec<f64> = psf.to_host(client).into_iter().map(to_f64).collect();
    let psf_sum: f64 = host_psf.iter().sum();
    let norm = if psf_sum.abs() > 1e-12 { 1.0 / psf_sum } else { 1.0 };
    let mut norm_psf = vec![0.0f64; kh * kw];
    let mut rev_psf = vec![0.0f64; kh * kw];
    for y in 0..kh {
        for x in 0..kw {
            let v = host_psf[y * kw + x] * norm;
            norm_psf[y * kw + x] = v;
            rev_psf[(kh - 1 - y) * kw + (kw - 1 - x)] = v;
        }
    }
    let psf_mirror = DeviceArray::from_host(client, psf.shape().clone(), cast_all::<F>(&norm_psf))?;
    let psf_conv = DeviceArray::from_host(client, psf.shape().clone(), cast_all::<F>(&rev_psf))?;

    let shape = input.shape();
    let est_a = DeviceArray::<F>::filled(client, shape.clone(), cast::<F>(0.5))?;
    let est_b = ws.array::<F>("rl_est_b", shape)?;
    let est_conv = ws.array::<F>("rl_est_conv", shape)?;
    let rel_blur = ws.array::<F>("rl_rel_blur", shape)?;
    let err_est = ws.array::<F>("rl_err_est", shape)?;

    let corr_settings = Correlate2d {
        mode: settings.mode,
        cval: 0.0,
        origin: (0, 0),
    };
    let eps = cast::<F>(settings.filter_epsilon.unwrap_or(1.0e-12).max(1.0e-15));
    let total = input.len() as u32;
    let geom = LaunchGeometry::elementwise(client, input.len());

    let mut ping = true;
    for iter in 0..settings.num_iter {
        let is_last = iter + 1 == settings.num_iter;
        let src = if ping { &est_a } else { &est_b };
        let dst = if is_last {
            output
        } else if ping {
            &est_b
        } else {
            &est_a
        };

        correlate_2d_into(client, src, &psf_conv, &est_conv, &corr_settings)?;
        unsafe {
            rl_div_clamp_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                input.binding(),
                est_conv.binding(),
                rel_blur.binding(),
                eps,
                total,
            );
        }
        correlate_2d_into(client, &rel_blur, &psf_mirror, &err_est, &corr_settings)?;
        unsafe {
            rl_mul_clamp_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                src.binding(),
                err_est.binding(),
                dst.binding(),
                total,
                is_last && settings.clip,
            );
        }
        ping = !ping;
    }
    Ok(())
}

/// Allocates and returns Richardson-Lucy deconvolved `input`.
pub fn richardson_lucy<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    settings: &RichardsonLucy,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    richardson_lucy_into(client, input, psf, &output, settings, &mut ws)?;
    Ok(output)
}

/// Settings for regularized frequency-domain Wiener deconvolution ([`wiener_deconvolve`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WienerDeconvolve {
    /// Regularization weight (`balance >= 0`) balancing data fidelity against high-frequency
    /// Laplacian roughness.
    pub balance: f64,
    /// Whether to clamp the restored output to `[-1.0, 1.0]`.
    pub clip: bool,
}

impl Default for WienerDeconvolve {
    fn default() -> Self {
        Self {
            balance: 0.1,
            clip: true,
        }
    }
}

/// Helper: computes the 2-D FFT of a small kernel centered at `(kh/2, kw/2)` inside `target_shape`.
fn centered_kernel_fft_2d<F: ImageFloat>(
    client: &Client,
    kernel: &DeviceArray<F>,
    target_shape: &Shape,
    re_out: &DeviceArray<F>,
    im_out: &DeviceArray<F>,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let padded = DeviceArray::<F>::zeros(client, target_shape.clone())?;
    let zero_im = DeviceArray::<F>::zeros(client, target_shape.clone())?;
    let geom = LaunchGeometry::elementwise(client, kernel.len());
    unsafe {
        fft_shift_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            kernel.binding(),
            padded.binding(),
            target_shape.dims()[0] as u32,
            target_shape.dims()[1] as u32,
            kernel.dims()[0] as u32,
            kernel.dims()[1] as u32,
            kernel.len() as u32,
        );
    }
    fft::fft_nd_into(client, &padded, &zero_im, re_out, im_out, false, ws)
}

/// Deconvolves 2-D `input` with `psf` into `output` using regularized frequency-domain Wiener
/// filtering:
/// $$\hat{X}(\omega) = \frac{\overline{H(\omega)}\,Y(\omega)}{|H(\omega)|^2 + \text{balance}\cdot |L(\omega)|^2}$$
/// where $L$ is the 2-D discrete Laplacian high-pass filter.
pub fn wiener_deconvolve_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &WienerDeconvolve,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::ndim_in("wiener_deconvolve input", input.shape(), &[2], "2")?;
    check::ndim_in("wiener_deconvolve psf", psf.shape(), &[2], "2")?;
    if psf.is_empty() {
        return Err(OpsError::InvalidWeights {
            what: "psf",
            reason: "PSF cannot be empty",
        });
    }
    if !settings.balance.is_finite() || settings.balance < 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "balance",
            reason: format!("balance must be non-negative and finite, got {}", settings.balance),
        });
    }
    if input.is_empty() {
        return Ok(());
    }
    let shape = input.shape();
    if psf.dims()[0] > shape.dims()[0] || psf.dims()[1] > shape.dims()[1] {
        return Err(OpsError::InvalidWeights {
            what: "psf",
            reason: "PSF dimensions must not exceed image dimensions",
        });
    }

    let zero_im = DeviceArray::<F>::zeros(client, shape.clone())?;
    let y_re = ws.array::<F>("wiener_y_re", shape)?;
    let y_im = ws.array::<F>("wiener_y_im", shape)?;
    fft::fft_nd_into(client, input, &zero_im, &y_re, &y_im, false, ws)?;

    let h_re = ws.array::<F>("wiener_h_re", shape)?;
    let h_im = ws.array::<F>("wiener_h_im", shape)?;
    centered_kernel_fft_2d(client, psf, shape, &h_re, &h_im, ws)?;

    // 2-D Laplacian impulse response [0, -1, 0; -1, 4, -1; 0, -1, 0]
    let lap_h = shape.dims()[0].min(3);
    let lap_w = shape.dims()[1].min(3);
    let lap_dev = if lap_h == 3 && lap_w == 3 {
        #[rustfmt::skip]
        let lap = [
            0.0, -1.0,  0.0,
           -1.0,  4.0, -1.0,
            0.0, -1.0,  0.0,
        ];
        DeviceArray::from_host(client, Shape::new([3, 3]), cast_all::<F>(&lap))?
    } else {
        DeviceArray::<F>::filled(client, Shape::new([1, 1]), cast::<F>(1.0))?
    };
    let l_re = ws.array::<F>("wiener_l_re", shape)?;
    let l_im = ws.array::<F>("wiener_l_im", shape)?;
    centered_kernel_fft_2d(client, &lap_dev, shape, &l_re, &l_im, ws)?;

    // |L(f)|^2 = hypot(l_re, l_im)^2; compute via complex_mul(L, conj(L))
    let l_mag_sq = ws.array::<F>("wiener_l_m2", shape)?;
    let l_zero = ws.array::<F>("wiener_l_z", shape)?;
    fft::complex_mul_into(client, &l_re, &l_im, &l_re, &l_im, &l_mag_sq, &l_zero, true)?;

    let x_re = ws.array::<F>("wiener_x_re", shape)?;
    let x_im = ws.array::<F>("wiener_x_im", shape)?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        wiener_freq_filter_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            y_re.binding(),
            y_im.binding(),
            h_re.binding(),
            h_im.binding(),
            l_mag_sq.binding(),
            x_re.binding(),
            x_im.binding(),
            cast::<F>(settings.balance),
            cast::<F>(1.0e-12),
            input.len() as u32,
        );
    }

    let raw_out = if settings.clip {
        ws.array::<F>("wiener_raw", shape)?
    } else {
        output.clone()
    };
    let out_im = ws.array::<F>("wiener_out_im", shape)?;
    fft::fft_nd_into(client, &x_re, &x_im, &raw_out, &out_im, true, ws)?;
    if settings.clip {
        elementwise::clip_into(client, &raw_out, output, -1.0, 1.0)?;
    }
    Ok(())
}

/// Allocates and returns Wiener-deconvolved `input`.
pub fn wiener_deconvolve<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    settings: &WienerDeconvolve,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    wiener_deconvolve_into(client, input, psf, &output, settings, &mut ws)?;
    Ok(output)
}

/// Settings for Bayesian unsupervised Wiener deconvolution ([`unsupervised_wiener`]).
#[derive(Debug, Clone, PartialEq)]
pub struct UnsupervisedWiener {
    /// Total number of Gibbs-sampling / empirical-Bayes iterations.
    pub num_iter: usize,
    /// Number of initial burn-in iterations discarded before averaging.
    pub burnin: usize,
    /// Whether to clamp the final deconvolved image to `[-1.0, 1.0]`.
    pub clip: bool,
    /// Deterministic RNG seed.
    pub seed: u64,
}

impl Default for UnsupervisedWiener {
    fn default() -> Self {
        Self {
            num_iter: 30,
            burnin: 10,
            clip: true,
            seed: 0,
        }
    }
}

/// Unsupervised Bayesian Wiener deconvolution (Orieux, Giovannelli & Rodet, 2010): jointly estimates
/// the restored image and the noise-to-prior regularization ratio `balance = sigma_n^2 / sigma_x^2`
/// across `settings.num_iter` iterations, returning `(deconvolved_image, balance_chain)`.
pub fn unsupervised_wiener<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    settings: &UnsupervisedWiener,
) -> OpsResult<(DeviceArray<F>, Vec<f64>)> {
    check::ndim_in("unsupervised_wiener input", input.shape(), &[2], "2")?;
    check::ndim_in("unsupervised_wiener psf", psf.shape(), &[2], "2")?;
    if psf.is_empty() {
        return Err(OpsError::InvalidWeights {
            what: "psf",
            reason: "PSF cannot be empty",
        });
    }
    if settings.num_iter == 0 || settings.burnin >= settings.num_iter {
        return Err(OpsError::InvalidParameter {
            name: "num_iter",
            reason: format!(
                "num_iter ({}) must be > burnin ({}) and >= 1",
                settings.num_iter, settings.burnin
            ),
        });
    }
    if input.is_empty() {
        return Ok((DeviceArray::<F>::empty(client, input.shape().clone())?, Vec::new()));
    }

    let shape = input.shape();
    let n = input.len() as f64;
    let mut ws = Workspace::new(client);

    let zero_im = DeviceArray::<F>::zeros(client, shape.clone())?;
    let y_re = DeviceArray::<F>::empty(client, shape.clone())?;
    let y_im = DeviceArray::<F>::empty(client, shape.clone())?;
    fft::fft_nd_into(client, input, &zero_im, &y_re, &y_im, false, &mut ws)?;

    let h_re = DeviceArray::<F>::empty(client, shape.clone())?;
    let h_im = DeviceArray::<F>::empty(client, shape.clone())?;
    centered_kernel_fft_2d(client, psf, shape, &h_re, &h_im, &mut ws)?;

    let lap_h = shape.dims()[0].min(3);
    let lap_w = shape.dims()[1].min(3);
    let lap_dev = if lap_h == 3 && lap_w == 3 {
        #[rustfmt::skip]
        let lap = [
            0.0, -1.0,  0.0,
           -1.0,  4.0, -1.0,
            0.0, -1.0,  0.0,
        ];
        DeviceArray::from_host(client, Shape::new([3, 3]), cast_all::<F>(&lap))?
    } else {
        DeviceArray::<F>::filled(client, Shape::new([1, 1]), cast::<F>(1.0))?
    };
    let l_re = DeviceArray::<F>::empty(client, shape.clone())?;
    let l_im = DeviceArray::<F>::empty(client, shape.clone())?;
    centered_kernel_fft_2d(client, &lap_dev, shape, &l_re, &l_im, &mut ws)?;

    let h_mag_sq = DeviceArray::<F>::empty(client, shape.clone())?;
    let l_mag_sq = DeviceArray::<F>::empty(client, shape.clone())?;
    let tmp_zero = DeviceArray::<F>::empty(client, shape.clone())?;
    fft::complex_mul_into(client, &h_re, &h_im, &h_re, &h_im, &h_mag_sq, &tmp_zero, true)?;
    fft::complex_mul_into(client, &l_re, &l_im, &l_re, &l_im, &l_mag_sq, &tmp_zero, true)?;

    // Pull spectral magnitudes and Conj(H)*Y to host for fast exact Gibbs / MAP updates
    let conj_hy_re = DeviceArray::<F>::empty(client, shape.clone())?;
    let conj_hy_im = DeviceArray::<F>::empty(client, shape.clone())?;
    fft::complex_mul_into(client, &y_re, &y_im, &h_re, &h_im, &conj_hy_re, &conj_hy_im, true)?;

    let hy_r: Vec<f64> = conj_hy_re.to_host(client).into_iter().map(to_f64).collect();
    let hy_i: Vec<f64> = conj_hy_im.to_host(client).into_iter().map(to_f64).collect();
    let yr_h: Vec<f64> = y_re.to_host(client).into_iter().map(to_f64).collect();
    let yi_h: Vec<f64> = y_im.to_host(client).into_iter().map(to_f64).collect();
    let hr_h: Vec<f64> = h_re.to_host(client).into_iter().map(to_f64).collect();
    let hi_h: Vec<f64> = h_im.to_host(client).into_iter().map(to_f64).collect();
    let hm2_h: Vec<f64> = h_mag_sq.to_host(client).into_iter().map(to_f64).collect();
    let lm2_h: Vec<f64> = l_mag_sq.to_host(client).into_iter().map(to_f64).collect();

    let (_, img_var) = reduce::mean_var(client, input, 0)?;
    let mut balance = 0.05;
    let mut balance_chain = Vec::with_capacity(settings.num_iter);
    let mut mean_x_re = vec![0.0f64; input.len()];
    let mut mean_x_im = vec![0.0f64; input.len()];
    let mut count_post = 0usize;

    // Deterministic PCG state for hyperparameter jitter
    let mut rng_state = settings.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);

    for iter in 0..settings.num_iter {
        let mut rss_sum = 0.0f64;
        let mut reg_sum = 0.0f64;
        let mut trace_ratio = 0.0f64;

        for k in 0..input.len() {
            let denom = (hm2_h[k] + balance * lm2_h[k]).max(1.0e-12);
            let xr = hy_r[k] / denom;
            let xi = hy_i[k] / denom;

            // Residual Y - H * X
            let hxr = hr_h[k] * xr - hi_h[k] * xi;
            let hxi = hr_h[k] * xi + hi_h[k] * xr;
            let dr = yr_h[k] - hxr;
            let di = yi_h[k] - hxi;
            rss_sum += (dr * dr + di * di) / n;

            let x_norm2 = (xr * xr + xi * xi) / n;
            reg_sum += lm2_h[k] * x_norm2;
            trace_ratio += hm2_h[k] / denom;

            if iter >= settings.burnin {
                mean_x_re[k] += xr;
                mean_x_im[k] += xi;
            }
        }

        if iter >= settings.burnin {
            count_post += 1;
        }

        // Empirical Bayes / Gibbs precision update:
        // sigma_n^2 = rss_sum / (N - trace_ratio), sigma_x^2 = reg_sum / trace_ratio
        let eff_dof = (n - trace_ratio).max(1.0);
        let sigma_n2 = (rss_sum / eff_dof).max(1.0e-8 * img_var.max(1.0e-6));
        let sigma_x2 = (reg_sum / trace_ratio.max(1.0)).max(1.0e-8);

        rng_state = rng_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 0.5;
        let jitter = (u * (2.0 / n.sqrt())).exp();
        balance = ((sigma_n2 / sigma_x2) * jitter).clamp(1.0e-6, 1.0e3);
        balance_chain.push(balance);
    }

    let inv_count = 1.0 / (count_post.max(1) as f64);
    for v in &mut mean_x_re {
        *v *= inv_count;
    }
    for v in &mut mean_x_im {
        *v *= inv_count;
    }

    let post_re = DeviceArray::from_host(client, shape.clone(), cast_all::<F>(&mean_x_re))?;
    let post_im = DeviceArray::from_host(client, shape.clone(), cast_all::<F>(&mean_x_im))?;
    let raw_out = DeviceArray::<F>::empty(client, shape.clone())?;
    let out_im = DeviceArray::<F>::empty(client, shape.clone())?;
    fft::fft_nd_into(client, &post_re, &post_im, &raw_out, &out_im, true, &mut ws)?;

    if settings.clip {
        let clipped = DeviceArray::<F>::empty(client, shape.clone())?;
        elementwise::clip_into(client, &raw_out, &clipped, -1.0, 1.0)?;
        Ok((clipped, balance_chain))
    } else {
        Ok((raw_out, balance_chain))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::convolve::{correlate_2d, Correlate2d};
    use crate::test_support::{assert_close, runtime_test};

    fn richardson_lucy_and_wiener_sharpen_blurred_impulse(client: &Client) {
        // Delta PSF (identity): both RL and Wiener with small balance recover the original image
        let shape = Shape::new([8, 8]);
        let img: Vec<f32> = (0..64).map(|i| ((i as f32) / 64.0) * 0.8 + 0.1).collect();
        let dev_in = DeviceArray::from_host(client, shape.clone(), img.clone()).unwrap();
        #[rustfmt::skip]
        let delta_psf = vec![
            0.0f32, 0.0, 0.0,
            0.0,    1.0, 0.0,
            0.0,    0.0, 0.0,
        ];
        let psf_dev = DeviceArray::from_host(client, Shape::new([3, 3]), delta_psf).unwrap();

        let rl_out = richardson_lucy(
            client,
            &dev_in,
            &psf_dev,
            &RichardsonLucy { num_iter: 15, clip: false, ..Default::default() },
        )
        .unwrap()
        .to_host(client);
        let want: Vec<f64> = img.iter().map(|&v| v as f64).collect();
        assert_close(&rl_out, &want, 1e-4, client.name());

        let w_out = wiener_deconvolve(
            client,
            &dev_in,
            &psf_dev,
            &WienerDeconvolve { balance: 1e-7, clip: false },
        )
        .unwrap()
        .to_host(client);
        assert_close(&w_out, &want, 1e-3, client.name());

        // Non-trivial 3x3 box PSF: RL increases peak contrast of a blurred central spot
        let mut spot = vec![0.1f32; 64];
        spot[4 * 8 + 4] = 0.9;
        let spot_dev = DeviceArray::from_host(client, shape, spot).unwrap();
        let box_psf = DeviceArray::from_host(client, Shape::new([3, 3]), vec![1.0f32 / 9.0; 9]).unwrap();
        let blurred = correlate_2d(client, &spot_dev, &box_psf, &Correlate2d::default()).unwrap();
        let blurred_peak = blurred.to_host(client)[4 * 8 + 4];
        let restored = richardson_lucy(
            client,
            &blurred,
            &box_psf,
            &RichardsonLucy { num_iter: 25, clip: true, ..Default::default() },
        )
        .unwrap()
        .to_host(client);
        assert!(restored[4 * 8 + 4] > blurred_peak + 0.15, "{}", client.name());

        let (uw_img, chain) = unsupervised_wiener(
            client,
            &blurred,
            &box_psf,
            &UnsupervisedWiener { num_iter: 12, burnin: 4, clip: true, seed: 7 },
        )
        .unwrap();
        assert_eq!(chain.len(), 12);
        assert_eq!(uw_img.dims(), &[8, 8]);
    }
    runtime_test!(test_restore_deconvolve, richardson_lucy_and_wiener_sharpen_blurred_impulse);
}

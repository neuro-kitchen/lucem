//! Spectral and frequency-domain filters: Fourier Gaussian, Fourier box, Fourier phase shift,
//! Butterworth filter, 2-D Gabor filter bank, and spectral windowing functions.

use std::f64::consts::PI;

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast_all, ImageFloat};
use lucem_core::memory::{DeviceArray, DeviceCache, Workspace};
use lucem_core::Shape;

use super::convolve::{correlate_2d, Correlate2d};
use super::kernels::{spectral_phase_axis_kernel, spectral_scale_axis_kernel};
use crate::core::{check, fft, PadMode};
use crate::error::{OpsError, OpsResult};

static SPECTRAL_CACHE: DeviceCache = DeviceCache::new();

/// Multiplies a Fourier-domain spectrum `(re, im)` in-place along each axis by the Gaussian
/// frequency transfer function $\exp(-2 \pi^2 \sigma_d^2 f_d^2)$.
pub fn fourier_gaussian_into<F: ImageFloat>(
    client: &Client,
    re: &DeviceArray<F>,
    im: &DeviceArray<F>,
    sigmas: &[f64],
) -> OpsResult<()> {
    check::same_shape("re", re, "im", im)?;
    let ndim = re.ndim();
    let per_axis = check::broadcast_per_axis("sigmas", sigmas, ndim)?;
    if re.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, re.len());
    for (ax, &sigma) in per_axis.iter().enumerate() {
        let split = re.shape().axis_split(ax);
        let n = split.len;
        let key = format!("fgauss-{n}-{sigma:.12}");
        let transfer = SPECTRAL_CACHE.get_or_upload::<F>(client, &key, Shape::vector(n), || {
            let weights: Vec<f64> = (0..n)
                .map(|k| {
                    let f = if k <= n / 2 { (k as f64) / (n as f64) } else { ((k as isize - n as isize) as f64) / (n as f64) };
                    (-2.0 * PI * PI * sigma * sigma * f * f).exp()
                })
                .collect();
            cast_all::<F>(&weights)
        });
        unsafe {
            spectral_scale_axis_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                re.binding(),
                im.binding(),
                transfer.binding(),
                n as u32,
                split.inner as u32,
                re.len() as u32,
            );
        }
    }
    Ok(())
}

/// Multiplies a Fourier-domain spectrum `(re, im)` by the uniform box sinc transfer function
/// $\prod_d \operatorname{sinc}(s_d f_d)$.
pub fn fourier_box_into<F: ImageFloat>(
    client: &Client,
    re: &DeviceArray<F>,
    im: &DeviceArray<F>,
    sizes: &[f64],
) -> OpsResult<()> {
    check::same_shape("re", re, "im", im)?;
    let ndim = re.ndim();
    let per_axis = check::broadcast_per_axis("sizes", sizes, ndim)?;
    if re.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, re.len());
    for (ax, &size) in per_axis.iter().enumerate() {
        let split = re.shape().axis_split(ax);
        let n = split.len;
        let key = format!("fbox-{n}-{size:.12}");
        let transfer = SPECTRAL_CACHE.get_or_upload::<F>(client, &key, Shape::vector(n), || {
            let weights: Vec<f64> = (0..n)
                .map(|k| {
                    let f = if k <= n / 2 { (k as f64) / (n as f64) } else { ((k as isize - n as isize) as f64) / (n as f64) };
                    let x = PI * size * f;
                    if x.abs() < 1e-12 { 1.0 } else { x.sin() / x }
                })
                .collect();
            cast_all::<F>(&weights)
        });
        unsafe {
            spectral_scale_axis_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                re.binding(),
                im.binding(),
                transfer.binding(),
                n as u32,
                split.inner as u32,
                re.len() as u32,
            );
        }
    }
    Ok(())
}

/// Multiplies a Fourier-domain spectrum `(re, im)` by the linear phase shift
/// $\exp(-i 2\pi \sum_d \text{shift}_d f_d)$.
pub fn fourier_shift_into<F: ImageFloat>(
    client: &Client,
    re: &DeviceArray<F>,
    im: &DeviceArray<F>,
    shifts: &[f64],
) -> OpsResult<()> {
    check::same_shape("re", re, "im", im)?;
    let ndim = re.ndim();
    let per_axis = check::broadcast_per_axis("shifts", shifts, ndim)?;
    if re.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, re.len());
    for (ax, &shift) in per_axis.iter().enumerate() {
        let split = re.shape().axis_split(ax);
        let n = split.len;
        let cos_key = format!("fshift-cos-{n}-{shift:.12}");
        let sin_key = format!("fshift-sin-{n}-{shift:.12}");
        let cos_phase = SPECTRAL_CACHE.get_or_upload::<F>(client, &cos_key, Shape::vector(n), || {
            let w: Vec<f64> = (0..n)
                .map(|k| {
                    let f = if k <= n / 2 { (k as f64) / (n as f64) } else { ((k as isize - n as isize) as f64) / (n as f64) };
                    (-2.0 * PI * shift * f).cos()
                })
                .collect();
            cast_all::<F>(&w)
        });
        let sin_phase = SPECTRAL_CACHE.get_or_upload::<F>(client, &sin_key, Shape::vector(n), || {
            let w: Vec<f64> = (0..n)
                .map(|k| {
                    let f = if k <= n / 2 { (k as f64) / (n as f64) } else { ((k as isize - n as isize) as f64) / (n as f64) };
                    (-2.0 * PI * shift * f).sin()
                })
                .collect();
            cast_all::<F>(&w)
        });
        unsafe {
            spectral_phase_axis_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                re.binding(),
                im.binding(),
                cos_phase.binding(),
                sin_phase.binding(),
                n as u32,
                split.inner as u32,
                re.len() as u32,
            );
        }
    }
    Ok(())
}

/// Applies an N-D frequency-domain Butterworth low-pass (`high_pass = false`) or high-pass
/// (`high_pass = true`) filter with normalized cutoff `cutoff_ratio` in `(0.0, 0.5]`.
pub fn butterworth<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    cutoff_ratio: f64,
    high_pass: bool,
    order: f64,
) -> OpsResult<DeviceArray<F>> {
    if !(0.0..=0.5).contains(&cutoff_ratio) || cutoff_ratio == 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "cutoff_ratio",
            reason: format!("cutoff_ratio must be in (0.0, 0.5], got {cutoff_ratio}"),
        });
    }
    let shape = input.shape().clone();
    let mut ws = Workspace::new(client);
    let zero_im = DeviceArray::<F>::zeros(client, shape.clone())?;
    let freq_re = ws.array::<F>("bw_re", &shape)?;
    let freq_im = ws.array::<F>("bw_im", &shape)?;
    fft::fft_nd_into(client, input, &zero_im, &freq_re, &freq_im, false, &mut ws)?;

    let dims = shape.dims();
    let strides = shape.strides();
    let ndim = dims.len();
    let mut h_weights = vec![0.0f64; shape.len()];
    for (idx, w) in h_weights.iter_mut().enumerate() {
        let mut r2 = 0.0;
        for d in 0..ndim {
            let n = dims[d];
            let k = (idx / strides[d]) % n;
            let f = if k <= n / 2 { (k as f64) / (n as f64) } else { ((k as isize - n as isize) as f64) / (n as f64) };
            r2 += (f / cutoff_ratio).powi(2);
        }
        let dist = r2.sqrt();
        let lp = 1.0 / (1.0 + dist.powf(2.0 * order));
        *w = if high_pass { 1.0 - lp } else { lp };
    }
    let w_dev = DeviceArray::from_host(client, shape.clone(), cast_all::<F>(&h_weights))?;
    let scaled_re = ws.array::<F>("bw_sre", &shape)?;
    let scaled_im = ws.array::<F>("bw_sim", &shape)?;
    let zero_w = DeviceArray::<F>::zeros(client, shape.clone())?;
    fft::complex_mul_into(client, &freq_re, &freq_im, &w_dev, &zero_w, &scaled_re, &scaled_im, false)?;

    let out_re = DeviceArray::<F>::empty(client, shape.clone())?;
    let out_im = ws.array::<F>("bw_oim", &shape)?;
    fft::fft_nd_into(client, &scaled_re, &scaled_im, &out_re, &out_im, true, &mut ws)?;
    Ok(out_re)
}

/// Generates a 2-D complex Gabor kernel `(real_kernel, imag_kernel)` on the host in `f64`.
pub fn gabor_kernel_2d(
    frequency: f64,
    theta: f64,
    sigma_x: f64,
    sigma_y: f64,
    n_stds: f64,
    offset: f64,
) -> (Shape, Vec<f64>, Vec<f64>) {
    let ct = theta.cos();
    let st = theta.sin();
    let rx = (n_stds * (sigma_x * ct).hypot(sigma_y * st)).ceil().max(1.0) as isize;
    let ry = (n_stds * (sigma_x * st).hypot(sigma_y * ct)).ceil().max(1.0) as isize;
    let h = (2 * ry + 1) as usize;
    let w = (2 * rx + 1) as usize;
    let norm = 1.0 / (2.0 * PI * sigma_x * sigma_y);
    let mut re = Vec::with_capacity(h * w);
    let mut im = Vec::with_capacity(h * w);
    for y in -ry..=ry {
        for x in -rx..=rx {
            let xr = (x as f64) * ct + (y as f64) * st;
            let yr = -(x as f64) * st + (y as f64) * ct;
            let env = norm * (-0.5 * (xr * xr / (sigma_x * sigma_x) + yr * yr / (sigma_y * sigma_y))).exp();
            let phase = 2.0 * PI * frequency * xr + offset;
            re.push(env * phase.cos());
            im.push(env * phase.sin());
        }
    }
    (Shape::new([h, w]), re, im)
}

/// Applies a 2-D Gabor filter, returning `(real_response, imag_response)` on the device.
#[allow(clippy::too_many_arguments)]
pub fn gabor_2d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    frequency: f64,
    theta: f64,
    sigma_x: f64,
    sigma_y: f64,
    n_stds: f64,
    offset: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)> {
    let (kshape, re_k, im_k) = gabor_kernel_2d(frequency, theta, sigma_x, sigma_y, n_stds, offset);
    let dev_re_k = DeviceArray::from_host(client, kshape.clone(), cast_all::<F>(&re_k))?;
    let dev_im_k = DeviceArray::from_host(client, kshape, cast_all::<F>(&im_k))?;
    let settings = Correlate2d { mode, cval, origin: (0, 0) };
    let out_re = correlate_2d(client, input, &dev_re_k, &settings)?;
    let out_im = correlate_2d(client, input, &dev_im_k, &settings)?;
    Ok((out_re, out_im))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};

    fn fourier_shift_zero_leaves_spectrum_unchanged(client: &Client) {
        let shape = Shape::new([4, 4]);
        let re = DeviceArray::from_host(client, shape.clone(), vec![1.0f32; 16]).unwrap();
        let im = DeviceArray::<f32>::zeros(client, shape).unwrap();
        fourier_shift_into(client, &re, &im, &[0.0, 0.0]).unwrap();
        assert_close(&re.to_host(client), &[1.0; 16], 1e-6, client.name());
    }
    runtime_test!(test_spectral_filters, fourier_shift_zero_leaves_spectrum_unchanged);
}

//! Full-reference image fidelity metrics: Mean Squared Error (`mse`), Normalized Root Mean Squared
//! Error (`nrmse`), Peak Signal-to-Noise Ratio (`psnr`), and Structural Similarity Index (`ssim`).

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::kernels::{mul_kernel, sq_diff_kernel, ssim_map_kernel};
use crate::core::{check, reduce, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::filter::box_blur::{box_blur_into, BoxBlur};
use crate::filter::gaussian::{gaussian_blur_into, GaussianBlur};

/// Normalization denominator for [`nrmse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NrmseNormalization {
    /// Euclidean norm of `image_true`: $\sqrt{\frac{1}{N}\sum I_{\text{true}}^2}$.
    #[default]
    Euclidean,
    /// Dynamic range of `image_true`: $\max(I_{\text{true}}) - \min(I_{\text{true}})$.
    MinMax,
    /// Mean intensity of `image_true`: $|\overline{I_{\text{true}}}|$.
    Mean,
}

/// Computes the Mean Squared Error (MSE) between `image_true` and `image_test`:
/// $$\operatorname{MSE}(I, J) = \frac{1}{N} \sum_{i=1}^{N} (I_i - J_i)^2$$
pub fn mse<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
) -> OpsResult<f64> {
    check::same_shape("image_true", image_true, "image_test", image_test)?;
    if image_true.is_empty() {
        return Err(OpsError::EmptyInput("mse"));
    }
    let sq_diff = DeviceArray::<F>::empty(client, image_true.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, image_true.len());
    unsafe {
        sq_diff_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            image_true.binding(),
            image_test.binding(),
            sq_diff.binding(),
            image_true.len() as u32,
        );
    }
    let stats = reduce::sum_min_max(client, &sq_diff)?;
    Ok(stats.sum / (image_true.len() as f64))
}

/// Computes the Normalized Root Mean Squared Error (NRMSE) between `image_true` and `image_test`.
pub fn nrmse<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
    normalization: NrmseNormalization,
) -> OpsResult<f64> {
    let err_mse = mse(client, image_true, image_test)?;
    let rmse = err_mse.sqrt();
    if rmse == 0.0 {
        return Ok(0.0);
    }
    let denom = match normalization {
        NrmseNormalization::Euclidean => {
            let (mean, var) = reduce::mean_var(client, image_true, 0)?;
            (var + mean * mean).sqrt()
        }
        NrmseNormalization::MinMax => {
            let stats = reduce::sum_min_max(client, image_true)?;
            stats.max - stats.min
        }
        NrmseNormalization::Mean => {
            let (mean, _) = reduce::mean_var(client, image_true, 0)?;
            mean.abs()
        }
    };
    if denom <= f64::EPSILON {
        Ok(f64::INFINITY)
    } else {
        Ok(rmse / denom)
    }
}

/// Computes the Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) between `image_true` and
/// `image_test`:
/// $$\operatorname{PSNR} = 10 \log_{10}\!\left(\frac{\text{data\_range}^2}{\operatorname{MSE}}\right)$$
///
/// When `data_range` is `None`, defaults to `max(image_true) - min(image_true)` (or `1.0` if
/// `image_true` is constant).
pub fn psnr<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
    data_range: Option<f64>,
) -> OpsResult<f64> {
    let err_mse = mse(client, image_true, image_test)?;
    let range = match data_range {
        Some(r) => {
            if !r.is_finite() || r <= 0.0 {
                return Err(OpsError::InvalidParameter {
                    name: "data_range",
                    reason: format!("data_range must be positive and finite, got {r}"),
                });
            }
            r
        }
        None => {
            let stats = reduce::sum_min_max(client, image_true)?;
            let span = stats.max - stats.min;
            if span > f64::EPSILON { span } else { 1.0 }
        }
    };
    if err_mse <= 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(10.0 * ((range * range) / err_mse).log10())
    }
}

/// Settings for the Structural Similarity Index ([`ssim`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Ssim {
    /// Sliding window side length (must be odd; defaults to `7` for box windows or `11` for Gaussian).
    pub win_size: Option<usize>,
    /// When `true`, uses separable Gaussian weighting with standard deviation `sigma`; when `false`,
    /// uses a uniform box window of size `win_size`.
    pub gaussian_weights: bool,
    /// Gaussian standard deviation when `gaussian_weights = true` (default `1.5`).
    pub sigma: f64,
    /// Explicit dynamic range $L$ of the input images (when `None`, inferred from `max - min` of
    /// `image_true`, or `1.0` if constant).
    pub data_range: Option<f64>,
    /// Whether to normalize local covariances by $N / (N - 1)$ (Bessel sample correction).
    pub use_sample_covariance: bool,
    /// Luminance stabilization constant $K_1$ (default `0.01`).
    pub k1: f64,
    /// Contrast/structure stabilization constant $K_2$ (default `0.03`).
    pub k2: f64,
}

impl Default for Ssim {
    fn default() -> Self {
        Self {
            win_size: None,
            gaussian_weights: false,
            sigma: 1.5,
            data_range: None,
            use_sample_covariance: true,
            k1: 0.01,
            k2: 0.03,
        }
    }
}

fn filter_ssim_moment<F: ImageFloat>(
    client: &Client,
    src: &DeviceArray<F>,
    dst: &DeviceArray<F>,
    win_size: usize,
    settings: &Ssim,
    ws: &mut Workspace,
) -> OpsResult<()> {
    if settings.gaussian_weights {
        let radius = win_size / 2;
        gaussian_blur_into(
            client,
            src,
            &[settings.sigma],
            dst,
            &GaussianBlur {
                mode: PadMode::Reflect,
                radii: Some(vec![radius]),
                ..Default::default()
            },
            ws,
        )
    } else {
        box_blur_into(
            client,
            src,
            &[win_size],
            dst,
            &BoxBlur {
                mode: PadMode::Reflect,
                ..Default::default()
            },
            ws,
        )
    }
}

/// Computes both the scalar mean SSIM and the full-resolution per-pixel SSIM map on the device.
pub fn ssim_with_map<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
    settings: &Ssim,
) -> OpsResult<(f64, DeviceArray<F>)> {
    check::same_shape("image_true", image_true, "image_test", image_test)?;
    if image_true.is_empty() {
        return Err(OpsError::EmptyInput("ssim"));
    }
    let win_size = match settings.win_size {
        Some(w) => {
            if w < 3 || w % 2 == 0 {
                return Err(OpsError::InvalidParameter {
                    name: "win_size",
                    reason: format!("win_size must be an odd integer >= 3, got {w}"),
                });
            }
            w
        }
        None => {
            if settings.gaussian_weights {
                let r = (3.5 * settings.sigma + 0.5).floor().max(1.0) as usize;
                2 * r + 1
            } else {
                7
            }
        }
    };
    if settings.gaussian_weights && (!settings.sigma.is_finite() || settings.sigma <= 0.0) {
        return Err(OpsError::InvalidParameter {
            name: "sigma",
            reason: format!("sigma must be positive and finite, got {}", settings.sigma),
        });
    }

    let range = match settings.data_range {
        Some(r) => {
            if !r.is_finite() || r <= 0.0 {
                return Err(OpsError::InvalidParameter {
                    name: "data_range",
                    reason: format!("data_range must be positive and finite, got {r}"),
                });
            }
            r
        }
        None => {
            let stats = reduce::sum_min_max(client, image_true)?;
            let span = stats.max - stats.min;
            if span > f64::EPSILON { span } else { 1.0 }
        }
    };

    let shape = image_true.shape();
    let ndim = shape.ndim();
    let n_total = image_true.len();
    let mut ws = Workspace::new(client);

    let xx = ws.array::<F>("ssim_xx", shape)?;
    let yy = ws.array::<F>("ssim_yy", shape)?;
    let xy = ws.array::<F>("ssim_xy", shape)?;
    let geom = LaunchGeometry::elementwise(client, n_total);
    unsafe {
        mul_kernel::launch::<F>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim,
            image_true.binding(),
            image_true.binding(),
            xx.binding(),
            n_total as u32,
        );
        mul_kernel::launch::<F>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim,
            image_test.binding(),
            image_test.binding(),
            yy.binding(),
            n_total as u32,
        );
        mul_kernel::launch::<F>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim,
            image_true.binding(),
            image_test.binding(),
            xy.binding(),
            n_total as u32,
        );
    }

    let ux = DeviceArray::<F>::empty(client, shape.clone())?;
    let uy = DeviceArray::<F>::empty(client, shape.clone())?;
    let uxx = DeviceArray::<F>::empty(client, shape.clone())?;
    let uyy = DeviceArray::<F>::empty(client, shape.clone())?;
    let uxy = DeviceArray::<F>::empty(client, shape.clone())?;

    filter_ssim_moment(client, image_true, &ux, win_size, settings, &mut ws)?;
    filter_ssim_moment(client, image_test, &uy, win_size, settings, &mut ws)?;
    filter_ssim_moment(client, &xx, &uxx, win_size, settings, &mut ws)?;
    filter_ssim_moment(client, &yy, &uyy, win_size, settings, &mut ws)?;
    filter_ssim_moment(client, &xy, &uxy, win_size, settings, &mut ws)?;

    let win_vol = (win_size as f64).powi(ndim as i32);
    let cov_norm = if settings.use_sample_covariance && win_vol > 1.0 {
        win_vol / (win_vol - 1.0)
    } else {
        1.0
    };
    let c1 = (settings.k1 * range).powi(2);
    let c2 = (settings.k2 * range).powi(2);

    let ssim_map = DeviceArray::<F>::empty(client, shape.clone())?;
    unsafe {
        ssim_map_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ux.binding(),
            uy.binding(),
            uxx.binding(),
            uyy.binding(),
            uxy.binding(),
            ssim_map.binding(),
            cast::<F>(cov_norm),
            cast::<F>(c1),
            cast::<F>(c2),
            n_total as u32,
        );
    }

    let stats = reduce::sum_min_max(client, &ssim_map)?;
    let mean_ssim = stats.sum / (n_total as f64);
    Ok((mean_ssim, ssim_map))
}

/// Computes the mean Structural Similarity Index (SSIM) between `image_true` and `image_test`
/// (Wang, Bovik, Sheikh & Simoncelli, 2004).
pub fn ssim<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
    settings: &Ssim,
) -> OpsResult<f64> {
    let (score, _) = ssim_with_map(client, image_true, image_test, settings)?;
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test, sample_positive_f64};
    use lumen_core::element::cast_all;
    use lumen_core::Shape;

    fn fidelity_metrics_match_exact_identities(client: &Client) {
        let shape = Shape::new([16, 16]);
        let x = sample_positive_f64(shape.len());
        let y: Vec<f64> = x.iter().map(|&v| v + 2.0).collect();
        let a = DeviceArray::from_host(client, shape.clone(), cast_all::<f32>(&x)).unwrap();
        let b = DeviceArray::from_host(client, shape, cast_all::<f32>(&y)).unwrap();

        assert_close(&[mse(client, &a, &a).unwrap()], &[0.0], 1e-7, client.name());
        assert_close(&[mse(client, &a, &b).unwrap()], &[4.0], 1e-5, client.name());
        assert!(psnr(client, &a, &a, Some(100.0)).unwrap().is_infinite());

        let want_psnr = 10.0 * (100.0f64 * 100.0 / 4.0).log10();
        assert_close(
            &[psnr(client, &a, &b, Some(100.0)).unwrap()],
            &[want_psnr],
            1e-4,
            client.name(),
        );

        let nrmse_mm = nrmse(client, &a, &b, NrmseNormalization::MinMax).unwrap();
        let min_x = x.iter().copied().fold(f64::INFINITY, f64::min);
        let max_x = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_close(&[nrmse_mm], &[2.0 / (max_x - min_x)], 1e-4, client.name());

        let ssim_ident = ssim(client, &a, &a, &Ssim::default()).unwrap();
        assert_close(&[ssim_ident], &[1.0], 1e-5, client.name());

        let ssim_gauss = ssim(
            client,
            &a,
            &b,
            &Ssim {
                gaussian_weights: true,
                data_range: Some(100.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(ssim_gauss > 0.95 && ssim_gauss < 1.0, "ssim_gauss = {ssim_gauss}");
    }
    runtime_test!(test_fidelity_metrics, fidelity_metrics_match_exact_identities);
}

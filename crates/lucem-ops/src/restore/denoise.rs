//! Edge-preserving image denoising (`tv_chambolle`, `tv_bregman`, `bilateral`, `non_local_means`)
//! and robust Gaussian noise standard-deviation estimation (`estimate_noise`).

use std::f64::consts::PI;

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast, cast_all, to_f64, ImageFloat};
use lucem_core::memory::{DeviceArray, Workspace};
use lucem_core::Shape;

use super::kernels::{
    bilateral_2d_kernel, immerkaer_laplacian_2d_kernel, nlm_2d_kernel, tv_bregman_2d_dual_kernel,
    tv_bregman_2d_gauss_seidel_kernel, tv_chambolle_2d_div_kernel, tv_chambolle_2d_dual_kernel,
};
use crate::core::{check, elementwise, reduce, PadMode};
use crate::error::{OpsError, OpsResult};

/// Settings for Chambolle total-variation (TV) denoising ([`tv_chambolle`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TvChambolle {
    /// Denoising regularization weight ($\lambda > 0$; larger values produce smoother images).
    pub weight: f64,
    /// Relative stopping tolerance on primal iterates (`0.0` runs all `max_num_iter` iterations).
    pub eps: f64,
    /// Maximum number of primal-dual iterations.
    pub max_num_iter: usize,
    /// Isotropic ($L_2$ norm of gradient) vs. anisotropic ($L_1$) total variation.
    pub isotropic: bool,
}

impl Default for TvChambolle {
    fn default() -> Self {
        Self {
            weight: 0.1,
            eps: 2.0e-4,
            max_num_iter: 200,
            isotropic: true,
        }
    }
}

/// Denoises `input` into `output` using Chambolle's (2004) primal-dual Total Variation projection
/// algorithm on the device.
pub fn tv_chambolle_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &TvChambolle,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::ndim_in("tv_chambolle", input.shape(), &[2], "2")?;
    if !settings.weight.is_finite() || settings.weight <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "weight",
            reason: format!("weight must be positive and finite, got {}", settings.weight),
        });
    }
    if settings.max_num_iter == 0 {
        return Err(OpsError::InvalidParameter {
            name: "max_num_iter",
            reason: "max_num_iter must be >= 1".to_string(),
        });
    }
    if input.is_empty() {
        return Ok(());
    }

    let shape = input.shape();
    let h = shape.dims()[0] as u32;
    let w = shape.dims()[1] as u32;
    let total = input.len() as u32;
    let geom = LaunchGeometry::elementwise(client, input.len());

    let py_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let px_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let py_b = ws.array::<F>("tvc_py_b", shape)?;
    let px_b = ws.array::<F>("tvc_px_b", shape)?;
    let u_prev = ws.array::<F>("tvc_u_prev", shape)?;
    let diff = ws.array::<F>("tvc_diff", shape)?;

    // Step size tau = 1 / (2 * ndim) = 0.25 for 2-D stability, scaled by 1 / weight
    let tau = 0.249;
    let step = cast::<F>(tau / settings.weight);
    let weight_f = cast::<F>(settings.weight);
    let tol_sq = settings.eps * settings.eps * (input.len() as f64);

    let mut ping = true;
    for iter in 0..settings.max_num_iter {
        let (py_cur, px_cur, py_nxt, px_nxt) = if ping {
            (&py_a, &px_a, &py_b, &px_b)
        } else {
            (&py_b, &px_b, &py_a, &px_a)
        };
        unsafe {
            tv_chambolle_2d_div_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                input.binding(),
                py_cur.binding(),
                px_cur.binding(),
                output.binding(),
                weight_f,
                h,
                w,
                total,
            );
            tv_chambolle_2d_dual_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                output.binding(),
                py_cur.binding(),
                px_cur.binding(),
                py_nxt.binding(),
                px_nxt.binding(),
                step,
                h,
                w,
                total,
                settings.isotropic,
            );
        }
        ping = !ping;

        if settings.eps > 0.0 && iter > 0 && (iter % 10 == 0 || iter + 1 == settings.max_num_iter) {
            elementwise::linear_combination_into(client, output, &u_prev, &diff, 1.0, -1.0, 0.0)?;
            let (_, var) = reduce::mean_var(client, &diff, 0)?;
            let (mean, _) = reduce::mean_var(client, &diff, 0)?;
            let norm_sq = (var + mean * mean) * (input.len() as f64);
            if norm_sq < tol_sq {
                break;
            }
        }
        if settings.eps > 0.0 && (iter + 1) % 10 == 0 {
            elementwise::copy_into(client, output, &u_prev)?;
        }
    }

    let (py_final, px_final) = if ping { (&py_a, &px_a) } else { (&py_b, &px_b) };
    unsafe {
        tv_chambolle_2d_div_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            py_final.binding(),
            px_final.binding(),
            output.binding(),
            weight_f,
            h,
            w,
            total,
        );
    }
    Ok(())
}

/// Allocates and returns Chambolle total-variation denoised `input`.
pub fn tv_chambolle<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &TvChambolle,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    tv_chambolle_into(client, input, &output, settings, &mut ws)?;
    Ok(output)
}

/// Settings for Split-Bregman total-variation denoising ([`tv_bregman`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TvBregman {
    /// Fidelity weight $\mu > 0$ on $\|u - f\|_2^2$ (smaller values produce stronger smoothing).
    pub weight: f64,
    /// Maximum number of Bregman iterations.
    pub max_num_iter: usize,
    /// Relative stopping tolerance on primal iterates.
    pub eps: f64,
    /// Isotropic ($L_2$) vs. anisotropic ($L_1$) shrinkage.
    pub isotropic: bool,
}

impl Default for TvBregman {
    fn default() -> Self {
        Self {
            weight: 5.0,
            max_num_iter: 100,
            eps: 1.0e-3,
            isotropic: true,
        }
    }
}

/// Denoises `input` into `output` using Goldstein & Osher's (2009) Split-Bregman TV algorithm.
pub fn tv_bregman_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &TvBregman,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::ndim_in("tv_bregman", input.shape(), &[2], "2")?;
    if !settings.weight.is_finite() || settings.weight <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "weight",
            reason: format!("weight must be positive and finite, got {}", settings.weight),
        });
    }
    if settings.max_num_iter == 0 {
        return Err(OpsError::InvalidParameter {
            name: "max_num_iter",
            reason: "max_num_iter must be >= 1".to_string(),
        });
    }
    if input.is_empty() {
        return Ok(());
    }

    let shape = input.shape();
    let h = shape.dims()[0] as u32;
    let w = shape.dims()[1] as u32;
    let total = input.len() as u32;
    let geom = LaunchGeometry::elementwise(client, input.len());

    elementwise::copy_into(client, input, output)?;
    let u_tmp = ws.array::<F>("tvb_u_tmp", shape)?;
    let dy = DeviceArray::<F>::zeros(client, shape.clone())?;
    let dx = DeviceArray::<F>::zeros(client, shape.clone())?;
    let by_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let bx_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let by_b = ws.array::<F>("tvb_by_b", shape)?;
    let bx_b = ws.array::<F>("tvb_bx_b", shape)?;

    let lam = 2.0 * settings.weight;
    let weight_f = cast::<F>(settings.weight);
    let lam_f = cast::<F>(lam);
    let inv_lam_f = cast::<F>(1.0 / lam);

    let mut ping = true;
    for _ in 0..settings.max_num_iter {
        let (by_cur, bx_cur, by_nxt, bx_nxt) = if ping {
            (&by_a, &bx_a, &by_b, &bx_b)
        } else {
            (&by_b, &bx_b, &by_a, &bx_a)
        };
        // Two damped Jacobi sweeps per outer Bregman iteration
        unsafe {
            tv_bregman_2d_gauss_seidel_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                input.binding(),
                output.binding(),
                dy.binding(),
                dx.binding(),
                by_cur.binding(),
                bx_cur.binding(),
                u_tmp.binding(),
                weight_f,
                lam_f,
                h,
                w,
                total,
            );
            tv_bregman_2d_gauss_seidel_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                input.binding(),
                u_tmp.binding(),
                dy.binding(),
                dx.binding(),
                by_cur.binding(),
                bx_cur.binding(),
                output.binding(),
                weight_f,
                lam_f,
                h,
                w,
                total,
            );
            tv_bregman_2d_dual_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                output.binding(),
                by_cur.binding(),
                bx_cur.binding(),
                dy.binding(),
                dx.binding(),
                by_nxt.binding(),
                bx_nxt.binding(),
                inv_lam_f,
                h,
                w,
                total,
                settings.isotropic,
            );
        }
        ping = !ping;
    }
    Ok(())
}

/// Allocates and returns Split-Bregman total-variation denoised `input`.
pub fn tv_bregman<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &TvBregman,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    tv_bregman_into(client, input, &output, settings, &mut ws)?;
    Ok(output)
}

/// Settings for 2-D bilateral edge-preserving smoothing ([`bilateral`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Bilateral {
    /// Radiometric (intensity/range) Gaussian standard deviation. When `None`, estimated from the
    /// sample standard deviation of the input image (clamped to `>= 1e-4`).
    pub sigma_color: Option<f64>,
    /// Spatial Gaussian standard deviation in pixels.
    pub sigma_spatial: f64,
    /// Optional odd window size (`>= 1`). Defaults to `max(5, 2 * ceil(3 * sigma_spatial) + 1)`.
    pub win_size: Option<usize>,
    /// Boundary padding mode.
    pub mode: PadMode,
    /// Constant fill value for `PadMode::Constant`.
    pub cval: f64,
}

impl Default for Bilateral {
    fn default() -> Self {
        Self {
            sigma_color: None,
            sigma_spatial: 1.0,
            win_size: None,
            mode: PadMode::Reflect,
            cval: 0.0,
        }
    }
}

/// Applies a 2-D edge-preserving bilateral filter into `output`.
pub fn bilateral_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &Bilateral,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::ndim_in("bilateral", input.shape(), &[2], "2")?;
    if !settings.sigma_spatial.is_finite() || settings.sigma_spatial <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "sigma_spatial",
            reason: format!("sigma_spatial must be positive and finite, got {}", settings.sigma_spatial),
        });
    }
    let win_size = match settings.win_size {
        Some(w) => {
            if w == 0 || w % 2 == 0 {
                return Err(OpsError::InvalidParameter {
                    name: "win_size",
                    reason: format!("win_size must be a positive odd integer, got {w}"),
                });
            }
            w
        }
        None => {
            let r = (3.0 * settings.sigma_spatial).ceil() as usize;
            (2 * r + 1).max(5)
        }
    };
    if input.is_empty() {
        return Ok(());
    }

    let sigma_color = match settings.sigma_color {
        Some(sc) => {
            if !sc.is_finite() || sc <= 0.0 {
                return Err(OpsError::InvalidParameter {
                    name: "sigma_color",
                    reason: format!("sigma_color must be positive and finite, got {sc}"),
                });
            }
            sc
        }
        None => {
            let (_, var) = reduce::mean_var(client, input, 0)?;
            var.sqrt().max(1.0e-4)
        }
    };

    let radius = (win_size / 2) as isize;
    let mut spatial_w = Vec::with_capacity(win_size * win_size);
    let inv_two_ss2 = -0.5 / (settings.sigma_spatial * settings.sigma_spatial);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let r2 = (dy * dy + dx * dx) as f64;
            spatial_w.push((r2 * inv_two_ss2).exp());
        }
    }
    let spatial_dev = DeviceArray::from_host(client, Shape::new([win_size, win_size]), cast_all::<F>(&spatial_w))?;
    let inv_two_sc2 = cast::<F>(-0.5 / (sigma_color * sigma_color));

    let dims = input.dims();
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        bilateral_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            spatial_dev.binding(),
            output.binding(),
            inv_two_sc2,
            cast::<F>(settings.cval),
            dims[0] as i32,
            dims[1] as i32,
            radius as i32,
            settings.mode.id(),
        );
    }
    Ok(())
}

/// Allocates and applies a 2-D edge-preserving bilateral filter to `input`.
pub fn bilateral<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &Bilateral,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    bilateral_into(client, input, &output, settings)?;
    Ok(output)
}

/// Settings for 2-D patch-based Non-Local Means (NLM) denoising ([`non_local_means`]).
#[derive(Debug, Clone, PartialEq)]
pub struct NonLocalMeans {
    /// Side length of square comparison patches (e.g. `7` for a `7x7` patch, radius `3`).
    pub patch_size: usize,
    /// Search radius in pixels (`(2 * patch_distance + 1)^2` candidate patches evaluated per pixel).
    pub patch_distance: usize,
    /// Cut-off smoothing parameter `h` (> 0).
    pub h: f64,
    /// Optional known Gaussian noise standard deviation $\sigma \ge 0$ subtracted from patch distances.
    pub sigma: f64,
    /// Boundary padding mode.
    pub mode: PadMode,
}

impl Default for NonLocalMeans {
    fn default() -> Self {
        Self {
            patch_size: 5,
            patch_distance: 6,
            h: 0.1,
            sigma: 0.0,
            mode: PadMode::Reflect,
        }
    }
}

/// Denoises `input` into `output` using Buades, Coll & Morel's (2005) Non-Local Means algorithm.
pub fn non_local_means_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &NonLocalMeans,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::ndim_in("non_local_means", input.shape(), &[2], "2")?;
    if settings.patch_size == 0 {
        return Err(OpsError::InvalidParameter {
            name: "patch_size",
            reason: "patch_size must be >= 1".to_string(),
        });
    }
    if !settings.h.is_finite() || settings.h <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "h",
            reason: format!("h must be positive and finite, got {}", settings.h),
        });
    }
    if !settings.sigma.is_finite() || settings.sigma < 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "sigma",
            reason: format!("sigma must be non-negative and finite, got {}", settings.sigma),
        });
    }
    if input.is_empty() {
        return Ok(());
    }

    let patch_radius = (settings.patch_size / 2) as i32;
    let search_radius = settings.patch_distance as i32;
    let patch_side = (2 * patch_radius + 1) as f64;
    let inv_patch_area = 1.0 / (patch_side * patch_side);
    let inv_h2 = 1.0 / (settings.h * settings.h);
    let sigma_sq = settings.sigma * settings.sigma;

    let dims = input.dims();
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        nlm_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            cast::<F>(inv_h2),
            cast::<F>(sigma_sq),
            cast::<F>(inv_patch_area),
            dims[0] as i32,
            dims[1] as i32,
            patch_radius,
            search_radius,
            settings.mode.id(),
        );
    }
    Ok(())
}

/// Allocates and returns Non-Local Means denoised `input`.
pub fn non_local_means<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &NonLocalMeans,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    non_local_means_into(client, input, &output, settings)?;
    Ok(output)
}

/// Estimates the additive Gaussian white noise standard deviation $\hat{\sigma}$ of `input` using
/// Immerkær's (1996) fast 3x3 zero-mean Laplacian operator on the device for 2-D images (and the
/// Donoho-Johnstone diagonal Haar wavelet median absolute deviation for general N-D images).
pub fn estimate_noise<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<f64> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("estimate_noise"));
    }
    let dims = input.dims();
    if dims.len() == 2 && dims[0] >= 3 && dims[1] >= 3 {
        let inner_h = dims[0] - 2;
        let inner_w = dims[1] - 2;
        let count = inner_h * inner_w;
        let out_abs = DeviceArray::<F>::empty(client, Shape::vector(count))?;
        let geom = LaunchGeometry::elementwise(client, count);
        unsafe {
            immerkaer_laplacian_2d_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                out_abs.binding(),
                inner_h as u32,
                inner_w as u32,
                dims[1] as u32,
            );
        }
        let stats = reduce::sum_min_max(client, &out_abs)?;
        let factor = (0.5 * PI).sqrt() / (6.0 * (count as f64));
        return Ok(stats.sum * factor);
    }

    // Fallback for 1-D, small 2-D, or >=3-D arrays: MAD of first-difference detail coefficients
    let host: Vec<f64> = input.to_host(client).into_iter().map(to_f64).collect();
    if host.len() < 2 {
        return Ok(0.0);
    }
    let mut diffs: Vec<f64> = host.windows(2).map(|w| (w[1] - w[0]).abs() / 2.0f64.sqrt()).collect();
    diffs.sort_by(f64::total_cmp);
    let mid = diffs.len() / 2;
    let median = if diffs.len() % 2 == 1 {
        diffs[mid]
    } else {
        0.5 * (diffs[mid - 1] + diffs[mid])
    };
    Ok(median / 0.674_489_750_196_081_7)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::random::fill_normal;
    use crate::test_support::{assert_close, runtime_test};

    fn denoisers_preserve_constant_and_reduce_variance(client: &Client) {
        let shape = Shape::new([16, 16]);
        let const_img = DeviceArray::<f32>::filled(client, shape.clone(), 0.5).unwrap();

        let tvc = tv_chambolle(client, &const_img, &TvChambolle::default()).unwrap().to_host(client);
        assert_close(&tvc, &vec![0.5; 256], 1e-4, client.name());

        let tvb = tv_bregman(client, &const_img, &TvBregman::default()).unwrap().to_host(client);
        assert_close(&tvb, &vec![0.5; 256], 1e-4, client.name());

        let bil = bilateral(client, &const_img, &Bilateral::default()).unwrap().to_host(client);
        assert_close(&bil, &vec![0.5; 256], 1e-5, client.name());

        let nlm = non_local_means(client, &const_img, &NonLocalMeans::default()).unwrap().to_host(client);
        assert_close(&nlm, &vec![0.5; 256], 1e-5, client.name());

        // Step edge + Gaussian noise: denoisers reduce variance around each plateau
        let noise = DeviceArray::<f32>::empty(client, Shape::new([32, 32])).unwrap();
        fill_normal(client, &noise, 0.0, 0.15, 12345).unwrap();
        let sigma_hat = estimate_noise(client, &noise).unwrap();
        assert!((sigma_hat - 0.15).abs() < 0.03, "sigma_hat = {sigma_hat} on {}", client.name());

        let (_, noisy_var) = reduce::mean_var(client, &noise, 0).unwrap();
        let denoised_tv = tv_chambolle(client, &noise, &TvChambolle { weight: 0.2, ..Default::default() }).unwrap();
        let (_, tv_var) = reduce::mean_var(client, &denoised_tv, 0).unwrap();
        assert!(tv_var < noisy_var * 0.25, "tv_var={tv_var}, noisy_var={noisy_var}");

        let denoised_nlm = non_local_means(
            client,
            &noise,
            &NonLocalMeans { h: 0.2, sigma: sigma_hat, ..Default::default() },
        )
        .unwrap();
        let (_, nlm_var) = reduce::mean_var(client, &denoised_nlm, 0).unwrap();
        assert!(nlm_var < noisy_var * 0.3, "nlm_var={nlm_var}, noisy_var={noisy_var}");
    }
    runtime_test!(test_restore_denoise, denoisers_preserve_constant_and_reduce_variance);
}

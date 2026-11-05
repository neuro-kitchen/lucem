//! Gaussian smoothing, Gaussian derivatives, Laplacian of Gaussian (LoG), Gaussian gradient
//! magnitude, Difference of Gaussians (DoG), and unsharp masking.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, cast_all, ImageFloat};
use lumen_core::memory::{DeviceArray, DeviceCache, Workspace};
use lumen_core::Shape;

use super::convolve::{correlate_1d_into, Correlate1d};
use super::kernels::{add_sq_kernel, sqrt_scale_kernel};
use crate::core::{check, elementwise, separable, PadMode};
use crate::error::{OpsError, OpsResult};

static GAUSSIAN_TAPS: DeviceCache = DeviceCache::new();

/// Computes the 1-D Gaussian (or Gaussian derivative of `order`) kernel of half-width `radius`
/// (`2 * radius + 1` taps) in `f64`.
///
/// Uses the exact recurrence for polynomial Hermite derivatives of the discrete normalized Gaussian
/// so orders 0, 1, 2, 3 match reference implementations to machine precision.
pub fn gaussian_kernel_1d(sigma: f64, order: u32, radius: usize) -> Vec<f64> {
    if radius == 0 {
        return if order == 0 { vec![1.0] } else { vec![0.0] };
    }
    let sigma2 = sigma * sigma;
    let r = radius as isize;
    let mut phi: Vec<f64> = (-r..=r).map(|x| (-0.5 * (x as f64).powi(2) / sigma2).exp()).collect();
    let sum: f64 = phi.iter().sum();
    for v in &mut phi {
        *v /= sum;
    }
    if order == 0 {
        return phi;
    }
    // Recurrence for polynomial q_k(x) where d^k/dx^k exp(-x^2/(2*sigma^2)) = q_k(x) * phi(x)
    // With correlation sign convention (matching finite-difference derivative orientation):
    let n = 2 * radius + 1;
    let mut out = vec![0.0; n];
    for (idx, &p) in phi.iter().enumerate() {
        let x = (idx as isize - r) as f64;
        let q = match order {
            1 => -x / sigma2,
            2 => (x * x - sigma2) / (sigma2 * sigma2),
            3 => -(x.powi(3) - 3.0 * x * sigma2) / sigma2.powi(3),
            _ => {
                // Higher order Hermite polynomial recurrence
                let mut h_prev = 1.0;
                let mut h_cur = -x / sigma2;
                for k in 1..order {
                    let h_next = (-x / sigma2) * h_cur - (k as f64 / sigma2) * h_prev;
                    h_prev = h_cur;
                    h_cur = h_next;
                }
                h_cur
            }
        };
        // Correlation reverses the kernel relative to convolution: negate odd derivative orders
        let sign = if order % 2 == 1 { -1.0 } else { 1.0 };
        out[idx] = sign * q * p;
    }
    out
}

/// Settings for 1-D Gaussian filtering ([`gaussian_1d`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Gaussian1d {
    pub axis: isize,
    pub order: u32,
    pub mode: PadMode,
    pub cval: f64,
    pub truncate: f64,
    pub radius: Option<usize>,
}

impl Default for Gaussian1d {
    fn default() -> Self {
        Self { axis: -1, order: 0, mode: PadMode::Reflect, cval: 0.0, truncate: 4.0, radius: None }
    }
}

impl Gaussian1d {
    pub fn resolved_radius(&self, sigma: f64) -> usize {
        self.radius.unwrap_or_else(|| (self.truncate * sigma + 0.5).floor().max(0.0) as usize)
    }
}

/// 1-D Gaussian filter along `settings.axis` into `output`, caching the tap weights per device.
pub fn gaussian_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigma: f64,
    output: &DeviceArray<F>,
    settings: &Gaussian1d,
) -> OpsResult<()> {
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "sigma",
            reason: format!("sigma must be non-negative and finite, got {sigma}"),
        });
    }
    if sigma == 0.0 {
        return elementwise::copy_into(client, input, output);
    }
    let radius = settings.resolved_radius(sigma);
    let len = 2 * radius + 1;
    let key = format!("gauss1d-{sigma:.12}-o{}-r{radius}", settings.order);
    let weights = GAUSSIAN_TAPS.get_or_upload::<F>(client, &key, Shape::vector(len), || {
        cast_all::<F>(&gaussian_kernel_1d(sigma, settings.order, radius))
    });
    correlate_1d_into(
        client,
        input,
        &weights,
        output,
        &Correlate1d { axis: settings.axis, mode: settings.mode, cval: settings.cval, origin: 0 },
    )
}

/// 1-D Gaussian filter allocating `output`.
pub fn gaussian_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigma: f64,
    settings: &Gaussian1d,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    gaussian_1d_into(client, input, sigma, &output, settings)?;
    Ok(output)
}

/// Settings for N-D separable Gaussian smoothing ([`gaussian_blur`]).
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianBlur {
    pub orders: Vec<u32>,
    pub mode: PadMode,
    pub cval: f64,
    pub truncate: f64,
    pub radii: Option<Vec<usize>>,
    pub axes: Option<Vec<isize>>,
    pub channel_axis: Option<isize>,
}

impl Default for GaussianBlur {
    fn default() -> Self {
        Self {
            orders: vec![0],
            mode: PadMode::Reflect,
            cval: 0.0,
            truncate: 4.0,
            radii: None,
            axes: None,
            channel_axis: None,
        }
    }
}

impl GaussianBlur {
    /// Halo `(before, after)` per axis for tiling large images.
    pub fn halo(&self, ndim: usize, sigmas: &[f64]) -> Vec<(usize, usize)> {
        (0..ndim)
            .map(|d| {
                let s = sigmas.get(d).copied().or_else(|| sigmas.first().copied()).unwrap_or(0.0);
                let r = if s > 0.0 {
                    (self.truncate * s + 0.5).floor().max(0.0) as usize
                } else {
                    0
                };
                (r, r)
            })
            .collect()
    }
}

/// N-D separable Gaussian blur into `output` using `ws` ping-pong slots and device-cached taps.
pub fn gaussian_blur_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    output: &DeviceArray<F>,
    settings: &GaussianBlur,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let mut axes = input.shape().axes(settings.axes.as_deref())?;
    if let Some(ch) = settings.channel_axis {
        let ch_ax = input.shape().axis(ch)?;
        axes.retain(|&a| a != ch_ax);
    }
    let per_axis_sigma = check::broadcast_per_axis("sigmas", sigmas, axes.len())?;
    for &s in &per_axis_sigma {
        if !s.is_finite() || s < 0.0 {
            return Err(OpsError::InvalidParameter {
                name: "sigmas",
                reason: format!("sigma must be non-negative and finite, got {s}"),
            });
        }
    }
    let orders = check::broadcast_per_axis("orders", &settings.orders, axes.len())?;
    let radii: Vec<Option<usize>> = match &settings.radii {
        Some(r) => check::broadcast_per_axis("radii", r, axes.len())?.into_iter().map(Some).collect(),
        None => vec![None; axes.len()],
    };

    #[derive(Clone, Copy)]
    struct Pass {
        axis: usize,
        sigma: f64,
        order: u32,
        radius: Option<usize>,
    }

    let passes: Vec<Pass> = axes
        .into_iter()
        .zip(per_axis_sigma)
        .zip(orders)
        .zip(radii)
        .filter(|&(((_, s), _), _)| s > 0.0)
        .map(|(((axis, sigma), order), radius)| Pass { axis, sigma, order, radius })
        .collect();

    separable::run(client, input, output, ws, &passes, |client, src, dst, pass| {
        gaussian_1d_into(
            client,
            src,
            pass.sigma,
            dst,
            &Gaussian1d {
                axis: pass.axis as isize,
                order: pass.order,
                mode: settings.mode,
                cval: settings.cval,
                truncate: settings.truncate,
                radius: pass.radius,
            },
        )
    })
}

/// N-D separable Gaussian blur allocating `output`.
pub fn gaussian_blur<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    settings: &GaussianBlur,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    gaussian_blur_into(client, input, sigmas, &output, settings, &mut ws)?;
    Ok(output)
}

/// Laplacian of Gaussian ($\nabla^2 G_\sigma * I$) into `output`: sums the second-derivative
/// Gaussian responses along each axis on the device without host round-trips.
pub fn gaussian_laplace_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    output: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
    truncate: f64,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let ndim = input.ndim();
    let per_axis_sigma = check::broadcast_per_axis("sigmas", sigmas, ndim)?;
    let term = ws.array::<F>("log_term", input.shape())?;

    for ax in 0..ndim {
        let mut orders = vec![0u32; ndim];
        orders[ax] = 2;
        let settings = GaussianBlur { orders, mode, cval, truncate, radii: None, axes: None, channel_axis: None };
        if ax == 0 {
            gaussian_blur_into(client, input, &per_axis_sigma, output, &settings, ws)?;
        } else {
            gaussian_blur_into(client, input, &per_axis_sigma, &term, &settings, ws)?;
            let accum = ws.array::<F>("log_accum", input.shape())?;
            elementwise::linear_combination_into(client, output, &term, &accum, 1.0, 1.0, 0.0)?;
            elementwise::copy_into(client, &accum, output)?;
        }
    }
    Ok(())
}

/// Allocates and computes the Laplacian of Gaussian ($\nabla^2 G_\sigma * I$).
pub fn gaussian_laplace<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    mode: PadMode,
    cval: f64,
    truncate: f64,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    gaussian_laplace_into(client, input, sigmas, &output, mode, cval, truncate, &mut ws)?;
    Ok(output)
}

/// Euclidean magnitude of the Gaussian gradient ($\|\nabla(G_\sigma * I)\|_2$) into `output`.
pub fn gaussian_grad_mag_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    output: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
    truncate: f64,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let ndim = input.ndim();
    let per_axis_sigma = check::broadcast_per_axis("sigmas", sigmas, ndim)?;
    let accum_sq = DeviceArray::<F>::zeros(client, input.shape().clone())?;
    let deriv = ws.array::<F>("ggm_deriv", input.shape())?;
    let geom = LaunchGeometry::elementwise(client, input.len());

    for ax in 0..ndim {
        let mut orders = vec![0u32; ndim];
        orders[ax] = 1;
        let settings = GaussianBlur { orders, mode, cval, truncate, radii: None, axes: None, channel_axis: None };
        gaussian_blur_into(client, input, &per_axis_sigma, &deriv, &settings, ws)?;
        unsafe {
            add_sq_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                deriv.binding(),
                accum_sq.binding(),
                input.len() as u32,
            );
        }
    }
    unsafe {
        sqrt_scale_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            accum_sq.binding(),
            output.binding(),
            cast::<F>(1.0),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Allocates and computes the Gaussian gradient magnitude ($\|\nabla(G_\sigma * I)\|_2$).
pub fn gaussian_grad_mag<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    mode: PadMode,
    cval: f64,
    truncate: f64,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    gaussian_grad_mag_into(client, input, sigmas, &output, mode, cval, truncate, &mut ws)?;
    Ok(output)
}

/// Difference of Gaussians: `G(low_sigma) * input - G(high_sigma) * input` into `output`.
pub fn difference_of_gaussians_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    low_sigmas: &[f64],
    high_sigmas: &[f64],
    output: &DeviceArray<F>,
    settings: &GaussianBlur,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let low = ws.array::<F>("dog_low", input.shape())?;
    let high = ws.array::<F>("dog_high", input.shape())?;
    gaussian_blur_into(client, input, low_sigmas, &low, settings, ws)?;
    gaussian_blur_into(client, input, high_sigmas, &high, settings, ws)?;
    elementwise::linear_combination_into(client, &low, &high, output, 1.0, -1.0, 0.0)
}

/// Allocates and computes the Difference of Gaussians (`G(low_sigma) - G(high_sigma)`).
pub fn difference_of_gaussians<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    low_sigmas: &[f64],
    high_sigmas: &[f64],
    settings: &GaussianBlur,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    difference_of_gaussians_into(client, input, low_sigmas, high_sigmas, &output, settings, &mut ws)?;
    Ok(output)
}

/// Unsharp masking: `input + amount * (input - G(sigma) * input)` into `output`.
pub fn unsharp_mask_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    amount: f64,
    output: &DeviceArray<F>,
    settings: &GaussianBlur,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let blurred = ws.array::<F>("unsharp_blur", input.shape())?;
    gaussian_blur_into(client, input, sigmas, &blurred, settings, ws)?;
    elementwise::linear_combination_into(client, input, &blurred, output, 1.0 + amount, -amount, 0.0)
}

/// Allocates and computes unsharp masking (`input + amount * (input - G(sigma) * input)`).
pub fn unsharp_mask<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    amount: f64,
    settings: &GaussianBlur,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    unsharp_mask_into(client, input, sigmas, amount, &output, settings, &mut ws)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test, sample_f64};

    fn gaussian_kernel_sums_to_one_and_blurs(client: &Client) {
        let k0 = gaussian_kernel_1d(2.0, 0, 8);
        assert!((k0.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let k1 = gaussian_kernel_1d(2.0, 1, 8);
        assert!(k1.iter().sum::<f64>().abs() < 1e-12);

        let shape = Shape::new([17, 21]);
        let x = sample_f64(shape.len());
        let input = DeviceArray::from_host(client, shape.clone(), cast_all::<f32>(&x)).unwrap();
        let blurred = gaussian_blur(client, &input, &[1.5], &GaussianBlur::default()).unwrap();
        assert_eq!(blurred.shape(), &shape);

        // Constant image stays constant under Reflect / Nearest
        let ones = DeviceArray::<f32>::filled(client, shape, 7.25).unwrap();
        let out = gaussian_blur(client, &ones, &[2.0], &GaussianBlur::default()).unwrap().to_host(client);
        assert_close(&out, &vec![7.25; ones.len()], 1e-5, client.name());
    }
    runtime_test!(test_gaussian_blur, gaussian_kernel_sums_to_one_and_blurs);
}

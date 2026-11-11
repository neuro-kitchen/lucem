---
title: "lumen_ops::restore"
description: "Image restoration: total-variation (`tv_chambolle`, `tv_bregman`), bilateral (`bilateral`), and non-local means (`non_local_means`) denoising, robust noise estimation (`estimate_noise`), rolling-ball background estimation (`rolling_ball`), Richardson-Lucy & Wiener deconvolution (`richardson_lucy`, `wiener_deconvolve`, `unsupervised_wiener`), biharmonic PDE inpainting (`biharmonic_inpaint`), and reliability-guided phase unwrapping (`unwrap_phase`)."
sidebar:
  label: "restore"
  order: 12
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Image restoration: total-variation (`tv_chambolle`, `tv_bregman`), bilateral (`bilateral`), and
non-local means (`non_local_means`) denoising, robust noise estimation (`estimate_noise`),
rolling-ball background estimation (`rolling_ball`), Richardson-Lucy & Wiener deconvolution
(`richardson_lucy`, `wiener_deconvolve`, `unsupervised_wiener`), biharmonic PDE inpainting
(`biharmonic_inpaint`), and reliability-guided phase unwrapping (`unwrap_phase`).

## Types

### `RollingBall`

```rust
pub struct RollingBall
```

Settings for rolling-ball background estimation (`rolling_ball`).

**Fields**

- `radius: f64`: Ball radius in pixels (used when `kernel` is `None`).
- `kernel: Option<(Shape, Vec<f64>)>`: Optional explicit non-flat kernel `(shape, values)` from `ball_kernel` or `ellipsoid_kernel`.
- `mode: PadMode`: Boundary padding mode.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `RichardsonLucy`

```rust
pub struct RichardsonLucy
```

Settings for iterative Richardson-Lucy deconvolution (`richardson_lucy`).

**Fields**

- `num_iter: usize`: Number of multiplicative expectation-maximization iterations.
- `clip: bool`: Whether to clamp the restored image to `[-1.0, 1.0]`.
- `filter_epsilon: Option<f64>`: Minimum denominator threshold below which ratios are stabilized (defaults to `1e-12`).
- `mode: PadMode`: Spatial boundary padding mode.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `WienerDeconvolve`

```rust
pub struct WienerDeconvolve
```

Settings for regularized frequency-domain Wiener deconvolution (`wiener_deconvolve`).

**Fields**

- `balance: f64`: Regularization weight (`balance >= 0`) balancing data fidelity against high-frequency Laplacian roughness.
- `clip: bool`: Whether to clamp the restored output to `[-1.0, 1.0]`.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `UnsupervisedWiener`

```rust
pub struct UnsupervisedWiener
```

Settings for Bayesian unsupervised Wiener deconvolution (`unsupervised_wiener`).

**Fields**

- `num_iter: usize`: Total number of Gibbs-sampling / empirical-Bayes iterations.
- `burnin: usize`: Number of initial burn-in iterations discarded before averaging.
- `clip: bool`: Whether to clamp the final deconvolved image to `[-1.0, 1.0]`.
- `seed: u64`: Deterministic RNG seed.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `TvChambolle`

```rust
pub struct TvChambolle
```

Settings for Chambolle total-variation (TV) denoising (`tv_chambolle`).

**Fields**

- `weight: f64`: Denoising regularization weight ($\lambda > 0$; larger values produce smoother images).
- `eps: f64`: Relative stopping tolerance on primal iterates (`0.0` runs all `max_num_iter` iterations).
- `max_num_iter: usize`: Maximum number of primal-dual iterations.
- `isotropic: bool`: Isotropic ($L_2$ norm of gradient) vs. anisotropic ($L_1$) total variation.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `TvBregman`

```rust
pub struct TvBregman
```

Settings for Split-Bregman total-variation denoising (`tv_bregman`).

**Fields**

- `weight: f64`: Fidelity weight $\mu > 0$ on $\|u - f\|_2^2$ (smaller values produce stronger smoothing).
- `max_num_iter: usize`: Maximum number of Bregman iterations.
- `eps: f64`: Relative stopping tolerance on primal iterates.
- `isotropic: bool`: Isotropic ($L_2$) vs. anisotropic ($L_1$) shrinkage.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `Bilateral`

```rust
pub struct Bilateral
```

Settings for 2-D bilateral edge-preserving smoothing (`bilateral`).

**Fields**

- `sigma_color: Option<f64>`: Radiometric (intensity/range) Gaussian standard deviation. When `None`, estimated from the sample standard deviation of the input image (clamped to `>= 1e-4`).
- `sigma_spatial: f64`: Spatial Gaussian standard deviation in pixels.
- `win_size: Option<usize>`: Optional odd window size (`>= 1`). Defaults to `max(5, 2 * ceil(3 * sigma_spatial) + 1)`.
- `mode: PadMode`: Boundary padding mode.
- `cval: f64`: Constant fill value for `PadMode::Constant`.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `NonLocalMeans`

```rust
pub struct NonLocalMeans
```

Settings for 2-D patch-based Non-Local Means (NLM) denoising (`non_local_means`).

**Fields**

- `patch_size: usize`: Side length of square comparison patches (e.g. `7` for a `7x7` patch, radius `3`).
- `patch_distance: usize`: Search radius in pixels (`(2 * patch_distance + 1)^2` candidate patches evaluated per pixel).
- `h: f64`: Cut-off smoothing parameter `h` (> 0).
- `sigma: f64`: Optional known Gaussian noise standard deviation $\sigma \ge 0$ subtracted from patch distances.
- `mode: PadMode`: Boundary padding mode.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `BiharmonicInpaint`

```rust
pub struct BiharmonicInpaint
```

Settings for 2-D biharmonic PDE inpainting (`biharmonic_inpaint`).

**Fields**

- `harmonic_iter: usize`: Number of harmonic ($\nabla^2 u = 0$) warm-start iterations used to propagate boundary values across missing regions before fourth-order refinement.
- `biharmonic_iter: usize`: Number of fourth-order biharmonic ($\nabla^4 u = 0$) relaxation iterations.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

## Functions

### `ball_kernel`

```rust
pub fn ball_kernel(radius: f64, ndim: usize) -> (Shape, Vec<f64>)
```

Generates an N-D spherical rolling-ball kernel of radius `radius`.

At offset $z \in [-r, r]^d$ ($r = \lfloor \text{radius} \rfloor$), the kernel height difference
from the ball apex is $R - \sqrt{R^2 - \|z\|_2^2}$ inside the ball ($\|z\|_2 \le R$) and
$+\infty$ outside.

### `ellipsoid_kernel`

```rust
pub fn ellipsoid_kernel(shape: &[usize], intensity: f64) -> (Shape, Vec<f64>)
```

Generates an N-D ellipsoidal rolling-ball kernel of bounding box `shape` and apex height
`intensity`.

Each dimension $d$ has semi-axis $a_d = \lfloor \text{shape}[d] / 2 \rfloor$. Inside the
ellipsoid $\sum_d (z_d / a_d)^2 \le 1$, the height difference from the apex is
$\text{intensity} \cdot \bigl(1 - \sqrt{1 - \sum_d (z_d / a_d)^2}\bigr)$, and $+\infty$ outside.

### `rolling_ball_into`

```rust
pub fn rolling_ball_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    background: &DeviceArray<F>,
    settings: &RollingBall,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Estimates the smooth background intensity surface of `input` into `background` by rolling a
non-flat ball or ellipsoid kernel underneath the intensity profile.

### `rolling_ball`

```rust
pub fn rolling_ball<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &RollingBall,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns the rolling-ball estimated background of `input`.

### `richardson_lucy_into`

```rust
pub fn richardson_lucy_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &RichardsonLucy,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Restores `input` blurred by 2-D `psf` into `output` using the Richardson-Lucy (1972/1974)
multiplicative update algorithm.

### `richardson_lucy`

```rust
pub fn richardson_lucy<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    settings: &RichardsonLucy,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns Richardson-Lucy deconvolved `input`.

### `wiener_deconvolve_into`

```rust
pub fn wiener_deconvolve_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &WienerDeconvolve,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Deconvolves 2-D `input` with `psf` into `output` using regularized frequency-domain Wiener
filtering:
$$\hat{X}(\omega) = \frac{\overline{H(\omega)}\,Y(\omega)}{|H(\omega)|^2 + \text{balance}\cdot |L(\omega)|^2}$$
where $L$ is the 2-D discrete Laplacian high-pass filter.

### `wiener_deconvolve`

```rust
pub fn wiener_deconvolve<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    settings: &WienerDeconvolve,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns Wiener-deconvolved `input`.

### `unsupervised_wiener`

```rust
pub fn unsupervised_wiener<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    psf: &DeviceArray<F>,
    settings: &UnsupervisedWiener,
) -> OpsResult<(DeviceArray<F>, Vec<f64>)>
```

Unsupervised Bayesian Wiener deconvolution (Orieux, Giovannelli & Rodet, 2010): jointly estimates
the restored image and the noise-to-prior regularization ratio `balance = sigma_n^2 / sigma_x^2`
across `settings.num_iter` iterations, returning `(deconvolved_image, balance_chain)`.

### `tv_chambolle_into`

```rust
pub fn tv_chambolle_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &TvChambolle,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Denoises `input` into `output` using Chambolle's (2004) primal-dual Total Variation projection
algorithm on the device.

### `tv_chambolle`

```rust
pub fn tv_chambolle<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &TvChambolle,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns Chambolle total-variation denoised `input`.

### `tv_bregman_into`

```rust
pub fn tv_bregman_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &TvBregman,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Denoises `input` into `output` using Goldstein & Osher's (2009) Split-Bregman TV algorithm.

### `tv_bregman`

```rust
pub fn tv_bregman<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &TvBregman,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns Split-Bregman total-variation denoised `input`.

### `bilateral_into`

```rust
pub fn bilateral_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &Bilateral,
) -> OpsResult<()>
```

Applies a 2-D edge-preserving bilateral filter into `output`.

### `bilateral`

```rust
pub fn bilateral<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &Bilateral,
) -> OpsResult<DeviceArray<F>>
```

Allocates and applies a 2-D edge-preserving bilateral filter to `input`.

### `non_local_means_into`

```rust
pub fn non_local_means_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &NonLocalMeans,
) -> OpsResult<()>
```

Denoises `input` into `output` using Buades, Coll & Morel's (2005) Non-Local Means algorithm.

### `non_local_means`

```rust
pub fn non_local_means<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &NonLocalMeans,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns Non-Local Means denoised `input`.

### `estimate_noise`

```rust
pub fn estimate_noise<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<f64>
```

Estimates the additive Gaussian white noise standard deviation $\hat{\sigma}$ of `input` using
Immerkær's (1996) fast 3x3 zero-mean Laplacian operator on the device for 2-D images (and the
Donoho-Johnstone diagonal Haar wavelet median absolute deviation for general N-D images).

### `biharmonic_inpaint_into`

```rust
pub fn biharmonic_inpaint_into<F: ImageFloat, M: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    mask: &DeviceArray<M>,
    output: &DeviceArray<F>,
    settings: &BiharmonicInpaint,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Inpaints missing/damaged regions of 2-D `input` (where `mask != 0`) into `output` by solving the
discrete biharmonic equation $\nabla^4 u = 0$ with reflective Neumann boundary conditions.

### `biharmonic_inpaint`

```rust
pub fn biharmonic_inpaint<F: ImageFloat, M: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    mask: &DeviceArray<M>,
    settings: &BiharmonicInpaint,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns biharmonic-inpainted `input` over pixels where `mask != 0`.

### `unwrap_phase_poisson_rhs_2d`

```rust
pub fn unwrap_phase_poisson_rhs_2d<F: ImageFloat>(
    client: &Client,
    phase: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Computes the 2-D wrapped-gradient divergence $\nabla \cdot W(\nabla \phi)$ on the device (the
right-hand side of the least-squares phase-unwrapping Poisson equation).

### `unwrap_phase_into`

```rust
pub fn unwrap_phase_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    wrap_around: Option<&[bool]>,
) -> OpsResult<()>
```

Unwraps a 1-D, 2-D, or 3-D phase array measured in radians modulo $2\pi$ into a continuous phase
field `output` using Herraez, Burton, Lalor & Gdeisat's (2002) reliability-guided non-continuous
path algorithm.

`wrap_around` specifies per-axis periodic boundary wrapping (or `false` on every axis when `None`).

### `unwrap_phase`

```rust
pub fn unwrap_phase<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    wrap_around: Option<&[bool]>,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns the unwrapped continuous phase of `input`.

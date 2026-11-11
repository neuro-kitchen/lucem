---
title: "lumen_ops::filter"
description: "Spatial, derivative, order-statistic, ridge, spectral, and windowed-histogram filters."
sidebar:
  label: "filter"
  order: 6
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Spatial, derivative, order-statistic, ridge, spectral, and windowed-histogram filters.

## Types

### `BoxBlur1d`

```rust
pub struct BoxBlur1d
```

Settings for `box_blur_1d`.

**Fields**

- `axis: isize`
- `mode: PadMode`
- `cval: f64`
- `origin: i32`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `BoxBlur`

```rust
pub struct BoxBlur
```

Settings for N-D separable `box_blur`.

**Fields**

- `mode: PadMode`
- `cval: f64`
- `origins: Option<Vec<i32>>`
- `axes: Option<Vec<isize>>`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `Correlate1d`

```rust
pub struct Correlate1d
```

Settings for 1-D correlation/convolution along an axis.

**Fields**

- `axis: isize`
- `mode: PadMode`
- `cval: f64`
- `origin: i32`

**Methods**

```rust
pub fn halo(&self, size: usize) -> (usize, usize)
```

Samples read past each side `(before, after)` along the filtered axis.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `Correlate2d`

```rust
pub struct Correlate2d
```

Settings for 2-D / N-D correlation and convolution.

**Fields**

- `mode: PadMode`
- `cval: f64`
- `origin: (i32, i32)`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `Gaussian1d`

```rust
pub struct Gaussian1d
```

Settings for 1-D Gaussian filtering (`gaussian_1d`).

**Fields**

- `axis: isize`
- `order: u32`
- `mode: PadMode`
- `cval: f64`
- `truncate: f64`
- `radius: Option<usize>`

**Methods**

```rust
pub fn resolved_radius(&self, sigma: f64) -> usize
```

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `GaussianBlur`

```rust
pub struct GaussianBlur
```

Settings for N-D separable Gaussian smoothing (`gaussian_blur`).

**Fields**

- `orders: Vec<u32>`
- `mode: PadMode`
- `cval: f64`
- `truncate: f64`
- `radii: Option<Vec<usize>>`
- `axes: Option<Vec<isize>>`
- `channel_axis: Option<isize>`

**Methods**

```rust
pub fn halo(&self, ndim: usize, sigmas: &[f64]) -> Vec<(usize, usize)>
```

Halo `(before, after)` per axis for tiling large images.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `GradientFilter`

```rust
pub struct GradientFilter
```

Settings for directional or isotropic edge stencils.

**Fields**

- `axis: Option<isize>`: Specific axis for a directional derivative, or `None` for the isotropic Euclidean magnitude $\sqrt{\frac{1}{D}\sum_{d=0}^{D-1} G_d^2}$ across all axes.
- `mode: PadMode`
- `cval: f64`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `Extrema1d`

```rust
pub struct Extrema1d
```

Settings for 1-D min/max filters (`min_filter_1d`, `max_filter_1d`).

**Fields**

- `axis: isize`
- `mode: PadMode`
- `cval: f64`
- `origin: i32`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `OrderFilter`

```rust
pub struct OrderFilter
```

Settings for N-D order-statistic filters.

**Fields**

- `mode: PadMode`
- `cval: f64`
- `origins: Option<Vec<i32>>`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `RidgeFilter`

```rust
pub struct RidgeFilter
```

Settings for multi-scale ridge filters (`frangi`, `sato`, `meijering`).

**Fields**

- `sigmas: Vec<f64>`: Gaussian scales $\sigma$ evaluated and combined via elementwise maximum.
- `black_ridges: bool`: `true` to detect dark ridges on a bright background; `false` for bright fluorescence neurites/vessels.
- `mode: PadMode`
- `cval: f64`
- `beta: f64`: Frangi blobness sensitivity parameter $\beta$.
- `gamma: f64`: Frangi structureness sensitivity parameter $\gamma$ (`c`).

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `StencilKind`

```rust
pub enum StencilKind
```

Separable derivative stencil family.

**Variants**

- `Sobel`: Central difference `[-1, 0, 1]` along derivative axis, `[1, 2, 1]` smoothing on orthogonal axes.
- `Prewitt`: Central difference `[-1, 0, 1]` along derivative axis, `[1, 1, 1]` smoothing on orthogonal axes.
- `Scharr`: Rotational-symmetry optimized 3-tap stencil (`[-1, 0, 1]` and `[3, 10, 3]`).
- `Farid`: 5-tap 1st-order optimal equiripple Farid stencil.

Implements `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`.

### `WindowedHistStat`

```rust
pub enum WindowedHistStat
```

Local statistic computed over the quantized sliding neighborhood.

**Variants**

- `Mean`
- `Min`
- `Max`
- `Gradient`
- `Autolevel`
- `Equalize`

Implements `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`.

## Functions

### `box_blur_1d_into`

```rust
pub fn box_blur_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    output: &DeviceArray<F>,
    settings: &BoxBlur1d,
) -> OpsResult<()>
```

1-D uniform box filter of window `size` along `settings.axis` into `output`.

### `box_blur_1d`

```rust
pub fn box_blur_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    settings: &BoxBlur1d,
) -> OpsResult<DeviceArray<F>>
```

1-D box filter allocating `output`.

### `box_blur_into`

```rust
pub fn box_blur_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sizes: &[usize],
    output: &DeviceArray<F>,
    settings: &BoxBlur,
    ws: &mut Workspace,
) -> OpsResult<()>
```

N-D separable uniform box filter across axes using `ws` ping-pong buffers.

### `box_blur`

```rust
pub fn box_blur<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sizes: &[usize],
    settings: &BoxBlur,
) -> OpsResult<DeviceArray<F>>
```

N-D separable uniform box filter allocating `output`.

### `correlate_1d_into`

```rust
pub fn correlate_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &Correlate1d,
) -> OpsResult<()>
```

1-D correlation into `output` using device-resident `weights`.

### `correlate_1d`

```rust
pub fn correlate_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &[f64],
    settings: &Correlate1d,
) -> OpsResult<DeviceArray<F>>
```

1-D correlation allocating `output` and caching `weights` on the device.

### `convolve_1d`

```rust
pub fn convolve_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &[f64],
    settings: &Correlate1d,
) -> OpsResult<DeviceArray<F>>
```

1-D convolution into `output` (correlation with reversed weights and mirrored origin).

### `correlate_2d_into`

```rust
pub fn correlate_2d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &Correlate2d,
) -> OpsResult<()>
```

2-D correlation into `output` with device-resident `[k_h, k_w]` `weights`.

### `correlate_2d`

```rust
pub fn correlate_2d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &DeviceArray<F>,
    settings: &Correlate2d,
) -> OpsResult<DeviceArray<F>>
```

2-D correlation allocating `output`.

### `gaussian_kernel_1d`

```rust
pub fn gaussian_kernel_1d(sigma: f64, order: u32, radius: usize) -> Vec<f64>
```

Computes the 1-D Gaussian (or Gaussian derivative of `order`) kernel of half-width `radius`
(`2 * radius + 1` taps) in `f64`.

Uses the exact recurrence for polynomial Hermite derivatives of the discrete normalized Gaussian
so orders 0, 1, 2, 3 match reference implementations to machine precision.

### `gaussian_1d_into`

```rust
pub fn gaussian_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigma: f64,
    output: &DeviceArray<F>,
    settings: &Gaussian1d,
) -> OpsResult<()>
```

1-D Gaussian filter along `settings.axis` into `output`, caching the tap weights per device.

### `gaussian_1d`

```rust
pub fn gaussian_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigma: f64,
    settings: &Gaussian1d,
) -> OpsResult<DeviceArray<F>>
```

1-D Gaussian filter allocating `output`.

### `gaussian_blur_into`

```rust
pub fn gaussian_blur_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    output: &DeviceArray<F>,
    settings: &GaussianBlur,
    ws: &mut Workspace,
) -> OpsResult<()>
```

N-D separable Gaussian blur into `output` using `ws` ping-pong slots and device-cached taps.

### `gaussian_blur`

```rust
pub fn gaussian_blur<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    settings: &GaussianBlur,
) -> OpsResult<DeviceArray<F>>
```

N-D separable Gaussian blur allocating `output`.

### `gaussian_laplace_into`

```rust
pub fn gaussian_laplace_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    output: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
    truncate: f64,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Laplacian of Gaussian ($\nabla^2 G_\sigma * I$) into `output`: sums the second-derivative
Gaussian responses along each axis on the device without host round-trips.

### `gaussian_laplace`

```rust
pub fn gaussian_laplace<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    mode: PadMode,
    cval: f64,
    truncate: f64,
) -> OpsResult<DeviceArray<F>>
```

Allocates and computes the Laplacian of Gaussian ($\nabla^2 G_\sigma * I$).

### `gaussian_grad_mag_into`

```rust
pub fn gaussian_grad_mag_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    output: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
    truncate: f64,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Euclidean magnitude of the Gaussian gradient ($\|\nabla(G_\sigma * I)\|_2$) into `output`.

### `gaussian_grad_mag`

```rust
pub fn gaussian_grad_mag<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    mode: PadMode,
    cval: f64,
    truncate: f64,
) -> OpsResult<DeviceArray<F>>
```

Allocates and computes the Gaussian gradient magnitude ($\|\nabla(G_\sigma * I)\|_2$).

### `difference_of_gaussians_into`

```rust
pub fn difference_of_gaussians_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    low_sigmas: &[f64],
    high_sigmas: &[f64],
    output: &DeviceArray<F>,
    settings: &GaussianBlur,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Difference of Gaussians: `G(low_sigma) * input - G(high_sigma) * input` into `output`.

### `difference_of_gaussians`

```rust
pub fn difference_of_gaussians<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    low_sigmas: &[f64],
    high_sigmas: &[f64],
    settings: &GaussianBlur,
) -> OpsResult<DeviceArray<F>>
```

Allocates and computes the Difference of Gaussians (`G(low_sigma) - G(high_sigma)`).

### `unsharp_mask_into`

```rust
pub fn unsharp_mask_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    amount: f64,
    output: &DeviceArray<F>,
    settings: &GaussianBlur,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Unsharp masking: `input + amount * (input - G(sigma) * input)` into `output`.

### `unsharp_mask`

```rust
pub fn unsharp_mask<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    sigmas: &[f64],
    amount: f64,
    settings: &GaussianBlur,
) -> OpsResult<DeviceArray<F>>
```

Allocates and computes unsharp masking (`input + amount * (input - G(sigma) * input)`).

### `stencil_filter_into`

```rust
pub fn stencil_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    kind: StencilKind,
    settings: &GradientFilter,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Applies a separable gradient filter (`Sobel`, `Prewitt`, `Scharr`, or `Farid`) into `output`.

### `sobel`

```rust
pub fn sobel<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &GradientFilter,
) -> OpsResult<DeviceArray<F>>
```

Sobel gradient filter (directional when `settings.axis = Some(ax)`, magnitude when `None`).

### `prewitt`

```rust
pub fn prewitt<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &GradientFilter,
) -> OpsResult<DeviceArray<F>>
```

Prewitt gradient filter.

### `scharr`

```rust
pub fn scharr<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &GradientFilter,
) -> OpsResult<DeviceArray<F>>
```

Scharr gradient filter.

### `farid`

```rust
pub fn farid<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &GradientFilter,
) -> OpsResult<DeviceArray<F>>
```

Farid 5-tap equiripple gradient filter.

### `roberts_into`

```rust
pub fn roberts_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
    ws: &mut Workspace,
) -> OpsResult<()>
```

2-D Roberts cross diagonal edge magnitude into `output`.

### `roberts`

```rust
pub fn roberts<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>>
```

2-D Roberts cross diagonal edge magnitude allocating `output`.

### `laplace_into`

```rust
pub fn laplace_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
) -> OpsResult<()>
```

N-D discrete Laplacian ($\sum_{d=0}^{D-1} [1, -2, 1]_d * I$) into `output`.

### `laplace`

```rust
pub fn laplace<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>>
```

N-D discrete Laplacian allocating `output`.

### `extrema_1d_into`

```rust
pub fn extrema_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    output: &DeviceArray<F>,
    settings: &Extrema1d,
    is_max: bool,
) -> OpsResult<()>
```

1-D minimum (`is_max = false`) or maximum (`is_max = true`) filter of window `size`.

### `min_filter_1d`

```rust
pub fn min_filter_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    settings: &Extrema1d,
) -> OpsResult<DeviceArray<F>>
```

1-D minimum filter allocating `output`.

### `max_filter_1d`

```rust
pub fn max_filter_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    settings: &Extrema1d,
) -> OpsResult<DeviceArray<F>>
```

1-D maximum filter allocating `output`.

### `extrema_filter_into`

```rust
pub fn extrema_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
    is_max: bool,
    ws: &mut Workspace,
) -> OpsResult<()>
```

N-D minimum (`is_max = false`) or maximum (`is_max = true`) filter over `footprint`.

Automatically decomposes rectangular footprints into separable 1-D passes across axes via
`core::separable::run`.

### `min_filter_into`

```rust
pub fn min_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
    ws: &mut Workspace,
) -> OpsResult<()>
```

N-D minimum filter over `footprint` into `output`.

### `min_filter`

```rust
pub fn min_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>>
```

N-D minimum filter allocating `output`.

### `max_filter_into`

```rust
pub fn max_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
    ws: &mut Workspace,
) -> OpsResult<()>
```

N-D maximum filter over `footprint` into `output`.

### `max_filter`

```rust
pub fn max_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>>
```

N-D maximum filter allocating `output`.

### `rank_filter_into`

```rust
pub fn rank_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    rank: isize,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
) -> OpsResult<()>
```

2-D rank filter selecting the `rank`-th smallest element (`0 <= rank < footprint.count_active()`;
negative `rank` counts from the end).

### `rank_filter`

```rust
pub fn rank_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    rank: isize,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>>
```

2-D rank filter allocating `output`.

### `median_filter_into`

```rust
pub fn median_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
    ws: &mut Workspace,
) -> OpsResult<()>
```

2-D median filter into `output`: uses `med9` for $3 \times 3$ windows and exact rank selection
for general footprints (averaging the two middle ranks when `count_active` is even).

### `median_filter`

```rust
pub fn median_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>>
```

2-D median filter allocating `output`.

### `percentile_filter`

```rust
pub fn percentile_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    percentile: f64,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>>
```

2-D percentile filter (`percentile` in `0.0..=100.0`) over `footprint`.

### `rank_order`

```rust
pub fn rank_order<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<(DeviceArray<u32>, Vec<f64>)>
```

Replaces every pixel by its non-negative integer rank `0..num_unique` among unique sorted values
of the image, returning `(ranked_image, sorted_unique_values)`.

### `frangi_into`

```rust
pub fn frangi_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &RidgeFilter,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Multi-scale 2-D Frangi vesselness filter into `output`.

### `frangi`

```rust
pub fn frangi<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &RidgeFilter,
) -> OpsResult<DeviceArray<F>>
```

Multi-scale 2-D Frangi vesselness filter allocating `output`.

### `sato_into`

```rust
pub fn sato_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &RidgeFilter,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Multi-scale 2-D Sato tubeness filter into `output`.

### `sato`

```rust
pub fn sato<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &RidgeFilter,
) -> OpsResult<DeviceArray<F>>
```

Multi-scale 2-D Sato tubeness filter allocating `output`.

### `meijering_into`

```rust
pub fn meijering_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &RidgeFilter,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Multi-scale 2-D Meijering neuriteness filter into `output`.

### `meijering`

```rust
pub fn meijering<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &RidgeFilter,
) -> OpsResult<DeviceArray<F>>
```

Multi-scale 2-D Meijering neuriteness filter allocating `output`.

### `fourier_gaussian_into`

```rust
pub fn fourier_gaussian_into<F: ImageFloat>(
    client: &Client,
    re: &DeviceArray<F>,
    im: &DeviceArray<F>,
    sigmas: &[f64],
) -> OpsResult<()>
```

Multiplies a Fourier-domain spectrum `(re, im)` in-place along each axis by the Gaussian
frequency transfer function $\exp(-2 \pi^2 \sigma_d^2 f_d^2)$.

### `fourier_box_into`

```rust
pub fn fourier_box_into<F: ImageFloat>(
    client: &Client,
    re: &DeviceArray<F>,
    im: &DeviceArray<F>,
    sizes: &[f64],
) -> OpsResult<()>
```

Multiplies a Fourier-domain spectrum `(re, im)` by the uniform box sinc transfer function
$\prod_d \operatorname{sinc}(s_d f_d)$.

### `fourier_shift_into`

```rust
pub fn fourier_shift_into<F: ImageFloat>(
    client: &Client,
    re: &DeviceArray<F>,
    im: &DeviceArray<F>,
    shifts: &[f64],
) -> OpsResult<()>
```

Multiplies a Fourier-domain spectrum `(re, im)` by the linear phase shift
$\exp(-i 2\pi \sum_d \text{shift}_d f_d)$.

### `butterworth`

```rust
pub fn butterworth<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    cutoff_ratio: f64,
    high_pass: bool,
    order: f64,
) -> OpsResult<DeviceArray<F>>
```

Applies an N-D frequency-domain Butterworth low-pass (`high_pass = false`) or high-pass
(`high_pass = true`) filter with normalized cutoff `cutoff_ratio` in `(0.0, 0.5]`.

### `gabor_kernel_2d`

```rust
pub fn gabor_kernel_2d(
    frequency: f64,
    theta: f64,
    sigma_x: f64,
    sigma_y: f64,
    n_stds: f64,
    offset: f64,
) -> (Shape, Vec<f64>, Vec<f64>)
```

Generates a 2-D complex Gabor kernel `(real_kernel, imag_kernel)` on the host in `f64`.

### `gabor_2d`

```rust
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
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)>
```

Applies a 2-D Gabor filter, returning `(real_response, imag_response)` on the device.

### `windowed_hist_stat_into`

```rust
pub fn windowed_hist_stat_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    stat: WindowedHistStat,
    max_bin: u32,
) -> OpsResult<()>
```

Applies a local windowed-histogram statistic (`Mean`, `Min`, `Max`, `Gradient`, `Autolevel`, or
`Equalize`) over `footprint` for a 2-D `u32` quantized image (`0..=max_bin`).

### `windowed_hist_stat`

```rust
pub fn windowed_hist_stat<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    stat: WindowedHistStat,
    max_bin: u32,
) -> OpsResult<DeviceArray<F>>
```

Allocates and computes a local windowed-histogram statistic over `footprint`.

### `windowed_entropy_into`

```rust
pub fn windowed_entropy_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    mode: PadMode,
) -> OpsResult<()>
```

Computes local Shannon entropy (in bits, base 2) over `footprint` into `output`.

### `windowed_entropy`

```rust
pub fn windowed_entropy<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    mode: PadMode,
) -> OpsResult<DeviceArray<F>>
```

Allocates and computes local Shannon entropy (in bits) over `footprint`.

## `lumen_ops::filter::gradient`

Edge and gradient filters: Sobel, Prewitt, Scharr, Farid, Roberts cross, and discrete Laplacian.

### `directional_stencil_into`

```rust
pub fn directional_stencil_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    deriv_axis: usize,
    kind: StencilKind,
    mode: PadMode,
    cval: f64,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Computes a single directional derivative along `deriv_axis` for `kind` into `output`.

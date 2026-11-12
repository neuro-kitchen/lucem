---
title: "lucem_ops::transform"
description: "Geometric image transforms, multi-resolution pyramids, integral images, and projection transforms (Hough & Radon)."
sidebar:
  label: "transform"
  order: 15
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Geometric image transforms, multi-resolution pyramids, integral images, and projection transforms
(Hough & Radon).

## Types

### `HoughLines`

```rust
pub struct HoughLines
```

Output of the 2-D straight-line Hough transform (`hough_lines`).

**Fields**

- `accumulator: DeviceArray<u32>`: Accumulator array of shape `[distances.len(), angles.len()]`.
- `angles: Vec<f64>`: Tested angles $\theta$ in radians.
- `distances: Vec<f64>`: Signed radial distance bins $\rho$ in pixels (`-diag ..= +diag`).

Implements `Clone`, `Debug`.

### `HoughLinePeak`

```rust
pub struct HoughLinePeak
```

A detected peak `(votes, angle_rad, distance_px)` in a Hough line accumulator.

**Fields**

- `votes: u32`
- `angle: f64`
- `distance: f64`

Implements `Clone`, `Copy`, `Debug`, `PartialEq`.

### `HoughEllipse`

```rust
pub struct HoughEllipse
```

Detected ellipse parameters from `hough_ellipses`.

**Fields**

- `votes: usize`
- `yc: f64`
- `xc: f64`
- `a: f64`
- `b: f64`
- `orientation: f64`

Implements `Clone`, `Copy`, `Debug`, `PartialEq`.

### `Pyramid`

```rust
pub struct Pyramid
```

Settings for `pyramid_reduce`, `pyramid_expand`, `gaussian_pyramid`, and `laplacian_pyramid`.

**Fields**

- `downscale: f64`
- `sigma: Option<f64>`: Optional Gaussian smoothing $\sigma$; defaults to `2 * downscale / 6.0`.
- `order: usize`
- `mode: PadMode`
- `cval: f64`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `WarpSettings`

```rust
pub struct WarpSettings
```

Shared interpolation and boundary settings for geometric warps.

**Fields**

- `order: usize`: B-spline interpolation order (`0..=5`, where `0` is nearest-neighbor, `1` bilinear, `3` bicubic).
- `mode: PadMode`
- `cval: f64`
- `prefilter: bool`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `BackprojectionFilter`

```rust
pub enum BackprojectionFilter
```

Frequency-domain filter window for `iradon` filtered backprojection.

**Variants**

- `None`: Unfiltered backprojection (lacks ramp weighting).
- `Ramp`: Ideal Ram-Lak $|\omega|$ ramp filter.
- `SheppLogan`: Shepp-Logan $\operatorname{sinc}$-windowed ramp filter.
- `Cosine`: Cosine-windowed ramp filter.
- `Hamming`: Hamming-windowed ramp filter.
- `Hann`: Hann-windowed ramp filter.

Implements `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`.

## Functions

### `integral_image_into`

```rust
pub fn integral_image_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Computes the N-D inclusive summed-area table (integral image) of `input` into `output`.

### `integral_image`

```rust
pub fn integral_image<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns the N-D inclusive summed-area table (integral image) of `input`.

### `integrate_box`

```rust
pub fn integrate_box<F: ImageFloat>(
    client: &Client,
    sat: &DeviceArray<F>,
    boxes: &[[usize; 4]],
) -> OpsResult<DeviceArray<F>>
```

Evaluates $O(1)$ rectangular sums $\sum_{r=r_0}^{r_1} \sum_{c=c_0}^{c_1} I[r, c]$ for a batch of
inclusive 2-D bounding boxes `&[[r0, c0, r1, c1]]` from a 2-D `integral_image` `sat`.

### `hough_lines`

```rust
pub fn hough_lines<E: Numeric + CubeElement>(
    client: &Client,
    edges: &DeviceArray<E>,
    angles: Option<&[f64]>,
) -> OpsResult<HoughLines>
```

Computes the straight-line Hough accumulator for non-zero edge pixels in a 2-D image over `angles`
(in radians; defaults to 180 uniformly spaced angles in $[-\pi/2, \pi/2)$ when `None`).

### `hough_line_peaks`

```rust
pub fn hough_line_peaks(
    client: &Client,
    hough: &HoughLines,
    min_distance: usize,
    min_angle: usize,
    threshold: Option<u32>,
    num_peaks: Option<usize>,
) -> OpsResult<Vec<HoughLinePeak>>
```

Extracts prominent peaks from a `HoughLines` accumulator with minimum `(min_xdistance, min_ydistance)`
bin separation between peaks.

### `hough_circles`

```rust
pub fn hough_circles<E: Numeric + CubeElement>(
    client: &Client,
    edges: &DeviceArray<E>,
    radii: &[f64],
    normalize: bool,
) -> OpsResult<DeviceArray<f32>>
```

Computes a 3-D circular Hough accumulator `[radii.len(), height, width]` for non-zero edge pixels
in `edges` across candidate `radii`.

### `hough_ellipses`

```rust
pub fn hough_ellipses<E: Numeric + CubeElement>(
    client: &Client,
    edges: &DeviceArray<E>,
    min_size: usize,
    max_size: Option<usize>,
    accuracy: f64,
    threshold: usize,
) -> OpsResult<Vec<HoughEllipse>>
```

Detects ellipses in a 2-D binary edge image using Xie & Ji's randomized/pair-voting randomized
Hough transform (voting over minor semi-axis $b$ for candidate antipodal pairs).

### `radon`

```rust
pub fn radon<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    thetas_deg: Option<&[f64]>,
) -> OpsResult<DeviceArray<F>>
```

Computes the 2-D forward Radon sinogram of `image` `[height, width]` across `thetas_deg` (in degrees,
defaulting to `0..180`), returning an array of shape `[n_bins, thetas_deg.len()]`.

### `iradon`

```rust
pub fn iradon<F: ImageFloat>(
    client: &Client,
    sinogram: &DeviceArray<F>,
    thetas_deg: &[f64],
    output_size: Option<usize>,
    filter: BackprojectionFilter,
) -> OpsResult<DeviceArray<F>>
```

Reconstructs a 2-D `[output_size, output_size]` image from a `sinogram` `[n_bins, n_angles]`
using filtered backprojection (`filter` in spatial/detector domain + `iradon_backproject_kernel`).

### `downsample_mean`

```rust
pub fn downsample_mean<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    factors: [usize; 2],
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>>
```

Downsamples a 2-D image by integer block factors `[factor_y, factor_x]` by averaging each tile,
padding partial border tiles with `mode` and `cval`.

### `resize_mean`

```rust
pub fn resize_mean<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    output_shape: [usize; 2],
) -> OpsResult<DeviceArray<F>>
```

Resizes a 2-D image to `output_shape` (`[out_h, out_w]`) by exact local box-overlap area averaging.

### `pyramid_reduce`

```rust
pub fn pyramid_reduce<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &Pyramid,
) -> OpsResult<DeviceArray<F>>
```

Smooths a 2-D image with a Gaussian filter and downsamples it by `settings.downscale`.

### `pyramid_expand`

```rust
pub fn pyramid_expand<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    output_shape: Option<[usize; 2]>,
    settings: &Pyramid,
) -> OpsResult<DeviceArray<F>>
```

Upsamples a 2-D image by `settings.downscale` (or into `output_shape` when specified) and smooths
with a Gaussian filter.

### `gaussian_pyramid`

```rust
pub fn gaussian_pyramid<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    max_layer: usize,
    settings: &Pyramid,
) -> OpsResult<Vec<DeviceArray<F>>>
```

Builds a multi-scale Gaussian pyramid `[level_0, level_1, ...]` where `level_0` is a copy of `image`
and at most `max_layer` reductions are performed (stopping early when any dimension reaches `1`).

### `laplacian_pyramid`

```rust
pub fn laplacian_pyramid<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    max_layer: usize,
    settings: &Pyramid,
) -> OpsResult<Vec<DeviceArray<F>>>
```

Builds a multi-scale Laplacian pyramid `[L_0, L_1, ..., G_N]` where each bandpass layer
$L_k = G_k - \operatorname{expand}(G_{k+1})$ and the final layer is the coarsest Gaussian residual.

### `sample_coords`

```rust
pub fn sample_coords<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    coords_y: &DeviceArray<F>,
    coords_x: &DeviceArray<F>,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>>
```

Samples a 2-D image at device coordinate arrays `(coords_y, coords_x)`.

### `affine_warp`

```rust
pub fn affine_warp<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    inverse_matrix: &[f64; 9],
    output_shape: Option<[usize; 2]>,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>>
```

Warps a 2-D image into `output_shape` (`[out_h, out_w]`, defaulting to `image.shape()`) using a
$3 \times 3$ homogeneous inverse transform `inverse_matrix` mapping output `[y_out, x_out, 1]` to
source `[y_in, x_in, w_in]`.

### `shift`

```rust
pub fn shift<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    shift_yx: [f64; 2],
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>>
```

Translates a 2-D image by `(shift_y, shift_x)` pixels (`output[y, x] = input[y - shift_y, x - shift_x]`).

### `zoom`

```rust
pub fn zoom<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    factors: [f64; 2],
    grid_mode: bool,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>>
```

Zooms a 2-D image by scale factors `[zoom_y, zoom_x]` (`> 0`).

### `rotate`

```rust
pub fn rotate<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    angle_deg: f64,
    resize_output: bool,
    center: Option<[f64; 2]>,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>>
```

Rotates a 2-D image counter-clockwise by `angle_deg` degrees around `center` (defaults to image center),
optionally expanding the output canvas when `resize_output = true`.

### `resize`

```rust
pub fn resize<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    output_shape: [usize; 2],
    anti_aliasing: bool,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>>
```

Resizes a 2-D image to `output_shape` (`[out_h, out_w]`), applying Gaussian anti-aliasing when
downsampling and `anti_aliasing = true`.

### `rescale`

```rust
pub fn rescale<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    scales: [f64; 2],
    anti_aliasing: bool,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>>
```

Rescales a 2-D image by `scales` (`[scale_y, scale_x]`) with optional Gaussian anti-aliasing.

### `polar_warp`

```rust
pub fn polar_warp<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    center: Option<[f64; 2]>,
    max_radius: Option<f64>,
    log_polar: bool,
    output_shape: Option<[usize; 2]>,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>>
```

Warps a 2-D image into polar (`log_polar = false`) or log-polar (`log_polar = true`) coordinates
of `output_shape` (`[n_angles, n_radii]`, defaulting to `[360, ceil(hypot(h/2, w/2))]`).

### `swirl`

```rust
pub fn swirl<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    center: Option<[f64; 2]>,
    strength: f64,
    radius: f64,
    rotation: f64,
    settings: &WarpSettings,
) -> OpsResult<DeviceArray<F>>
```

Applies a non-linear 2-D swirl warp around `center` with `strength` and radial extent `radius` (`> 0`).

---
title: "lumen_ops::feature"
description: "Feature detection and local descriptors: local maxima, multi-scale blobs (LoG, DoG, DoH), Hessian & structure tensors, Canny edges, corner detectors, template matching, LBP, GLCM, and HOG."
sidebar:
  label: "feature"
  order: 5
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Feature detection and local descriptors: local maxima, multi-scale blobs (LoG, DoG, DoH),
Hessian & structure tensors, Canny edges, corner detectors, template matching, LBP, GLCM, and HOG.

## Types

### `Blob`

```rust
pub struct Blob
```

A detected multi-scale blob in a 2-D image.

**Fields**

- `y: usize`
- `x: usize`
- `sigma: f64`
- `response: f64`

**Methods**

```rust
pub fn radius(&self) -> f64
```

Approximate visual radius $\sqrt{2}\,\sigma$ of a 2-D isotropic Gaussian blob.

Implements `Clone`, `Copy`, `Debug`, `PartialEq`.

### `BlobDetector`

```rust
pub struct BlobDetector
```

Settings for multi-scale blob detection (`blobs_log`, `blobs_dog`, `blobs_doh`).

**Fields**

- `min_sigma: f64`
- `max_sigma: f64`
- `num_sigma: usize`: Number of scales for LoG / DoH (`>= 1`).
- `sigma_ratio: f64`: Scale ratio $k > 1$ between successive Gaussians in `blobs_dog`.
- `threshold: f64`: Absolute response threshold for scale-space maxima.
- `overlap: f64`: Maximum allowed area overlap fraction (`0.0..=1.0`); blobs overlapping more than `overlap` with a stronger blob are pruned.
- `log_scale: bool`: Use logarithmic scale spacing when `true`, linear spacing when `false`.
- `exclude_border: usize`: Exclude blobs within `exclude_border` pixels of the image boundary.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `Canny`

```rust
pub struct Canny
```

Settings for the `canny` edge detector.

**Fields**

- `sigma: f64`: Standard deviation of the Gaussian pre-smoothing filter (`>= 0.0`).
- `low_threshold: Option<f64>`: Lower hysteresis threshold (`None` defaults to `0.1 * max_grad`).
- `high_threshold: Option<f64>`: Upper hysteresis threshold (`None` defaults to `0.2 * max_grad`).
- `use_quantiles: bool`: When `true`, `low_threshold` and `high_threshold` are interpreted as fractions in `0.0..=1.0` of the maximum gradient magnitude.
- `mode: PadMode`
- `cval: f64`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `Harris`

```rust
pub struct Harris
```

Settings for `harris`.

**Fields**

- `method: HarrisMethod`
- `k: f64`
- `eps: f64`
- `sigma: f64`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `LocalPeaks`

```rust
pub struct LocalPeaks
```

Settings for `local_peaks`.

**Fields**

- `min_distance: usize`: Minimum Chebyshev/Euclidean distance between returned peaks (`>= 1`).
- `threshold_abs: Option<f64>`: Optional absolute minimum intensity threshold; defaults to the image minimum.
- `threshold_rel: Option<f64>`: Optional relative threshold in `0.0..=1.0` scaled by the image maximum.
- `exclude_border: Option<usize>`: Number of border pixels to exclude (`None` defaults to `min_distance`; `Some(0)` keeps borders).
- `num_peaks: Option<usize>`: Maximum number of peaks to return, ordered by descending intensity then raster order.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `GlcmProps`

```rust
pub struct GlcmProps
```

Haralick summary statistics of a single $(d, \theta)$ slice of a GLCM.

**Fields**

- `contrast: f64`
- `dissimilarity: f64`
- `homogeneity: f64`
- `asm: f64`
- `energy: f64`
- `correlation: f64`

Implements `Clone`, `Copy`, `Debug`, `PartialEq`.

### `Hog`

```rust
pub struct Hog
```

Settings for Histogram of Oriented Gradients (`hog`).

**Fields**

- `orientations: usize`
- `pixels_per_cell: (usize, usize)`
- `cells_per_block: (usize, usize)`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `HarrisMethod`

```rust
pub enum HarrisMethod
```

Harris corner response variant.

**Variants**

- `K`: $\det(A) - k\,\operatorname{tr}(A)^2$.
- `Noble`: Noble's parameter-free harmonic mean $\frac{2\det(A)}{\operatorname{tr}(A) + \varepsilon}$.

Implements `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`.

### `LbpMethod`

```rust
pub enum LbpMethod
```

Encoding variant for circular Local Binary Patterns (`lbp`).

**Variants**

- `Default`: Raw $P$-bit unsigned code $\sum_{p=0}^{P-1} s(g_p - g_c) 2^p$.
- `Ror`: Rotation-invariant minimum cyclic bit shift.
- `Uniform`: Uniform rotation-invariant pattern in `0..=P+1` ($\le 2$ bitwise transitions around the ring).

Implements `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`.

## Functions

### `blobs_log`

```rust
pub fn blobs_log<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &BlobDetector,
) -> OpsResult<Vec<Blob>>
```

Scale-normalized Laplacian of Gaussian ($-\sigma^2 \nabla^2 G_\sigma * I$) blob detector on 2-D bright blobs.

### `blobs_dog`

```rust
pub fn blobs_dog<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &BlobDetector,
) -> OpsResult<Vec<Blob>>
```

Difference of Gaussians (DoG) multi-scale blob detector on 2-D bright blobs.

### `blobs_doh`

```rust
pub fn blobs_doh<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &BlobDetector,
) -> OpsResult<Vec<Blob>>
```

Scale-normalized Determinant of Hessian ($\sigma^4 \det H_\sigma$) blob detector on 2-D images.

### `canny`

```rust
pub fn canny<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &Canny,
) -> OpsResult<DeviceArray<u32>>
```

Runs the 2-D Canny edge detector on `image`, returning a binary `u32` edge map (`1` on edges, `0` elsewhere).

### `harris`

```rust
pub fn harris<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &Harris,
) -> OpsResult<DeviceArray<F>>
```

Computes the 2-D Harris corner response image.

### `shi_tomasi`

```rust
pub fn shi_tomasi<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
) -> OpsResult<DeviceArray<F>>
```

Computes the 2-D Shi-Tomasi (Kanade-Tomasi) minimum-eigenvalue corner response at scale `sigma`.

### `kitchen_rosenfeld`

```rust
pub fn kitchen_rosenfeld<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>>
```

Computes the 2-D Kitchen & Rosenfeld gradient-curvature corner response.

### `foerstner`

```rust
pub fn foerstner<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)>
```

Computes the 2-D Förstner interest operator `(w, q)` (error ellipse size `w` and roundness `q`)
at Gaussian scale `sigma`.

### `moravec`

```rust
pub fn moravec<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    window_radius: usize,
) -> OpsResult<DeviceArray<F>>
```

Computes the 2-D Moravec minimum directional SSD corner response over a `(2 * window_radius + 1)^2` patch.

### `fast`

```rust
pub fn fast<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    n: usize,
    threshold: f64,
) -> OpsResult<DeviceArray<F>>
```

Computes Rosten & Drummond's FAST corner response on the 16-pixel Bresenham circle of radius 3,
requiring at least `n` (`1..=16`, typically `9` or `12`) contiguous brighter or darker pixels
exceeding `threshold`.

### `corner_subpix`

```rust
pub fn corner_subpix<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    corners: &[Vec<usize>],
    window_size: usize,
    sigma: f64,
) -> OpsResult<Vec<[f64; 2]>>
```

Refines integer 2-D corner coordinates `&[[y, x]]` to subpixel precision `(y, x)` by fitting a
local 2-D quadratic surface over a `(2 * window_size + 1)^2` Gaussian-weighted neighborhood.

### `local_peaks`

```rust
pub fn local_peaks<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &LocalPeaks,
) -> OpsResult<Vec<Vec<usize>>>
```

Finds local maxima in a 2-D or N-D image separated by at least `settings.min_distance`, returning
coordinates `[row, col, ...]` sorted by descending peak intensity (ties broken in raster order).

### `hessian_matrix`

```rust
pub fn hessian_matrix<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>, DeviceArray<F>)>
```

Computes the 2-D Gaussian Hessian components `(H_rr, H_rc, H_cc)` at scale `sigma`.

### `hessian_det`

```rust
pub fn hessian_det<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>>
```

Computes the determinant $\det(H_\sigma) = H_{rr} H_{cc} - H_{rc}^2$ of the 2-D Gaussian Hessian.

### `hessian_eigvals`

```rust
pub fn hessian_eigvals<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)>
```

Computes ordered eigenvalues $(\lambda_1 \ge \lambda_2)$ of the 2-D Gaussian Hessian at scale `sigma`.

### `structure_tensor`

```rust
pub fn structure_tensor<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>, DeviceArray<F>)>
```

Computes the 2-D structure tensor `(A_rr, A_rc, A_cc)` by smoothing outer products of Sobel gradients
with a Gaussian window of scale `sigma`.

### `structure_tensor_eigvals`

```rust
pub fn structure_tensor_eigvals<F: ImageFloat>(
    client: &Client,
    arr: &DeviceArray<F>,
    arc: &DeviceArray<F>,
    acc: &DeviceArray<F>,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)>
```

Computes ordered eigenvalues $(\lambda_1 \ge \lambda_2)$ of the 2-D structure tensor `(arr, arc, acc)`.

### `shape_index`

```rust
pub fn shape_index<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigma: f64,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>>
```

Computes Koenderink & van Doorn's local 2-D shape index $s \in [-1, 1]$ at scale `sigma`:
cup (`-1.0`), rut (`-0.5`), saddle (`0.0`), ridge (`+0.5`), dome/cap (`+1.0`).

### `match_template`

```rust
pub fn match_template<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    template: &DeviceArray<F>,
    pad_input: bool,
) -> OpsResult<DeviceArray<F>>
```

Zero-mean Normalized Cross-Correlation (ZNCC) template matching in 2-D.

When `pad_input = false`, returns the valid correlation map of shape
`[img_h - tpl_h + 1, img_w - tpl_w + 1]`; when `pad_input = true`, returns an array of the same
shape as `image` centered at `(tpl_h / 2, tpl_w / 2)`.

### `lbp`

```rust
pub fn lbp<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    num_points: usize,
    radius: f64,
    method: LbpMethod,
) -> OpsResult<DeviceArray<u32>>
```

Computes circular 2-D Local Binary Patterns (`lbp`) with `num_points` (`1..=32`) at `radius > 0`.

### `glcm`

```rust
pub fn glcm<I: ImageInt>(
    client: &Client,
    image: &DeviceArray<I>,
    distances: &[usize],
    angles: &[f64],
    levels: usize,
    symmetric: bool,
    normed: bool,
) -> OpsResult<DeviceArray<f64>>
```

Computes the 4-D Gray-Level Co-occurrence Matrix of shape `[levels, levels, distances.len(), angles.len()]`
from a 2-D integer image with values in `0..levels`.

### `glcm_props`

```rust
pub fn glcm_props(
    client: &Client,
    p_matrix: &DeviceArray<f64>,
) -> OpsResult<Vec<Vec<GlcmProps>>>
```

Extracts Haralick texture properties (`contrast`, `dissimilarity`, `homogeneity`, `asm`, `energy`,
`correlation`) for every `(distance, angle)` slice `[d_idx, a_idx]` of a 4-D `glcm` matrix.

### `multiscale_features`

```rust
pub fn multiscale_features<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigmas: &[f64],
    include_intensity: bool,
    include_edges: bool,
    include_texture: bool,
) -> OpsResult<Vec<DeviceArray<F>>>
```

Computes a bank of multi-scale basic features (intensity, gradient magnitude, and Hessian eigenvalues)
across each Gaussian scale in `sigmas`.

### `hog`

```rust
pub fn hog<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &Hog,
) -> OpsResult<Vec<f64>>
```

Computes the L2-Hys normalized Histogram of Oriented Gradients (HOG) feature vector for a 2-D image.

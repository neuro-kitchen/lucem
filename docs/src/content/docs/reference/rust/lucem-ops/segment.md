---
title: "lucem_ops::segment"
description: "Image segmentation: watershed, label post-processing, superpixels, random walker, and active contours."
sidebar:
  label: "segment"
  order: 13
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Image segmentation: watershed, label post-processing, superpixels, random walker, and active contours.

- `watershed`: marker-controlled flooding `watershed` and Image Foresting Transform `watershed_ift`.
- `labels`: `expand_labels`, `clear_border`, `find_boundaries`, `mark_boundaries`, and `join_segmentations`.
- `superpixel`: `slic`, `felzenszwalb`, `quickshift`, and `random_walker`.
- `active_contour`: `chan_vese`, `morph_chan_vese`, `morph_geodesic_contour`, `checkerboard_level_set`, and `disk_level_set`.

## Types

### `ChanVeseOptions`

```rust
pub struct ChanVeseOptions
```

Parameters for the Chan-Vese active contour model (`chan_vese`).

**Fields**

- `mu: f64`
- `lambda1: f64`
- `lambda2: f64`
- `tol: f64`
- `max_iter: usize`
- `dt: f64`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `SlicOptions`

```rust
pub struct SlicOptions
```

Parameters for Simple Linear Iterative Clustering (`slic`).

**Fields**

- `n_segments: usize`
- `compactness: f64`
- `max_iter: usize`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `WatershedOptions`

```rust
pub struct WatershedOptions
```

Options for marker-controlled `watershed`.

**Fields**

- `compactness: f64`: Compact watershed weighting factor ($\ge 0$). `0.0` gives classic Meyer flooding.
- `watershed_line: bool`: If `true`, keeps a 1-pixel dam (`0`) separating distinct catchment basins.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `BoundaryMode`

```rust
pub enum BoundaryMode
```

Boundary localization mode for `find_boundaries`.

**Variants**

- `Thick`: Both inner and outer boundary pixels around label transitions.
- `Inner`: Foreground pixels (`!= background`) adjacent to a different label.
- `Outer`: Pixels adjacent to a different foreground label.

Implements `Clone`, `Copy`, `Debug`, `Default`, `Eq`, `PartialEq`.

## Functions

### `checkerboard_level_set`

```rust
pub fn checkerboard_level_set<F: ImageFloat>(
    client: &Client,
    shape: Shape,
    square_size: usize,
) -> OpsResult<DeviceArray<F>>
```

Generates a 2-D checkerboard binary level-set (`1.0` / `0.0`) of `shape = [h, w]` with tile
size `square_size` on the device.

### `disk_level_set`

```rust
pub fn disk_level_set<F: ImageFloat>(
    client: &Client,
    shape: Shape,
    center: Option<(f64, f64)>,
    radius: Option<f64>,
) -> OpsResult<DeviceArray<F>>
```

Generates a 2-D binary disk level-set (`1.0` inside `radius` around `center`, `0.0` outside) on
the device. Defaults to the image center and `0.375 * min(h, w)` when `None`.

### `chan_vese`

```rust
pub fn chan_vese<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    init_level_set: Option<&DeviceArray<F>>,
    options: &ChanVeseOptions,
) -> OpsResult<(DeviceArray<u32>, DeviceArray<F>)>
```

Segments a 2-D image using the Chan-Vese active contour without edges, returning the binary
mask (`1` where $\phi > 0$, `0` elsewhere) and the evolved level-set $\phi$.

### `morph_chan_vese`

```rust
pub fn morph_chan_vese<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    iterations: usize,
    init_level_set: Option<&DeviceArray<F>>,
    smoothing: usize,
    lambda1: f64,
    lambda2: f64,
) -> OpsResult<DeviceArray<u32>>
```

Morphological Active Contours Without Edges (MorphACWE / `morph_chan_vese`) on a 2-D image.

### `morph_geodesic_contour`

```rust
pub fn morph_geodesic_contour<F: ImageFloat>(
    client: &Client,
    gimage: &DeviceArray<F>,
    iterations: usize,
    init_level_set: Option<&DeviceArray<F>>,
    smoothing: usize,
    threshold: f64,
    balloon: f64,
) -> OpsResult<DeviceArray<u32>>
```

Morphological Geodesic Active Contours (MorphGAC / `morph_geodesic_contour`) on a 2-D inverse
edge-indicator map `gimage`.

### `expand_labels`

```rust
pub fn expand_labels(
    client: &Client,
    labels: &DeviceArray<u32>,
    distance: f64,
    spacing: Option<&[f64]>,
) -> OpsResult<DeviceArray<u32>>
```

Dilates positive labels in `labels` into background (`0`) pixels by at most `distance`
Euclidean distance without overwriting existing labels, using
`crate::core::distance::nearest_background_indices` on the inverted foreground mask.

### `clear_border`

```rust
pub fn clear_border(
    client: &Client,
    labels: &DeviceArray<u32>,
    buffer_size: usize,
    bgval: u32,
) -> OpsResult<DeviceArray<u32>>
```

Clears any labeled region (`!= bgval`) that touches the outer `buffer_size`-pixel border of
`labels`, setting those regions to `bgval`.

### `find_boundaries`

```rust
pub fn find_boundaries(
    client: &Client,
    labels: &DeviceArray<u32>,
    connectivity: usize,
    mode: BoundaryMode,
    background: u32,
) -> OpsResult<DeviceArray<u32>>
```

Finds boundary pixels (`1` on boundaries, `0` elsewhere) between distinct regions in a 2-D
`labels` array.

### `mark_boundaries`

```rust
pub fn mark_boundaries<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    labels: &DeviceArray<u32>,
    color: [f64; 3],
    mode: BoundaryMode,
) -> OpsResult<DeviceArray<F>>
```

Returns an RGB `[h, w, 3]` visualization with boundaries of `labels` drawn in `color` on top of
a 2-D grayscale `[h, w]` or RGB `[h, w, 3]` `image`.

### `join_segmentations`

```rust
pub fn join_segmentations(
    client: &Client,
    s1: &DeviceArray<u32>,
    s2: &DeviceArray<u32>,
) -> OpsResult<DeviceArray<u32>>
```

Computes the common refinement (overlay intersection) of two label maps `s1` and `s2`,
sequentially relabeling every non-zero combined pair into `1..=num_joined`.

### `slic`

```rust
pub fn slic<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    options: &SlicOptions,
) -> OpsResult<DeviceArray<u32>>
```

Simple Linear Iterative Clustering (SLIC) superpixel over-segmentation on a 2-D image `[h, w]`.
Runs each pixel-to-cluster assignment iteration on the device via `slic_assign_2d_kernel`.

### `felzenszwalb`

```rust
pub fn felzenszwalb<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    scale: f64,
    min_size: usize,
) -> OpsResult<DeviceArray<u32>>
```

Graph-based minimum spanning forest segmentation (Felzenszwalb & Huttenlocher 2004) on a 2-D
image `[h, w]`.

### `quickshift`

```rust
pub fn quickshift<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    kernel_size: f64,
    max_dist: f64,
    ratio: f64,
) -> OpsResult<DeviceArray<u32>>
```

Quickshift mode-seeking medoid shift segmentation (Vedaldi & Soatto 2008) on a 2-D image `[h, w]`.

### `random_walker`

```rust
pub fn random_walker<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    markers: &DeviceArray<u32>,
    beta: f64,
    tol: f64,
    max_iter: usize,
) -> OpsResult<DeviceArray<u32>>
```

Anisotropic harmonic random-walker segmentation (Grady 2006) on a 2-D image `[h, w]` from
seed labels in `markers` (`> 0` = seed label, `0` = unlabeled), using edge weights
$w_{ij} = \exp(-\beta (I_i - I_j)^2)$.

### `watershed`

```rust
pub fn watershed<F: ImageFloat>(
    client: &Client,
    relief: &DeviceArray<F>,
    markers: &DeviceArray<u32>,
    footprint: Option<&StructuringElement>,
    mask: Option<&DeviceArray<u32>>,
    options: &WatershedOptions,
) -> OpsResult<DeviceArray<u32>>
```

Marker-controlled watershed segmentation of an N-D topographic `relief` array starting from
positive seed labels in `markers` (`0` = unlabeled), optionally restricted to non-zero pixels
of `mask`.

### `watershed_ift`

```rust
pub fn watershed_ift<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    markers: &DeviceArray<u32>,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<u32>>
```

Image Foresting Transform (IFT) shortest-path watershed under the minimax path-cost
$f_{\max}(\pi) = \max_{e \in \pi} w(e)$ from positive seeds in `markers`.

---
title: "lumen_ops::core"
description: "Shared device primitives: the single home for every building block reused across `lumen-ops`."
sidebar:
  label: "core"
  order: 3
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Shared device primitives: the single home for every building block reused across `lumen-ops`.

- `pad`: `PadMode` and device boundary indexing (`kernels::pad_index`, `kernels::read_padded`).
- `check`: shape, aliasing, axis, origin, and weight validation.
- `separable`: multi-axis 1-D pass runner over `Workspace` ping-pong buffers.
- `structuring`: `StructuringElement` (boolean footprints, separable rectangle detection, device-cached offsets).
- `elementwise`: copy, affine scale/offset, clip, rescale, cast, LUT, and threshold comparison.
- `reduce`: global and axis reductions (`sum`, `min`, `max`, Welford `mean`/`var`, `argmin`, `argmax`).
- `select`: exact $k$-th element and percentile selection via IEEE-754 radix histograms.
- `histogram`: global, ranged, and integer histograms.
- `scan`: axis prefix sums and N-D integral images.
- `compact`: stream compaction of non-zero indices and coordinates in raster order.
- `components`: connected-component labeling, sequential relabeling, and deterministic per-region reductions.
- `distance`: exact separable Euclidean distance transform (EDT), feature transform, and chamfer distances.
- `spline`: B-spline prefiltering and N-D coordinate interpolation (orders 0–5).
- `fft`: N-D forward and inverse Fourier transforms on split complex buffers.
- `eigen`: per-pixel symmetric $2 \times 2$ and $3 \times 3$ eigenvalues.
- `random`: counter-based uniform and Gaussian RNG on the device.

## Types

### `StructuringElement`

```rust
pub struct StructuringElement
```

An N-D boolean neighborhood footprint.

**Methods**

```rust
pub fn from_mask(shape: impl Into<Shape>, mask: Vec<bool>) -> OpsResult<Self>
```

Creates a structuring element from `shape` and a C-order boolean `mask`.

```rust
pub fn rectangle(dims: impl Into<Shape>) -> OpsResult<Self>
```

Hyper-rectangular footprint with every element active (`dims` lengths per axis).

```rust
pub fn hypercube(ndim: usize, width: usize) -> OpsResult<Self>
```

Square (2-D) or hyper-cubic (N-D) footprint of side `width` along `ndim` dimensions.

```rust
pub fn connectivity(ndim: usize, connectivity: usize) -> OpsResult<Self>
```

N-D connectivity structure: a $3^D$ element where a neighbor at offset $\Delta$ is active iff
$\sum_d |\Delta_d|^2 \le \text{connectivity}$ (`connectivity` in `1..=ndim`).

```rust
pub fn disk(radius: usize) -> Self
```

2-D circular disk footprint of radius `radius` (`(2*radius + 1) x (2*radius + 1)`), active where
$\Delta y^2 + \Delta x^2 \le r^2$.

```rust
pub fn ball(radius: usize) -> Self
```

3-D spherical ball footprint of radius `radius` (`(2*radius + 1)^3`), active where
$\Delta z^2 + \Delta y^2 + \Delta x^2 \le r^2$.

```rust
pub fn diamond(radius: usize) -> Self
```

2-D L1 diamond of radius `radius`, active where $|\Delta y| + |\Delta x| \le r$.

```rust
pub fn octahedron(radius: usize) -> Self
```

3-D L1 octahedron of radius `radius`, active where $|\Delta z| + |\Delta y| + |\Delta x| \le r$.

```rust
pub fn ellipse(radius_y: usize, radius_x: usize) -> Self
```

2-D axis-aligned ellipse with semi-axes `(radius_y, radius_x)`.

```rust
pub fn is_rectangle(&self) -> bool
```

Returns `true` if every position in the bounding box is active (so min/max filters can use
separable 1-D passes along each axis).

```rust
pub fn mirrored(&self) -> Self
```

Reverses the footprint along every axis (turning correlation into convolution or dilation's
reflected structuring element).

```rust
pub fn shape(&self) -> &Shape
```

```rust
pub fn dims(&self) -> &[usize]
```

```rust
pub fn ndim(&self) -> usize
```

```rust
pub fn mask(&self) -> &[bool]
```

```rust
pub fn count_active(&self) -> usize
```

Number of active (`true`) positions in the footprint.

```rust
pub fn halo(&self, origins: Option<&[i32]>) -> Vec<(usize, usize)>
```

Halo `(before, after)` along each axis for given `origins` (defaults to `0` per axis).

```rust
pub fn coordinate_offsets(&self, origins: Option<&[i32]>) -> Vec<Vec<i32>>
```

Coordinate offsets `(delta_0, ..., delta_{D-1})` of every active element relative to the
footprint center `(dims[d] / 2 + origin[d])`.

```rust
pub fn device_offsets(&self, client: &Client, origins: Option<&[i32]>) -> DeviceArray<i32>
```

Device-cached flattened `i32` coordinate offsets `[num_active, ndim]` for `client`.

Implements `Clone`, `Debug`, `Eq`, `Hash`, `PartialEq`.

### `PadMode`

```rust
pub enum PadMode
```

How a stencil or interpolation kernel reads coordinates outside `0..len` along an axis.

Shown for the left boundary of `a b c d`:

**Variants**

- `Reflect`: Half-sample symmetric reflection: `d c b a | a b c d | d c b a`.
- `Constant`: Constant fill value (`cval`): `k k k k | a b c d | k k k k`.
- `Nearest`: Replicate edge pixel: `a a a a | a b c d | d d d d`.
- `Mirror`: Whole-sample symmetric reflection: `d c b | a b c d | c b a`.
- `Wrap`: Periodic wrap-around: `a b c d | a b c d | a b c d`.

**Methods**

```rust
pub const fn id(self) -> u32
```

Comptime integer identifier passed to `#[cube]` kernels.

```rust
pub fn parse(name: &str) -> OpsResult<Self>
```

Parses a boundary mode name (accepting common aliases).

```rust
pub const fn name(self) -> &'static str
```

Implements `Clone`, `Copy`, `Debug`, `Default`, `Deserialize`, `Eq`, `Hash`, `PartialEq`, `Serialize`.

## Constants

### `PAD_REFLECT`

```rust
pub const PAD_REFLECT: u32 = 0;
```

### `PAD_CONSTANT`

```rust
pub const PAD_CONSTANT: u32 = 1;
```

### `PAD_NEAREST`

```rust
pub const PAD_NEAREST: u32 = 2;
```

### `PAD_MIRROR`

```rust
pub const PAD_MIRROR: u32 = 3;
```

### `PAD_WRAP`

```rust
pub const PAD_WRAP: u32 = 4;
```

## `lumen_ops::core::check`

Shared argument validation for image operations.

### `same_shape`

```rust
pub fn same_shape<A: CubeElement, B: CubeElement>(
    first_name: &'static str,
    first: &DeviceArray<A>,
    second_name: &'static str,
    second: &DeviceArray<B>,
) -> OpsResult<()>
```

Verifies that `a` and `b` have identical shapes.

### `expect_shape`

```rust
pub fn expect_shape(
    first_name: &'static str,
    first: &Shape,
    second_name: &'static str,
    second: &Shape,
) -> OpsResult<()>
```

Verifies that `shape_a` and `shape_b` are equal.

### `not_aliased`

```rust
pub fn not_aliased<A: CubeElement, B: CubeElement>(
    input: &DeviceArray<A>,
    output: &DeviceArray<B>,
) -> OpsResult<()>
```

Verifies that `input` and `output` do not share the same underlying device buffer.

### `origin`

```rust
pub fn origin(origin: i32, size: usize) -> OpsResult<()>
```

Verifies that `origin` lies in `-(size as i32 / 2) ..= ((size - 1) as i32 / 2)`.

### `weights`

```rust
pub fn weights(w: &[f64]) -> OpsResult<()>
```

Verifies that a host slice of weights is non-empty and every weight is finite.

### `ndim_in`

```rust
pub fn ndim_in(
    what: &'static str,
    shape: &Shape,
    allowed: &[usize],
    expected: &'static str,
) -> OpsResult<()>
```

Verifies that `ndim` matches one of the allowed ranks.

### `broadcast_per_axis`

```rust
pub fn broadcast_per_axis<T: Copy>(
    name: &'static str,
    values: &[T],
    num_axes: usize,
) -> OpsResult<Vec<T>>
```

Broadcasts a per-axis parameter slice `values` to `axes.len()`: a length-1 slice is repeated for
every axis; otherwise its length must equal `axes.len()`.

## `lumen_ops::core::compact`

Stream compaction: extracts flat indices and N-D coordinates of non-zero elements in canonical
C-contiguous raster order.

### `flag_nonzero`

```rust
pub fn flag_nonzero<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
) -> OpsResult<DeviceArray<u32>>
```

Flags non-zero elements of `input` into a `u32` device array (`1` for non-zero, `0` for zero).

### `nonzero_coordinates`

```rust
pub fn nonzero_coordinates<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
) -> OpsResult<Vec<Vec<usize>>>
```

Returns the N-D coordinates `[num_nonzero, ndim]` in C raster order of all non-zero elements in
`input`.

### `nonzero_coordinates_device`

```rust
pub fn nonzero_coordinates_device<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
) -> OpsResult<DeviceArray<u32>>
```

Returns a device array `[num_nonzero, ndim]` (`u32`) of non-zero coordinates in raster order.

## `lumen_ops::core::components`

Connected-component labeling, sequential relabeling, and deterministic per-region reductions.

### `ConnectedComponents`

```rust
pub struct ConnectedComponents
```

Connected-component labeling result: the `u32` label map (`0` for background, `1..=num_labels`
in raster order of first appearance) and `num_labels`.

**Fields**

- `labels: DeviceArray<u32>`
- `num_labels: usize`

Implements `Clone`, `Debug`.

### `RegionSummary`

```rust
pub struct RegionSummary
```

Summary statistics for a single labeled region.

**Fields**

- `label: u32`
- `count: usize`
- `sum: f64`
- `mean: f64`
- `variance: f64`
- `std_dev: f64`
- `min: f64`
- `max: f64`
- `min_index: usize`
- `max_index: usize`
- `bbox: Vec<(usize, usize)>`: Minimum and maximum coordinate along each axis `[(min_0, max_0), ...]`, where `max` is exclusive (`min..max` slice bounds).
- `centroid: Vec<f64>`: Intensity-unweighted centroid coordinates along each axis.
- `weighted_centroid: Vec<f64>`: Intensity-weighted center of mass along each axis.

Implements `Clone`, `Debug`, `PartialEq`.

### `label_components`

```rust
pub fn label_components<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    footprint: Option<&StructuringElement>,
) -> OpsResult<ConnectedComponents>
```

Finds connected components of non-zero elements in `input` using `footprint` (or orthogonal
1-connectivity when `None`), returning canonical raster-ordered labels `1..=num_labels`.

### `relabel_sequential`

```rust
pub fn relabel_sequential(
    client: &Client,
    labels: &DeviceArray<u32>,
    offset: u32,
) -> OpsResult<(DeviceArray<u32>, BTreeMap<u32, u32>)>
```

Relabels an arbitrary integer label array sequentially into `offset .. offset + num_unique` in
order of ascending positive label values (`0` remains `0`).

### `summarize_regions`

```rust
pub fn summarize_regions<F: ImageFloat>(
    client: &Client,
    labels: &DeviceArray<u32>,
    intensity: Option<&DeviceArray<F>>,
) -> OpsResult<Vec<RegionSummary>>
```

Computes deterministic per-label region summaries in a single raster pass (avoiding non-associative
float atomic additions).

## `lumen_ops::core::distance`

Exact Euclidean distance transform (EDT), nearest-background feature transform, and chamfer
distance transforms (`Taxicab` / `Chessboard`).

### `ChamferMetric`

```rust
pub enum ChamferMetric
```

Metric for discrete chamfer distance transforms.

**Variants**

- `Chessboard`: $L_\infty$ (8-connected / 26-connected chessboard distance).
- `Taxicab`: $L_1$ (4-connected / 6-connected Manhattan distance).

Implements `Clone`, `Copy`, `Debug`, `Default`, `Eq`, `PartialEq`.

### `euclidean_distance_into`

```rust
pub fn euclidean_distance_into<E: Numeric + CubeElement, F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<E>,
    output: &DeviceArray<F>,
    spacing: Option<&[f64]>,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Computes the exact Euclidean distance from every foreground (`!= 0`) element to the nearest
background (`== 0`) element into `output`, using separable 1-D squared-distance passes on the
device and `ws` ping-pong slots.

### `euclidean_distance`

```rust
pub fn euclidean_distance<E: Numeric + CubeElement, F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<E>,
    spacing: Option<&[f64]>,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns the exact Euclidean distance transform of `input`.

### `nearest_background_indices`

```rust
pub fn nearest_background_indices<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    spacing: Option<&[f64]>,
) -> OpsResult<DeviceArray<u32>>
```

Nearest-feature index transform: for every pixel, returns the flat raster index of the nearest
background (`== 0`) pixel under exact Euclidean distance (with first-index tie breaking).

### `chamfer_distance`

```rust
pub fn chamfer_distance<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    metric: ChamferMetric,
) -> OpsResult<DeviceArray<i32>>
```

Computes the exact chamfer (`Taxicab` or `Chessboard`) integer distance transform.

## `lumen_ops::core::eigen`

Per-pixel analytic eigenvalues of symmetric $2 \times 2$ and $3 \times 3$ matrices (used by
Hessian, structure tensor, ridge filters, shape index, and inertia tensors).

### `symmetric_eigenvalues_2x2_into`

```rust
pub fn symmetric_eigenvalues_2x2_into<F: ImageFloat>(
    client: &Client,
    axx: &DeviceArray<F>,
    axy: &DeviceArray<F>,
    ayy: &DeviceArray<F>,
    eig1: &DeviceArray<F>,
    eig2: &DeviceArray<F>,
) -> OpsResult<()>
```

Computes ordered eigenvalues $\lambda_1 \ge \lambda_2$ for each pixel of a symmetric $2 \times 2$
field `(axx, axy, ayy)` into `(eig1, eig2)`.

### `symmetric_eigenvalues_3x3_into`

```rust
pub fn symmetric_eigenvalues_3x3_into<F: ImageFloat>(
    client: &Client,
    a00: &DeviceArray<F>,
    a01: &DeviceArray<F>,
    a02: &DeviceArray<F>,
    a11: &DeviceArray<F>,
    a12: &DeviceArray<F>,
    a22: &DeviceArray<F>,
    eig1: &DeviceArray<F>,
    eig2: &DeviceArray<F>,
    eig3: &DeviceArray<F>,
) -> OpsResult<()>
```

Computes ordered eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$ for each pixel of a symmetric
$3 \times 3$ field `(a00, a01, a02, a11, a12, a22)` into `(eig1, eig2, eig3)`.

## `lumen_ops::core::elementwise`

Host dispatchers for shared elementwise operations on `DeviceArray`s.

### `copy_into`

```rust
pub fn copy_into<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    output: &DeviceArray<E>,
) -> OpsResult<()>
```

Copies `input` into `output` on the device.

### `copy`

```rust
pub fn copy<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
) -> OpsResult<DeviceArray<E>>
```

Allocates and returns a device copy of `input`.

### `scale_offset_into`

```rust
pub fn scale_offset_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    scale: f64,
    offset: f64,
) -> OpsResult<()>
```

Computes `output = input * scale + offset` on the device.

### `clip_into`

```rust
pub fn clip_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    lo: f64,
    hi: f64,
) -> OpsResult<()>
```

Clamps `input` into `[lo, hi]` and writes into `output`.

### `rescale_clip_into`

```rust
pub fn rescale_clip_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    in_range: (f64, f64),
    out_range: (f64, f64),
) -> OpsResult<()>
```

Linearly maps `[in_lo, in_hi]` to `[out_lo, out_hi]`, clamping values outside the range.

### `cast_into`

```rust
pub fn cast_into<Src: Numeric + CubeElement, Dst: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<Src>,
    output: &DeviceArray<Dst>,
) -> OpsResult<()>
```

Casts elements from `Src` to `Dst` on the device.

### `cast`

```rust
pub fn cast<Src: Numeric + CubeElement, Dst: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<Src>,
) -> OpsResult<DeviceArray<Dst>>
```

Allocates and casts elements from `Src` to `Dst` on the device.

### `affine_into`

```rust
pub fn affine_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    scale: f64,
    offset: f64,
) -> OpsResult<()>
```

Alias for `scale_offset_into`: `output = input * scale + offset`.

### `threshold_scalar_into`

```rust
pub fn threshold_scalar_into<F: ImageFloat, Out: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    threshold: f64,
    inclusive: bool,
    output: &DeviceArray<Out>,
) -> OpsResult<()>
```

Convenience wrapper for `compare_threshold_scalar_into` with `(threshold, inclusive, output)` argument order.

### `uint_to_unit_float_into`

```rust
pub fn uint_to_unit_float_into<Src: ImageInt, F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<Src>,
    output: &DeviceArray<F>,
    max_val: f64,
) -> OpsResult<()>
```

Converts an unsigned integer image in `0..=max_val` to float in `[0.0, 1.0]`.

### `unit_float_to_uint_into`

```rust
pub fn unit_float_to_uint_into<F: ImageFloat, Dst: ImageInt>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<Dst>,
    max_val: f64,
) -> OpsResult<()>
```

Converts a unit float image (`[0.0, 1.0]`, clamped) to unsigned integer `0..=max_val` with nearest
rounding.

### `apply_lut_into`

```rust
pub fn apply_lut_into<Idx: ImageInt, Val: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<Idx>,
    lut: &DeviceArray<Val>,
    output: &DeviceArray<Val>,
) -> OpsResult<()>
```

Maps integer indices through a device-resident look-up table `lut`.

### `compare_threshold_device_into`

```rust
pub fn compare_threshold_device_into<F: ImageFloat, Out: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    threshold: &DeviceArray<F>,
    output: &DeviceArray<Out>,
    inclusive: bool,
) -> OpsResult<()>
```

Compares `input` against a 1-element device array `threshold` (`>` or `>=` when `inclusive`) without
synchronizing with the host.

### `compare_threshold_scalar_into`

```rust
pub fn compare_threshold_scalar_into<F: ImageFloat, Out: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    threshold: f64,
    output: &DeviceArray<Out>,
    inclusive: bool,
) -> OpsResult<()>
```

Compares `input` against a host scalar `threshold` (`>` or `>=` when `inclusive`).

### `linear_combination_into`

```rust
pub fn linear_combination_into<F: ImageFloat>(
    client: &Client,
    a: &DeviceArray<F>,
    b: &DeviceArray<F>,
    output: &DeviceArray<F>,
    alpha: f64,
    beta: f64,
    bias: f64,
) -> OpsResult<()>
```

Computes `output = alpha * a + beta * b + bias` on the device.

### `hypot_into`

```rust
pub fn hypot_into<F: ImageFloat>(
    client: &Client,
    a: &DeviceArray<F>,
    b: &DeviceArray<F>,
    output: &DeviceArray<F>,
) -> OpsResult<()>
```

Computes `output = sqrt(a^2 + b^2)` on the device.

## `lumen_ops::core::fft`

N-D forward and inverse Fourier transforms on split complex `DeviceArray` pairs.

### `fft_1d_into`

```rust
pub fn fft_1d_into<F: ImageFloat>(
    client: &Client,
    re_in: &DeviceArray<F>,
    im_in: &DeviceArray<F>,
    re_out: &DeviceArray<F>,
    im_out: &DeviceArray<F>,
    axis: isize,
    inverse: bool,
) -> OpsResult<()>
```

1-D forward (`inverse = false`) or inverse (`inverse = true`) DFT along `axis`.

### `fft_nd_into`

```rust
pub fn fft_nd_into<F: ImageFloat>(
    client: &Client,
    re_in: &DeviceArray<F>,
    im_in: &DeviceArray<F>,
    re_out: &DeviceArray<F>,
    im_out: &DeviceArray<F>,
    inverse: bool,
    ws: &mut Workspace,
) -> OpsResult<()>
```

N-D forward (`inverse = false`) or inverse (`inverse = true`) FFT across all axes of `(re_in, im_in)`
into `(re_out, im_out)` using `ws` ping-pong slots.

### `complex_mul_into`

```rust
pub fn complex_mul_into<F: ImageFloat>(
    client: &Client,
    a_re: &DeviceArray<F>,
    a_im: &DeviceArray<F>,
    b_re: &DeviceArray<F>,
    b_im: &DeviceArray<F>,
    out_re: &DeviceArray<F>,
    out_im: &DeviceArray<F>,
    conj_b: bool,
) -> OpsResult<()>
```

Multiplies two split-complex arrays pointwise (`conj_b = true` multiplies by the complex conjugate
of `b`, for cross-correlation).

## `lumen_ops::core::histogram`

Global and ranged histograms on the device.

### `histogram_ranged_device`

```rust
pub fn histogram_ranged_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
    range: (f64, f64),
) -> OpsResult<DeviceArray<u32>>
```

Computes a `nbins`-bin histogram of `input` over `[min_val, max_val]` on the device.

### `histogram`

```rust
pub fn histogram<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
    range: Option<(f64, f64)>,
) -> OpsResult<(DeviceArray<u32>, Vec<f64>)>
```

Computes a histogram and bin centers over `range` (or `(min, max)` of `input` when `None`).

### `histogram_uint_device`

```rust
pub fn histogram_uint_device<I: ImageInt>(
    client: &Client,
    input: &DeviceArray<I>,
    nbins: usize,
) -> OpsResult<DeviceArray<u32>>
```

Direct integer histogram of unsigned values in `0..nbins`.

## `lumen_ops::core::random`

Seeded counter-based random number generation on the device.

### `fill_uniform`

```rust
pub fn fill_uniform<F: ImageFloat>(
    client: &Client,
    output: &DeviceArray<F>,
    seed: u64,
) -> OpsResult<()>
```

Fills `output` with deterministic uniform random values in `[0.0, 1.0)` seeded by `seed`.

### `fill_normal`

```rust
pub fn fill_normal<F: ImageFloat>(
    client: &Client,
    output: &DeviceArray<F>,
    mean: f64,
    std_dev: f64,
    seed: u64,
) -> OpsResult<()>
```

Fills `output` with deterministic Gaussian $\mathcal{N}(\text{mean}, \text{std\_dev}^2)$ values
seeded by `seed`.

## `lumen_ops::core::reduce`

Global and per-axis reductions on `DeviceArray`s.

### `SummaryStats`

```rust
pub struct SummaryStats
```

Summary statistics of an array (`sum`, `min`, `max`).

**Fields**

- `sum: f64`
- `min: f64`
- `max: f64`

Implements `Clone`, `Copy`, `Debug`, `PartialEq`.

### `ReduceOp`

```rust
pub enum ReduceOp
```

Axis reduction mode.

**Variants**

- `Sum`
- `Mean`
- `Min`
- `Max`

Implements `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`.

### `row_sum_min_max_into`

```rust
pub fn row_sum_min_max_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    rows: usize,
    cols: usize,
    out_sum: &DeviceArray<F>,
    out_min: &DeviceArray<F>,
    out_max: &DeviceArray<F>,
) -> OpsResult<()>
```

Computes `(sum, min, max)` per row of a `[rows, cols]` view into 1-D device arrays of length `rows`.

### `sum_min_max_device`

```rust
pub fn sum_min_max_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>, DeviceArray<F>)>
```

Computes global `(sum, min, max)` as three 1-element `DeviceArray`s (no host read).

### `sum_min_max`

```rust
pub fn sum_min_max<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<SummaryStats>
```

Host summary `(sum, min, max)` of `input`.

### `row_mean_var_into`

```rust
pub fn row_mean_var_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    rows: usize,
    cols: usize,
    ddof: usize,
    out_mean: &DeviceArray<F>,
    out_var: &DeviceArray<F>,
) -> OpsResult<()>
```

Computes `(mean, variance)` per row of `[rows, cols]` into 1-D device arrays of length `rows`.

### `mean_var_device`

```rust
pub fn mean_var_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    ddof: usize,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>)>
```

Computes global `(mean, variance)` as two 1-element `DeviceArray`s (no host read).

### `mean_var`

```rust
pub fn mean_var<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    ddof: usize,
) -> OpsResult<(f64, f64)>
```

Host `(mean, variance)` of `input`.

### `argmin_argmax_device`

```rust
pub fn argmin_argmax_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<(DeviceArray<u32>, DeviceArray<u32>)>
```

Computes global `(argmin, argmax)` flat indices as 1-element `u32` device arrays.

### `argmin_argmax`

```rust
pub fn argmin_argmax<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<(usize, usize)>
```

Host `(argmin, argmax)` flat indices of `input` (first-index tie rule).

### `reduce_axis_into`

```rust
pub fn reduce_axis_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    axis: isize,
    op: ReduceOp,
) -> OpsResult<()>
```

Reduces `input` along `axis` into `output` (whose shape removes `axis`).

### `reduce_axis`

```rust
pub fn reduce_axis<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    axis: isize,
    op: ReduceOp,
    _ws: &mut Workspace,
) -> OpsResult<DeviceArray<F>>
```

Reduces `input` along `axis`, allocating the reduced output array.

## `lumen_ops::core::scan`

Prefix sums (scans) and N-D integral scans on `DeviceArray`s.

### `scan_axis_into`

```rust
pub fn scan_axis_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    axis: isize,
    inclusive: bool,
) -> OpsResult<()>
```

Computes an inclusive (`inclusive = true`) or exclusive (`inclusive = false`) prefix sum along
`axis` into `output`.

### `integral_scan_into`

```rust
pub fn integral_scan_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Computes an N-D inclusive prefix sum (integral image / summed-area table) across every axis of
`input` into `output` using `ws` ping-pong buffers.

### `integral_scan`

```rust
pub fn integral_scan<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns the N-D integral image of `input`.

## `lumen_ops::core::select`

Exact $k$-th smallest element and percentile selection.

### `kth_f32`

```rust
pub fn kth_f32(client: &Client, input: &DeviceArray<f32>, k: usize) -> OpsResult<f32>
```

Selects the exact `k`-th smallest `f32` element (`0 <= k < input.len()`) using 4-pass 8-bit
radix histograms on the device.

### `percentiles`

```rust
pub fn percentiles<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    qs: &[f64],
) -> OpsResult<Vec<f64>>
```

Computes exact percentiles (`q` in `0.0..=100.0`) with linear interpolation.

### `percentiles_device`

```rust
pub fn percentiles_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    qs: &[f64],
) -> OpsResult<DeviceArray<F>>
```

Computes exact percentiles and uploads them as a 1-D `DeviceArray`.

## `lumen_ops::core::separable`

Multi-axis separable pass driver using `Workspace` ping-pong buffers.

### `run`

```rust
pub fn run<F, P, Step>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    ws: &mut Workspace,
    passes: &[P],
    step: Step,
) -> OpsResult<()>
where
    F: ImageFloat,
    Step: FnMut(&Client, &DeviceArray<F>, &DeviceArray<F>, &P) -> OpsResult<()>
```

Runs `passes` sequentially from `input` into `output`, alternating `"sep_ping"` and `"sep_pong"`
slots from `ws` for intermediate passes.

- When `passes` is empty, copies `input` into `output` on the device.
- When `passes.len() == 1`, runs directly from `input` into `output` without touching `ws`.
- When `passes.len() > 1`, alternates workspace buffers and writes the final pass directly into
  `output`.

## `lumen_ops::core::spline`

B-spline prefiltering and N-D coordinate sampling (orders 0–5).

### `spline_poles`

```rust
pub fn spline_poles(order: usize) -> OpsResult<&'static [f64]>
```

Returns the B-spline filter poles for interpolation `order` (`2..=5`).

### `spline_prefilter_1d_into`

```rust
pub fn spline_prefilter_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    order: usize,
    axis: isize,
) -> OpsResult<()>
```

Applies 1-D B-spline prefiltering of `order` (`2..=5`) along `axis` from `input` into `output`.

### `spline_prefilter_into`

```rust
pub fn spline_prefilter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    order: usize,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Applies N-D B-spline prefiltering across all axes of `input` into `output`.

### `sample_2d_into`

```rust
pub fn sample_2d_into<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    coords_y: &DeviceArray<F>,
    coords_x: &DeviceArray<F>,
    output: &DeviceArray<F>,
    order: usize,
    mode: PadMode,
    cval: f64,
    prefilter: bool,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Samples a 2-D image at device coordinate arrays `(coords_y, coords_x)` into `output`.

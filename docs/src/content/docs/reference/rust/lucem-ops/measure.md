---
title: "lucem_ops::measure"
description: "Quantitative image and region measurements."
sidebar:
  label: "measure"
  order: 8
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Quantitative image and region measurements.

- `regions`: connected-component labeling, bounding boxes, and region feature extraction (`regionprops`).
- `moments`: raw, central, scale-normalized, and Hu rotation-invariant moments, plus inertia tensors.
- `topology`: perimeter, Crofton perimeter, Euler characteristic, block reduction, line profiles, Shannon entropy, and blur score.
- `colocalize`: fluorescence colocalization coefficients (Pearson's $r$, Manders' $M_1, M_2$, overlap, intersection).
- `contour`: 2-D marching squares and 3-D marching cubes isosurface extraction.

## Types

### `SurfaceMesh`

```rust
pub struct SurfaceMesh
```

Triangle mesh extracted by `marching_cubes`.

**Fields**

- `vertices: Vec<[f64; 3]>`: Vertex coordinates `[z, y, x]` in physical/voxel space.
- `faces: Vec<[usize; 3]>`: Triangle index triples `[i0, i1, i2]` into `vertices`.
- `normals: Vec<[f64; 3]>`: Unit outward vertex normals `[nz, ny, nx]`.

Implements `Clone`, `Debug`, `PartialEq`.

### `RegionFeatures`

```rust
pub struct RegionFeatures
```

Rich geometric and intensity features for a single labeled region (analogous to `regionprops`).

**Fields**

- `label: u32`
- `area: usize`
- `bbox: Vec<(usize, usize)>`
- `bbox_area: usize`
- `extent: f64`
- `centroid: Vec<f64>`
- `weighted_centroid: Vec<f64>`
- `equivalent_diameter: f64`
- `major_axis_length: f64`
- `minor_axis_length: f64`
- `eccentricity: f64`
- `orientation: f64`
- `inertia_tensor_eigvals: Vec<f64>`
- `perimeter: f64`
- `mean_intensity: f64`
- `min_intensity: f64`
- `max_intensity: f64`
- `std_intensity: f64`
- `sum_intensity: f64`

Implements `Clone`, `Debug`, `PartialEq`.

### `Contour2d`

```rust
pub type Contour2d = Vec<[f64; 2]>;
```

A 2-D polygonal contour extracted by `marching_squares`: an ordered sequence of `(y, x)`
subpixel vertices.

## Functions

### `pearson_r`

```rust
pub fn pearson_r<F: ImageFloat>(
    client: &Client,
    channel_a: &DeviceArray<F>,
    channel_b: &DeviceArray<F>,
) -> OpsResult<f64>
```

Computes Pearson's linear correlation coefficient $r \in [-1, 1]$ between two identically
shaped intensity channels `channel_a` and `channel_b`.

### `manders_coeff`

```rust
pub fn manders_coeff<F: ImageFloat>(
    client: &Client,
    channel_a: &DeviceArray<F>,
    channel_b: &DeviceArray<F>,
    threshold_a: f64,
    threshold_b: f64,
) -> OpsResult<(f64, f64)>
```

Computes Manders' split colocalization coefficients `(M1, M2)`:
- `M1`: fraction of `channel_a` intensity in pixels where `channel_b > threshold_b`
- `M2`: fraction of `channel_b` intensity in pixels where `channel_a > threshold_a`

### `manders_overlap`

```rust
pub fn manders_overlap<F: ImageFloat>(
    client: &Client,
    channel_a: &DeviceArray<F>,
    channel_b: &DeviceArray<F>,
) -> OpsResult<f64>
```

Computes Manders' uncentered overlap coefficient
$R = \frac{\sum I_A I_B}{\sqrt{\sum I_A^2 \sum I_B^2}}$.

### `intersection_coeff`

```rust
pub fn intersection_coeff<E: Numeric + CubeElement>(
    client: &Client,
    mask_a: &DeviceArray<E>,
    mask_b: &DeviceArray<E>,
) -> OpsResult<f64>
```

Computes the binary intersection (pixel-overlap) fraction $|A \cap B| / |A|$ between two binary
masks `mask_a` and `mask_b` (non-zero treated as foreground).

### `marching_squares`

```rust
pub fn marching_squares<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    level: f64,
) -> OpsResult<Vec<Contour2d>>
```

Extracts 2-D iso-valued contours at `level` from a `[h, w]` image using the Marching Squares
algorithm. Active cells are classified on the device before assembling connected polylines.

### `marching_cubes`

```rust
pub fn marching_cubes<F: ImageFloat>(
    client: &Client,
    volume: &DeviceArray<F>,
    level: f64,
    spacing: Option<[f64; 3]>,
) -> OpsResult<SurfaceMesh>
```

Extracts a 3-D triangulated isosurface at `level` from a `[d, h, w]` volume using tetrahedral
marching-cubes decomposition (guaranteed manifold, crack-free, and free of ambiguous faces),
classifying active voxel cells on the device first.

### `mesh_surface_area`

```rust
pub fn mesh_surface_area(mesh: &SurfaceMesh) -> f64
```

Computes the total surface area of a triangulated 3-D `SurfaceMesh`.

### `moments`

```rust
pub fn moments<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    order: usize,
    center: Option<(f64, f64)>,
) -> OpsResult<Vec<Vec<f64>>>
```

Computes raw 2-D spatial moments $M_{p,q}$ up to `order` around `center` (defaults to `(0.0, 0.0)`).

### `central_moments`

```rust
pub fn central_moments<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    order: usize,
) -> OpsResult<Vec<Vec<f64>>>
```

Computes central 2-D moments $\mu_{p,q}$ up to `order` about the image centroid $(\bar{y}, \bar{x})$.

### `normalized_moments`

```rust
pub fn normalized_moments<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    order: usize,
) -> OpsResult<Vec<Vec<f64>>>
```

Computes scale-normalized central 2-D moments $\eta_{p,q} = \mu_{p,q} / \mu_{0,0}^{1 + (p + q)/2}$
for $p + q \ge 2$ (`NaN` for $p + q < 2$ when `mu00 == 0`, `0.0` at $(0,1),(1,0)$, `1.0` at $(0,0)$).

### `hu_moments`

```rust
pub fn hu_moments<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<[f64; 7]>
```

Computes the 7 classic Hu rotation-, scale-, and translation-invariant moments of a 2-D image.

### `inertia_tensor`

```rust
pub fn inertia_tensor<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<[[f64; 2]; 2]>
```

Computes the 2-D central inertia tensor `[[I_yy, I_yx], [I_xy, I_xx]]` normalized by total mass:
`[[mu02 / mu00, -mu11 / mu00], [-mu11 / mu00, mu20 / mu00]]`.

### `inertia_tensor_eigvals`

```rust
pub fn inertia_tensor_eigvals<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<[f64; 2]>
```

Computes the eigenvalues `[lambda_1, lambda_2]` (`lambda_1 >= lambda_2`) of the 2-D inertia tensor.

### `label_components`

```rust
pub fn label_components<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    footprint: Option<&StructuringElement>,
) -> OpsResult<ConnectedComponents>
```

Labels connected foreground (`!= 0`) components in `input` using `footprint` (or orthogonal
1-connectivity when `None`), returning canonical raster-ordered labels `1..=num_labels`.

### `bounding_boxes`

```rust
pub fn bounding_boxes(
    client: &Client,
    labels: &DeviceArray<u32>,
) -> OpsResult<BTreeMap<u32, Vec<(usize, usize)>>>
```

Returns the tight axis-aligned bounding box `[(min_0, max_0), ...]` (`max` exclusive) for every
positive label in `labels`, sorted by ascending label ID.

### `region_stats`

```rust
pub fn region_stats<F: ImageFloat>(
    client: &Client,
    labels: &DeviceArray<u32>,
    intensity: Option<&DeviceArray<F>>,
) -> OpsResult<Vec<RegionSummary>>
```

Computes deterministic per-label intensity and coordinate summaries (`RegionSummary`) for every
positive label in `labels`.

### `region_features`

```rust
pub fn region_features<F: ImageFloat>(
    client: &Client,
    labels: &DeviceArray<u32>,
    intensity: Option<&DeviceArray<F>>,
) -> OpsResult<Vec<RegionFeatures>>
```

Computes comprehensive geometric and intensity features (`RegionFeatures`) for every positive
label in `labels`.

### `perimeter`

```rust
pub fn perimeter<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    connectivity: usize,
) -> OpsResult<f64>
```

Computes the boundary perimeter of non-zero foreground pixels in a 2-D binary image using
discrete marching-squares edge lengths (4- or 8-connectivity).

### `crofton_perimeter`

```rust
pub fn crofton_perimeter<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    directions: usize,
) -> OpsResult<f64>
```

Estimates the perimeter of a 2-D binary image using the discrete Crofton formula in `directions`
(`2` or `4` orientations).

### `euler_number`

```rust
pub fn euler_number<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    connectivity: usize,
) -> OpsResult<i64>
```

Computes the topological Euler characteristic $\chi = C - H$ (connected components minus holes
in 2-D) of a 2-D binary image using 2x2 bit-quad counts.

### `block_reduce`

```rust
pub fn block_reduce<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    block_size: (usize, usize),
    op: ReduceOp,
    cval: f64,
) -> OpsResult<DeviceArray<F>>
```

Downsamples a 2-D array by pooling non-overlapping `(block_h, block_w)` tiles with `op`
(`Sum`, `Mean`, `Min`, or `Max`), padding incomplete trailing blocks with `cval`.

### `profile_line`

```rust
pub fn profile_line<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    src: (f64, f64),
    dst: (f64, f64),
    num_samples: Option<usize>,
    order: usize,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>>
```

Samples a 1-D intensity profile along the line segment from `src = (y0, x0)` to `dst = (y1, x1)`
of a 2-D image using B-spline interpolation of `order` (`0..=5`).

### `shannon_entropy`

```rust
pub fn shannon_entropy<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
    base: f64,
) -> OpsResult<f64>
```

Computes the Shannon entropy $H = -\sum_k p_k \log_{\text{base}}(p_k)$ of `input` from a
device-computed `nbins`-bin histogram.

### `blur_score`

```rust
pub fn blur_score<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<f64>
```

Computes the variance of the discrete 2-D Laplacian as a sharpness / focus metric (higher values
indicate sharper edges; lower values indicate greater blur).

---
title: "lucem_ops::morphology"
description: "Mathematical morphology: grayscale, binary, exact-Euclidean isotropic, geodesic reconstruction, and skeletonization."
sidebar:
  label: "morphology"
  order: 10
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Mathematical morphology: grayscale, binary, exact-Euclidean isotropic, geodesic reconstruction,
and skeletonization.

## Types

### `BinaryMorphology`

```rust
pub struct BinaryMorphology
```

Settings for binary morphology operations.

**Fields**

- `iterations: usize`
- `border_value: bool`
- `origins: Option<Vec<i32>>`

Implements `Clone`, `Debug`, `Default`, `Eq`, `PartialEq`.

### `GrayMorphology`

```rust
pub struct GrayMorphology
```

Settings for grayscale morphological operations.

**Fields**

- `mode: PadMode`
- `cval: f64`
- `origins: Option<Vec<i32>>`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `ReconstructMethod`

```rust
pub enum ReconstructMethod
```

Geodesic reconstruction method.

**Variants**

- `Dilation`: Iterative geodesic dilation of `seed` under `mask` (`seed <= mask`).
- `Erosion`: Iterative geodesic erosion of `seed` above `mask` (`seed >= mask`).

Implements `Clone`, `Copy`, `Debug`, `Default`, `Eq`, `PartialEq`.

## Functions

### `binary_step_into`

```rust
pub fn binary_step_into(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    output: &DeviceArray<u32>,
    settings: &BinaryMorphology,
    is_dilation: bool,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Binary erosion (`is_dilation = false`) or dilation (`is_dilation = true`) for `settings.iterations`
steps using `ws` ping-pong slots.

### `binary_erode`

```rust
pub fn binary_erode(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    settings: &BinaryMorphology,
) -> OpsResult<DeviceArray<u32>>
```

Binary erosion allocating `output`.

### `binary_dilate`

```rust
pub fn binary_dilate(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    settings: &BinaryMorphology,
) -> OpsResult<DeviceArray<u32>>
```

Binary dilation allocating `output`.

### `binary_open`

```rust
pub fn binary_open(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    settings: &BinaryMorphology,
) -> OpsResult<DeviceArray<u32>>
```

Binary opening (erosion followed by dilation).

### `binary_close`

```rust
pub fn binary_close(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: &StructuringElement,
    settings: &BinaryMorphology,
) -> OpsResult<DeviceArray<u32>>
```

Binary closing (dilation followed by erosion).

### `hit_or_miss`

```rust
pub fn hit_or_miss(
    client: &Client,
    input: &DeviceArray<u32>,
    fg_structure: &StructuringElement,
    bg_structure: &StructuringElement,
) -> OpsResult<DeviceArray<u32>>
```

Binary hit-or-miss transform matching foreground `fg_structure` and background `bg_structure`.

### `fill_holes`

```rust
pub fn fill_holes(
    client: &Client,
    input: &DeviceArray<u32>,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<u32>>
```

Fills internal background (`0`) holes enclosed by foreground (`!= 0`) pixels: labels background
components and fills every background component that does not touch the array border.

### `reconstruct_into`

```rust
pub fn reconstruct_into<F: ImageFloat>(
    client: &Client,
    seed: &DeviceArray<F>,
    mask: &DeviceArray<F>,
    output: &DeviceArray<F>,
    method: ReconstructMethod,
    eight_connected: bool,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Morphological geodesic reconstruction of `seed` under/above `mask` into `output`.

### `reconstruct`

```rust
pub fn reconstruct<F: ImageFloat>(
    client: &Client,
    seed: &DeviceArray<F>,
    mask: &DeviceArray<F>,
    method: ReconstructMethod,
    eight_connected: bool,
) -> OpsResult<DeviceArray<F>>
```

Allocates and computes morphological geodesic reconstruction of `seed` under/above `mask`.

### `h_maxima`

```rust
pub fn h_maxima<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    h: f64,
    eight_connected: bool,
) -> OpsResult<DeviceArray<u32>>
```

H-maxima transform: suppresses all regional maxima whose height is `< h` by reconstructing
`input - h` under `input` and returning the binary mask (`1u32`) where `input - reconstructed >= h`.

### `h_minima`

```rust
pub fn h_minima<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    h: f64,
    eight_connected: bool,
) -> OpsResult<DeviceArray<u32>>
```

H-minima transform: suppresses all regional minima shallower than `h`.

### `regional_maxima`

```rust
pub fn regional_maxima<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    eight_connected: bool,
) -> OpsResult<DeviceArray<u32>>
```

Regional maxima binary mask (`1u32` at connected plateaus strictly greater than all surrounding
neighbors).

### `regional_minima`

```rust
pub fn regional_minima<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    eight_connected: bool,
) -> OpsResult<DeviceArray<u32>>
```

Regional minima binary mask (`1u32` at connected valleys strictly smaller than all surrounding
neighbors).

### `remove_small_objects`

```rust
pub fn remove_small_objects(
    client: &Client,
    input: &DeviceArray<u32>,
    min_size: usize,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<u32>>
```

Removes connected foreground components with fewer than `min_size` pixels from a binary mask
or labeled array.

### `remove_small_holes`

```rust
pub fn remove_small_holes(
    client: &Client,
    input: &DeviceArray<u32>,
    area_threshold: usize,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<u32>>
```

Fills enclosed background holes smaller than `area_threshold` pixels in a binary mask.

### `flood_fill`

```rust
pub fn flood_fill<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    seed_point: &[usize],
    new_value: F,
    tolerance: f64,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<F>>
```

Flood-fills connected pixels starting at `seed_point` within `tolerance` of the seed intensity.

### `erode_into`

```rust
pub fn erode_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Grayscale erosion ($I \ominus B$) into `output`.

### `erode`

```rust
pub fn erode<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>>
```

Grayscale erosion allocating `output`.

### `dilate_into`

```rust
pub fn dilate_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Grayscale dilation ($I \oplus B$) into `output` (maximum filter over the reflected structuring
element).

### `dilate`

```rust
pub fn dilate<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>>
```

Grayscale dilation allocating `output`.

### `open_into`

```rust
pub fn open_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Grayscale opening ($(I \ominus B) \oplus B$) into `output`.

### `open`

```rust
pub fn open<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>>
```

Grayscale opening allocating `output`.

### `close_into`

```rust
pub fn close_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Grayscale closing ($(I \oplus B) \ominus B$) into `output`.

### `close`

```rust
pub fn close<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>>
```

Grayscale closing allocating `output`.

### `white_tophat_into`

```rust
pub fn white_tophat_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()>
```

White top-hat transform ($I - (I \ominus B) \oplus B$) into `output`, isolating bright features
smaller than `footprint`.

### `white_tophat`

```rust
pub fn white_tophat<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>>
```

White top-hat transform allocating `output`.

### `black_tophat_into`

```rust
pub fn black_tophat_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Black top-hat transform ($(I \oplus B) \ominus B - I$) into `output`, isolating dark features
smaller than `footprint`.

### `black_tophat`

```rust
pub fn black_tophat<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>>
```

Black top-hat transform allocating `output`.

### `morph_gradient_into`

```rust
pub fn morph_gradient_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Beucher morphological gradient ($(I \oplus B) - (I \ominus B)$) into `output`.

### `morph_gradient`

```rust
pub fn morph_gradient<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>>
```

Beucher morphological gradient allocating `output`.

### `morph_laplace_into`

```rust
pub fn morph_laplace_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Morphological Laplacian ($(I \oplus B) + (I \ominus B) - 2 I$) into `output`.

### `morph_laplace`

```rust
pub fn morph_laplace<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>>
```

Morphological Laplacian allocating `output`.

### `isotropic_dilate`

```rust
pub fn isotropic_dilate(
    client: &Client,
    input: &DeviceArray<u32>,
    radius: f64,
) -> OpsResult<DeviceArray<u32>>
```

Isotropic binary dilation of `input` by Euclidean ball of radius `radius`: foreground where the
Euclidean distance to the nearest foreground pixel is `<= radius`.

### `isotropic_erode`

```rust
pub fn isotropic_erode(
    client: &Client,
    input: &DeviceArray<u32>,
    radius: f64,
) -> OpsResult<DeviceArray<u32>>
```

Isotropic binary erosion of `input` by Euclidean ball of radius `radius`: foreground where the
Euclidean distance to the nearest background pixel (with outside treated as background) is `> radius`.

### `isotropic_open`

```rust
pub fn isotropic_open(
    client: &Client,
    input: &DeviceArray<u32>,
    radius: f64,
) -> OpsResult<DeviceArray<u32>>
```

Isotropic binary opening (isotropic erosion followed by isotropic dilation).

### `isotropic_close`

```rust
pub fn isotropic_close(
    client: &Client,
    input: &DeviceArray<u32>,
    radius: f64,
) -> OpsResult<DeviceArray<u32>>
```

Isotropic binary closing (isotropic dilation followed by isotropic erosion).

### `skeletonize_into`

```rust
pub fn skeletonize_into(
    client: &Client,
    input: &DeviceArray<u32>,
    output: &DeviceArray<u32>,
    ws: &mut Workspace,
) -> OpsResult<()>
```

2-D parallel binary skeletonization (Zhang-Suen thinning) into `output`, polling the device
convergence flag every `RECONSTRUCT_POLL_INTERVAL` iterations.

### `skeletonize`

```rust
pub fn skeletonize(client: &Client, input: &DeviceArray<u32>) -> OpsResult<DeviceArray<u32>>
```

Allocates and computes the 2-D binary skeleton of `input`.

### `medial_axis`

```rust
pub fn medial_axis(
    client: &Client,
    input: &DeviceArray<u32>,
) -> OpsResult<(DeviceArray<u32>, DeviceArray<f32>)>
```

Computes the 2-D medial axis skeleton and the exact Euclidean distance transform at every pixel:
returns `(skeleton_mask, distance_map)`.

### `convex_hull_image`

```rust
pub fn convex_hull_image(
    client: &Client,
    input: &DeviceArray<u32>,
) -> OpsResult<DeviceArray<u32>>
```

Computes the 2-D binary convex hull mask enclosing all foreground (`!= 0`) pixels of `input`
using Andrew's monotone chain on compacted foreground coordinates and half-plane rasterization.

## Constants

### `RECONSTRUCT_POLL_INTERVAL`

```rust
pub const RECONSTRUCT_POLL_INTERVAL: usize = 8;
```

Number of device geodesic iterations between host convergence flag checks (§2.A rule 2b).

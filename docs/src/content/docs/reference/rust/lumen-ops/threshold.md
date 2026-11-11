---
title: "lumen_ops::threshold"
description: "Global, adaptive local, and hysteresis thresholding."
sidebar:
  label: "threshold"
  order: 14
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Global, adaptive local, and hysteresis thresholding.

## Types

### `LocalThreshold`

```rust
pub struct LocalThreshold
```

Settings for `local_threshold`.

**Fields**

- `method: LocalMethod`
- `offset: f64`: Constant subtracted from the local weighted average (`T(x) = local_stat(x) - offset`).
- `mode: PadMode`
- `cval: f64`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `WindowThreshold`

```rust
pub struct WindowThreshold
```

Settings for `niblack` and `sauvola` local thresholding.

**Fields**

- `window_size: usize`
- `k: f64`
- `r: f64`: Dynamic range of standard deviation (used by Sauvola; defaults to `0.5` for float images).
- `mode: PadMode`
- `cval: f64`

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `LocalMethod`

```rust
pub enum LocalMethod
```

Method for computing the local reference surface in `local_threshold`.

**Variants**

- `Gaussian`
- `Mean`
- `Median`

Implements `Clone`, `Copy`, `Debug`, `Default`, `Eq`, `PartialEq`.

## Functions

### `otsu_device`

```rust
pub fn otsu_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>>
```

Computes the Otsu threshold on the device as a 1-element `DeviceArray<F>`.

### `otsu`

```rust
pub fn otsu<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<f64>
```

Computes the global Otsu threshold maximizing inter-class variance.

### `multi_otsu`

```rust
pub fn multi_otsu<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    classes: usize,
    nbins: usize,
) -> OpsResult<Vec<f64>>
```

Multi-class Otsu thresholding for `classes` classes (`classes >= 2`), returning `classes - 1`
sorted thresholds maximizing between-class variance via dynamic programming over the histogram.

### `mean_threshold_device`

```rust
pub fn mean_threshold_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Global mean threshold as a 1-element `DeviceArray<F>`.

### `mean_threshold`

```rust
pub fn mean_threshold<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<f64>
```

Global mean threshold.

### `isodata`

```rust
pub fn isodata<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<f64>
```

ISODATA iterative threshold ($\theta = \frac{1}{2}(\mu_0(\theta) + \mu_1(\theta))$).

### `isodata_device`

```rust
pub fn isodata_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>>
```

ISODATA threshold on the device.

### `li`

```rust
pub fn li<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<f64>
```

Li's iterative minimum cross-entropy threshold.

### `li_device`

```rust
pub fn li_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>>
```

Li threshold as a 1-element `DeviceArray<F>`.

### `yen`

```rust
pub fn yen<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<f64>
```

Yen's maximum correlation criterion threshold.

### `yen_device`

```rust
pub fn yen_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>>
```

Yen threshold as a 1-element `DeviceArray<F>`.

### `triangle`

```rust
pub fn triangle<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<f64>
```

Geometric triangle threshold: constructs a chord from the histogram peak to the farthest
non-zero tail and picks the bin at maximum perpendicular distance.

### `triangle_device`

```rust
pub fn triangle_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>>
```

Triangle threshold as a 1-element `DeviceArray<F>`.

### `minimum_threshold`

```rust
pub fn minimum_threshold<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
    max_num_iter: usize,
) -> OpsResult<f64>
```

Bimodal minimum threshold: iteratively smooths the histogram with a 3-point moving average
until at most two local maxima remain, then returns the valley minimum between them.

### `hysteresis_threshold`

```rust
pub fn hysteresis_threshold<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    low: f64,
    high: f64,
) -> OpsResult<DeviceArray<u32>>
```

Applies hysteresis thresholding with `low` and `high` thresholds: pixels `> high` are strong
foreground, and pixels `> low` connected (with full connectivity) to at least one strong pixel
are retained (`1u32`, else `0u32`).

### `local_threshold_into`

```rust
pub fn local_threshold_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    block_size: usize,
    output: &DeviceArray<F>,
    settings: &LocalThreshold,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Computes the local adaptive threshold surface $T(x) = \text{local\_stat}(x) - \text{offset}$
over windows of side `block_size` into `output`.

### `local_threshold`

```rust
pub fn local_threshold<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    block_size: usize,
    settings: &LocalThreshold,
) -> OpsResult<DeviceArray<F>>
```

Allocates and returns the local adaptive threshold surface.

### `niblack_into`

```rust
pub fn niblack_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &WindowThreshold,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Niblack local threshold surface $T = \mu - k \sigma$ into `output`.

### `niblack`

```rust
pub fn niblack<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &WindowThreshold,
) -> OpsResult<DeviceArray<F>>
```

Niblack local threshold surface allocating `output`.

### `sauvola_into`

```rust
pub fn sauvola_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &WindowThreshold,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Sauvola local threshold surface $T = \mu \cdot \left(1 + k \left(\frac{\sigma}{R} - 1\right)\right)$
into `output`.

### `sauvola`

```rust
pub fn sauvola<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &WindowThreshold,
) -> OpsResult<DeviceArray<F>>
```

Sauvola local threshold surface allocating `output`.

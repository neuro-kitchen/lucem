---
title: "lumen_ops::intensity"
description: "Intensity conversions, contrast stretching, equalization, and stochastic noise synthesis."
sidebar:
  label: "intensity"
  order: 7
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Intensity conversions, contrast stretching, equalization, and stochastic noise synthesis.

## Types

### `ClaheSettings`

```rust
pub struct ClaheSettings
```

Settings for `clahe` (Contrast Limited Adaptive Histogram Equalization).

**Fields**

- `grid_size: (usize, usize)`: Number of contextual tiles `(tiles_y, tiles_x)`. Default `(8, 8)`.
- `clip_limit: f64`: Normalized clip limit in `[0, 1]`. Default `0.01`.
- `num_bins: usize`: Number of histogram bins per tile. Default `256`.

Implements `Clone`, `Copy`, `Debug`, `Default`.

### `IntensityRange`

```rust
pub enum IntensityRange
```

Input range specification for `rescale_range`.

**Variants**

- `Image`: Automatically stretch from the image's minimum to maximum intensity on the device.
- `Explicit(f64, f64)`: Stretch between explicit `(min, max)` bounds.

Implements `Clone`, `Copy`, `Debug`, `PartialEq`.

### `NoiseModel`

```rust
pub enum NoiseModel
```

Noise distribution model for `add_noise`.

**Variants**

- `Gaussian { mean: f64, std: f64 }`: Additive Gaussian white noise with `(mean, std)`.
- `Speckle { mean: f64, std: f64 }`: Multiplicative speckle noise `x + x * N(mean, std^2)`.
- `Salt { amount: f64 }`: Impulse salt noise replacing `amount` fraction of pixels with `1.0`.
- `Pepper { amount: f64 }`: Impulse pepper noise replacing `amount` fraction of pixels with `0.0`.
- `SaltAndPepper { amount: f64, salt_vs_pepper: f64 }`: Impulse salt-and-pepper noise replacing `amount` fraction of pixels with `1.0` (`salt_vs_pepper`) or `0.0`.
- `Poisson { scale: f64 }`: Shot / Poisson noise scaled by `scale` photons per unit intensity.

Implements `Clone`, `Copy`, `Debug`, `PartialEq`.

## Functions

### `rescale_range`

```rust
pub fn rescale_range<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    in_range: IntensityRange,
    out_min: f64,
    out_max: f64,
) -> OpsResult<DeviceArray<F>>
```

Rescale image intensities from `in_range` into `(out_min, out_max)` on the device.

### `adjust_gamma`

```rust
pub fn adjust_gamma<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    gamma: f64,
    gain: f64,
) -> OpsResult<DeviceArray<F>>
```

Power-law (gamma) transformation: `gain * (x ^ gamma)`.

### `adjust_log`

```rust
pub fn adjust_log<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    gain: f64,
    inv: bool,
) -> OpsResult<DeviceArray<F>>
```

Logarithmic intensity adjustment: `gain * log2(1 + x)` (or `gain * (2^x - 1)` when `inv` is true).

### `adjust_sigmoid`

```rust
pub fn adjust_sigmoid<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    cutoff: f64,
    gain: f64,
    inv: bool,
) -> OpsResult<DeviceArray<F>>
```

Sigmoid contrast adjustment: `1 / (1 + exp(gain * (cutoff - x)))`.

### `is_low_contrast`

```rust
pub fn is_low_contrast<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    fraction_threshold: f64,
    lower_percentile: f64,
    upper_percentile: f64,
) -> OpsResult<bool>
```

Determine whether an image has low contrast by comparing its `[lower_percentile, upper_percentile]`
intensity span to `fraction_threshold` of the dynamic range.

### `to_float`

```rust
pub fn to_float<Src: ImageInt, F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<Src>,
    max_val: f64,
) -> OpsResult<DeviceArray<F>>
```

Convert an unsigned integer `DeviceArray` in `0..=max_val` to a unit floating-point `DeviceArray<F>` in `[0, 1]`.

### `to_u8`

```rust
pub fn to_u8<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<DeviceArray<u8>>
```

Rescale a unit floating-point array in `[0, 1]` to `u8` (`[0, 255]`) on the device.

### `to_u16`

```rust
pub fn to_u16<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<DeviceArray<u16>>
```

Rescale a unit floating-point array in `[0, 1]` to `u16` (`[0, 65535]`) on the device.

### `to_u32`

```rust
pub fn to_u32<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    max_val: f64,
) -> OpsResult<DeviceArray<u32>>
```

Rescale a unit floating-point array in `[0, 1]` to `u32` (`[0, max_val]`) on the device.

### `to_bool`

```rust
pub fn to_bool<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<DeviceArray<u8>>
```

Convert a floating-point array to a binary `u8` mask (`1` where `x > 0`, `0` otherwise).

### `invert`

```rust
pub fn invert<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    max_val: f64,
) -> OpsResult<DeviceArray<F>>
```

Invert a floating-point image around `[0, max_val]` (`max_val - x`).

### `crop`

```rust
pub fn crop<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    crop_widths: &[(usize, usize)],
) -> OpsResult<DeviceArray<E>>
```

Crop `[(before_0, after_0), ...]` elements from each spatial axis of an N-D array (1-D, 2-D, or 3-D).

### `pad`

```rust
pub fn pad<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    pad_widths: &[(usize, usize)],
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>>
```

Pad a floating-point array on each spatial axis according to `pad_widths` and `mode`.

### `cdf_device`

```rust
pub fn cdf_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    num_bins: usize,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>, DeviceArray<F>)>
```

Compute the normalized cumulative distribution function (CDF) and device min/max bounds.

### `cdf`

```rust
pub fn cdf<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    num_bins: usize,
) -> OpsResult<DeviceArray<F>>
```

Compute the normalized cumulative distribution function (CDF) of an image (`[num_bins]`).

### `equalize_hist`

```rust
pub fn equalize_hist<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    num_bins: usize,
) -> OpsResult<DeviceArray<F>>
```

Global histogram equalization mapping intensities to `[0, 1]` via the device-computed CDF.

### `clahe`

```rust
pub fn clahe<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: ClaheSettings,
) -> OpsResult<DeviceArray<F>>
```

Contrast Limited Adaptive Histogram Equalization (CLAHE) for 2-D images on the device.

### `match_histograms`

```rust
pub fn match_histograms<F: ImageFloat>(
    client: &Client,
    source: &DeviceArray<F>,
    reference: &DeviceArray<F>,
    num_bins: usize,
) -> OpsResult<DeviceArray<F>>
```

Match the histogram of `source` to that of `reference` on the device.

### `add_noise`

```rust
pub fn add_noise<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    model: NoiseModel,
    seed: u64,
    clip: bool,
) -> OpsResult<DeviceArray<F>>
```

Add synthetic random noise to an image on the device.

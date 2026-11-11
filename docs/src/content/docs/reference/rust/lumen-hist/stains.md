---
title: "lumen_hist::stains"
description: "Brightfield stains: optical density (Beer–Lambert), stain vectors and colour deconvolution."
sidebar:
  label: "stains"
  order: 20
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Brightfield stains: optical density (Beer–Lambert), stain vectors and colour deconvolution.

A pixel's optical density per channel is `-log10(I / I₀)` with `I₀` the background (white)
value. Each stain has a unit OD vector (how much it absorbs in red, green and blue); three
vectors form a matrix whose inverse separates a pixel's OD into the amount of each stain.
Presets are Ruifrok & Johnston's (2001); `estimate_macenko` fits two stains to a region of
the slide itself (Macenko et al., 2009), which follows the actual staining better.

## Types

### `StainVector`

```rust
pub struct StainVector
```

**Fields**

- `name: String`
- `od: [f64; 3]`: Unit optical density in red, green, blue.

**Methods**

```rust
pub fn new(name: &str, od: [f64; 3]) -> Self
```

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `StainVectors`

```rust
pub struct StainVectors
```

Three stains (the third may be the residual: what the first two do not explain) and the
background RGB.

**Fields**

- `name: String`
- `stains: [StainVector; 3]`
- `background: [f64; 3]`

**Methods**

```rust
pub fn two(name: &str, a: StainVector, b: StainVector, background: [f64; 3]) -> Self
```

Two stains and the residual (their cross product).

```rust
pub fn h_dab() -> Self
```

Hematoxylin and DAB (immunohistochemistry).

```rust
pub fn h_e() -> Self
```

Hematoxylin and eosin.

```rust
pub fn h_e_dab() -> Self
```

Hematoxylin, eosin and DAB.

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

## Functions

### `od`

```rust
pub fn od(value: f64, background: f64) -> f64
```

The optical density of a value against its background (0 at the background, larger darker).

### `deconvolve`

```rust
pub fn deconvolve(rgb: &[u8], stains: &StainVectors) -> HistResult<[Vec<f32>; 3]>
```

Separates interleaved RGB pixels (0–255) into the amount of each stain (optical density
units): three planes, in the order of `stains.stains`.

### `od_sum`

```rust
pub fn od_sum(rgb: &[u8], background: [f64; 3]) -> Vec<f32>
```

The summed optical density of each pixel (how much any stain is there: tissue detection,
"OD sum" measurements).

### `estimate_background`

```rust
pub fn estimate_background(rgb: &[u8]) -> [f64; 3]
```

The background (white) of a brightfield region: the mean of its brightest 1 % of pixels.

### `estimate_macenko`

```rust
pub fn estimate_macenko(
    rgb: &[u8],
    background: [f64; 3],
    names: (&str, &str),
    min_od: f64,
    alpha: f64,
) -> HistResult<StainVectors>
```

Two stains fitted to a region (Macenko): the optical densities of its stained pixels (OD
above `min_od` in every channel) span a plane; the stains are its extreme directions (the
`alpha` and `100 − alpha` percentiles of their angles). The first stain is the one that
absorbs more red (hematoxylin before eosin or DAB). Fewer than 100 stained pixels: an error.

---
title: "lucem_hist::pixel"
description: "Pixel classifiers: every pixel of an image (at an analysis resolution) gets a class from features of its neighbourhood at several scales — Gaussian-smoothed values, gradient magnitude and Laplacian of Gaussian of each channel, computed on the GPU through `lucem-ops` — and Random Trees trained on pixels of regions drawn and given a class. The class map then measures areas (`crate::area`) or becomes regions (`class_regions`)."
sidebar:
  label: "pixel"
  order: 15
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Pixel classifiers: every pixel of an image (at an analysis resolution) gets a class from
features of its neighbourhood at several scales — Gaussian-smoothed values, gradient
magnitude and Laplacian of Gaussian of each channel, computed on the GPU through
`lucem-ops` — and Random Trees trained on pixels of regions drawn and given a class. The
class map then measures areas (`crate::area`) or becomes regions (`class_regions`).

## Types

### `PixelFeatures`

```rust
pub struct PixelFeatures
```

**Fields**

- `sigmas: Vec<f64>`: Scales (pixels of the analysed image).
- `smoothed: bool`
- `gradient: bool`
- `laplacian: bool`

Implements `Clone`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

### `PixelTraining`

```rust
pub struct PixelTraining
```

How a pixel classifier learns: its features, its trees, and at most how many pixels of each
class (evenly spread) it learns from.

**Fields**

- `features: PixelFeatures`
- `forest: ForestSettings`
- `max_per_class: usize`

Implements `Clone`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

### `PixelClassifier`

```rust
pub struct PixelClassifier
```

**Fields**

- `features: PixelFeatures`
- `channels: Vec<String>`: The channels it was trained on, in order (the same must be given to classify).
- `classes: Vec<String>`
- `forest: RandomForest`

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

## Functions

### `features`

```rust
pub fn features(
    client: &Client,
    channels: &[(String, &[f32])],
    w: usize,
    h: usize,
    f: &PixelFeatures,
) -> HistResult<Vec<(String, Vec<f32>)>>
```

The feature planes of `channels` (each `w × h`): names and values.

### `train`

```rust
pub fn train(
    client: &Client,
    channels: &[(String, &[f32])],
    w: usize,
    h: usize,
    labels: &[u8],
    classes: &[String],
    t: &PixelTraining,
) -> HistResult<PixelClassifier>
```

Trains on the pixels `labels` marks (`0`: none, `k`: `classes[k − 1]`).

### `classify`

```rust
pub fn classify(
    client: &Client,
    c: &PixelClassifier,
    channels: &[(String, &[f32])],
    w: usize,
    h: usize,
) -> HistResult<Vec<u8>>
```

The class map of an image (`k`: `classes[k − 1]`).

### `class_regions`

```rust
pub fn class_regions(
    map: &[u8],
    grid: Grid,
    class: u8,
    min_area_px: f64,
    min_hole_px: f64,
) -> Vec<Polygon>
```

The regions of one class of a class map lying on the slide as `grid` (pieces under
`min_area_px` map pixels dropped, holes under `min_hole_px` filled), as level-0 outlines.

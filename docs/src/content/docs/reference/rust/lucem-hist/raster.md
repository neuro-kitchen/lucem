---
title: "lucem_hist::raster"
description: "Small raster helpers on the host, for images already reduced to an analysis resolution (tissue at a coarse level, density maps, class maps): smoothing, Otsu, connected components, small pieces and holes."
sidebar:
  label: "raster"
  order: 17
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Small raster helpers on the host, for images already reduced to an analysis resolution
(tissue at a coarse level, density maps, class maps): smoothing, Otsu, connected components,
small pieces and holes.

## Types

### `Grid`

```rust
pub struct Grid
```

Where a raster lies on the slide: its size, the level-0 position of its first pixel and how
many level-0 pixels one of its pixels covers.

**Fields**

- `width: usize`
- `height: usize`
- `origin: Point`
- `downsample: f64`

**Methods**

```rust
pub fn new(width: usize, height: usize, origin: Point, downsample: f64) -> Self
```

```rust
pub fn len(&self) -> usize
```

```rust
pub fn is_empty(&self) -> bool
```

Implements `Clone`, `Copy`, `Debug`, `PartialEq`.

## Functions

### `gaussian`

```rust
pub fn gaussian(values: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32>
```

A separable Gaussian blur (σ in pixels; edges repeat the border).

### `otsu`

```rust
pub fn otsu(values: &[f32]) -> f32
```

Otsu's threshold of finite values (256 bins between their min and max).

### `components`

```rust
pub fn components(mask: &[u8], w: usize, h: usize) -> (Vec<u32>, u32)
```

8-connected components of the non-zero pixels: labels `1..=n` and `n`.

### `label_areas`

```rust
pub fn label_areas(labels: &[u32], n: u32) -> Vec<usize>
```

Pixels per label (index 0: background).

### `clean`

```rust
pub fn clean(mask: &mut [u8], w: usize, h: usize, min_area: f64, min_hole: f64)
```

Drops pieces of `mask` smaller than `min_area` pixels and fills holes smaller than
`min_hole` pixels (background pieces not touching the border).

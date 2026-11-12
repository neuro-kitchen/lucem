---
title: "lucem_hist::superpixel"
description: "Superpixels: SLIC (on the GPU through `lucem-ops`) cuts an image into pieces of about a given size that follow its edges; each becomes a tile object with its measurements (area, mean and std dev of every channel), ready for an object classifier — a classification by region without detecting cells."
sidebar:
  label: "superpixel"
  order: 21
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Superpixels: SLIC (on the GPU through `lucem-ops`) cuts an image into pieces of about a given
size that follow its edges; each becomes a tile object with its measurements (area, mean and
std dev of every channel), ready for an object classifier — a classification by region
without detecting cells.

## Types

### `SuperpixelSettings`

```rust
pub struct SuperpixelSettings
```

**Fields**

- `spacing_um: f64`: About how wide a superpixel is (µm).
- `compactness: f64`: Higher: more square pieces; lower: closer to the edges.
- `iterations: usize`

Implements `Clone`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

## Functions

### `superpixels`

```rust
pub fn superpixels(
    client: &Client,
    channels: &[(String, &[f32])],
    grid: Grid,
    pixel_um: f64,
    s: &SuperpixelSettings,
) -> HistResult<(Vec<u32>, Vec<HistObject>)>
```

The superpixels of `channels` (SLIC runs on their mean) lying on the slide as `grid`, and
their tile objects; `pixel_um` µm per level-0 pixel.

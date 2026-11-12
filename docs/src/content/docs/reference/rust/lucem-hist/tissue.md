---
title: "lucem_hist::tissue"
description: "Tissue detection: on a low-resolution image of the whole slide, a smoothed value per pixel (how much stain for brightfield, how bright for fluorescence) above a threshold (Otsu's when none is given), pieces and holes smaller than given areas dropped and filled, then outlined."
sidebar:
  label: "tissue"
  order: 22
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Tissue detection: on a low-resolution image of the whole slide, a smoothed value per pixel
(how much stain for brightfield, how bright for fluorescence) above a threshold (Otsu's when
none is given), pieces and holes smaller than given areas dropped and filled, then outlined.

## Types

### `TissueSettings`

```rust
pub struct TissueSettings
```

**Fields**

- `brightfield: bool`: Brightfield (tissue absorbs: the optical density sum) or fluorescence (tissue shines: the mean of the channels).
- `threshold: Option<f64>`: The threshold of the smoothed value; `None`: Otsu's.
- `sigma_um: f64`: Smoothing (µm).
- `min_area_um2: f64`: Pieces under this area go, holes under this area fill (µm²).
- `min_hole_um2: f64`

Implements `Clone`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

### `Tissue`

```rust
pub struct Tissue
```

The tissue of a low-resolution image: its mask, the threshold used, and its regions.

**Fields**

- `mask: Vec<u8>`
- `threshold: f64`
- `regions: Vec<HistObject>`

Implements `Clone`, `Debug`, `PartialEq`.

## Functions

### `detect`

```rust
pub fn detect(
    image: &[f32],
    channels: usize,
    grid: Grid,
    pixel_um: f64,
    s: &TissueSettings,
) -> Tissue
```

The tissue of an interleaved image (`channels` values per pixel; RGB 0–255 for brightfield)
lying on the slide as `grid`, `pixel_um` µm per level-0 pixel; regions in level-0 pixels,
with their area (`Area µm^2`).

---
title: "lumen_hist::cells"
description: "Cell detection and measurement."
sidebar:
  label: "cells"
  order: 2
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Cell detection and measurement.

Nuclei are found on one channel where they are bright (hematoxylin optical density from
`crate::stains::deconvolve`, or a nuclear stain such as DAPI), on the GPU through
`lumen-ops`: an optional rolling-ball background is subtracted, a median and a Gaussian
smooth it, a threshold marks the nuclei, and touching nuclei are split by shape (distance
transform peaks as markers of a watershed). Nuclei out of the area range go; each grows by
`cell_expansion_um` (without crossing a neighbour) into its cell.

Every cell is measured in its compartments (nucleus, cytoplasm, cell) for every channel
given (mean, std dev, min, max) with its shape (area, perimeter, circularity, eccentricity),
named as `Nucleus: Area µm^2`, `Cytoplasm: DAB OD mean`, `Nucleus/Cell area ratio`.

A model's nucleus labels (a plugin of `plugins/ml`) become cells with `cells_from_nuclei`:
the same filtering, expansion and measurements.

## Types

### `CellDetection`

```rust
pub struct CellDetection
```

**Fields**

- `background_radius_um: f64`: Rolling-ball radius of the background subtracted (µm; 0: none).
- `median_radius_um: f64`: Median filter radius (µm; 0: none).
- `sigma_um: f64`: Gaussian smoothing (µm).
- `min_area_um2: f64`: Nuclei kept (µm²).
- `max_area_um2: f64`
- `threshold: f64`: Of the smoothed detection channel (an optical density, or the channel's units).
- `split_by_shape: bool`: Touching nuclei split by their shape.
- `split_by_intensity: bool`: Split at the dark gaps between nuclei instead: seeds at each nucleus' brightest point (of the smoothed image), boundaries along the valleys between them. Better in dense tissue, where the threshold leaves nuclei joined in large pieces.
- `cell_expansion_um: f64`: How far a nucleus grows into its cell (µm; 0: nuclei only).

Implements `Clone`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

### `CellImage`

```rust
pub struct CellImage<'a>
```

A region to detect cells in: the channel nuclei are bright in, and the channels measured
(name, values), all `width × height`, at `downsample` level-0 pixels per pixel from `origin`
(level-0 pixels), `pixel_um` µm per level-0 pixel.

**Fields**

- `width: usize`
- `height: usize`
- `detection: &'a [f32]`
- `channels: Vec<(String, &'a [f32])>`
- `origin: Point`
- `downsample: f64`
- `pixel_um: f64`

### `DetectedCells`

```rust
pub struct DetectedCells
```

Cells found: the objects, and the nucleus and cell labels (`1..=n`, object `id` = label).

**Fields**

- `objects: Vec<HistObject>`
- `nuclei: Vec<u32>`
- `cells: Vec<u32>`

Implements `Clone`, `Debug`.

## Functions

### `detect_cells`

```rust
pub fn detect_cells(
    client: &Client,
    img: &CellImage<'_>,
    s: &CellDetection,
) -> HistResult<DetectedCells>
```

Finds and measures the cells of `img`.

### `cells_from_nuclei`

```rust
pub fn cells_from_nuclei(
    client: &Client,
    img: &CellImage<'_>,
    nuclei: &[u32],
    s: &CellDetection,
) -> HistResult<DetectedCells>
```

Cells from nucleus labels (`0`: none; any ids), from `detect_cells` or from a model: out
of the area range they go, the rest is relabelled `1..=n`, expanded into cells and measured.

---
title: "lumen_hist::slide"
description: "Whole-slide analysis: a region of any size is read tile by tile (with an overlap, so cells on a tile's edge are whole in one of them), cells are detected in each, and a cell is kept by the tile whose core (the tile without its overlap) holds its centroid — each cell once. Inside a drawn region, only the cells whose centroid is in it are kept."
sidebar:
  label: "slide"
  order: 19
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Whole-slide analysis: a region of any size is read tile by tile (with an overlap, so cells on
a tile's edge are whole in one of them), cells are detected in each, and a cell is kept by
the tile whose core (the tile without its overlap) holds its centroid — each cell once.
Inside a drawn region, only the cells whose centroid is in it are kept.

The image's channels become what detection needs through `Staining`: brightfield RGB is
separated into stains (the first finds the nuclei, all are measured as optical densities);
fluorescence finds the nuclei on one channel and measures every channel.

## Types

### `SlideRegion`

```rust
pub struct SlideRegion
```

A region of one pyramid level to analyse.

**Fields**

- `x0: usize`: In pixels of the level read.
- `y0: usize`
- `width: usize`
- `height: usize`
- `downsample: f64`: Level-0 pixels per pixel of the level, and µm per level-0 pixel.
- `pixel_um: f64`
- `within: Option<Polygon>`: Only cells whose centroid is inside (level-0 outline).

Implements `Clone`, `Debug`, `PartialEq`.

### `Tiling`

```rust
pub struct Tiling
```

Tiling of a region.

**Fields**

- `tile: usize`: Tile side (pixels of the level read).
- `overlap: usize`: Pixels shared with the neighbours on each side (more than a cell's diameter).

Implements `Clone`, `Copy`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

### `Staining`

```rust
pub enum Staining
```

How the image's channels become a detection channel and the channels measured.

**Variants**

- `Brightfield { stains: StainVectors }`: RGB (the first three channels) separated with these stains; nuclei on the first stain.
- `Fluorescence { nuclear: usize }`: Nuclei on channel `nuclear`; every channel measured under its name.

**Methods**

```rust
pub fn prepare(
    &self,
    raw: &[Vec<f32>],
    names: &[String],
    ranges: &[f32],
) -> HistResult<Prepared>
```

The detection values and the measured channels of one tile, from its raw channels
(`ranges`: each channel's full range, so 8-bit, 16-bit and float RGB all read as 0–255).

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `Prepared`

```rust
pub type Prepared = (Vec<f32>, Vec<(String, Vec<f32>)>);
```

A tile ready for detection: the values nuclei are found on, and the channels measured.

### `ReadTile`

```rust
pub type ReadTile<'a> = dyn FnMut(usize, usize, usize, usize) -> HistResult<Vec<Vec<f32>>> + 'a;
```

What reads the image: `(x0, y0, w, h)` in pixels of the level → one plane per channel.

### `Labeller`

```rust
pub type Labeller<'a> = dyn FnMut(&[Vec<f32>], usize, usize) -> HistResult<Vec<u32>> + 'a;
```

Nucleus labels of one tile from a model: the tile's raw channels, its width and height →
labels (`0`: none) of the same size.

## Functions

### `auto_threshold`

```rust
pub fn auto_threshold(coarse: &[f32], background_subtracted: bool) -> (f64, f64, f64)
```

A detection threshold for a fluorescence channel from a coarse look at it: Otsu's, less the
median (the background) when a rolling ball takes the background away first. Returns the
threshold, Otsu's and the median.

### `detect_cells_in_region`

```rust
pub fn detect_cells_in_region(
    client: &Client,
    read: &mut ReadTile<'_>,
    names: &[String],
    ranges: &[f32],
    staining: &Staining,
    region: &SlideRegion,
    settings: &CellDetection,
    tiling: Tiling,
    progress: &mut dyn FnMut(usize, usize),
) -> HistResult<Vec<HistObject>>
```

Detects the cells of a region, tile by tile; `progress(done, total)` after each tile. Cell
ids run `1..` over the whole region.

### `cells_in_region`

```rust
pub fn cells_in_region(
    client: &Client,
    read: &mut ReadTile<'_>,
    names: &[String],
    ranges: &[f32],
    staining: &Staining,
    region: &SlideRegion,
    settings: &CellDetection,
    tiling: Tiling,
    labeller: Option<&mut Labeller<'_>>,
    progress: &mut dyn FnMut(usize, usize),
) -> HistResult<Vec<HistObject>>
```

As `detect_cells_in_region`, the nuclei found by `labeller` (a model) when given, else by
the threshold of `settings`; the cells grow from them and are measured the same way.

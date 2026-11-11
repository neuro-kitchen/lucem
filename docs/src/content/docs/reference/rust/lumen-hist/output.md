---
title: "lumen_hist::output"
description: "Where an analysis is stored, and how: next to the pixels it was made from."
sidebar:
  label: "output"
  order: 14
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Where an analysis is stored, and how: next to the pixels it was made from.

- **Where**: inside the image's own OME-Zarr when it is one and can be written (labels
  where other viewers look for them), else `<fallback>/<image>.lumen.zarr`.
- **What**:
  - `labels/cells`, `labels/nuclei`: label images (OME-Zarr `labels`, multiscale, `u32`; each
    object its id), rasterised chunk by chunk in parallel;
  - `lumen/cells.parquet` (and `tiles.parquet`): one row per object — id, kind, class,
    intensity class, region, position (px, µm), then every measurement — Arrow / Parquet,
    read by Polars, PyArrow, pandas, DuckDB;
  - `lumen/shapes.parquet`: GeoParquet (WKB polygons with the `geo` metadata): the objects'
    outlines and nuclei, and the regions;
  - `lumen/provenance.json`: the session (every step and its parameters) and the versions.

## Types

### `RegionShape`

```rust
pub struct RegionShape<'a>
```

A region of the image for the shapes table.

**Fields**

- `id: &'a str`
- `class: &'a str`
- `outline: &'a [[f64; 2]]`

## Functions

### `store_for`

```rust
pub fn store_for(image: &Path, name: &str, fallback: &Path) -> PathBuf
```

The store of an image: its own OME-Zarr when writable, else under `fallback`.

### `write_objects`

```rust
pub fn write_objects(
    store: &Path,
    name: &str,
    objects: &[HistObject],
    region: &[Option<String>],
    pixel_um: f64,
) -> HistResult<PathBuf>
```

Writes `lumen/<name>.parquet`: the objects and their measurements (`region`: each object's
region, `None` the image).

### `write_shapes`

```rust
pub fn write_shapes(
    store: &Path,
    objects: &[HistObject],
    regions: &[RegionShape<'_>],
) -> HistResult<PathBuf>
```

Writes `lumen/shapes.parquet` (GeoParquet 1.1, WKB, level-0 pixel coordinates): every
object's outline (`cell`, `tile`) and nucleus, and the regions.

### `write_labels`

```rust
pub fn write_labels(
    store: &Path,
    name: &str,
    objects: &[HistObject],
    nucleus: bool,
    (width, height): (usize, usize),
    pixel_um: Option<f64>,
) -> HistResult<PathBuf>
```

Writes `labels/<name>`: an OME-Zarr label image of the objects (`nucleus`: their nuclei,
else their outlines), `width × height` level-0 pixels, chunks of 1024 rasterised in parallel.

### `write_provenance`

```rust
pub fn write_provenance(store: &Path, provenance: &Value) -> HistResult<PathBuf>
```

Writes `lumen/provenance.json`.

### `ensure_group`

```rust
pub fn ensure_group(store: &Path) -> HistResult<()>
```

Creates the fallback store's group when it is not the image's own.

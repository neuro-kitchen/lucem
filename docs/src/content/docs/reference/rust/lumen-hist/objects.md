---
title: "lumen_hist::objects"
description: "The objects of an analysis: annotations, detections, cells and tiles, with their measurements."
sidebar:
  label: "objects"
  order: 13
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

The objects of an analysis: annotations, detections, cells and tiles, with their measurements.

## Types

### `HistObject`

```rust
pub struct HistObject
```

**Fields**

- `id: u32`
- `kind: ObjectKind`
- `roi: Polygon`: The outline (a cell's whole cell).
- `nucleus: Option<Polygon>`
- `centroid: Point`
- `class: Option<String>`: Its class (`Tumor`, `CD3: CD8`, …).
- `intensity: Option<IntensityClass>`
- `measurements: Measurements`

**Methods**

```rust
pub fn new(id: u32, kind: ObjectKind, roi: Polygon) -> Self
```

An object from its outline (centroid from the outline).

```rust
pub fn get(&self, name: &str) -> Option<f64>
```

```rust
pub fn full_class(&self) -> Option<String>
```

The class with its intensity, as shown: `Tumor: 2+`, `Positive`, `Stroma`.

```rust
pub fn has_class(&self, class: &str) -> bool
```

Whether its class is `class` or starts with it (`CD3` matches `CD3: CD8`).

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `ObjectKind`

```rust
pub enum ObjectKind
```

**Variants**

- `Annotation`: A region drawn or found (tissue, a class region of a pixel classifier).
- `Detection`: A detected object without a nucleus / cell split.
- `Cell`: A cell: its outline is the cell, `nucleus` its nucleus.
- `Tile`: A tile or superpixel.

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `Eq`, `PartialEq`, `Serialize`.

### `IntensityClass`

```rust
pub enum IntensityClass
```

The intensity class of an object (staining strength of one measurement).

**Variants**

- `Negative`
- `Positive`: One threshold: above it.
- `OnePlus`
- `TwoPlus`
- `ThreePlus`

**Methods**

```rust
pub fn name(self) -> &'static str
```

```rust
pub fn is_positive(self) -> bool
```

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `Eq`, `Ord`, `PartialEq`, `PartialOrd`, `Serialize`.

### `Point`

```rust
pub type Point = [f64; 2];
```

A point in level-0 pixels.

### `Polygon`

```rust
pub type Polygon = Vec<Point>;
```

A closed outline in level-0 pixels (the last point joins the first).

### `Measurements`

```rust
pub type Measurements = BTreeMap<String, f64>;
```

Measurements by name (`Nucleus: Area µm^2`, `Cell: DAB OD mean`, …).

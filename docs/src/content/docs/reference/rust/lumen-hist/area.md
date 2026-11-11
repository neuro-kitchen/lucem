---
title: "lumen_hist::area"
description: "Areas: of each class of a class map, and of positive staining, inside a region."
sidebar:
  label: "area"
  order: 1
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Areas: of each class of a class map, and of positive staining, inside a region.

## Types

### `ClassArea`

```rust
pub struct ClassArea
```

**Fields**

- `class: String`
- `area_um2: f64`
- `percent: f64`: Of the classified area inside the region (unclassified pixels left out).

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

## Functions

### `class_areas`

```rust
pub fn class_areas(
    class_map: &[u8],
    classes: &[String],
    within: Option<&[u8]>,
    pixel_area_um2: f64,
) -> Vec<ClassArea>
```

The area of each class of `class_map` (0: none, `k`: `classes[k − 1]`), only where `within`
is non-zero when given; `pixel_area_um2` is the area of one map pixel.

### `positive_area`

```rust
pub fn positive_area(
    values: &[f32],
    threshold: f32,
    within: Option<&[u8]>,
    pixel_area_um2: f64,
) -> (f64, f64, f64)
```

Positive area: pixels of `values` (a stain's optical density, say) above `threshold` inside
`within`; `(positive µm², region µm², percent)`.

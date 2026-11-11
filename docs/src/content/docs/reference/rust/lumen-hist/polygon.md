---
title: "lumen_hist::polygon"
description: "Outlines: area, perimeter, centroid, point inside, and labels traced into outlines."
sidebar:
  label: "polygon"
  order: 16
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Outlines: area, perimeter, centroid, point inside, and labels traced into outlines.

## Functions

### `area`

```rust
pub fn area(p: &[Point]) -> f64
```

### `perimeter`

```rust
pub fn perimeter(p: &[Point]) -> f64
```

### `centroid`

```rust
pub fn centroid(p: &[Point]) -> Point
```

The centroid of the enclosed area (the mean of the points for a degenerate outline).

### `contains`

```rust
pub fn contains(p: &[Point], q: Point) -> bool
```

Whether `q` is inside `p` (even-odd rule).

### `bounds`

```rust
pub fn bounds(p: &[Point]) -> (f64, f64, f64, f64)
```

The bounding box `(x0, y0, x1, y1)`.

### `trace_labels`

```rust
pub fn trace_labels(
    labels: &[u32],
    w: usize,
    h: usize,
    origin: Point,
    ds: f64,
) -> Vec<(u32, Polygon)>
```

The outer outline of every label of a `w × h` raster (pixel corners, level-0 pixels:
`origin + corner · ds`), by label. Each pixel side between a label and anything else is an
edge; edges chain into loops and the largest loop of a label is its outline (holes are left
out). Points along straight runs are dropped.

### `fill`

```rust
pub fn fill(
    p: &[Point],
    raster: &mut [u32],
    w: usize,
    h: usize,
    origin: Point,
    ds: f64,
    value: u32,
)
```

Fills `p` (level-0 pixels) into a `w × h` raster of a region at `origin` with `ds` level-0
pixels per raster pixel: pixels whose centre is inside get `value`.

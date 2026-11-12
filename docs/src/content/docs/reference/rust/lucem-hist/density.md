---
title: "lucem_hist::density"
description: "Density maps: on a grid over a region, how many objects (of a class, or positive ones) are within a radius of each grid pixel — or what percentage of them are positive; then the hotspots (the highest maxima, apart from each other) and the regions above a value."
sidebar:
  label: "density"
  order: 4
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Density maps: on a grid over a region, how many objects (of a class, or positive ones) are
within a radius of each grid pixel — or what percentage of them are positive; then the
hotspots (the highest maxima, apart from each other) and the regions above a value.

## Types

### `DensitySettings`

```rust
pub struct DensitySettings
```

**Fields**

- `pixel_um: f64`: The map's pixel (µm).
- `radius_um: f64`: The radius counted around each map pixel (µm).
- `class: Option<String>`: Only objects of this class.
- `value: DensityValue`

Implements `Clone`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

### `DensityMap`

```rust
pub struct DensityMap
```

**Fields**

- `width: usize`
- `height: usize`
- `origin: Point`: Level-0 position of the first map pixel, and level-0 pixels per map pixel.
- `ds: f64`
- `values: Vec<f32>`

**Methods**

```rust
pub fn centre_of(&self, i: usize) -> Point
```

Implements `Clone`, `Debug`, `PartialEq`.

### `DensityValue`

```rust
pub enum DensityValue
```

**Variants**

- `Count`: Objects within the radius.
- `PerMm2`: Objects per mm².
- `PercentPositive`: Positive objects (intensity class) as a percentage of those within the radius.

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `Eq`, `PartialEq`, `Serialize`.

## Functions

### `density_map`

```rust
pub fn density_map(
    objects: &[HistObject],
    bounds: (f64, f64, f64, f64),
    pixel_um: f64,
    s: &DensitySettings,
) -> DensityMap
```

The density of `objects` over `bounds` (level-0 `x0, y0, x1, y1`); `pixel_um` µm per level-0
pixel.

### `hotspots`

```rust
pub fn hotspots(
    map: &DensityMap,
    count: usize,
    min_distance_um: f64,
    min_value: f32,
    pixel_um: f64,
) -> Vec<(Point, f32)>
```

The `count` highest local maxima of the map at least `min_value`, each at least
`min_distance_um` from the ones before it (level-0 points, highest first).

### `regions_above`

```rust
pub fn regions_above(
    map: &DensityMap,
    value: f32,
    min_area_um2: f64,
    pixel_um: f64,
) -> Vec<Polygon>
```

The regions where the map is at least `value` (pieces under `min_area_um2` dropped), as
level-0 outlines.

---
title: "lucem_hist::export"
description: "Exporting measurements: one row per object (TSV or CSV), a summary per region (counts per class, positive %, H-score, densities, area), and objects as GeoJSON."
sidebar:
  label: "export"
  order: 7
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Exporting measurements: one row per object (TSV or CSV), a summary per region (counts per
class, positive %, H-score, densities, area), and objects as GeoJSON.

## Types

### `Table`

```rust
pub struct Table
```

A table: its header and rows (text cells).

**Fields**

- `header: Vec<String>`
- `rows: Vec<Vec<String>>`

**Methods**

```rust
pub fn to_delimited(&self, sep: char) -> String
```

The table as text, cells separated by `sep` (`'\t'`, `','`); a cell holding the separator,
a quote or a line break is quoted.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

## Functions

### `measurements_table`

```rust
pub fn measurements_table(objects: &[HistObject], image: &str, pixel_um: f64) -> Table
```

One row per object: image, id, kind, class, centroid (µm), then every measurement any of
them has (sorted by name; empty where one lacks it).

### `region_summary`

```rust
pub fn region_summary(region: &Polygon, objects: &[HistObject], pixel_um: f64) -> Measurements
```

What a region reports about the objects whose centroid is in it: their number, per class,
the intensity summary (when they have intensity classes), the region's area and the
densities per mm².

### `geojson`

```rust
pub fn geojson(objects: &[HistObject]) -> Value
```

The objects as a GeoJSON FeatureCollection (level-0 pixels): the outline as the geometry,
a cell's nucleus as `nucleusGeometry`, the class and measurements as properties.

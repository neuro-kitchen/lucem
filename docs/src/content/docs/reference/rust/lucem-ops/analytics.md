---
title: "lucem_ops::analytics"
description: "Columnar & tabular analytics powered by **Apache Arrow (`arrow`)** and **Polars (`polars`)**."
sidebar:
  label: "analytics"
  order: 1
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Columnar & tabular analytics powered by **Apache Arrow (`arrow`)** and **Polars (`polars`)**.

Bridges GPU measurement outputs from `crate::measure`, `crate::core::histogram`, and
`crate::metric` into typed Arrow `RecordBatch`es and Polars `DataFrame`s for zero-copy
tabular inspection, filtering, sorting, and export in `lucem-app` and `lucem-cli`.

## Functions

### `region_features_arrow_schema`

```rust
pub fn region_features_arrow_schema() -> Schema
```

Builds the canonical Apache Arrow `Schema` for connected-component `RegionFeatures`.

### `regions_to_record_batch`

```rust
pub fn regions_to_record_batch(regions: &[RegionFeatures]) -> OpsResult<RecordBatch>
```

Converts a slice of `RegionFeatures` into an Apache Arrow `RecordBatch`.

### `regions_to_dataframe`

```rust
pub fn regions_to_dataframe(regions: &[RegionFeatures]) -> OpsResult<DataFrame>
```

Converts a slice of `RegionFeatures` into a Polars `DataFrame` for fast filtering, sorting,
and statistical aggregation.

### `histogram_to_dataframe`

```rust
pub fn histogram_to_dataframe(
    bin_centers: &[f64],
    counts: &[u32],
) -> OpsResult<(RecordBatch, DataFrame)>
```

Converts device histogram `bin_centers` and `counts` into both an Apache Arrow `RecordBatch`
and a Polars `DataFrame` with normalized probability density `density`.

### `profile_to_dataframe`

```rust
pub fn profile_to_dataframe(distances: &[f64], intensities: &[f64]) -> OpsResult<DataFrame>
```

Converts a 1-D intensity line profile into a Polars `DataFrame` with columns
`("distance_px", "intensity")`.

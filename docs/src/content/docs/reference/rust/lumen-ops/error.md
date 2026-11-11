---
title: "lumen_ops::error"
description: "Errors of `lumen-ops`."
sidebar:
  label: "error"
  order: 4
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Errors of `lumen-ops`.

## Types

### `OpsError`

```rust
pub enum OpsError
```

**Variants**

- `Core(CoreError)`: Error from `lumen-core` (shapes, axes, element limits, device memory budget, runtime).
- `AliasedBuffers`: Input and output arrays alias the same underlying device buffer.
- `ShapeMismatch { first_name: &'static str, first: Vec<usize>, second_name: &'static str, second: Vec<usize> }`: Two arrays that must have identical shapes disagree.
- `InvalidDimension { what: &'static str, expected: &'static str, actual: usize, shape: Vec<usize> }`: Unexpected array dimensionality (`ndim`).
- `UnknownPadMode(String)`: Unknown boundary padding mode name.
- `InvalidOrigin { origin: i32, size: usize, min: i32, max: i32 }`: Invalid filter origin offset for a window of given size.
- `InvalidWeights { what: &'static str, reason: &'static str }`: Empty or non-finite filter weights or structuring element.
- `InvalidParameter { name: &'static str, reason: String }`: Invalid parameter value (e.g. negative sigma, zero window size, percentile outside `0..=100`).
- `EmptyInput(&'static str)`: Operation is not supported on an empty array (e.g. global min/max or threshold without elements).
- `Analytics(String)`: Error building an Apache Arrow `RecordBatch` or Polars `DataFrame` in analytics.

Implements `Clone`, `Debug`, `Display`, `Error`, `From`, `PartialEq`.

### `OpsResult`

```rust
pub type OpsResult<T> = Result<T, OpsError>;
```

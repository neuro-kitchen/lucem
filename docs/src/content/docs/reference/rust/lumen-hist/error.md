---
title: "lumen_hist::error"
description: "Errors of `lumen-hist`."
sidebar:
  label: "error"
  order: 6
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Errors of `lumen-hist`.

## Types

### `HistError`

```rust
pub enum HistError
```

**Variants**

- `Ops(OpsError)`
- `Core(CoreError)`
- `Invalid(String)`: A setting or an input that cannot work.

Implements `Clone`, `Debug`, `Display`, `Error`, `From`, `PartialEq`.

### `HistResult`

```rust
pub type HistResult<T> = Result<T, HistError>;
```

---
title: "lucem_hist::multiplex"
description: "Multiplexed analysis: one classifier per marker (a measurement of its channel above a threshold makes the cell positive for it), combined into a composite so a cell's class lists its markers (`CD3: CD8`); phenotypes are named marker combinations counted over a region."
sidebar:
  label: "multiplex"
  order: 12
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Multiplexed analysis: one classifier per marker (a measurement of its channel above a
threshold makes the cell positive for it), combined into a composite so a cell's class lists
its markers (`CD3: CD8`); phenotypes are named marker combinations counted over a region.

## Types

### `MarkerRule`

```rust
pub struct MarkerRule
```

**Fields**

- `marker: String`: The marker (`CD8`): the class part a positive cell gets.
- `measurement: String`: What decides it (`Cell: CD8 mean`).
- `threshold: f64`

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `Phenotype`

```rust
pub struct Phenotype
```

A named combination: the markers a cell must have and the ones it must not.

**Fields**

- `name: String`
- `positive: Vec<String>`
- `negative: Vec<String>`

**Methods**

```rust
pub fn matches(&self, o: &HistObject) -> bool
```

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

## Functions

### `marker_classifier`

```rust
pub fn marker_classifier(name: &str, rules: &[MarkerRule]) -> ObjectClassifier
```

The composite classifier of the markers, in their order.

### `phenotype_counts`

```rust
pub fn phenotype_counts<'a>(
    objects: impl IntoIterator<Item = &'a HistObject> + Clone,
    phenotypes: &[Phenotype],
) -> Vec<(String, usize)>
```

How many objects each phenotype has (an object may count in several).

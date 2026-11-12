---
title: "lucem_hist::forest"
description: "Random Trees: an ensemble of decision trees, each grown on a bootstrap sample with a random subset of the features tried at every split (Gini impurity); a prediction is the mean of the trees' class fractions. Deterministic for a seed, and saved as JSON with its classifier."
sidebar:
  label: "forest"
  order: 9
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Random Trees: an ensemble of decision trees, each grown on a bootstrap sample with a random
subset of the features tried at every split (Gini impurity); a prediction is the mean of
the trees' class fractions. Deterministic for a seed, and saved as JSON with its classifier.

## Types

### `ForestSettings`

```rust
pub struct ForestSettings
```

**Fields**

- `trees: usize`
- `max_depth: usize`: 0: no limit.
- `min_samples_leaf: usize`
- `features_per_split: Option<usize>`: Features tried per split; `None`: the square root of their number.
- `seed: u64`

Implements `Clone`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

### `RandomForest`

```rust
pub struct RandomForest
```

**Fields**

- `n_classes: usize`
- `n_features: usize`

**Methods**

```rust
pub fn train(
    x: &[Vec<f64>],
    y: &[usize],
    n_classes: usize,
    s: &ForestSettings,
) -> HistResult<Self>
```

Trains on rows `x` (all the same length, finite: replace missing values first) and their
classes `y` (`0..n_classes`).

```rust
pub fn predict_proba(&self, row: &[f64]) -> Vec<f64>
```

The mean of the trees' class fractions.

```rust
pub fn predict(&self, row: &[f64]) -> (usize, f64)
```

The most likely class and its probability.

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

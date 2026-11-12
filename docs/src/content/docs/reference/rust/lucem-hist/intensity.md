---
title: "lucem_hist::intensity"
description: "Intensity classes: one measurement (a cell's `Nucleus: DAB OD mean`, say) against one threshold (negative / positive) or three (negative, 1+, 2+, 3+), and what a region reports from them: Positive % (a labelling index such as Ki67's) and the H-score (0–300)."
sidebar:
  label: "intensity"
  order: 10
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Intensity classes: one measurement (a cell's `Nucleus: DAB OD mean`, say) against one
threshold (negative / positive) or three (negative, 1+, 2+, 3+), and what a region reports
from them: Positive % (a labelling index such as Ki67's) and the H-score (0–300).

## Types

### `IntensityThresholds`

```rust
pub struct IntensityThresholds
```

**Fields**

- `measurement: String`
- `thresholds: Vec<f64>`: One threshold, or three increasing ones.
- `filter: Option<String>`: Only objects of this class (`Tumor`) get an intensity class.

**Methods**

```rust
pub fn class_of(&self, v: f64) -> IntensityClass
```

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `IntensitySummary`

```rust
pub struct IntensitySummary
```

**Fields**

- `total: usize`: Objects with an intensity class.
- `negative: usize`
- `positive: usize`
- `one_plus: usize`
- `two_plus: usize`
- `three_plus: usize`
- `positive_percent: f64`
- `h_score: f64`: `1 · %1+ + 2 · %2+ + 3 · %3+` (0–300). With one threshold there are no 1+ / 2+ / 3+ classes, so it stays 0.

Implements `Clone`, `Copy`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

## Functions

### `classify`

```rust
pub fn classify(objects: &mut [HistObject], t: &IntensityThresholds) -> usize
```

Sets the intensity class of the objects it applies to (their main class stays); the others
lose theirs. How many were classified.

### `summarize`

```rust
pub fn summarize<'a>(objects: impl IntoIterator<Item = &'a HistObject>) -> IntensitySummary
```

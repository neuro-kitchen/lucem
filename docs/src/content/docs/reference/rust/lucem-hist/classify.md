---
title: "lucem_hist::classify"
description: "Object classifiers: a class for every object (cell, detection, tile) from its measurements."
sidebar:
  label: "classify"
  order: 3
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Object classifiers: a class for every object (cell, detection, tile) from its measurements.

- `ObjectClassifier::SingleMeasurement`: one measurement against a threshold.
- `ObjectClassifier::RandomTrees`: trained from examples, `train_random_trees`: objects
  whose centroid lies in a region drawn and given a class (or under a point of that class)
  are that class's examples. The measurements used are all those the examples share, or the
  ones chosen; they may be normalized (mean 0, std dev 1), which trees do not need.
- `ObjectClassifier::Composite`: several applied in turn, their classes joined (`CD3: CD8`)
  — the multiplexed case.

A classifier is saved as JSON (`ObjectClassifier::to_json`) and applied to other images.

## Types

### `ObjectClassifier`

```rust
pub enum ObjectClassifier
```

**Variants**

- `SingleMeasurement { measurement: String, threshold: f64, above: Option<String>, below: Option<String>, filter: Option<String> }`
- `RandomTrees { features: Vec<String>, classes: Vec<String>, normalize: Option<Vec<(f64, f64)>>, forest: RandomForest, filter: Option<String> }`
- `Composite { name: String, parts: Vec<ObjectClassifier> }`

**Methods**

```rust
pub fn classify(&self, o: &HistObject) -> Option<String>
```

The class `o` gets (`None`: none, or not for this classifier).

```rust
pub fn apply(&self, objects: &mut [HistObject]) -> usize
```

Classifies every object it applies to (the others keep their class); how many changed.

```rust
pub fn to_json(&self) -> String
```

```rust
pub fn from_json(text: &str) -> HistResult<Self>
```

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `FeatureChoice`

```rust
pub enum FeatureChoice
```

Which measurements train a classifier.

**Variants**

- `All`: Every measurement all the examples have.
- `Selected(Vec<String>)`

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `Example`

```rust
pub enum Example
```

An example region or point and its class.

**Variants**

- `Region(Polygon, String)`
- `Point(Point, String)`

Implements `Clone`, `Debug`, `PartialEq`.

## Functions

### `train_random_trees`

```rust
pub fn train_random_trees(
    objects: &[HistObject],
    examples: &[Example],
    features: &FeatureChoice,
    normalize: bool,
    filter: Option<String>,
    s: &ForestSettings,
) -> HistResult<ObjectClassifier>
```

Trains Random Trees from the objects the examples cover.

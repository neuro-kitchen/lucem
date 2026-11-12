---
title: "lucem_hist::models"
description: "Histology models as plugins."
sidebar:
  label: "models"
  order: 11
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Histology models as plugins.

lucem never links a model: each upstream family is a crate `models/<family>` (its code under
its licence, weights fetched and verified through `lucem-hub`) and a program
`plugins/ml/lucem-ml-<family>` the app finds and runs (`lucem-ml`'s protocol). A histology
model says so in its manifest (`domains: ["histology"]`) and answers:

- `lucem_ml::Request::Instances` (nucleus and cell segmentation): a label raster, and the
  class of each label when the model classifies (tumour, inflammatory, …). lucem turns the
  labels into cells with `crate::cells::cells_from_nuclei` — the same measurements,
  expansion and classifiers as its own detection — with `labels_from_result` and
  `apply_instance_classes`.
- `Segment` (interactive: SAM) and, for patch encoders, an embedding per tile that object
  classifiers take as features.

`FAMILIES` lists the families planned, in the order they are worth bringing in. Before one
is added, its licence and its weights' are checked at the source (`licence_checked`): some
weights are for non-commercial research only, or gated behind an agreement, and the plugin
must say so in its manifest.

## Types

### `Family`

```rust
pub struct Family
```

A model family to bring in as a plugin.

**Fields**

- `id: &'static str`: `plugins/ml/lucem-ml-<id>`, `models/<id>`.
- `name: &'static str`
- `task: Task`
- `domains: &'static [&'static str]`: Images it is trained for.
- `licence: &'static str`: The code's licence (SPDX) as last known; the weights' may differ.
- `licence_checked: bool`: Whether code and weights licences were confirmed at the source for lucem's use.
- `what: &'static str`

Implements `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`.

## Functions

### `labels_from_result`

```rust
pub fn labels_from_result(r: &InstancesResult) -> HistResult<(Vec<u32>, usize, usize)>
```

The label raster a plugin wrote (`u32` little-endian, `width × height`).

### `apply_instance_classes`

```rust
pub fn apply_instance_classes(
    objects: &mut [HistObject],
    model_labels: &[u32],
    kept: &[u32],
    classes: &[InstanceClass],
)
```

Gives cells the class the model gave their nucleus. `original` maps each object's id (its
label after `crate::cells::cells_from_nuclei` renumbered them) back to the model's label:
the model's label of a pixel inside the object's nucleus.

### `model_nuclei`

```rust
pub fn model_nuclei(
    send: &mut dyn FnMut(&Request) -> Result<Value, String>,
    model: &str,
    channels: usize,
    nuclear: usize,
    raw: &[Vec<f32>],
    ranges: &[f32],
    w: usize,
    h: usize,
    pixel_um: f64,
) -> HistResult<Vec<u32>>
```

The nuclei a model plugin finds in one tile, sent as `instances` through `send` (the
plugin's server): a 3-channel model gets RGB from the first three channels (scaled by their
`ranges` to 0–255); a 1-channel model gets channel `nuclear`, its 0.1st to 99.9th percentile
to 0–255 (a 16-bit channel kept dim by its range would otherwise lose its nuclei to
rounding). Its labels read back (none when it found none).

### `model_level`

```rust
pub fn model_level(downsamples: &[f64], pixel_um: f64) -> usize
```

The pyramid level a nucleus model reads: the coarsest whose pixels are at most 0.55 µm
(models are trained at about 0.5 µm; finer is resampled by the plugin).

## Constants

### `FAMILIES`

```rust
pub const FAMILIES: &[Family] = _;
```

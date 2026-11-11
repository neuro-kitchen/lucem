---
title: "lumen_hist::engine"
description: "The one way a histology step runs. A step is a serializable `Action` (what the session records); `apply` runs it on an image's `Analysis` (what the steps before it left) and says what it did as an `Outcome` (data: shown by the app, returned to an agent, printed by the command line). The app, `lumen session run` and an agent all go through here, so a step does the same wherever it runs."
sidebar:
  label: "engine"
  order: 5
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

The one way a histology step runs. A step is a serializable `Action` (what the session
records); `apply` runs it on an image's `Analysis` (what the steps before it left) and
says what it did as an `Outcome` (data: shown by the app, returned to an agent, printed by
the command line). The app, `lumen session run` and an agent all go through here, so a step
does the same wherever it runs.

Training a classifier is not a step: it makes a step's parameters (the classifier) from the
user's examples; applying it is the step.

## Types

### `Analysis`

```rust
pub struct Analysis
```

What the steps so far left on one image.

**Fields**

- `tissue: Vec<Polygon>`: Tissue regions (level-0 outlines).
- `cells: Vec<HistObject>`
- `tiles: Vec<HistObject>`
- `classifier: Option<ObjectClassifier>`: The classifiers in use: new cells (tiles) are classed by them as they are found.
- `tile_classifier: Option<ObjectClassifier>`
- `class_map: Option<ClassMap>`
- `density: Option<Density>`

Implements `Clone`, `Debug`, `Default`.

### `ClassMap`

```rust
pub struct ClassMap
```

Every pixel's class over a region.

**Fields**

- `origin: [f64; 2]`: Level-0 position of the map's first pixel, and level-0 pixels per map pixel.
- `ds: f64`
- `width: usize`
- `height: usize`
- `map: Vec<u8>`: 0: none (outside the regions), k: `classes[k - 1]`.
- `classes: Vec<String>`
- `areas: Vec<ClassArea>`

Implements `Clone`, `Debug`.

### `Density`

```rust
pub struct Density
```

A density map and how to read it.

**Fields**

- `map: DensityMap`
- `max: f32`
- `label: String`: What a value means ("Tumor: cells per mm²").

Implements `Clone`, `Debug`.

### `NoModels`

```rust
pub struct NoModels
```

No model plugins: a step that names a model fails, saying so.

Implements `Models`.

### `PluginModels`

```rust
pub struct PluginModels
```

The plugin programs found on this machine, each started once when a step needs it.

Implements `Default`, `Models`.

### `Ctx`

```rust
pub struct Ctx<'a>
```

What a step runs with.

**Fields**

- `client: &'a Client`
- `pyramid: &'a Pyramid`
- `z: usize`: The plane (of a volume) it reads.
- `models: &'a mut dyn Models`
- `progress: &'a mut dyn FnMut(&str, Option<f32>)`: What it is doing, and how far (0..1) when known.
- `cancel: &'a AtomicBool`: Set to stop the step after its current tile.

### `Outcome`

```rust
pub enum Outcome
```

What a step did, as data.

**Variants**

- `Tissue { regions: usize, area_mm2: f64, threshold: f64 }`
- `Cells { found: usize, total: usize, threshold: Option<f64> }`
- `Classified { tiles: bool, counts: BTreeMap<String, usize> }`
- `Intensity { total: usize, positive_percent: f64, one_plus: usize, two_plus: usize, three_plus: usize, h_score: f64 }`
- `PixelClasses { areas: Vec<ClassArea> }`
- `Superpixels { found: usize, total: usize }`
- `Density { max: f32, label: String }`
- `Export`: Writing is the front end's (the output store); nothing to compute.

**Methods**

```rust
pub fn summary(&self) -> String
```

One line for people (the status bar, the command line).

Implements `Clone`, `Debug`, `PartialEq`, `Serialize`.

## Traits

### `Models`

```rust
pub trait Models
```

How a step reaches nucleus models (plugins): the app through its running servers, the
command line by starting the plugin programs.

**Methods**

```rust
fn channels(&mut self, id: &str) -> HistResult<usize>;
fn request(&mut self, id: &str, req: &Request) -> Result<Value, String>;
```

## Types

### `Channels`

```rust
pub type Channels = (Vec<(String, Vec<f32>)>, [usize; 4], f64);
```

Every channel of a level-0 rectangle read at `level` (plane `z`), scaled to 0..1 by its
range: names and values, the level region `[x, y, w, h]` and its downsample.

## Functions

### `level_for`

```rust
pub fn level_for(p: &Pyramid, um: f64, target_um: f64, w: f64, h: f64, max_side: f64) -> usize
```

The finest level whose pixels are at most `target_um` (a little over is fine), coarser
still while the `w × h` level-0 rectangle would be over `max_side` pixels.

### `read_channels`

```rust
pub fn read_channels(
    p: &Pyramid,
    level: usize,
    z: usize,
    b: (f64, f64, f64, f64),
) -> HistResult<Channels>
```

### `apply`

```rust
pub fn apply(ctx: &mut Ctx<'_>, a: &mut Analysis, action: &Action) -> HistResult<Outcome>
```

Runs `action` on `a`.

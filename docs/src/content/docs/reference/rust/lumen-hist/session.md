---
title: "lumen_hist::session"
description: "The session: every step that changed the analysis, in order, with its parameters — what was done, kept as JSON (`session.json` next to the project, and in each output's provenance), so the same analysis runs again without the app: `lumen session run session.json <images…>` on a server, image after image."
sidebar:
  label: "session"
  order: 18
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

The session: every step that changed the analysis, in order, with its parameters — what
was done, kept as JSON (`session.json` next to the project, and in each output's
provenance), so the same analysis runs again without the app: `lumen session run
session.json <images…>` on a server, image after image.

Only steps that change data are recorded (finding tissue or cells, training, scoring,
classing pixels, exporting); views, panels and selections are not.

## Types

### `Session`

```rust
pub struct Session
```

**Fields**

- `lumen_session: u32`
- `steps: Vec<Step>`

**Methods**

```rust
pub fn load(path: &Path) -> HistResult<Self>
```

```rust
pub fn save(&self, path: &Path) -> HistResult<()>
```

```rust
pub fn push(&mut self, image: &str, action: Action)
```

Records a step on `image` now.

```rust
pub fn recipe(&self) -> Vec<&Action>
```

The steps to run on another image: the last of each kind, in their order (a step done
again replaces the earlier one: the session keeps what was settled on).

Implements `Clone`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

### `Step`

```rust
pub struct Step
```

One step: when (seconds since 1970), on which image (its name), and what.

**Fields**

- `at: u64`
- `image: String`
- `action: Action`

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `RunResult`

```rust
pub struct RunResult
```

What a run gives for one image.

**Fields**

- `tissue: Vec<Polygon>`
- `cells: Vec<HistObject>`
- `tiles: Vec<HistObject>`
- `notes: Vec<String>`: What each step did (one line each).

Implements `Debug`, `Default`.

### `Within`

```rust
pub enum Within
```

Where a step looks.

**Variants**

- `Image`: The whole image.
- `Tissue`: The tissue found (a `tissue` step before it, or found then).
- `Regions { outlines: Vec<Polygon> }`: These regions (level-0 outlines); on another image, the tissue is used instead.

Implements `Clone`, `Debug`, `Default`, `Deserialize`, `PartialEq`, `Serialize`.

### `Action`

```rust
pub enum Action
```

**Variants**

- `Tissue { settings: TissueSettings }`: Regions of class Tissue.
- `Cells { within: Within, staining: Staining, detection: CellDetection, auto_threshold: bool, model: Option<String> }`: Cells found and measured (`model`: a nucleus model plugin, else the threshold).
- `ClassifyObjects { classifier: ObjectClassifier, tiles: bool }`: The cells (or tiles) classed by a trained classifier.
- `Intensity { thresholds: IntensityThresholds }`: Intensity classes (negative, 1+, 2+, 3+).
- `PixelClasses { classifier: PixelClassifier, pixel_um: f64, within: Within }`: Every pixel classed (inside `within`).
- `Superpixels { within: Within, settings: SuperpixelSettings, pixel_um: f64 }`: Superpixel tiles.
- `Density { settings: DensitySettings }`: A density map.
- `Export`: Everything written to the output store.

**Methods**

```rust
pub fn title(&self) -> &'static str
```

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

## Functions

### `run`

```rust
pub fn run(
    client: &Client,
    p: &Pyramid,
    recipe: &[&Action],
    progress: &mut dyn FnMut(&str),
) -> HistResult<RunResult>
```

Runs `recipe` (see `Session::recipe`) on one image through the engine (the same steps
the app runs). Regions drawn on the image the session was recorded on become this image's
tissue (`Within::Tissue`).

## Constants

### `SESSION_VERSION`

```rust
pub const SESSION_VERSION: u32 = 1;
```

The format this crate reads and writes (`lumen_session` field).

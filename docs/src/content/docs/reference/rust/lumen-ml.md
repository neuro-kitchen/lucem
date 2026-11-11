---
title: "lumen_ml"
description: "The model-plugin protocol: discovery, client, messages."
sidebar:
  label: "lumen-ml"
  order: 6
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

`lumen-ml`: how lumen talks to machine-learning models, without containing any.

Models come as **plugins**: separate programs named `lumen-ml-<family>` (as dsp-kitchen's
sorters), each built from one crate per upstream model family under `models/`, each under its
upstream's licence, with its own runtime (ONNX Runtime, Burn, …). The app and the command line
never link them, so a model never adds weight, licences or a runtime to lumen itself, and a
plugin can be installed, updated or removed on its own.

- `manifest`: what a plugin offers — its models, their task (interactive segmentation,
  segmentation, detection, classification), weights, licence and provenance — as data.
- `protocol`: the requests and events, JSON lines over the program's stdin and stdout; images
  and masks travel as raw files in a scratch folder.
- `discover`: finding the plugin programs (`$LUMEN_PLUGINS`, the user's plugin folder, next
  to the app) and asking each for its manifest.
- `client`: a plugin running in server mode (`<program> serve`), kept alive between requests
  so a model stays loaded (and SAM keeps its image embedding between clicks).

Workspace layout (see `.tasks/2026-11-10_round5-tasks.md`):
```text
crates/lumen-ml            this contract (the app depends on it; serde only)
hub/lumen-hub              verified downloads and the model cache (all network access)
models/<family>            one crate per upstream family (models/sam: SAM 2, Apache-2.0)
plugins/ml/lumen-ml-<fam>  the program the app runs, over the protocol
```

## Types

### `PluginProcess`

```rust
pub struct PluginProcess
```

**Fields**

- `program: String`

**Methods**

```rust
pub fn start(program: &str) -> Result<Self, String>
```

Starts `program serve`.

```rust
pub fn alive(&mut self) -> bool
```

Whether the program is still running.

```rust
pub fn request(
    &mut self,
    req: &Request,
    progress: &mut dyn FnMut(&str, Option<f32>),
) -> Result<Value, String>
```

Sends a request and reads events until its answer; `progress` sees the progress events.

Implements `Drop`.

### `Weight`

```rust
pub struct Weight
```

A file a model needs, verified by hash before use.

**Fields**

- `file: String`
- `url: String`
- `sha256: String`: Lowercase hex SHA-256 (a file without one is refused).
- `bytes: u64`

Implements `Clone`, `Debug`, `Deserialize`, `Eq`, `PartialEq`, `Serialize`.

### `ModelInfo`

```rust
pub struct ModelInfo
```

One model of a plugin.

**Fields**

- `id: String`: Unique among all plugins (`sam2-hiera-tiny`).
- `name: String`
- `family: String`: The upstream family (`sam`, `stardist`, …).
- `task: Task`
- `domains: Vec<String>`: The images it is made for (`histology`, `fluorescence`, `natural`): the app lists a histology model where histology is analysed.
- `description: String`
- `licence: String`: The licence of the model and its weights (SPDX).
- `paper: Option<String>`: Whom to credit: the paper and the upstream code.
- `code: Option<String>`
- `weights: Vec<Weight>`
- `installed: bool`: Its weights are downloaded and verified.
- `default: bool`: The plugin's suggestion when the user has not chosen.
- `channels: u32`: Channels its images have: 3 (RGB, e.g. H&E) or 1 (one fluorescence channel); 0: not said (RGB).

**Methods**

```rust
pub fn download_bytes(&self) -> u64
```

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `RuntimeStatus`

```rust
pub struct RuntimeStatus
```

Whether the plugin's runtime can run here, and why not.

**Fields**

- `ready: bool`
- `detail: String`: What runs the models (`ONNX Runtime 1.22.1 (CUDA, CPU)`), or what is missing and how to get it.
- `gpu: Option<GpuRuntime>`: A GPU build of the runtime the plugin can install (`install runtime-cuda`), when it has one.

Implements `Clone`, `Debug`, `Deserialize`, `Eq`, `PartialEq`, `Serialize`.

### `GpuRuntime`

```rust
pub struct GpuRuntime
```

A plugin's optional GPU runtime.

**Fields**

- `installed: bool`
- `usable: bool`: Installed and everything it needs is there (a driver, libraries such as cuDNN).
- `bytes: u64`
- `detail: String`: What it is, or what it still needs.

Implements `Clone`, `Debug`, `Deserialize`, `Eq`, `PartialEq`, `Serialize`.

### `PluginInfo`

```rust
pub struct PluginInfo
```

A plugin program and what it offers.

**Fields**

- `name: String`
- `version: String`
- `licence: String`: The program's licence (it may differ from lumen's).
- `runtime: RuntimeStatus`
- `models: Vec<ModelInfo>`
- `program: String`: Where it was found (set by `crate::discover`, not by the plugin).

**Methods**

```rust
pub fn model(&self, id: &str) -> Option<&ModelInfo>
```

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `RasterRef`

```rust
pub struct RasterRef
```

A raster in a file: `width × height × channels` bytes, row by row (RGB for images, one byte
per pixel for masks).

**Fields**

- `path: String`
- `width: u32`
- `height: u32`
- `channels: u32`

**Methods**

```rust
pub fn len(&self) -> usize
```

```rust
pub fn is_empty(&self) -> bool
```

Implements `Clone`, `Debug`, `Deserialize`, `Eq`, `PartialEq`, `Serialize`.

### `InstanceClass`

```rust
pub struct InstanceClass
```

A label's class, when the model classifies what it finds (`Neoplastic`, `Inflammatory`).

**Fields**

- `label: u32`
- `class: String`
- `score: Option<f32>`

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `InstancesResult`

```rust
pub struct InstancesResult
```

The answer to `Request::Instances`: the labels (`channels` 1, 4 bytes per pixel,
little-endian `u32`) and the classes of those it classified.

**Fields**

- `labels: RasterRef`
- `classes: Vec<InstanceClass>`
- `ms: f64`

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `MaskOut`

```rust
pub struct MaskOut
```

A mask the plugin wrote, and how sure it is.

**Fields**

- `raster: RasterRef`
- `score: f32`

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `SegmentResult`

```rust
pub struct SegmentResult
```

The answer to `Request::Segment`, best first.

**Fields**

- `masks: Vec<MaskOut>`
- `encode_ms: f64`: Milliseconds spent encoding the image (0 when reused) and decoding the prompts.
- `decode_ms: f64`

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `Task`

```rust
pub enum Task
```

What a model does.

**Variants**

- `InteractiveSegmentation`: A mask from prompts (points, boxes): SAM.
- `Segmentation`: Every object of a region as labels: nuclei, cells.
- `Detection`: Points or boxes of objects.
- `Classification`: A class per object or region.
- `Embedding`: A feature vector per patch (a foundation model's embedding), for classifiers downstream.

**Methods**

```rust
pub fn title(self) -> &'static str
```

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `Eq`, `PartialEq`, `Serialize`.

### `Prompt`

```rust
pub enum Prompt
```

A prompt, in pixels of the image sent.

**Variants**

- `Point { x: f32, y: f32, positive: bool }`: A point on (`positive`) or off the object.
- `Box { x0: f32, y0: f32, x1: f32, y1: f32 }`: A box around the object.

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `Request`

```rust
pub enum Request
```

**Variants**

- `Info`: The plugin's manifest (as `<program> info`).
- `Install { model: String }`: Downloads and verifies a model's weights.
- `Segment { model: String, image: RasterRef, image_key: String, prompts: Vec<Prompt>, multimask: bool, out_dir: String }`: Masks of the object the prompts point at. `image_key` names the image's content: the same key lets the plugin reuse what it computed for that image (SAM's embedding).
- `Instances { model: String, image: RasterRef, pixel_um: Option<f64>, out_dir: String }`: Every object of an image as labels (nuclei, cells): a `u32` raster, `0` background. `pixel_um` lets the model work at the scale it was trained for.
- `Quit`: Ends a server.

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `Event`

```rust
pub enum Event
```

**Variants**

- `Progress { stage: String, fraction: Option<f32> }`
- `Done { result: Value }`: The request's answer (its JSON depends on the request).
- `Error { message: String }`

**Methods**

```rust
pub fn line(&self) -> String
```

One line of the protocol.

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

## Functions

### `plugin_dirs`

```rust
pub fn plugin_dirs() -> Vec<PathBuf>
```

The folders searched, in order.

### `discover`

```rust
pub fn discover() -> Vec<Result<PluginInfo, String>>
```

Every plugin found and its manifest; a program that does not answer is reported as an error
(its name and why), not left out silently.

## Constants

### `PROGRAM_PREFIX`

```rust
pub const PROGRAM_PREFIX: &str = "lumen-ml-";
```

Plugin programs start with this.

### `RUNTIME_ID`

```rust
pub const RUNTIME_ID: &str = "runtime";
```

The id `install` takes (and `Request::Install` carries) for a plugin's runtime itself rather
than a model: a plugin whose runtime is not ready installs it under this id.

### `RUNTIME_CUDA_ID`

```rust
pub const RUNTIME_CUDA_ID: &str = "runtime-cuda";
```

The id for a plugin's GPU build of its runtime (see `manifest::GpuRuntime`).

## `lumen_ml::discover`

Finding the plugin programs: every executable named `lumen-ml-*` in, in order,
`$LUMEN_PLUGINS` (folders separated as in `PATH`), the user's plugin folder
(`$XDG_DATA_HOME/lumen/plugins` or `~/.local/share/lumen/plugins`), the folder of the running
program (an installed app with its plugins beside it, or `target/<profile>` in development)
and `PATH`. The first program of each name wins; each is asked for its manifest
(`<program> info`).

### `programs`

```rust
pub fn programs() -> BTreeMap<String, PathBuf>
```

The plugin programs found, by name.

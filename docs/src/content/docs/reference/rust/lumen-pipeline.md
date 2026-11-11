---
title: "lumen_pipeline"
description: "Pipelines as data: the operation catalog, the executor, tables."
sidebar:
  label: "lumen-pipeline"
  order: 4
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Pipeline engine of `lumen`: operators as typed graph nodes (`ops`), a graph evaluator over
image regions (`eval`), your own functions over images of any size (`custom`), pipelines as
data for headless runs (`spec`), and the older single-input modifier stack with live analytics
(`modifier`, `executor`).

## Types

### `RunOptions`

```rust
pub struct RunOptions
```

How to walk an image.

**Fields**

- `level: usize`
- `channels: Option<Vec<usize>>`: Channels to process; all of them when `None`.
- `walk: Walk`
- `block: Option<usize>`: Side of a block in y and x; the image's tile size when `None` (at least 256).
- `block_z: Option<usize>`: Depth of a block in `Walk::Volume`; the image's chunk depth when `None` (at least 32).
- `halo: usize`: Pixels of context on each side, in y and x.
- `halo_z: Option<usize>`: Planes of context on each side in `Walk::Volume`; `halo` when `None`.
- `read_ahead: usize`: Blocks read ahead of the one being processed.
- `threads: usize`
- `cancel: Option<Arc<AtomicBool>>`
- `progress: Option<ProgressFn>`

**Methods**

```rust
pub fn blocks(&self, py: &Pyramid) -> Result<(Zyx, Vec<Block>), RunError>
```

The grid and the blocks of a walk.

Implements `Clone`, `Default`.

### `BlockContext`

```rust
pub struct BlockContext
```

Where a block sits, for the function that receives it.

**Fields**

- `index: usize`
- `total: usize`
- `level: usize`
- `channel: usize`
- `block: Block`
- `downsample: f64`: Level-0 pixels per pixel of this level.
- `pixel_size_um: Option<f64>`: Size of a pixel of this level (µm), if known.
- `z_spacing_um: Option<f64>`: Distance between planes of this level (µm), if known.

Implements `Clone`, `Debug`.

### `RunSummary`

```rust
pub struct RunSummary
```

What a headless run wrote.

**Fields**

- `output: PathBuf`
- `spec: PathBuf`: The spec, written next to the output.
- `blocks: usize`
- `warnings: Vec<String>`
- `objects: Option<usize>`: Objects found (labels and tables).

Implements `Clone`, `Debug`.

### `PlanNode`

```rust
pub struct PlanNode
```

One node of a plan.

**Fields**

- `op: Option<Op>`: `None`: a source; its value comes from the caller.
- `enabled: bool`: Off: the node passes its first input through (when the types allow it).
- `inputs: Vec<Option<String>>`: The node feeding each input port.

Implements `Clone`, `Debug`.

### `PipelineExecutionResult`

```rust
pub struct PipelineExecutionResult
```

Complete output of evaluating the non-destructive GPU pipeline + analytics pass.

**Fields**

- `width: usize`
- `height: usize`
- `output_host: Vec<f32>`: Processed `[height, width]` image on the host for display tonemapping & pixel inspection.
- `min_val: f64`: Global min, max, mean, and std of the processed image.
- `max_val: f64`
- `mean_val: f64`
- `std_val: f64`
- `otsu_level: f64`
- `entropy_bits: f64`
- `sharpness_score: f64`
- `hist_centers: Vec<f64>`: 64-bin histogram counts and centers for the Analytics chart.
- `hist_counts: Vec<u32>`
- `hist_df: DataFrame`
- `contours: Vec<Contour2d>`: Marching Squares iso-contours at the Otsu threshold level (for VectorCraft contour overlay).
- `regions: Vec<RegionFeatures>`: Extracted connected-component region features + Polars DataFrame + Arrow RecordBatch.
- `regions_df: DataFrame`
- `regions_batch: RecordBatch`
- `elapsed_ms: f64`: Total GPU pipeline execution time in milliseconds.

Implements `Clone`, `Debug`.

### `PipelineExecutor`

```rust
pub struct PipelineExecutor
```

**Methods**

```rust
pub fn execute(
    client: &Client,
    input_dev: &DeviceArray<f32>,
    modifiers: &[ModifierNode],
    extract_contours: bool,
) -> OpsResult<PipelineExecutionResult>
```

Runs all enabled `modifiers` sequentially on `input_dev` on the GPU, then extracts
histogram, Marching Squares contours, and connected-component `RegionFeatures` into
Polars `DataFrame`s and Arrow `RecordBatch`es.

### `ModifierNode`

```rust
pub struct ModifierNode
```

An instance of a non-destructive GPU modifier in the Studio stack.

**Fields**

- `id: u64`
- `kind: ModifierKind`
- `enabled: bool`
- `primary_param: f64`
- `secondary_param: f64`

**Methods**

```rust
pub fn new(id: u64, kind: ModifierKind) -> Self
```

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `SpecNode`

```rust
pub struct SpecNode
```

One operator of a spec.

**Fields**

- `op: Op`
- `inputs: Vec<String>`: The nodes or sources feeding each input port.
- `enabled: bool`

Implements `Clone`, `Debug`, `Deserialize`, `JsonSchema`, `PartialEq`, `Serialize`.

### `PipelineSpec`

```rust
pub struct PipelineSpec
```

A pipeline to run over an image.

**Fields**

- `sources: BTreeMap<String, usize>`: Source name to channel index.
- `nodes: BTreeMap<String, SpecNode>`
- `target: String`: The node whose result is written.
- `level: usize`: The pyramid level it runs on.
- `normalize: bool`: Scale each channel to 0–1 by its type's range, as the app does.

**Methods**

```rust
pub fn plan(&self) -> Plan
```

The graph to evaluate.

```rust
pub fn validate(&self) -> Result<(), String>
```

Names resolve, parameters are valid, ports are all connected and no node feeds itself.

Implements `Clone`, `Debug`, `Deserialize`, `JsonSchema`, `PartialEq`, `Serialize`.

### `LabelStitcher`

```rust
pub struct LabelStitcher
```

Provisional ids, their joins and measurements. Add blocks in raster order.

**Methods**

```rust
pub fn new(grid: Zyx, link_z: bool) -> Self
```

For blocks on a grid of `grid` (`[z, y, x]`). With `link_z` off, objects are joined within
each plane only (a walk plane by plane, where each plane was segmented by itself).

```rust
pub fn add(&mut self, block: &Block, labels: &[u32], intensity: Option<&[f32]>) -> Vec<u32>
```

Adds the owned part of a block. `labels` holds its local labels (0: background) and
`intensity`, when given, the matching image, both C-ordered of `block.size`. Returns the
block with provisional ids.

#### Panics
If the slices are not of the block's size.

```rust
pub fn finish(self) -> Stitched
```

Joins everything, numbers the objects 1, 2, … in order of first appearance and tabulates
them. Positions are in the pixels of the level walked.

Implements `Debug`.

### `Stitched`

```rust
pub struct Stitched
```

The final labels and a table of the objects.

**Fields**

- `map: Vec<u32>`: `map[provisional]` is the final label (1, 2, …; 0 stays background).
- `table: Table`

Implements `Clone`, `Debug`.

### `Table`

```rust
pub struct Table
```

A table of `f64` columns (missing values are NaN).

**Fields**

- `columns: Vec<(String, Vec<f64>)>`

**Methods**

```rust
pub fn new() -> Self
```

```rust
pub fn rows(&self) -> usize
```

```rust
pub fn is_empty(&self) -> bool
```

```rust
pub fn column(&self, name: &str) -> Option<&[f64]>
```

The column called `name`.

```rust
pub fn push_row(&mut self, row: &[(String, f64)])
```

Adds a row; a column not seen before is NaN in the earlier rows, and a column this row
does not name is NaN in it.

```rust
pub fn to_frame(&self) -> PolarsResult<DataFrame>
```

As a Polars frame, for analysis.

```rust
pub fn write_csv(&self, path: &Path) -> Result<()>
```

Writes the table as CSV at `path`, which must not exist yet.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `RunError`

```rust
pub enum RunError
```

**Variants**

- `Io(IoError)`
- `Function(String)`
- `Invalid(String)`
- `Cancelled`

Implements `Debug`, `Display`, `Error`, `From`.

### `Walk`

```rust
pub enum Walk
```

How a volume is walked (a 2D image has only planes).

**Variants**

- `Planes`: Each plane by itself: the function gets `[h, w]`.
- `Volume`: In 3D blocks: the function gets `[d, h, w]`.

Implements `Clone`, `Copy`, `Debug`, `Default`, `Eq`, `PartialEq`.

### `ModifierKind`

```rust
pub enum ModifierKind
```

Supported non-destructive GPU operators in the Studio modifier stack and Node Editor.

**Variants**

- `GaussianBlur`
- `SobelEdges`
- `ClaheContrast`
- `GammaCurve`
- `TvChambolleDenoise`
- `RollingBallBackground`
- `CannyEdges`
- `OtsuThreshold`
- `MorphGradient`

**Methods**

```rust
pub fn title(self) -> &'static str
```

```rust
pub fn category(self) -> &'static str
```

```rust
pub fn primary_label(self) -> &'static str
```

```rust
pub fn default_params(self) -> (f64, f64)
```

```rust
pub fn step(self) -> f64
```

```rust
pub fn bounds(self) -> (f64, f64)
```

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `Eq`, `Hash`, `PartialEq`, `Serialize`.

### `PortType`

```rust
pub enum PortType
```

What flows along an edge.

**Variants**

- `Image`: Intensities (`f32`, one channel).
- `Mask`: 0 / 1 per pixel.
- `Labels`: An object id per pixel (0: background).
- `Table`: One row per object (measurements).

**Methods**

```rust
pub fn name(self) -> &'static str
```

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `Eq`, `Hash`, `JsonSchema`, `PartialEq`, `Serialize`.

### `Value`

```rust
pub enum Value
```

A value on the device (or, for tables, on the host).

**Variants**

- `Image(DeviceArray<f32>)`
- `Mask(DeviceArray<u32>)`
- `Labels(DeviceArray<u32>)`
- `Table(Vec<RegionFeatures>)`

**Methods**

```rust
pub fn port(&self) -> PortType
```

```rust
pub fn dims(&self) -> Option<(usize, usize)>
```

`(height, width)` of a raster value.

```rust
pub fn to_host(&self, client: &Client) -> Option<Vec<f32>>
```

A raster value's pixels as `f32` on the host (labels as their ids, masks 0 / 1).

Implements `Clone`.

### `Op`

```rust
pub enum Op
```

One operator and its parameters.

**Variants**

- `GaussianBlur { sigma: f64 }`: Gaussian smoothing.
- `Median { radius: usize }`: Median in a disk of `radius` px.
- `UnsharpMask { sigma: f64, amount: f64 }`: Sharpening: `x + amount · (x − blur(x))`.
- `DifferenceOfGaussians { low_sigma: f64, high_sigma: f64 }`: Band-pass: `blur(low) − blur(high)` (blob enhancement).
- `Sobel {  }`: Gradient magnitude.
- `Laplace {  }`: Second derivative (edges, spots).
- `Gamma { gamma: f64, gain: f64 }`: `gain · x^gamma` (values in 0–1).
- `Rescale { min: f64, max: f64 }`: Maps `[min, max]` to `[0, 1]`, clipped.
- `Clahe { clip_limit: f64, tiles: usize }`: Contrast-limited adaptive histogram equalization.
- `RollingBall { radius: f64 }`: Subtracts the rolling-ball background.
- `TvDenoise { weight: f64, iterations: usize }`: Total-variation denoising.
- `WhiteTophat { radius: usize }`: Bright details smaller than the disk (background removal).
- `MorphGradient { radius: usize }`: Dilation − erosion (outlines).
- `Canny { sigma: f64 }`: Edges as a mask.
- `OtsuThreshold { bins: usize }`: Pixels above the Otsu threshold.
- `Threshold { value: f64 }`: Pixels at or above `value`.
- `Open { radius: usize }`: Mask opening (removes specks).
- `Close { radius: usize }`: Mask closing (fills gaps).
- `Erode { radius: usize }`
- `Dilate { radius: usize }`
- `RemoveSmall { min_size: usize }`: Drops objects smaller than `min_size` px.
- `MaskUnion {  }`: On where A or B is on.
- `MaskIntersect {  }`: On where A and B are on.
- `MaskSubtract {  }`: On where A is on and B is off.
- `MaskInvert {  }`: On where the mask is off.
- `Label {  }`: Connected objects of a mask, numbered.
- `Watershed { compactness: f64 }`: Grows the marker labels over the relief image (splits touching objects).
- `MaskImage {  }`: The image where the mask is on, 0 elsewhere.
- `Combine { a: f64, b: f64 }`: `a · A + b · B`.
- `Measure {  }`: One row per labelled object: area, shape, intensity.

**Methods**

```rust
pub fn tag(&self) -> &'static str
```

The `op` tag (`gaussian_blur`, …).

```rust
pub fn title(&self) -> &'static str
```

Its name for people.

```rust
pub fn family(&self) -> &'static str
```

Its family (menus, colours).

```rust
pub fn inputs(&self) -> Vec<(PortType, &'static str)>
```

Input types, in port order, with a name for each port.

```rust
pub fn output(&self) -> PortType
```

```rust
pub fn halo(&self) -> usize
```

Pixels of context read around each output pixel (at the level it runs on). Ops that look
at the whole input (thresholds, CLAHE, labelling) give their usual margin; their result on a
region can differ from the whole image's.

```rust
pub fn at_downsample(&self, ds: f64) -> Op
```

The op for a level whose pixels are `ds` level-0 pixels: sizes (sigmas, radii, areas) are
given in level-0 px, so a result looks the same at every pyramid level.

```rust
pub fn is_global(&self) -> bool
```

Whether its result depends on statistics of its whole input (so a region of the image
needs the value computed on the whole image).

```rust
pub fn validate(&self) -> Result<(), String>
```

Checks the parameters (positive sizes, finite numbers).

```rust
pub fn run(&self, client: &Client, inputs: &[Value]) -> Result<Value, String>
```

Runs the op on its inputs (in port order, already type-checked by the caller).

Implements `Clone`, `Debug`, `Deserialize`, `JsonSchema`, `PartialEq`, `Serialize`.

### `Plan`

```rust
pub type Plan = BTreeMap<String, PlanNode>;
```

The graph to run, by node id.

## Functions

### `measure`

```rust
pub fn measure<F>(
    client: &Client,
    py: &Arc<Pyramid>,
    options: &RunOptions,
    f: F,
) -> Result<Table, RunError>
where
    F: FnMut(&Client, &DeviceArray<f32>, &BlockContext) -> Result<Vec<(String, f64)>, String>
```

Calls `f` on every block and gathers what it returns into a table, one row per block and
channel (with `channel`, `z`, `y`, `x` of the block).

### `transform`

```rust
pub fn transform<F>(
    client: &Client,
    py: &Arc<Pyramid>,
    options: &RunOptions,
    output: &Path,
    f: F,
) -> Result<PathBuf, RunError>
where
    F: FnMut(&Client, &DeviceArray<f32>, &BlockContext) -> Result<DeviceArray<f32>, String>
```

Writes `f` of every block to an OME-Zarr store at `output` (which must not exist), channel by
channel. `f` returns an array of the shape it was given.

### `segment`

```rust
pub fn segment<F>(
    client: &Client,
    py: &Arc<Pyramid>,
    options: &RunOptions,
    output: &Path,
    f: F,
) -> Result<Table, RunError>
where
    F: FnMut(&Client, &DeviceArray<f32>, &BlockContext) -> Result<DeviceArray<u32>, String>
```

Writes labels from `f` of every block of one channel to an OME-Zarr store at `output` (which
must not exist), joining objects cut by block borders, and returns one row per object. Labels
need not be numbered consistently between blocks. In `Walk::Planes` each plane's objects are
separate; in `Walk::Volume` objects are joined along z too.

### `run_spec`

```rust
pub fn run_spec(
    client: &Client,
    py: &Arc<Pyramid>,
    spec: &PipelineSpec,
    output: &Path,
    options: RunOptions,
) -> Result<RunSummary, RunError>
```

Runs a pipeline spec over an image, plane by plane: its target's result is written to `output`
(OME-Zarr for images, masks and labels; CSV for tables), and the spec next to it.

### `evaluate`

```rust
pub fn evaluate(
    client: &Client,
    plan: &Plan,
    target: &str,
    source: &mut dyn FnMut(&str) -> Result<Value, String>,
) -> Result<Value, String>
```

The value of `target`; `source(id)` gives each source node's value.

### `evaluate_all`

```rust
pub fn evaluate_all(
    client: &Client,
    plan: &Plan,
    targets: &[String],
    source: &mut dyn FnMut(&str) -> Result<Value, String>,
) -> BTreeMap<String, Result<Value, String>>
```

Every node of `targets` (and what they need), each computed once; a node that fails gives its
error and the others go on.

### `at_downsample`

```rust
pub fn at_downsample(plan: &Plan, ds: f64) -> Plan
```

The plan with every op set for a level of downsample `ds` (see `Op::at_downsample`).

### `halo`

```rust
pub fn halo(plan: &Plan, target: &str) -> usize
```

Pixels of context `target` needs around a region: the largest sum of halos on any path to a
source.

### `catalog`

```rust
pub fn catalog() -> Vec<Op>
```

Every op with its default parameters, in menu order.

## `lumen_pipeline::custom`

Your own operations over an image of any size, built from lumen's core methods.

### `ProgressFn`

```rust
pub type ProgressFn = Arc<dyn Fn(usize, usize) + Send + Sync>;
```

Called with blocks done and blocks in all.

### `DEFAULT_BLOCK_SIDE`

```rust
pub const DEFAULT_BLOCK_SIDE: usize = 1024;
```

Default side of a block when the image's chunks are not used.

## `lumen_pipeline::eval`

Runs a graph of ops: `evaluate` computes one node's value from its inputs, recursively, each
node once (memoized), with the source rasters given by the caller (an image region, a mask
drawn by hand, …). The caller builds the plan from its own document, so this crate knows
nothing about projects or the app.

### `output_type`

```rust
pub fn output_type(
    plan: &Plan,
    id: &str,
    source_type: &dyn Fn(&str) -> Option<PortType>,
) -> Option<PortType>
```

What `id` gives (sources: `source_type`).

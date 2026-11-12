---
title: "lucem_hist"
description: "Histology: tissue, cells, classifiers, densities, the analysis engine and sessions."
sidebar:
  label: "Overview"
  order: 0
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

`lucem-hist`: the histology workflows of lucem, as a library the app, the command line and
scripts share. Each module is one step of a whole-slide analysis:

| module | what it does |
|---|---|
| `stains` | stain vectors (presets, estimated from a region, background) and colour deconvolution |
| `tissue` | tissue detection on a low-resolution image: smoothed threshold, small pieces and holes removed |
| `area` | areas of classes and of positive staining inside a region |
| `cells` | nucleus detection (background, smoothing, threshold, split by shape), cell expansion, measurements per compartment and channel (GPU, through `lucem-ops`); cells from a model's nucleus labels too |
| `features` | smoothed features (a Gaussian-weighted mean of the neighbours' measurements) |
| `classify` | object classifiers: one measurement and a threshold, Random Trees trained from class-labelled regions, composites; saved as JSON |
| `intensity` | intensity classes (negative, 1+, 2+, 3+), Positive % and H-score |
| `density` | density maps of objects, their hotspots and thresholded regions |
| `multiplex` | per-marker classifiers combined, phenotypes from marker combinations |
| `pixel` | pixel classifiers: multiscale features (GPU) and Random Trees, class maps, regions and areas |
| `slide` | whole regions tile by tile (overlap; each cell once, by the tile whose core holds it), inside a drawn region; brightfield or fluorescence input |
| `superpixel` | SLIC superpixels as tile objects with their measurements |
| `export` | measurement tables (TSV / CSV), region summaries, GeoJSON |
| `models` | histology models as plugins (`plugins/ml/lucem-ml-<family>`): the families, their tasks and licences |

Coordinates of objects are level-0 pixels of the slide; lengths and areas in measurements are
micrometres (from the image's pixel size), as the measurement names say.

## At the crate root

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

### `HistObject`

```rust
pub struct HistObject
```

**Fields**

- `id: u32`
- `kind: ObjectKind`
- `roi: Polygon`: The outline (a cell's whole cell).
- `nucleus: Option<Polygon>`
- `centroid: Point`
- `class: Option<String>`: Its class (`Tumor`, `CD3: CD8`, …).
- `intensity: Option<IntensityClass>`
- `measurements: Measurements`

**Methods**

```rust
pub fn new(id: u32, kind: ObjectKind, roi: Polygon) -> Self
```

An object from its outline (centroid from the outline).

```rust
pub fn get(&self, name: &str) -> Option<f64>
```

```rust
pub fn full_class(&self) -> Option<String>
```

The class with its intensity, as shown: `Tumor: 2+`, `Positive`, `Stroma`.

```rust
pub fn has_class(&self, class: &str) -> bool
```

Whether its class is `class` or starts with it (`CD3` matches `CD3: CD8`).

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `IntensityClass`

```rust
pub enum IntensityClass
```

The intensity class of an object (staining strength of one measurement).

**Variants**

- `Negative`
- `Positive`: One threshold: above it.
- `OnePlus`
- `TwoPlus`
- `ThreePlus`

**Methods**

```rust
pub fn name(self) -> &'static str
```

```rust
pub fn is_positive(self) -> bool
```

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `Eq`, `Ord`, `PartialEq`, `PartialOrd`, `Serialize`.

### `Measurements`

```rust
pub type Measurements = BTreeMap<String, f64>;
```

Measurements by name (`Nucleus: Area µm^2`, `Cell: DAB OD mean`, …).

### `ObjectKind`

```rust
pub enum ObjectKind
```

**Variants**

- `Annotation`: A region drawn or found (tissue, a class region of a pixel classifier).
- `Detection`: A detected object without a nucleus / cell split.
- `Cell`: A cell: its outline is the cell, `nucleus` its nucleus.
- `Tile`: A tile or superpixel.

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `Eq`, `PartialEq`, `Serialize`.

### `Point`

```rust
pub type Point = [f64; 2];
```

A point in level-0 pixels.

### `Polygon`

```rust
pub type Polygon = Vec<Point>;
```

A closed outline in level-0 pixels (the last point joins the first).

## Modules

| Module | What it holds |
|---|---|
| [`lucem_hist::area`](/lucem/reference/rust/lucem-hist/area/) | Areas: of each class of a class map, and of positive staining, inside a region. |
| [`lucem_hist::cells`](/lucem/reference/rust/lucem-hist/cells/) | Cell detection and measurement. |
| [`lucem_hist::classify`](/lucem/reference/rust/lucem-hist/classify/) | Object classifiers: a class for every object (cell, detection, tile) from its measurements. |
| [`lucem_hist::density`](/lucem/reference/rust/lucem-hist/density/) | Density maps: on a grid over a region, how many objects (of a class, or positive ones) are within a radius of each grid pixel — or what percentage of them are positive; then the hotspots (the highest maxima, apart from each other) and the regions above a value. |
| [`lucem_hist::engine`](/lucem/reference/rust/lucem-hist/engine/) | The one way a histology step runs. A step is a serializable `Action` (what the session records); `apply` runs it on an image's `Analysis` (what the steps before it left) and says what it did as an `Outcome` (data: shown by the app, returned to an agent, printed by the command line). The app, `lucem session run` and an agent all go through here, so a step does the same wherever it runs. |
| [`lucem_hist::error`](/lucem/reference/rust/lucem-hist/error/) | Errors of `lucem-hist`. |
| [`lucem_hist::export`](/lucem/reference/rust/lucem-hist/export/) | Exporting measurements: one row per object (TSV or CSV), a summary per region (counts per class, positive %, H-score, densities, area), and objects as GeoJSON. |
| [`lucem_hist::features`](/lucem/reference/rust/lucem-hist/features/) | Smoothed features: each object gets, for every measurement, the mean of its neighbours' values weighted by a Gaussian of their centroid distance (FWHM in µm), and how many neighbours are near. A cell then carries what its surroundings look like, which helps a classifier tell regions apart (tumour vs stroma) where single cells are ambiguous. |
| [`lucem_hist::forest`](/lucem/reference/rust/lucem-hist/forest/) | Random Trees: an ensemble of decision trees, each grown on a bootstrap sample with a random subset of the features tried at every split (Gini impurity); a prediction is the mean of the trees' class fractions. Deterministic for a seed, and saved as JSON with its classifier. |
| [`lucem_hist::intensity`](/lucem/reference/rust/lucem-hist/intensity/) | Intensity classes: one measurement (a cell's `Nucleus: DAB OD mean`, say) against one threshold (negative / positive) or three (negative, 1+, 2+, 3+), and what a region reports from them: Positive % (a labelling index such as Ki67's) and the H-score (0–300). |
| [`lucem_hist::models`](/lucem/reference/rust/lucem-hist/models/) | Histology models as plugins. |
| [`lucem_hist::multiplex`](/lucem/reference/rust/lucem-hist/multiplex/) | Multiplexed analysis: one classifier per marker (a measurement of its channel above a threshold makes the cell positive for it), combined into a composite so a cell's class lists its markers (`CD3: CD8`); phenotypes are named marker combinations counted over a region. |
| [`lucem_hist::objects`](/lucem/reference/rust/lucem-hist/objects/) | The objects of an analysis: annotations, detections, cells and tiles, with their measurements. |
| [`lucem_hist::output`](/lucem/reference/rust/lucem-hist/output/) | Where an analysis is stored, and how: next to the pixels it was made from. |
| [`lucem_hist::pixel`](/lucem/reference/rust/lucem-hist/pixel/) | Pixel classifiers: every pixel of an image (at an analysis resolution) gets a class from features of its neighbourhood at several scales — Gaussian-smoothed values, gradient magnitude and Laplacian of Gaussian of each channel, computed on the GPU through `lucem-ops` — and Random Trees trained on pixels of regions drawn and given a class. The class map then measures areas (`crate::area`) or becomes regions (`class_regions`). |
| [`lucem_hist::polygon`](/lucem/reference/rust/lucem-hist/polygon/) | Outlines: area, perimeter, centroid, point inside, and labels traced into outlines. |
| [`lucem_hist::raster`](/lucem/reference/rust/lucem-hist/raster/) | Small raster helpers on the host, for images already reduced to an analysis resolution (tissue at a coarse level, density maps, class maps): smoothing, Otsu, connected components, small pieces and holes. |
| [`lucem_hist::session`](/lucem/reference/rust/lucem-hist/session/) | The session: every step that changed the analysis, in order, with its parameters — what was done, kept as JSON (`session.json` next to the project, and in each output's provenance), so the same analysis runs again without the app: `lucem session run session.json <images…>` on a server, image after image. |
| [`lucem_hist::slide`](/lucem/reference/rust/lucem-hist/slide/) | Whole-slide analysis: a region of any size is read tile by tile (with an overlap, so cells on a tile's edge are whole in one of them), cells are detected in each, and a cell is kept by the tile whose core (the tile without its overlap) holds its centroid — each cell once. Inside a drawn region, only the cells whose centroid is in it are kept. |
| [`lucem_hist::stains`](/lucem/reference/rust/lucem-hist/stains/) | Brightfield stains: optical density (Beer–Lambert), stain vectors and colour deconvolution. |
| [`lucem_hist::superpixel`](/lucem/reference/rust/lucem-hist/superpixel/) | Superpixels: SLIC (on the GPU through `lucem-ops`) cuts an image into pieces of about a given size that follow its edges; each becomes a tile object with its measurements (area, mean and std dev of every channel), ready for an object classifier — a classification by region without detecting cells. |
| [`lucem_hist::tissue`](/lucem/reference/rust/lucem-hist/tissue/) | Tissue detection: on a low-resolution image of the whole slide, a smoothed value per pixel (how much stain for brightfield, how bright for fluorescence) above a threshold (Otsu's when none is given), pieces and holes smaller than given areas dropped and filled, then outlined. |

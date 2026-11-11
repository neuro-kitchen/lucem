---
title: "lumen_io"
description: "Reading and writing images: TIFF, OME-Zarr, pyramids, projects."
sidebar:
  label: "lumen-io"
  order: 2
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Image readers, writers, procedural phantoms, and project manifest I/O for `lumen`.

## Types

### `LoadedImage`

```rust
pub struct LoadedImage
```

A decoded image ready for GPU upload or display.

**Fields**

- `name: String`: Source file path, or synthetic identifier for generated canvases/phantoms.
- `path: Option<PathBuf>`
- `width: usize`
- `height: usize`
- `channels: usize`
- `dtype_label: String`: Original bit depth description (e.g. `"uint8"`, `"uint16"`, `"float32"`).
- `luminance_f32: Vec<f32>`: Normalized `[0.0, 1.0]` luminance plane `[height, width]` in row-major order for single-channel GPU operators.
- `rgba_f32: Vec<f32>`: Interleaved RGBA `[height, width, 4]` in `[0.0, 1.0]` for color-space and stain ops.
- `ome_zarr: Option<OmeZarrContainerInfo>`: Optional OME-Zarr container metadata when loaded from an `.ome.zarr` bundle.

**Methods**

```rust
pub fn blank_canvas(
    name: impl Into<String>,
    width: usize,
    height: usize,
    fill_intensity: f32,
) -> IoResult<Self>
```

Creates a new uniform blank canvas of `width x height` filled with `fill_intensity` in `[0.0, 1.0]`.

```rust
pub fn to_device_luminance(&self, client: &Client) -> IoResult<DeviceArray<f32>>
```

Uploads the `[height, width]` normalized luminance plane to the active CubeCL `Client`.

```rust
pub fn to_device_rgb(&self, client: &Client) -> IoResult<DeviceArray<f32>>
```

Uploads the `[height, width, 3]` RGB tensor to the active CubeCL `Client`.

Implements `Clone`, `Debug`.

### `LoadedBlock`

```rust
pub struct LoadedBlock
```

One block of one channel.

**Fields**

- `block: Block`
- `channel: usize`
- `data: Vec<f32>`: The read part of the block, C-ordered.

Implements `Clone`, `Debug`.

### `BlockLoader`

```rust
pub struct BlockLoader
```

Blocks in walk order: for each block, each channel in turn.

**Methods**

```rust
pub fn new(
    pyramid: Arc<Pyramid>,
    level: usize,
    channels: Vec<usize>,
    blocks: Vec<Block>,
    read_ahead: usize,
    threads: usize,
) -> Self
```

Starts reading `blocks` of `channels` at `level` on `threads` threads, holding at most
`read_ahead` finished blocks.

```rust
pub fn total(&self) -> usize
```

Blocks times channels.

Implements `Drop`, `Iterator`.

### `OmeZarrContainerInfo`

```rust
pub struct OmeZarrContainerInfo
```

Metadata for an opened `.ome.zarr` container attached to `LoadedImage`.

**Fields**

- `root_path: PathBuf`: Path to the `.ome.zarr` root directory on disk.
- `container_name: String`: Display name of the `.ome.zarr` container (e.g. `"TDTNI1_P4_M_SAL_S1_stack.ome.zarr"`).
- `pyramid_levels: Vec<String>`: Available multiscale pyramid levels from `.zattrs` (e.g. `["s0", "s1", "s2", "s3"]`).
- `level_dimensions: Vec<(String, usize, usize)>`: `(level_name, width, height)` for each multiscale pyramid level.
- `active_level: String`: Currently loaded pyramid level (e.g. `"s2"`).
- `auto_pyramid: bool`: Whether zooming in/out automatically switches the active pyramid level (`s3 ↔ s2 ↔ s1 ↔ s0`).
- `channel_labels: Vec<String>`: Channel labels from `.zattrs` `omero.channels` (e.g. `["DAPI_deblur", "RFP"]`).
- `active_channel: Option<usize>`: `None` = composite across channels; `Some(c)` = single channel `c`.
- `label_groups: Vec<String>`: Available segmentation label groups from `labels/.zattrs` (e.g. `["cells", "nuclei"]`).
- `active_label_group: Option<String>`: `None` = primary image pyramid; `Some(name)` = `labels/<name>/<active_level>`.

**Methods**

```rust
pub fn select_level_for_display_width(&self, displayed_width: f32) -> String
```

Chooses the appropriate pyramid level (`s3..s0`) for a given on-screen `displayed_width`
in viewport pixels, with hysteresis so scrolling near a threshold does not oscillate.

Implements `Clone`, `Debug`.

### `SavedModifier`

```rust
pub struct SavedModifier
```

A saved modifier node inside a `.lumen` project file.

**Fields**

- `kind: String`
- `enabled: bool`
- `primary_param: f64`
- `secondary_param: f64`

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `ProjectManifest`

```rust
pub struct ProjectManifest
```

Top-level `.lumen` project manifest.

**Fields**

- `name: String`
- `canvas_width: usize`
- `canvas_height: usize`
- `source_path: Option<String>`
- `phantom_kind: Option<String>`
- `colormap: String`
- `exposure: f32`
- `gamma: f32`
- `modifiers: Vec<SavedModifier>`

**Methods**

```rust
pub fn save_to_file(&self, path: impl AsRef<Path>) -> IoResult<()>
```

```rust
pub fn load_from_file(path: impl AsRef<Path>) -> IoResult<Self>
```

Implements `Clone`, `Debug`, `Deserialize`, `PartialEq`, `Serialize`.

### `LevelInfo`

```rust
pub struct LevelInfo
```

One resolution level.

**Fields**

- `path: String`: The array's path in the container (`s0`, `0`, …), or `memory/<n>`.
- `width: usize`
- `height: usize`
- `tile_w: usize`: Tile size (the chunk size along y and x).
- `tile_h: usize`
- `downsample: f64`: Level-0 pixels per pixel of this level (1 for level 0, about 2 for the next, …).
- `depth: usize`: Planes along z (1 for a 2D image).
- `tile_d: usize`: Chunk size along z.
- `z_downsample: f64`: Level-0 planes per plane of this level.

**Methods**

```rust
pub fn tiles_x(&self) -> usize
```

```rust
pub fn tiles_y(&self) -> usize
```

Implements `Clone`, `Debug`, `PartialEq`.

### `ChannelInfo`

```rust
pub struct ChannelInfo
```

One channel and how the file suggests to show it.

**Fields**

- `label: String`
- `color: [u8; 3]`: Display colour (RGB, 0–255).
- `window: (f32, f32)`: Suggested display window (raw values).
- `range: (f32, f32)`: The range the values can take (raw values).

Implements `Clone`, `Debug`, `PartialEq`.

### `Plane`

```rust
pub struct Plane
```

A plane of one channel: `width × height` raw values, row-major.

**Fields**

- `width: usize`
- `height: usize`
- `data: Vec<f32>`

Implements `Clone`, `Debug`.

### `Tile`

```rust
pub struct Tile
```

One decoded tile of one channel: `width × height` raw values, row-major.

**Fields**

- `level: usize`
- `channel: usize`
- `tx: usize`
- `ty: usize`
- `width: usize`
- `height: usize`
- `data: Vec<f32>`

Implements `Clone`, `Debug`.

### `Pyramid`

```rust
pub struct Pyramid
```

A multiscale image opened for tile reads.

**Fields**

- `name: String`
- `path: Option<PathBuf>`: The file or container on disk (`None` for generated images).
- `levels: Vec<LevelInfo>`: Finest first.
- `channels: Vec<ChannelInfo>`
- `pixel_size_um: Option<f64>`: Size of a level-0 pixel, if the metadata says (µm).
- `z_spacing_um: Option<f64>`: Distance between level-0 planes, if the metadata says (µm).
- `dtype: String`: Stored element type (`uint16`, `float32`, …).
- `label_groups: Vec<String>`: Label images in the container (`labels/<name>`).

**Methods**

```rust
pub fn open(path: &Path) -> IoResult<Self>
```

Opens an OME-Zarr container (or any path inside one), or an image file.

```rust
pub fn open_zarr(root: &Path) -> IoResult<Self>
```

Reads the container's metadata (no pixel data).

```rust
pub fn from_loaded(img: LoadedImage) -> Self
```

An in-memory pyramid of a decoded image: one channel for grey images, three (R, G, B)
for colour ones; each level halves the previous until it fits one tile.

```rust
pub fn from_volume(
    name: impl Into<String>,
    channel_info: Vec<ChannelInfo>,
    channels: Vec<Vec<f32>>,
    [d, h, w]: Zyx,
    pixel_size_um: Option<f64>,
    z_spacing_um: Option<f64>,
    dtype: impl Into<String>,
) -> IoResult<Self>
```

An in-memory pyramid of a volume: each element of `channels` holds `depth × height × width`
values. Levels halve x and y, and z too while the voxels stay about cubic.

```rust
pub fn depth(&self) -> usize
```

Planes along z at level 0 (1 for a 2D image).

```rust
pub fn is_volume(&self) -> bool
```

Whether the image has more than one plane along z.

```rust
pub fn width(&self) -> usize
```

```rust
pub fn height(&self) -> usize
```

```rust
pub fn coarsest(&self) -> usize
```

The coarsest level.

```rust
pub fn level_for_zoom(&self, zoom: f64) -> usize
```

The level to draw at `zoom` screen px per level-0 px: the coarsest whose pixels are not
larger than a screen pixel (sharp), never coarser than the last level.

```rust
pub fn read_tile(&self, level: usize, channel: usize, tx: usize, ty: usize) -> IoResult<Tile>
```

One tile of the first plane of one channel at `level`.

```rust
pub fn read_tile_at(
    &self,
    level: usize,
    channel: usize,
    z: usize,
    tx: usize,
    ty: usize,
) -> IoResult<Tile>
```

One tile of plane `z` of one channel at `level`; edge tiles are cut to the image.

```rust
pub fn read_region(
    &self,
    level: usize,
    channel: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> IoResult<Vec<f32>>
```

A region of the first plane of one channel at `level`.

```rust
pub fn read_region_at(
    &self,
    level: usize,
    channel: usize,
    z: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> IoResult<Vec<f32>>
```

A region of plane `z` of one channel at `level` (`x0, y0, w, h` in that level's pixels,
clipped to it), assembled from its tiles.

```rust
pub fn read_block(
    &self,
    level: usize,
    channel: usize,
    origin: Zyx,
    size: Zyx,
) -> IoResult<Vec<f32>>
```

A box of one channel at `level`: `size` values from `origin` (both `[z, y, x]`, in that
level's pixels), C-ordered. It must lie inside the level.

```rust
pub fn read_plane(
    &self,
    level: usize,
    channel: usize,
    axis: PlaneAxis,
    index: usize,
) -> IoResult<Plane>
```

A plane of one channel at `level`, across `axis` at `index` (that level's pixels): an XY
plane is `width × height`, an XZ plane `width × depth`, a YZ plane `height × depth`.

```rust
pub fn auto_window(&self, channel: usize, low: f64, high: f64) -> IoResult<(f32, f32)>
```

Low and high percentiles of a channel at the coarsest level (an automatic window).

Implements `Debug`.

### `WriterOptions`

```rust
pub struct WriterOptions
```

What to create.

**Fields**

- `name: String`
- `channels: Vec<String>`
- `shape: Zyx`: `[depth, height, width]`; depth 1 is a 2D image.
- `chunks: Zyx`: Chunk size, `[z, y, x]`.
- `pixel_size_um: Option<f64>`
- `z_spacing_um: Option<f64>`

Implements `Clone`, `Debug`.

### `ZarrWriter`

```rust
pub struct ZarrWriter
```

A store being written.

**Methods**

```rust
pub fn create<T: Sample>(root: impl AsRef<Path>, options: WriterOptions) -> IoResult<Self>
```

Starts a store for samples of type `T` at `root`, which must not exist yet.

```rust
pub fn root(&self) -> &Path
```

```rust
pub fn write<T: Sample>(
    &self,
    channel: usize,
    origin: Zyx,
    size: Zyx,
    data: &[T],
) -> IoResult<()>
```

Stores the block of `size` at `origin` of `channel` as one chunk of level 0. `origin` must
be on the chunk grid and `size` a whole chunk, or what remains of one at the image's edge.

```rust
pub fn write_many<T: Sample>(&self, blocks: &[(usize, Zyx, Zyx, Vec<T>)]) -> IoResult<()>
```

Stores many chunks at once, in parallel (each chunk is its own file, so they are
independent): `(channel, origin, size, values)` each, as `Self::write` takes them.

```rust
pub fn remap_labels(&self, map: &[u32]) -> IoResult<()>
```

Replaces every label of level 0 by `map[label]` (labels past its end become 0). Used once
objects cut by block borders have been joined.

```rust
pub fn finish<T: Sample>(self) -> IoResult<PathBuf>
```

Builds the coarser levels, writes the metadata and returns the store's path.

Implements `Debug`.

### `IoError`

```rust
pub enum IoError
```

**Variants**

- `Core(CoreError)`
- `StdIo(Error)`
- `Image(ImageError)`
- `Json(Error)`
- `InvalidDimensions { width: usize, height: usize, reason: &'static str }`
- `OmeZarr(String)`

Implements `Debug`, `Display`, `Error`, `From`.

### `PhantomKind`

```rust
pub enum PhantomKind
```

Built-in synthetic datasets available from the Startup Project Hub and Left Option Rail.

**Variants**

- `FluorescenceNuclei`: Confocal fluorescence brain-section nuclei & neurite filaments (`512 x 512`).
- `SiemensStarTarget`: Siemens star + resolution bar target for filter, deconvolution & edge-detection testing (`512 x 512`).
- `VectorContourShapes`: VectorCraft geometric shapes, rings & SDF contours for Marching Squares / active contours (`512 x 512`).
- `InterferenceFringes`: Smooth multi-frequency interference & phase fringes for restoration / unwrapping (`512 x 512`).

**Methods**

```rust
pub fn title(self) -> &'static str
```

```rust
pub fn subtitle(self) -> &'static str
```

```rust
pub fn badge(self) -> &'static str
```

Implements `Clone`, `Copy`, `Debug`, `Eq`, `Hash`, `PartialEq`.

### `PlaneAxis`

```rust
pub enum PlaneAxis
```

An axis a plane is cut across.

**Variants**

- `Z`: An XY plane, at a position along z.
- `Y`: An XZ plane, at a position along y.
- `X`: A YZ plane, at a position along x.

Implements `Clone`, `Copy`, `Debug`, `Eq`, `Hash`, `PartialEq`.

### `SampleType`

```rust
pub enum SampleType
```

What a store holds.

**Variants**

- `F32`: Intensities, averaged for coarser levels.
- `U32`: Masks and labels, sampled for coarser levels (an average of ids means nothing).

Implements `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`.

## Traits

### `Sample`

```rust
pub trait Sample
```

An element a store can hold.

**Methods**

```rust
fn to_bytes(values: &[Self], out: &mut Vec<u8>);
fn from_bytes(bytes: &[u8]) -> Vec<Self>;
fn reduce(values: &[Self]) -> Self;
```

## Types

### `IoResult`

```rust
pub type IoResult<T> = Result<T, IoError>;
```

## Functions

### `load_image_file`

```rust
pub fn load_image_file(path: impl AsRef<Path>) -> IoResult<LoadedImage>
```

Loads an image file or `.ome.zarr` pyramid directory from disk (`.ome.zarr`, `.png`, `.tif`,
`.tiff`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif`, `.hdr`, `.exr`).

### `save_image_file`

```rust
pub fn save_image_file(
    path: impl AsRef<Path>,
    width: usize,
    height: usize,
    luminance_f32: &[f32],
) -> IoResult<()>
```

Saves a `[height, width]` normalized `f32` buffer to disk as an 8-bit grayscale PNG/TIFF/JPEG.

### `find_ome_zarr_root`

```rust
pub fn find_ome_zarr_root(path: &Path) -> Option<PathBuf>
```

Walks up `path` ancestors to find the root `.ome.zarr` container directory.

### `is_ome_zarr_path`

```rust
pub fn is_ome_zarr_path(path: &Path) -> bool
```

Returns `true` if `path` is (or is inside) an `.ome.zarr` container.

### `inspect_ome_zarr_container`

```rust
pub fn inspect_ome_zarr_container(root_dir: &Path) -> IoResult<OmeZarrContainerInfo>
```

Parses `.zattrs` and `labels/.zattrs` from the `.ome.zarr` container root.

### `load_ome_zarr`

```rust
pub fn load_ome_zarr(path: &Path) -> IoResult<LoadedImage>
```

Loads an `.ome.zarr` container (or any path inside it) using its default view settings.

### `load_ome_zarr_with_options`

```rust
pub fn load_ome_zarr_with_options(info: &OmeZarrContainerInfo) -> IoResult<LoadedImage>
```

Loads the specified pyramid level, channel selection, or label group from an `OmeZarrContainerInfo`.

### `discover_workspace_data_paths`

```rust
pub fn discover_workspace_data_paths(workspace_dir: &Path) -> Vec<PathBuf>
```

Scans `.data/` in the workspace for top-level `.ome.zarr` containers and standard image files.

### `generate_phantom`

```rust
pub fn generate_phantom(kind: PhantomKind, width: usize, height: usize) -> LoadedImage
```

Generates a `width x height` synthetic `LoadedImage` for `kind`.

### `sample_volume`

```rust
pub fn sample_volume(size: usize, depth: usize) -> IoResult<Pyramid>
```

A synthetic volume for trying the 3D paths: bright spheres in one channel and a helical tube
in another, `depth` planes of `size × size` with voxels twice as deep as they are wide.

## Constants

### `MEMORY_TILE`

```rust
pub const MEMORY_TILE: usize = 512;
```

Edge of the tiles of an in-memory pyramid (logical tile; Zarr levels use their chunk size).

## `lumen_io::pyramid`

Multiscale images, 2D or 3D, read piece by piece: an OME-Zarr container (v0.4, Zarr v2), a
TIFF z-stack, or a plain image file (PNG, TIFF, JPEG, …) turned into an in-memory pyramid.

### `VOLUME_COARSEST`

```rust
pub const VOLUME_COARSEST: usize = 64;
```

An in-memory volume's levels halve until they fit this many voxels a side.

### `MAX_ARRAY_ELEMENTS`

```rust
pub const MAX_ARRAY_ELEMENTS: u128 = _;
```

Most elements of an array a file may declare.

### `MAX_CHUNK_BYTES`

```rust
pub const MAX_CHUNK_BYTES: u128 = _;
```

Most bytes one decoded chunk may hold.

### `MAX_READ_ELEMENTS`

```rust
pub const MAX_READ_ELEMENTS: usize = _;
```

Most elements of one read.

### `MAX_LEVELS`

```rust
pub const MAX_LEVELS: usize = 32;
```

Most resolution levels a file may declare.

### `CHUNK_CACHE_BYTES`

```rust
pub const CHUNK_CACHE_BYTES: usize = _;
```

Decoded multi-plane chunks kept for repeated plane reads (bytes).

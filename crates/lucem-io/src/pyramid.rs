//! Multiscale images, 2D or 3D, read piece by piece: an OME-Zarr container (v0.4, Zarr v2), a
//! TIFF z-stack, or a plain image file (PNG, TIFF, JPEG, …) turned into an in-memory pyramid.
//!
//! Nothing is decoded until a piece is asked for: [`Pyramid::read_tile_at`] reads one tile of one
//! plane, [`Pyramid::read_block`] any box and [`Pyramid::read_plane`] a plane along any axis. Values
//! are raw `f32` (not normalized; the display decides the window). Axes, levels, channel names,
//! colours and display windows, and the voxel size come from the metadata (`axes`, `multiscales`,
//! `omero`), whatever the levels are called. A 3D image is never guessed from its array's shape.
//!
//! Everything in a file's metadata is untrusted: paths must stay inside the container and sizes
//! are checked before anything is allocated.
//!
//! The reader is `Sync`: a pool of threads reads tiles of one pyramid at once.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lucem_core::Zyx;

use serde::Deserialize;

use crate::error::{IoError, IoResult};
use crate::image_io::{load_image_file, LoadedImage};
use crate::ome_zarr::{blosc_decompress, find_ome_zarr_root, is_safe_relative_path};

/// A cached chunk and when it was last used.
type CachedChunk = (Arc<Vec<f32>>, u64);

/// Edge of the tiles of an in-memory pyramid (logical tile; Zarr levels use their chunk size).
pub const MEMORY_TILE: usize = 512;
/// An in-memory volume's levels halve until they fit this many voxels a side.
pub const VOLUME_COARSEST: usize = 64;
/// Most elements of an array a file may declare.
pub const MAX_ARRAY_ELEMENTS: u128 = 1 << 48;
/// Most bytes one decoded chunk may hold.
pub const MAX_CHUNK_BYTES: u128 = 1 << 31;
/// Most elements of one read.
pub const MAX_READ_ELEMENTS: usize = 1 << 34;
/// Most resolution levels a file may declare.
pub const MAX_LEVELS: usize = 32;
/// Decoded multi-plane chunks kept for repeated plane reads (bytes).
pub const CHUNK_CACHE_BYTES: usize = 1 << 30;

/// One resolution level.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelInfo {
    /// The array's path in the container (`s0`, `0`, …), or `memory/<n>`.
    pub path: String,
    pub width: usize,
    pub height: usize,
    /// Tile size (the chunk size along y and x).
    pub tile_w: usize,
    pub tile_h: usize,
    /// Level-0 pixels per pixel of this level (1 for level 0, about 2 for the next, …).
    pub downsample: f64,
    /// Planes along z (1 for a 2D image).
    pub depth: usize,
    /// Chunk size along z.
    pub tile_d: usize,
    /// Level-0 planes per plane of this level.
    pub z_downsample: f64,
}

impl LevelInfo {
    pub fn tiles_x(&self) -> usize {
        self.width.div_ceil(self.tile_w.max(1))
    }

    pub fn tiles_y(&self) -> usize {
        self.height.div_ceil(self.tile_h.max(1))
    }
}

/// One channel and how the file suggests to show it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelInfo {
    pub label: String,
    /// Display colour (RGB, 0–255).
    pub color: [u8; 3],
    /// Suggested display window (raw values).
    pub window: (f32, f32),
    /// The range the values can take (raw values).
    pub range: (f32, f32),
    /// Shown when the image opens (OME-Zarr `omero` `active`; on when the file does not say): a
    /// highly multiplexed image opens with a few markers, not all of them added together.
    pub active: bool,
}

/// An axis a plane is cut across.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneAxis {
    /// An XY plane, at a position along z.
    Z,
    /// An XZ plane, at a position along y.
    Y,
    /// A YZ plane, at a position along x.
    X,
}

/// A plane of one channel: `width × height` raw values, row-major.
#[derive(Debug, Clone)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

/// One decoded tile of one channel: `width × height` raw values, row-major.
#[derive(Debug, Clone)]
pub struct Tile {
    pub level: usize,
    pub channel: usize,
    pub tx: usize,
    pub ty: usize,
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

/// A multiscale image opened for tile reads.
#[derive(Debug)]
pub struct Pyramid {
    pub name: String,
    /// The file or container on disk (`None` for generated images).
    pub path: Option<PathBuf>,
    /// Finest first.
    pub levels: Vec<LevelInfo>,
    pub channels: Vec<ChannelInfo>,
    /// Size of a level-0 pixel, if the metadata says (µm).
    pub pixel_size_um: Option<f64>,
    /// Distance between level-0 planes, if the metadata says (µm).
    pub z_spacing_um: Option<f64>,
    /// Stored element type (`uint16`, `float32`, …).
    pub dtype: String,
    /// Label images in the container (`labels/<name>`).
    pub label_groups: Vec<String>,
    backend: Backend,
    chunks: ChunkCache,
}

/// Decoded chunks by `(level, chunk index)`, least recently used dropped first.
#[derive(Debug, Default)]
struct ChunkCache(Mutex<ChunkCacheState>);

#[derive(Debug, Default)]
struct ChunkCacheState {
    map: HashMap<(usize, Vec<usize>), CachedChunk>,
    bytes: usize,
    clock: u64,
}

impl ChunkCache {
    fn get(&self, key: &(usize, Vec<usize>)) -> Option<Arc<Vec<f32>>> {
        let mut c = self.0.lock().unwrap_or_else(|e| e.into_inner());
        c.clock += 1;
        let clock = c.clock;
        c.map.get_mut(key).map(|e| {
            e.1 = clock;
            e.0.clone()
        })
    }

    fn put(&self, key: (usize, Vec<usize>), data: Arc<Vec<f32>>) {
        let mut c = self.0.lock().unwrap_or_else(|e| e.into_inner());
        c.clock += 1;
        let clock = c.clock;
        c.bytes += data.len() * 4;
        if let Some(old) = c.map.insert(key, (data, clock)) {
            c.bytes -= old.0.len() * 4;
        }
        while c.bytes > CHUNK_CACHE_BYTES && c.map.len() > 1 {
            let Some(oldest) = c.map.iter().min_by_key(|(_, e)| e.1).map(|(k, _)| k.clone()) else { break };
            if let Some(e) = c.map.remove(&oldest) {
                c.bytes -= e.0.len() * 4;
            }
        }
    }
}

#[derive(Debug)]
enum Backend {
    Zarr(Vec<ZarrArray>),
    /// `planes[level][channel]`: `depth × height × width` values.
    Memory(Vec<Vec<Vec<f32>>>),
}

/// One Zarr v2 array (a level).
#[derive(Debug, Clone)]
struct ZarrArray {
    dir: PathBuf,
    shape: Vec<usize>,
    chunks: Vec<usize>,
    dtype: Dtype,
    compressor: Compressor,
    sep: String,
    fill: f32,
    /// Index of the channel, z, y and x axes in `shape`.
    c_axis: Option<usize>,
    z_axis: Option<usize>,
    y_axis: usize,
    x_axis: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Compressor {
    None,
    Blosc,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Dtype {
    kind: char,
    size: usize,
    little: bool,
}

impl Dtype {
    fn parse(s: &str) -> IoResult<Self> {
        let b = s.as_bytes();
        if b.len() < 3 {
            return Err(IoError::OmeZarr(format!("unsupported dtype {s:?}")));
        }
        let little = b[0] != b'>';
        let kind = b[1] as char;
        let size: usize = s[2..].parse().map_err(|_| IoError::OmeZarr(format!("unsupported dtype {s:?}")))?;
        match (kind, size) {
            ('u', 1 | 2 | 4) | ('i', 1 | 2 | 4) | ('f', 4 | 8) | ('b', 1) => Ok(Self { kind, size, little }),
            _ => Err(IoError::OmeZarr(format!("unsupported dtype {s:?}"))),
        }
    }

    fn label(&self) -> String {
        match self.kind {
            'u' => format!("uint{}", self.size * 8),
            'i' => format!("int{}", self.size * 8),
            'f' => format!("float{}", self.size * 8),
            _ => "bool".into(),
        }
    }

    /// The values this type can hold.
    fn range(&self) -> (f32, f32) {
        match (self.kind, self.size) {
            ('u', 1) | ('b', _) => (0.0, 255.0),
            ('u', 2) => (0.0, 65535.0),
            ('u', 4) => (0.0, u32::MAX as f32),
            ('i', 1) => (-128.0, 127.0),
            ('i', 2) => (-32768.0, 32767.0),
            ('i', 4) => (i32::MIN as f32, i32::MAX as f32),
            _ => (0.0, 1.0),
        }
    }

    fn decode(&self, raw: &[u8], out: &mut Vec<f32>) {
        let n = raw.len() / self.size;
        out.clear();
        out.reserve(n);
        macro_rules! read {
            ($t:ty, $n:expr) => {
                for c in raw.chunks_exact($n) {
                    let mut a = [0u8; $n];
                    a.copy_from_slice(c);
                    let v = if self.little { <$t>::from_le_bytes(a) } else { <$t>::from_be_bytes(a) };
                    out.push(v as f32);
                }
            };
        }
        match (self.kind, self.size) {
            ('u', 1) | ('b', 1) => out.extend(raw.iter().map(|&b| b as f32)),
            ('i', 1) => out.extend(raw.iter().map(|&b| b as i8 as f32)),
            ('u', 2) => read!(u16, 2),
            ('i', 2) => read!(i16, 2),
            ('u', 4) => read!(u32, 4),
            ('i', 4) => read!(i32, 4),
            ('f', 4) => read!(f32, 4),
            ('f', 8) => read!(f64, 8),
            _ => {}
        }
    }
}

#[derive(Debug, Deserialize)]
struct ZArray {
    shape: Vec<usize>,
    chunks: Vec<usize>,
    dtype: String,
    #[serde(default)]
    dimension_separator: Option<String>,
    #[serde(default)]
    compressor: Option<ZCompressor>,
    #[serde(default)]
    fill_value: Option<serde_json::Value>,
    #[serde(default)]
    order: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ZCompressor {
    id: String,
    #[serde(default)]
    cname: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RootAttrs {
    #[serde(default)]
    multiscales: Vec<Multiscale>,
    #[serde(default)]
    omero: Option<Omero>,
}

#[derive(Debug, Deserialize, Default)]
struct Multiscale {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    axes: Vec<serde_json::Value>,
    #[serde(default)]
    datasets: Vec<Dataset>,
}

#[derive(Debug, Deserialize)]
struct Dataset {
    path: String,
    #[serde(default, rename = "coordinateTransformations")]
    transforms: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
struct Omero {
    #[serde(default)]
    channels: Vec<OmeroChannel>,
}

#[derive(Debug, Deserialize, Default)]
struct OmeroChannel {
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    color: Option<String>,
    #[serde(default)]
    window: Option<OmeroWindow>,
    #[serde(default)]
    active: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
struct OmeroWindow {
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    end: Option<f64>,
    #[serde(default)]
    min: Option<f64>,
    #[serde(default)]
    max: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
struct LabelsAttrs {
    #[serde(default)]
    labels: Vec<String>,
}

/// Default colours of channels without one (fluorescence first: blue, green, red, …).
const DEFAULT_COLORS: [[u8; 3]; 7] = [[64, 128, 255], [64, 255, 96], [255, 64, 64], [255, 0, 255], [0, 255, 255], [255, 255, 0], [255, 255, 255]];

impl Pyramid {
    /// Opens an OME-Zarr container (or any path inside one), or an image file.
    pub fn open(path: &Path) -> IoResult<Self> {
        if let Some(root) = find_ome_zarr_root(path) {
            return Self::open_zarr(&root);
        }
        if path.is_dir() && path.join(".zgroup").is_file() {
            return Self::open_zarr(path);
        }
        if let Some(stack) = read_tiff_stack(path)? {
            return Ok(stack);
        }
        let img = load_image_file(path)?;
        Ok(Self::from_loaded(img))
    }

    /// Reads the container's metadata (no pixel data).
    pub fn open_zarr(root: &Path) -> IoResult<Self> {
        let attrs: RootAttrs = match fs::read_to_string(root.join(".zattrs")) {
            Ok(text) => serde_json::from_str(&text)?,
            Err(_) => RootAttrs::default(),
        };
        let ms = attrs.multiscales.into_iter().next().unwrap_or_default();
        let axis_names: Vec<String> = ms
            .axes
            .iter()
            .map(|a| match a {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(o) => o.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string(),
                _ => String::new(),
            })
            .collect();
        let unit = ms.axes.iter().rev().find_map(|a| a.get("unit").and_then(|u| u.as_str()).map(str::to_string));
        let mut datasets: Vec<Dataset> = ms.datasets;
        if datasets.len() > MAX_LEVELS {
            return Err(IoError::OmeZarr(format!("{} resolution levels declared (at most {MAX_LEVELS})", datasets.len())));
        }
        if let Some(bad) = datasets.iter().find(|d| !is_safe_relative_path(&d.path)) {
            return Err(IoError::OmeZarr(format!("level path {:?} leaves the container", bad.path)));
        }
        if datasets.is_empty() {
            for cand in ["0", "1", "2", "3", "4", "5", "s0", "s1", "s2", "s3", "s4", "s5"] {
                if root.join(cand).join(".zarray").is_file() {
                    datasets.push(Dataset { path: cand.into(), transforms: Vec::new() });
                }
            }
        }
        if datasets.is_empty() {
            return Err(IoError::OmeZarr(format!("no multiscale levels in {}", root.display())));
        }

        let mut arrays = Vec::new();
        let mut levels = Vec::new();
        let mut pixel_size_um = None;
        let mut z_spacing_um = None;
        for (i, ds) in datasets.iter().enumerate() {
            let dir = root.join(&ds.path);
            let arr = ZarrArray::open(&dir, &axis_names)?;
            let (h, w) = (arr.shape[arr.y_axis], arr.shape[arr.x_axis]);
            let (th, tw) = (arr.chunks[arr.y_axis], arr.chunks[arr.x_axis]);
            if i == 0 {
                // Level 0's scale along x is the pixel size
                pixel_size_um = scale_of(&ds.transforms).and_then(|s| s.last().copied()).filter(|v| *v > 0.0).map(|v| to_micrometres(v, unit.as_deref()));
            }
            let (d, td) = arr.z_axis.map_or((1, 1), |z| (arr.shape[z], arr.chunks[z]));
            if i == 0 {
                z_spacing_um = arr.z_axis.and_then(|z| scale_of(&ds.transforms).and_then(|s| s.get(z).copied())).filter(|v| *v > 0.0).map(|v| to_micrometres(v, unit.as_deref()));
            }
            levels.push(LevelInfo { path: ds.path.clone(), width: w, height: h, tile_w: tw, tile_h: th, downsample: 1.0, depth: d, tile_d: td, z_downsample: 1.0 });
            arrays.push(arr);
        }
        let (w0, d0) = (levels[0].width.max(1) as f64, levels[0].depth.max(1) as f64);
        for l in &mut levels {
            l.downsample = w0 / l.width.max(1) as f64;
            l.z_downsample = d0 / l.depth.max(1) as f64;
        }
        // Finest first, whatever order the metadata lists them in
        let mut order: Vec<usize> = (0..levels.len()).collect();
        order.sort_by(|&a, &b| levels[b].width.cmp(&levels[a].width));
        let levels: Vec<LevelInfo> = order.iter().map(|&i| levels[i].clone()).collect();
        let arrays: Vec<ZarrArray> = order.iter().map(|&i| arrays[i].clone()).collect();

        let a0 = &arrays[0];
        let n_channels = a0.c_axis.map_or(1, |c| a0.shape[c]);
        let range = a0.dtype.range();
        let omero = attrs.omero.unwrap_or_default();
        let channels = (0..n_channels)
            .map(|c| {
                let o = omero.channels.get(c);
                let label = o.and_then(|o| o.label.clone()).filter(|l| !l.is_empty()).unwrap_or_else(|| format!("Channel {c}"));
                let color = o.and_then(|o| o.color.as_deref()).and_then(parse_hex).unwrap_or(if n_channels == 1 { [255, 255, 255] } else { DEFAULT_COLORS[c % DEFAULT_COLORS.len()] });
                let win = o.and_then(|o| o.window.as_ref());
                let window = match win {
                    Some(w) => (w.start.unwrap_or(range.0 as f64) as f32, w.end.unwrap_or(range.1 as f64) as f32),
                    None => range,
                };
                let range = match win {
                    Some(w) => (w.min.unwrap_or(range.0 as f64) as f32, w.max.unwrap_or(range.1 as f64) as f32),
                    None => range,
                };
                let active = o.and_then(|o| o.active).unwrap_or(true);
                ChannelInfo { label, color, window, range, active }
            })
            .collect();

        let mut label_groups = Vec::new();
        if let Ok(text) = fs::read_to_string(root.join("labels").join(".zattrs"))
            && let Ok(l) = serde_json::from_str::<LabelsAttrs>(&text) {
                label_groups = l.labels.into_iter().filter(|g| is_safe_relative_path(g) && root.join("labels").join(g).is_dir()).collect();
            }
        let name = ms.name.filter(|n| !n.is_empty()).unwrap_or_else(|| root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into()));
        Ok(Self { name, path: Some(root.to_path_buf()), levels, channels, pixel_size_um, z_spacing_um, dtype: a0.dtype.label(), label_groups, backend: Backend::Zarr(arrays), chunks: ChunkCache::default() })
    }

    /// An in-memory pyramid of a decoded image: one channel for grey images, three (R, G, B)
    /// for colour ones; each level halves the previous until it fits one tile.
    pub fn from_loaded(img: LoadedImage) -> Self {
        let (w, h) = (img.width, img.height);
        let n = w * h;
        let colour = img.channels >= 3 && img.rgba_f32.len() >= n * 4;
        let base: Vec<Vec<f32>> = if colour {
            (0..3).map(|c| (0..n).map(|i| img.rgba_f32[i * 4 + c]).collect()).collect()
        } else {
            vec![img.luminance_f32.clone()]
        };
        let channels = if colour {
            vec![
                ChannelInfo { label: "Red".into(), color: [255, 0, 0], window: (0.0, 1.0), range: (0.0, 1.0), active: true },
                ChannelInfo { label: "Green".into(), color: [0, 255, 0], window: (0.0, 1.0), range: (0.0, 1.0), active: true },
                ChannelInfo { label: "Blue".into(), color: [0, 0, 255], window: (0.0, 1.0), range: (0.0, 1.0), active: true },
            ]
        } else {
            vec![ChannelInfo { label: "Intensity".into(), color: [255, 255, 255], window: (0.0, 1.0), range: (0.0, 1.0), active: true }]
        };
        let mut levels = vec![LevelInfo { path: "memory/0".into(), width: w, height: h, tile_w: MEMORY_TILE, tile_h: MEMORY_TILE, downsample: 1.0, depth: 1, tile_d: 1, z_downsample: 1.0 }];
        let mut planes = vec![base];
        while {
            let l = levels.last().expect("one level");
            l.width.max(l.height) > MEMORY_TILE && l.width > 1 && l.height > 1
        } {
            let prev = levels.last().expect("one level").clone();
            let (nw, nh) = (prev.width.div_ceil(2), prev.height.div_ceil(2));
            let down: Vec<Vec<f32>> = planes.last().expect("one level").iter().map(|p| downsample2(p, prev.width, prev.height)).collect();
            levels.push(LevelInfo { path: format!("memory/{}", levels.len()), width: nw, height: nh, tile_w: MEMORY_TILE, tile_h: MEMORY_TILE, downsample: w as f64 / nw as f64, depth: 1, tile_d: 1, z_downsample: 1.0 });
            planes.push(down);
        }
        Self {
            name: img.name,
            path: img.path,
            levels,
            channels,
            pixel_size_um: None,
            z_spacing_um: None,
            dtype: img.dtype_label,
            label_groups: Vec::new(),
            backend: Backend::Memory(planes),
            chunks: ChunkCache::default(),
        }
    }

    /// An in-memory pyramid of a volume: each element of `channels` holds `depth × height × width`
    /// values. Levels halve x and y, and z too while the voxels stay about cubic.
    pub fn from_volume(name: impl Into<String>, channel_info: Vec<ChannelInfo>, channels: Vec<Vec<f32>>, [d, h, w]: Zyx, pixel_size_um: Option<f64>, z_spacing_um: Option<f64>, dtype: impl Into<String>) -> IoResult<Self> {
        let n = d.checked_mul(h).and_then(|v| v.checked_mul(w)).filter(|n| *n > 0 && *n <= MAX_READ_ELEMENTS).ok_or(IoError::InvalidDimensions { width: w, height: h, reason: "volume is empty or too large" })?;
        if channels.is_empty() || channels.len() != channel_info.len() || channels.iter().any(|c| c.len() != n) {
            return Err(IoError::InvalidDimensions { width: w, height: h, reason: "each channel must hold depth × height × width values" });
        }
        let mut levels = vec![LevelInfo { path: "memory/0".into(), width: w, height: h, tile_w: MEMORY_TILE, tile_h: MEMORY_TILE, downsample: 1.0, depth: d, tile_d: 1, z_downsample: 1.0 }];
        let mut planes = vec![channels];
        let mut ratio = match (pixel_size_um, z_spacing_um) {
            (Some(p), Some(z)) if p > 0.0 => z / p,
            _ => 1.0,
        };
        loop {
            let prev = levels.last().expect("one level").clone();
            if prev.width.max(prev.height).max(prev.depth) <= VOLUME_COARSEST || (prev.width <= 1 && prev.height <= 1 && prev.depth <= 1) {
                break;
            }
            let halve_z = prev.depth > 1 && ratio <= 1.5;
            let down: Vec<Vec<f32>> = planes.last().expect("one level").iter().map(|p| downsample_volume(p, [prev.depth, prev.height, prev.width], halve_z)).collect();
            let (nw, nh, nd) = (prev.width.div_ceil(2), prev.height.div_ceil(2), if halve_z { prev.depth.div_ceil(2) } else { prev.depth });
            ratio = if halve_z { ratio } else { ratio / 2.0 };
            levels.push(LevelInfo { path: format!("memory/{}", levels.len()), width: nw, height: nh, tile_w: MEMORY_TILE, tile_h: MEMORY_TILE, downsample: w as f64 / nw as f64, depth: nd, tile_d: 1, z_downsample: d as f64 / nd as f64 });
            planes.push(down);
        }
        Ok(Self { name: name.into(), path: None, levels, channels: channel_info, pixel_size_um, z_spacing_um, dtype: dtype.into(), label_groups: Vec::new(), backend: Backend::Memory(planes), chunks: ChunkCache::default() })
    }

    /// Planes along z at level 0 (1 for a 2D image).
    pub fn depth(&self) -> usize {
        self.levels[0].depth
    }

    /// Whether the image has more than one plane along z.
    pub fn is_volume(&self) -> bool {
        self.depth() > 1
    }

    pub fn width(&self) -> usize {
        self.levels[0].width
    }

    pub fn height(&self) -> usize {
        self.levels[0].height
    }

    /// The coarsest level.
    pub fn coarsest(&self) -> usize {
        self.levels.len() - 1
    }

    /// The level to draw at `zoom` screen px per level-0 px: the coarsest whose pixels are not
    /// larger than a screen pixel (sharp), never coarser than the last level.
    pub fn level_for_zoom(&self, zoom: f64) -> usize {
        let want = 1.0 / zoom.max(1e-9);
        let mut best = 0;
        for (i, l) in self.levels.iter().enumerate() {
            if l.downsample <= want * 1.001 {
                best = i;
            }
        }
        best
    }

    /// One tile of the first plane of one channel at `level`.
    pub fn read_tile(&self, level: usize, channel: usize, tx: usize, ty: usize) -> IoResult<Tile> {
        self.read_tile_at(level, channel, 0, tx, ty)
    }

    /// One tile of plane `z` of one channel at `level`; edge tiles are cut to the image.
    pub fn read_tile_at(&self, level: usize, channel: usize, z: usize, tx: usize, ty: usize) -> IoResult<Tile> {
        let l = self.levels.get(level).ok_or_else(|| IoError::OmeZarr(format!("no level {level}")))?;
        if z >= l.depth {
            return Err(IoError::OmeZarr(format!("plane {z} is outside level {level} ({} planes)", l.depth)));
        }
        let x0 = tx * l.tile_w;
        let y0 = ty * l.tile_h;
        if x0 >= l.width || y0 >= l.height {
            return Err(IoError::OmeZarr(format!("tile {tx},{ty} is outside level {level}")));
        }
        let (w, h) = ((l.width - x0).min(l.tile_w), (l.height - y0).min(l.tile_h));
        let data = match &self.backend {
            Backend::Memory(planes) => {
                let p = &planes[level][channel.min(planes[level].len() - 1)];
                let base = z * l.height * l.width;
                let mut out = Vec::with_capacity(w * h);
                for y in y0..y0 + h {
                    out.extend_from_slice(&p[base + y * l.width + x0..base + y * l.width + x0 + w]);
                }
                out
            }
            Backend::Zarr(arrays) => arrays[level].read_chunk(level, &self.chunks, channel, z, ty, tx, w, h)?,
        };
        Ok(Tile { level, channel, tx, ty, width: w, height: h, data })
    }

    /// A region of the first plane of one channel at `level`.
    pub fn read_region(&self, level: usize, channel: usize, x0: usize, y0: usize, w: usize, h: usize) -> IoResult<Vec<f32>> {
        self.read_region_at(level, channel, 0, x0, y0, w, h)
    }

    /// A region of plane `z` of one channel at `level` (`x0, y0, w, h` in that level's pixels,
    /// clipped to it), assembled from its tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn read_region_at(&self, level: usize, channel: usize, z: usize, x0: usize, y0: usize, w: usize, h: usize) -> IoResult<Vec<f32>> {
        let l = self.levels.get(level).ok_or_else(|| IoError::OmeZarr(format!("no level {level}")))?.clone();
        let (x1, y1) = ((x0 + w).min(l.width), (y0 + h).min(l.height));
        checked_read_len(1, h, w)?;
        let mut out = vec![0.0f32; w * h];
        if x0 >= x1 || y0 >= y1 {
            return Ok(out);
        }
        // The chunks under the region, read in parallel (each is its own file), then assembled
        let coords: Vec<(usize, usize)> = (y0 / l.tile_h..=(y1 - 1) / l.tile_h).flat_map(|ty| (x0 / l.tile_w..=(x1 - 1) / l.tile_w).map(move |tx| (tx, ty))).collect();
        let tiles: Vec<_> = {
            use rayon::prelude::*;
            coords.par_iter().map(|&(tx, ty)| self.read_tile_at(level, channel, z, tx, ty).map(|t| (tx, ty, t))).collect::<IoResult<Vec<_>>>()?
        };
        for (tx, ty, t) in tiles {
            {
                let (tx0, ty0) = (tx * l.tile_w, ty * l.tile_h);
                let (ax0, ax1) = (x0.max(tx0), x1.min(tx0 + t.width));
                let (ay0, ay1) = (y0.max(ty0), y1.min(ty0 + t.height));
                for y in ay0..ay1 {
                    let src = (y - ty0) * t.width + (ax0 - tx0);
                    let dst = (y - y0) * w + (ax0 - x0);
                    out[dst..dst + (ax1 - ax0)].copy_from_slice(&t.data[src..src + (ax1 - ax0)]);
                }
            }
        }
        Ok(out)
    }

    /// A box of one channel at `level`: `size` values from `origin` (both `[z, y, x]`, in that
    /// level's pixels), C-ordered. It must lie inside the level.
    pub fn read_block(&self, level: usize, channel: usize, origin: Zyx, size: Zyx) -> IoResult<Vec<f32>> {
        let l = self.levels.get(level).ok_or_else(|| IoError::OmeZarr(format!("no level {level}")))?;
        let dims = [l.depth, l.height, l.width];
        if (0..3).any(|a| origin[a].checked_add(size[a]).is_none_or(|end| end > dims[a])) {
            return Err(IoError::OmeZarr(format!("block {size:?} at {origin:?} is outside level {level} {dims:?}")));
        }
        let n = checked_read_len(size[0], size[1], size[2])?;
        let mut out = Vec::with_capacity(n);
        for z in origin[0]..origin[0] + size[0] {
            out.extend(self.read_region_at(level, channel, z, origin[2], origin[1], size[2], size[1])?);
        }
        Ok(out)
    }

    /// A plane of one channel at `level`, across `axis` at `index` (that level's pixels): an XY
    /// plane is `width × height`, an XZ plane `width × depth`, a YZ plane `height × depth`.
    pub fn read_plane(&self, level: usize, channel: usize, axis: PlaneAxis, index: usize) -> IoResult<Plane> {
        let l = self.levels.get(level).ok_or_else(|| IoError::OmeZarr(format!("no level {level}")))?.clone();
        let (d, h, w) = (l.depth, l.height, l.width);
        Ok(match axis {
            PlaneAxis::Z => Plane { width: w, height: h, data: self.read_block(level, channel, [index, 0, 0], [1, h, w])? },
            PlaneAxis::Y => Plane { width: w, height: d, data: self.read_block(level, channel, [0, index, 0], [d, 1, w])? },
            PlaneAxis::X => Plane { width: h, height: d, data: self.read_block(level, channel, [0, 0, index], [d, h, 1])? },
        })
    }

    /// Low and high percentiles of a channel at the coarsest level (an automatic window).
    pub fn auto_window(&self, channel: usize, low: f64, high: f64) -> IoResult<(f32, f32)> {
        let level = self.coarsest();
        let l = &self.levels[level];
        let mut values = self.read_block(level, channel, [0, 0, 0], [l.depth, l.height, l.width])?;
        values.retain(|v| v.is_finite());
        if values.is_empty() {
            return Ok((0.0, 1.0));
        }
        values.sort_by(|a, b| a.total_cmp(b));
        let at = |q: f64| values[((values.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize];
        let (lo, hi) = (at(low), at(high));
        Ok(if hi > lo { (lo, hi) } else { (lo, lo + 1.0) })
    }
}

impl ZarrArray {
    fn open(dir: &Path, axis_names: &[String]) -> IoResult<Self> {
        let text = fs::read_to_string(dir.join(".zarray")).map_err(|e| IoError::OmeZarr(format!("{}: {e}", dir.join(".zarray").display())))?;
        let meta: ZArray = serde_json::from_str(&text)?;
        if meta.order.as_deref() == Some("F") {
            return Err(IoError::OmeZarr("Fortran-ordered Zarr arrays are not supported".into()));
        }
        let dtype = Dtype::parse(&meta.dtype)?;
        let compressor = match meta.compressor.as_ref().map(|c| c.id.as_str()) {
            None => Compressor::None,
            Some("blosc") => {
                let cname = meta.compressor.as_ref().and_then(|c| c.cname.as_deref()).unwrap_or("lz4");
                if cname != "lz4" {
                    return Err(IoError::OmeZarr(format!("Blosc with {cname} is not supported yet (lz4 is)")));
                }
                Compressor::Blosc
            }
            Some(other) => return Err(IoError::OmeZarr(format!("Zarr compressor {other:?} is not supported yet (blosc-lz4 and none are)"))),
        };
        let nd = meta.shape.len();
        if !(2..=6).contains(&nd) || meta.chunks.len() != nd {
            return Err(IoError::OmeZarr(format!("unsupported Zarr shape {:?}", meta.shape)));
        }
        if meta.chunks.contains(&0) || meta.shape.contains(&0) {
            return Err(IoError::OmeZarr(format!("Zarr shape {:?} / chunks {:?} has an empty axis", meta.shape, meta.chunks)));
        }
        let elements = meta.shape.iter().try_fold(1u128, |a, &v| a.checked_mul(v as u128));
        if elements.is_none_or(|e| e > MAX_ARRAY_ELEMENTS) {
            return Err(IoError::OmeZarr(format!("Zarr shape {:?} is too large", meta.shape)));
        }
        let chunk_bytes = meta.chunks.iter().try_fold(dtype.size as u128, |a, &v| a.checked_mul(v as u128));
        if chunk_bytes.is_none_or(|b| b > MAX_CHUNK_BYTES) {
            return Err(IoError::OmeZarr(format!("Zarr chunks {:?} are too large", meta.chunks)));
        }
        let find = |name: &str| axis_names.iter().position(|a| a == name).filter(|&i| i < nd);
        let (y_axis, x_axis) = (find("y").unwrap_or(nd - 2), find("x").unwrap_or(nd - 1));
        let c_axis = find("c").or(if nd >= 3 && axis_names.is_empty() { Some(nd - 3) } else { None });
        let z_axis = find("z");
        if meta.dimension_separator.as_deref().is_some_and(|s| s != "." && s != "/") {
            return Err(IoError::OmeZarr(format!("unsupported dimension separator {:?}", meta.dimension_separator)));
        }
        let fill = meta.fill_value.as_ref().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let sep = meta.dimension_separator.unwrap_or_else(|| ".".into());
        Ok(Self { dir: dir.to_path_buf(), shape: meta.shape, chunks: meta.chunks, dtype, compressor, sep, fill, c_axis, z_axis, y_axis, x_axis })
    }

    /// Chunk `(ty, tx)` of `channel` at plane `z` (other axes at 0), cut to `w × h`.
    #[allow(clippy::too_many_arguments)]
    fn read_chunk(&self, level: usize, cache: &ChunkCache, channel: usize, z: usize, ty: usize, tx: usize, w: usize, h: usize) -> IoResult<Vec<f32>> {
        let nd = self.shape.len();
        let mut idx = vec![0usize; nd];
        idx[self.y_axis] = ty;
        idx[self.x_axis] = tx;
        let mut c_in_chunk = 0;
        if let Some(c) = self.c_axis {
            let ch = channel.min(self.shape[c].saturating_sub(1));
            idx[c] = ch / self.chunks[c];
            c_in_chunk = ch % self.chunks[c];
        }
        let mut z_in_chunk = 0;
        if let Some(a) = self.z_axis {
            idx[a] = z / self.chunks[a];
            z_in_chunk = z % self.chunks[a];
        }
        // Chunks holding several planes are read again for each plane: keep them decoded
        let multi_plane = self.z_axis.is_some_and(|a| self.chunks[a] > 1);
        let key = (level, idx);
        let values = match multi_plane.then(|| cache.get(&key)).flatten() {
            Some(v) => v,
            None => {
                let Some(v) = self.decode_chunk(&key.1)? else { return Ok(vec![self.fill; w * h]) };
                let v = Arc::new(v);
                if multi_plane {
                    cache.put(key, v.clone());
                }
                v
            }
        };
        // The chunk is a full block (edges padded): strides of the chunk's dims, C order
        let mut strides = vec![1usize; nd];
        for d in (0..nd - 1).rev() {
            strides[d] = strides[d + 1] * self.chunks[d + 1];
        }
        let base = self.c_axis.map_or(0, |c| c_in_chunk * strides[c]) + self.z_axis.map_or(0, |a| z_in_chunk * strides[a]);
        let (sy, sx) = (strides[self.y_axis], strides[self.x_axis]);
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                out.push(values.get(base + y * sy + x * sx).copied().unwrap_or(self.fill));
            }
        }
        Ok(out)
    }

    /// The values of a chunk; `None` when its file does not exist (it holds the fill value).
    fn decode_chunk(&self, idx: &[usize]) -> IoResult<Option<Vec<f32>>> {
        let name = idx.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(&self.sep);
        let raw = match fs::read(self.dir.join(name)) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let expected = self.chunks.iter().product::<usize>() * self.dtype.size;
        let bytes = match self.compressor {
            Compressor::None => raw,
            Compressor::Blosc => {
                let declared = raw.get(4..8).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
                if declared != Some(expected) {
                    return Err(IoError::OmeZarr(format!("chunk declares {declared:?} bytes, its array needs {expected}")));
                }
                blosc_decompress(&raw)?
            }
        };
        let mut values = Vec::new();
        self.dtype.decode(&bytes, &mut values);
        Ok(Some(values))
    }
}

/// Elements of a read, refused when it would be unreasonably large.
fn checked_read_len(d: usize, h: usize, w: usize) -> IoResult<usize> {
    d.checked_mul(h).and_then(|v| v.checked_mul(w)).filter(|n| *n <= MAX_READ_ELEMENTS).ok_or_else(|| IoError::OmeZarr(format!("a read of {d} × {h} × {w} is too large")))
}

/// A TIFF file with several same-sized grey pages, as one volume; `None` for anything else (the
/// usual image reader takes those).
fn read_tiff_stack(path: &Path) -> IoResult<Option<Pyramid>> {
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::ColorType;
    let is_tiff = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"));
    if !is_tiff || !path.is_file() {
        return Ok(None);
    }
    let Ok(mut decoder) = Decoder::new(std::io::BufReader::new(fs::File::open(path)?)) else { return Ok(None) };
    let Ok((w, h)) = decoder.dimensions() else { return Ok(None) };
    let (w, h) = (w as usize, h as usize);
    if !decoder.more_images() {
        return Ok(None);
    }
    let mut values = Vec::new();
    let mut depth = 0usize;
    let mut range = (0.0f32, 1.0f32);
    let mut dtype = "float32";
    loop {
        if !matches!(decoder.colortype(), Ok(ColorType::Gray(_))) || decoder.dimensions().ok() != Some((w as u32, h as u32)) {
            return Ok(None);
        }
        let page = match decoder.read_image() {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };
        let before = values.len();
        match page {
            DecodingResult::U8(v) => {
                range = (0.0, 255.0);
                dtype = "uint8";
                values.extend(v.into_iter().map(|x| x as f32));
            }
            DecodingResult::U16(v) => {
                range = (0.0, 65535.0);
                dtype = "uint16";
                values.extend(v.into_iter().map(|x| x as f32));
            }
            DecodingResult::F32(v) => values.extend(v),
            _ => return Ok(None),
        }
        if values.len() - before != w * h {
            return Ok(None);
        }
        depth += 1;
        checked_read_len(depth, h, w)?;
        if !decoder.more_images() {
            break;
        }
        if decoder.next_image().is_err() {
            return Ok(None);
        }
    }
    if depth < 2 {
        return Ok(None);
    }
    let name = path.file_name().map_or_else(|| "stack".to_string(), |n| n.to_string_lossy().into_owned());
    let info = vec![ChannelInfo { label: "Intensity".into(), color: [255, 255, 255], window: range, range, active: true }];
    let mut py = Pyramid::from_volume(name, info, vec![values], [depth, h, w], None, None, dtype)?;
    py.path = Some(path.to_path_buf());
    Ok(Some(py))
}

/// The `scale` transform of a dataset.
fn scale_of(transforms: &[serde_json::Value]) -> Option<Vec<f64>> {
    transforms.iter().find(|t| t.get("type").and_then(|t| t.as_str()) == Some("scale")).and_then(|t| t.get("scale")).and_then(|s| s.as_array()).map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
}

fn to_micrometres(v: f64, unit: Option<&str>) -> f64 {
    match unit {
        Some("nanometer") => v / 1000.0,
        Some("millimeter") => v * 1000.0,
        Some("centimeter") => v * 10_000.0,
        Some("meter") => v * 1_000_000.0,
        _ => v,
    }
}

fn parse_hex(s: &str) -> Option<[u8; 3]> {
    let s = s.trim_start_matches('#');
    if s.len() != 6 {
        return None;
    }
    let p = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).ok();
    Some([p(0)?, p(2)?, p(4)?])
}

/// 2 × 2 box average.
fn downsample2(p: &[f32], w: usize, h: usize) -> Vec<f32> {
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let (mut s, mut n) = (0.0f32, 0.0f32);
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let (yy, xx) = (2 * y + dy, 2 * x + dx);
                if yy < h && xx < w {
                    s += p[yy * w + xx];
                    n += 1.0;
                }
            }
            out[y * nw + x] = s / n.max(1.0);
        }
    }
    out
}

/// Halves x and y (and z when `halve_z`) with a box average.
fn downsample_volume(p: &[f32], [d, h, w]: Zyx, halve_z: bool) -> Vec<f32> {
    let nd = if halve_z { d.div_ceil(2) } else { d };
    let (nh, nw) = (h.div_ceil(2), w.div_ceil(2));
    let mut out = Vec::with_capacity(nd * nh * nw);
    let zs: &[usize] = if halve_z { &[0, 1] } else { &[0] };
    let zstep = if halve_z { 2 } else { 1 };
    for z in 0..nd {
        for y in 0..nh {
            for x in 0..nw {
                let (mut s, mut n) = (0.0f32, 0.0f32);
                for &dz in zs {
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let (zz, yy, xx) = (zstep * z + dz, 2 * y + dy, 2 * x + dx);
                        if zz < d && yy < h && xx < w {
                            s += p[(zz * h + yy) * w + xx];
                            n += 1.0;
                        }
                    }
                }
                out.push(s / n.max(1.0));
            }
        }
    }
    out
}

/// A synthetic volume for trying the 3D paths: bright spheres in one channel and a helical tube
/// in another, `depth` planes of `size × size` with voxels twice as deep as they are wide.
pub fn sample_volume(size: usize, depth: usize) -> IoResult<Pyramid> {
    let (w, h, d) = (size.max(8), size.max(8), depth.max(8));
    let n = checked_read_len(d, h, w)?;
    let mut blobs = vec![0.0f32; n];
    let mut tube = vec![0.0f32; n];
    let mut seed = 0x2545_F491u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) as f32 / (1u64 << 31) as f32
    };
    let centres: Vec<[f32; 4]> = (0..48).map(|_| [rnd() * d as f32, rnd() * h as f32, rnd() * w as f32, 3.0 + rnd() * size as f32 / 16.0]).collect();
    for z in 0..d {
        // Scaled so the spheres are round in micrometres (voxels are twice as deep as wide)
        let (zf, turn) = (z as f32, z as f32 / d as f32 * std::f32::consts::TAU * 2.0);
        let (ty, tx) = (h as f32 * (0.5 + 0.3 * turn.sin()), w as f32 * (0.5 + 0.3 * turn.cos()));
        for y in 0..h {
            for x in 0..w {
                let i = (z * h + y) * w + x;
                let (yf, xf) = (y as f32, x as f32);
                let dt = (yf - ty).powi(2) + (xf - tx).powi(2);
                tube[i] = (-dt / (2.0 * (size as f32 / 40.0).powi(2))).exp();
                let mut v = 0.0f32;
                for c in &centres {
                    let r2 = (2.0 * (zf - c[0])).powi(2) + (yf - c[1]).powi(2) + (xf - c[2]).powi(2);
                    v = v.max((-r2 / (2.0 * c[3] * c[3])).exp());
                }
                blobs[i] = v;
            }
        }
    }
    let info = vec![
        ChannelInfo { label: "Nuclei".into(), color: [64, 128, 255], window: (0.0, 1.0), range: (0.0, 1.0), active: true },
        ChannelInfo { label: "Filament".into(), color: [64, 255, 96], window: (0.0, 1.0), range: (0.0, 1.0), active: true },
    ];
    Pyramid::from_volume("Sample volume", info, vec![blobs, tube], [d, h, w], Some(1.0), Some(2.0), "float32")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Option<PathBuf> {
        let ws = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?.parent()?.to_path_buf();
        let p = ws.join(".data/TDTNI1_P4_M_SAL_S1_stack.ome.zarr");
        p.exists().then_some(p)
    }

    #[test]
    fn opens_the_sample_container_tile_by_tile() {
        let Some(p) = sample() else { return };
        let py = Pyramid::open(&p).unwrap();
        assert_eq!(py.levels.len(), 4);
        assert_eq!((py.width(), py.height()), (4063, 3050));
        assert_eq!(py.channels.len(), 2);
        assert_eq!(py.channels[0].label, "DAPI_deblur");
        assert_eq!(py.channels[1].color, [255, 0, 0]);
        let t = py.read_tile(0, 1, 3, 2).unwrap();
        assert_eq!((t.width, t.height), (4063 - 3 * 1024, 3050 - 2 * 1024));
        assert!(py.pixel_size_um.is_some());
        assert_eq!(py.level_for_zoom(1.0), 0);
        assert_eq!(py.level_for_zoom(0.25), 2);
    }

    #[test]
    fn a_volume_reads_planes_along_every_axis() {
        let (d, h, w) = (6usize, 5usize, 4usize);
        let values: Vec<f32> = (0..d * h * w).map(|i| i as f32).collect();
        let info = vec![ChannelInfo { label: "a".into(), color: [255, 255, 255], window: (0.0, 1.0), range: (0.0, 1.0), active: true }];
        let py = Pyramid::from_volume("v", info, vec![values], [d, h, w], None, None, "float32").unwrap();
        assert!(py.is_volume());
        let at = |z: usize, y: usize, x: usize| ((z * h + y) * w + x) as f32;
        assert_eq!(py.read_plane(0, 0, PlaneAxis::Z, 2).unwrap().data[7], at(2, 1, 3));
        let xz = py.read_plane(0, 0, PlaneAxis::Y, 3).unwrap();
        assert_eq!((xz.width, xz.height), (w, d));
        assert_eq!(xz.data[(4 * w) + 1], at(4, 3, 1));
        let yz = py.read_plane(0, 0, PlaneAxis::X, 1).unwrap();
        assert_eq!((yz.width, yz.height), (h, d));
        assert_eq!(yz.data[(5 * h) + 2], at(5, 2, 1));
        assert!(py.read_block(0, 0, [5, 0, 0], [2, 1, 1]).is_err(), "outside the level");
    }

    #[test]
    fn metadata_that_leaves_the_container_or_lies_about_size_is_refused() {
        let root = std::env::temp_dir().join(format!("lucem-bad-zarr-{}", std::process::id()));
        fs::create_dir_all(root.join("0")).unwrap();
        fs::write(root.join(".zgroup"), r#"{"zarr_format":2}"#).unwrap();
        fs::write(root.join(".zattrs"), r#"{"multiscales":[{"axes":["y","x"],"datasets":[{"path":"../../etc"}]}]}"#).unwrap();
        assert!(Pyramid::open_zarr(&root).unwrap_err().to_string().contains("leaves the container"));
        fs::write(root.join(".zattrs"), r#"{"multiscales":[{"axes":["y","x"],"datasets":[{"path":"0"}]}]}"#).unwrap();
        fs::write(root.join("0/.zarray"), r#"{"shape":[18446744073709551615,18446744073709551615],"chunks":[1,1],"dtype":"<f4"}"#).unwrap();
        assert!(Pyramid::open_zarr(&root).unwrap_err().to_string().contains("too large"));
        fs::write(root.join("0/.zarray"), r#"{"shape":[10,10],"chunks":[0,1],"dtype":"<f4"}"#).unwrap();
        assert!(Pyramid::open_zarr(&root).is_err(), "an empty chunk axis");
        fs::remove_dir_all(&root).unwrap();
    }

    /// `omero` `active: false` opens a channel hidden; a channel the file says nothing about is shown.
    #[test]
    fn channels_open_as_the_file_marks_them() {
        let root = std::env::temp_dir().join(format!("lucem-active-zarr-{}", std::process::id()));
        fs::create_dir_all(root.join("0")).unwrap();
        fs::write(root.join(".zgroup"), r#"{"zarr_format":2}"#).unwrap();
        fs::write(root.join(".zattrs"), r#"{"multiscales":[{"axes":["c","y","x"],"datasets":[{"path":"0"}]}],
            "omero":{"channels":[{"label":"DAPI","active":true},{"label":"CD3","active":false},{"label":"CD20"}]}}"#).unwrap();
        fs::write(root.join("0/.zarray"), r#"{"shape":[3,4,4],"chunks":[1,4,4],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","zarr_format":2}"#).unwrap();
        let py = Pyramid::open_zarr(&root).unwrap();
        let shown: Vec<(&str, bool)> = py.channels.iter().map(|c| (c.label.as_str(), c.active)).collect();
        assert_eq!(shown, [("DAPI", true), ("CD3", false), ("CD20", true)]);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn memory_pyramid_halves_until_one_tile() {
        let img = crate::phantoms::generate_phantom(crate::phantoms::PhantomKind::FluorescenceNuclei, 1500, 1100);
        let py = Pyramid::from_loaded(img);
        assert!(py.levels.last().unwrap().width.max(py.levels.last().unwrap().height) <= MEMORY_TILE);
        let r = py.read_region(0, 0, 500, 500, 600, 30).unwrap();
        assert_eq!(r.len(), 600 * 30);
    }
}

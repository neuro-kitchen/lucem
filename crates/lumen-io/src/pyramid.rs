//! Multiscale images read tile by tile: an OME-Zarr container (v0.4, Zarr v2) or a plain image
//! file (PNG, TIFF, JPEG, …) turned into an in-memory pyramid.
//!
//! Nothing is decoded until a tile is asked for: [`Pyramid::read_tile`] reads one chunk of one
//! channel at one level and returns its raw values as `f32` (not normalized; the display decides
//! the window). Levels, channel names, colours and display windows, and the pixel size come from
//! the metadata (`multiscales`, `omero`), whatever the levels are called.
//!
//! The reader is `Sync`: a pool of threads reads tiles of one pyramid at once.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{IoError, IoResult};
use crate::image_io::{load_image_file, LoadedImage};
use crate::ome_zarr::{blosc_decompress, find_ome_zarr_root};

/// Edge of the tiles of an in-memory pyramid (logical tile; Zarr levels use their chunk size).
pub const MEMORY_TILE: usize = 512;

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
    /// Stored element type (`uint16`, `float32`, …).
    pub dtype: String,
    /// Label images in the container (`labels/<name>`).
    pub label_groups: Vec<String>,
    backend: Backend,
}

#[derive(Debug)]
enum Backend {
    Zarr(Vec<ZarrArray>),
    /// `planes[level][channel]`: `width × height` values.
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
    /// Index of the channel, y and x axes in `shape`.
    c_axis: Option<usize>,
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
        for (i, ds) in datasets.iter().enumerate() {
            let dir = root.join(&ds.path);
            let arr = ZarrArray::open(&dir, &axis_names)?;
            let (h, w) = (arr.shape[arr.y_axis], arr.shape[arr.x_axis]);
            let (th, tw) = (arr.chunks[arr.y_axis], arr.chunks[arr.x_axis]);
            if i == 0 {
                // Level 0's scale along x is the pixel size
                pixel_size_um = scale_of(&ds.transforms).and_then(|s| s.last().copied()).filter(|v| *v > 0.0).map(|v| to_micrometres(v, unit.as_deref()));
            }
            levels.push(LevelInfo { path: ds.path.clone(), width: w, height: h, tile_w: tw, tile_h: th, downsample: 1.0 });
            arrays.push(arr);
        }
        let w0 = levels[0].width.max(1) as f64;
        for l in &mut levels {
            l.downsample = w0 / l.width.max(1) as f64;
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
                ChannelInfo { label, color, window, range }
            })
            .collect();

        let mut label_groups = Vec::new();
        if let Ok(text) = fs::read_to_string(root.join("labels").join(".zattrs")) {
            if let Ok(l) = serde_json::from_str::<LabelsAttrs>(&text) {
                label_groups = l.labels.into_iter().filter(|g| root.join("labels").join(g).is_dir()).collect();
            }
        }
        let name = ms.name.filter(|n| !n.is_empty()).unwrap_or_else(|| root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into()));
        Ok(Self { name, path: Some(root.to_path_buf()), levels, channels, pixel_size_um, dtype: a0.dtype.label(), label_groups, backend: Backend::Zarr(arrays) })
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
                ChannelInfo { label: "Red".into(), color: [255, 0, 0], window: (0.0, 1.0), range: (0.0, 1.0) },
                ChannelInfo { label: "Green".into(), color: [0, 255, 0], window: (0.0, 1.0), range: (0.0, 1.0) },
                ChannelInfo { label: "Blue".into(), color: [0, 0, 255], window: (0.0, 1.0), range: (0.0, 1.0) },
            ]
        } else {
            vec![ChannelInfo { label: "Intensity".into(), color: [255, 255, 255], window: (0.0, 1.0), range: (0.0, 1.0) }]
        };
        let mut levels = vec![LevelInfo { path: "memory/0".into(), width: w, height: h, tile_w: MEMORY_TILE, tile_h: MEMORY_TILE, downsample: 1.0 }];
        let mut planes = vec![base];
        while {
            let l = levels.last().expect("one level");
            l.width.max(l.height) > MEMORY_TILE && l.width > 1 && l.height > 1
        } {
            let prev = levels.last().expect("one level").clone();
            let (nw, nh) = (prev.width.div_ceil(2), prev.height.div_ceil(2));
            let down: Vec<Vec<f32>> = planes.last().expect("one level").iter().map(|p| downsample2(p, prev.width, prev.height)).collect();
            levels.push(LevelInfo { path: format!("memory/{}", levels.len()), width: nw, height: nh, tile_w: MEMORY_TILE, tile_h: MEMORY_TILE, downsample: w as f64 / nw as f64 });
            planes.push(down);
        }
        Self {
            name: img.name,
            path: img.path,
            levels,
            channels,
            pixel_size_um: None,
            dtype: img.dtype_label,
            label_groups: Vec::new(),
            backend: Backend::Memory(planes),
        }
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

    /// One tile of one channel at `level`; edge tiles are cut to the image.
    pub fn read_tile(&self, level: usize, channel: usize, tx: usize, ty: usize) -> IoResult<Tile> {
        let l = self.levels.get(level).ok_or_else(|| IoError::OmeZarr(format!("no level {level}")))?;
        let x0 = tx * l.tile_w;
        let y0 = ty * l.tile_h;
        if x0 >= l.width || y0 >= l.height {
            return Err(IoError::OmeZarr(format!("tile {tx},{ty} is outside level {level}")));
        }
        let (w, h) = ((l.width - x0).min(l.tile_w), (l.height - y0).min(l.tile_h));
        let data = match &self.backend {
            Backend::Memory(planes) => {
                let p = &planes[level][channel.min(planes[level].len() - 1)];
                let mut out = Vec::with_capacity(w * h);
                for y in y0..y0 + h {
                    out.extend_from_slice(&p[y * l.width + x0..y * l.width + x0 + w]);
                }
                out
            }
            Backend::Zarr(arrays) => arrays[level].read_chunk(channel, ty, tx, w, h)?,
        };
        Ok(Tile { level, channel, tx, ty, width: w, height: h, data })
    }

    /// A region of one channel at `level` (`x0, y0, w, h` in that level's pixels, clipped to
    /// it), assembled from its tiles.
    pub fn read_region(&self, level: usize, channel: usize, x0: usize, y0: usize, w: usize, h: usize) -> IoResult<Vec<f32>> {
        let l = self.levels.get(level).ok_or_else(|| IoError::OmeZarr(format!("no level {level}")))?.clone();
        let (x1, y1) = ((x0 + w).min(l.width), (y0 + h).min(l.height));
        let mut out = vec![0.0f32; w * h];
        if x0 >= x1 || y0 >= y1 {
            return Ok(out);
        }
        for ty in y0 / l.tile_h..=(y1 - 1) / l.tile_h {
            for tx in x0 / l.tile_w..=(x1 - 1) / l.tile_w {
                let t = self.read_tile(level, channel, tx, ty)?;
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

    /// Low and high percentiles of a channel at the coarsest level (an automatic window).
    pub fn auto_window(&self, channel: usize, low: f64, high: f64) -> IoResult<(f32, f32)> {
        let level = self.coarsest();
        let l = &self.levels[level];
        let mut values = self.read_region(level, channel, 0, 0, l.width, l.height)?;
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
        if nd < 2 || meta.chunks.len() != nd {
            return Err(IoError::OmeZarr(format!("unsupported Zarr shape {:?}", meta.shape)));
        }
        let find = |name: &str| axis_names.iter().position(|a| a == name).filter(|&i| i < nd);
        let (y_axis, x_axis) = (find("y").unwrap_or(nd - 2), find("x").unwrap_or(nd - 1));
        let c_axis = find("c").or(if nd >= 3 && axis_names.is_empty() { Some(nd - 3) } else { None });
        let fill = meta.fill_value.as_ref().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let sep = meta.dimension_separator.unwrap_or_else(|| ".".into());
        Ok(Self { dir: dir.to_path_buf(), shape: meta.shape, chunks: meta.chunks, dtype, compressor, sep, fill, c_axis, y_axis, x_axis })
    }

    /// Chunk `(ty, tx)` of `channel` (other axes at 0), cut to `w × h`.
    fn read_chunk(&self, channel: usize, ty: usize, tx: usize, w: usize, h: usize) -> IoResult<Vec<f32>> {
        let nd = self.shape.len();
        let mut idx = vec![0usize; nd];
        idx[self.y_axis] = ty;
        idx[self.x_axis] = tx;
        let mut c_in_chunk = 0;
        if let Some(c) = self.c_axis {
            let ch = channel.min(self.shape[c].saturating_sub(1));
            idx[c] = ch / self.chunks[c].max(1);
            c_in_chunk = ch % self.chunks[c].max(1);
        }
        let key = idx.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(&self.sep);
        let file = self.dir.join(key);
        let raw = match fs::read(&file) {
            Ok(b) => b,
            // A missing chunk holds the fill value
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![self.fill; w * h]),
            Err(e) => return Err(e.into()),
        };
        let bytes = match self.compressor {
            Compressor::None => raw,
            Compressor::Blosc => blosc_decompress(&raw)?,
        };
        let mut values = Vec::new();
        self.dtype.decode(&bytes, &mut values);
        // The chunk is a full block (edges padded): strides of the chunk's dims, C order
        let mut strides = vec![1usize; nd];
        for d in (0..nd - 1).rev() {
            strides[d] = strides[d + 1] * self.chunks[d + 1];
        }
        let base = self.c_axis.map_or(0, |c| c_in_chunk * strides[c]);
        let (sy, sx) = (strides[self.y_axis], strides[self.x_axis]);
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                out.push(values.get(base + y * sy + x * sx).copied().unwrap_or(self.fill));
            }
        }
        Ok(out)
    }
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
    fn memory_pyramid_halves_until_one_tile() {
        let img = crate::phantoms::generate_phantom(crate::phantoms::PhantomKind::FluorescenceNuclei, 1500, 1100);
        let py = Pyramid::from_loaded(img);
        assert_eq!(py.levels.last().unwrap().width.max(py.levels.last().unwrap().height) <= MEMORY_TILE, true);
        let r = py.read_region(0, 0, 500, 500, 600, 30).unwrap();
        assert_eq!(r.len(), 600 * 30);
    }
}

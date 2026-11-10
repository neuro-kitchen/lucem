//! Writing an OME-Zarr image block by block, without holding it in memory.
//!
//! [`ZarrWriter::create`] starts a store (it never replaces an existing path);
//! [`ZarrWriter::write`] stores one chunk of level 0 at a time; [`ZarrWriter::finish`] builds the
//! coarser levels from the chunks on disk and writes the metadata. The result is OME-Zarr v0.4
//! (Zarr v2, uncompressed chunks), which [`crate::Pyramid`] and other tools read.
//!
//! Axes are `c, y, x` for one plane and `c, z, y, x` for a volume.

use std::fs;
use std::path::{Path, PathBuf};

use lumen_core::Zyx;
use serde_json::json;

use crate::error::{IoError, IoResult};

/// What a store holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleType {
    /// Intensities, averaged for coarser levels.
    F32,
    /// Masks and labels, sampled for coarser levels (an average of ids means nothing).
    U32,
}

impl SampleType {
    fn dtype(self) -> &'static str {
        match self {
            SampleType::F32 => "<f4",
            SampleType::U32 => "<u4",
        }
    }
}

/// An element a store can hold.
pub trait Sample: Copy + Default + 'static {
    const TYPE: SampleType;
    fn to_bytes(values: &[Self], out: &mut Vec<u8>);
    fn from_bytes(bytes: &[u8]) -> Vec<Self>;
    /// The value of a coarser pixel from the pixels it covers (never empty).
    fn reduce(values: &[Self]) -> Self;
}

impl Sample for f32 {
    const TYPE: SampleType = SampleType::F32;
    fn to_bytes(values: &[Self], out: &mut Vec<u8>) {
        out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    }
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    fn reduce(values: &[Self]) -> Self {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

impl Sample for u32 {
    const TYPE: SampleType = SampleType::U32;
    fn to_bytes(values: &[Self], out: &mut Vec<u8>) {
        out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    }
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    fn reduce(values: &[Self]) -> Self {
        values[0]
    }
}

/// What to create.
#[derive(Debug, Clone)]
pub struct WriterOptions {
    pub name: String,
    pub channels: Vec<String>,
    /// `[depth, height, width]`; depth 1 is a 2D image.
    pub shape: Zyx,
    /// Chunk size, `[z, y, x]`.
    pub chunks: Zyx,
    pub pixel_size_um: Option<f64>,
    pub z_spacing_um: Option<f64>,
}

#[derive(Debug, Clone)]
struct Level {
    shape: Zyx,
    scale: [f64; 3],
}

/// A store being written.
#[derive(Debug)]
pub struct ZarrWriter {
    root: PathBuf,
    options: WriterOptions,
    sample: SampleType,
}

impl ZarrWriter {
    /// Starts a store for samples of type `T` at `root`, which must not exist yet.
    pub fn create<T: Sample>(root: impl AsRef<Path>, options: WriterOptions) -> IoResult<Self> {
        let root = root.as_ref().to_path_buf();
        if root.exists() {
            return Err(IoError::OmeZarr(format!("{} already exists", root.display())));
        }
        if options.channels.is_empty() || options.shape.contains(&0) || options.chunks.contains(&0) {
            return Err(IoError::OmeZarr("a store needs at least one channel and non-empty shape and chunks".into()));
        }
        if options.shape.iter().chain(options.chunks.iter()).try_fold(1u128, |a, &v| a.checked_mul(v as u128)).is_none() {
            return Err(IoError::OmeZarr("the store's shape is too large".into()));
        }
        fs::create_dir_all(root.join("0"))?;
        fs::write(root.join(".zgroup"), "{\n  \"zarr_format\": 2\n}\n")?;
        Ok(Self { root, options, sample: T::TYPE })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The chunk grid position of `origin`.
    fn chunk_of(&self, origin: Zyx) -> IoResult<Zyx> {
        let c = self.options.chunks;
        if (0..3).any(|a| !origin[a].is_multiple_of(c[a]) || origin[a] >= self.options.shape[a]) {
            return Err(IoError::OmeZarr(format!("block at {origin:?} is not on the chunk grid {c:?} of shape {:?}", self.options.shape)));
        }
        Ok([origin[0] / c[0], origin[1] / c[1], origin[2] / c[2]])
    }

    /// Stores the block of `size` at `origin` of `channel` as one chunk of level 0. `origin` must
    /// be on the chunk grid and `size` a whole chunk, or what remains of one at the image's edge.
    pub fn write<T: Sample>(&self, channel: usize, origin: Zyx, size: Zyx, data: &[T]) -> IoResult<()> {
        self.check_type::<T>()?;
        if channel >= self.options.channels.len() {
            return Err(IoError::OmeZarr(format!("no channel {channel}")));
        }
        let index = self.chunk_of(origin)?;
        let (c, s) = (self.options.chunks, self.options.shape);
        let whole = (0..3).all(|a| size[a] == c[a].min(s[a] - origin[a]));
        if !whole || data.len() != size[0] * size[1] * size[2] {
            return Err(IoError::OmeZarr(format!("block of {size:?} with {} values at {origin:?} is not a whole chunk of {c:?}", data.len())));
        }
        write_chunk(&self.chunk_path(0, channel, index), data, size, c)
    }

    fn check_type<T: Sample>(&self) -> IoResult<()> {
        if T::TYPE == self.sample { Ok(()) } else { Err(IoError::OmeZarr("this store holds another element type".into())) }
    }

    fn chunk_path(&self, level: usize, channel: usize, [z, y, x]: Zyx) -> PathBuf {
        let dir = self.root.join(level.to_string()).join(channel.to_string());
        if self.options.shape[0] > 1 { dir.join(z.to_string()).join(y.to_string()).join(x.to_string()) } else { dir.join(y.to_string()).join(x.to_string()) }
    }

    /// Replaces every label of level 0 by `map[label]` (labels past its end become 0). Used once
    /// objects cut by block borders have been joined.
    pub fn remap_labels(&self, map: &[u32]) -> IoResult<()> {
        if self.sample != SampleType::U32 {
            return Err(IoError::OmeZarr("only a store of labels can be remapped".into()));
        }
        let (c, s) = (self.options.chunks, self.options.shape);
        let total: usize = c.iter().product();
        for channel in 0..self.options.channels.len() {
            for z in 0..s[0].div_ceil(c[0]) {
                for y in 0..s[1].div_ceil(c[1]) {
                    for x in 0..s[2].div_ceil(c[2]) {
                        let path = self.chunk_path(0, channel, [z, y, x]);
                        if !path.is_file() {
                            continue;
                        }
                        let mut values = u32::from_bytes(&fs::read(&path)?);
                        if values.len() != total {
                            return Err(IoError::OmeZarr(format!("{} has the wrong size", path.display())));
                        }
                        values.iter_mut().for_each(|v| *v = map.get(*v as usize).copied().unwrap_or(0));
                        write_raw(&path, &values)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Builds the coarser levels, writes the metadata and returns the store's path.
    pub fn finish<T: Sample>(self) -> IoResult<PathBuf> {
        self.check_type::<T>()?;
        let mut levels = vec![Level { shape: self.options.shape, scale: [1.0; 3] }];
        let (px, dz) = (self.options.pixel_size_um.unwrap_or(1.0), self.options.z_spacing_um.unwrap_or(self.options.pixel_size_um.unwrap_or(1.0)));
        let largest_chunk = self.options.chunks[1].max(self.options.chunks[2]);
        while {
            let l = levels.last().expect("one level");
            l.shape[1].max(l.shape[2]) > largest_chunk && levels.len() < 12
        } {
            let prev = levels.last().expect("one level").clone();
            let halve_z = prev.shape[0] > 1 && (dz * prev.scale[0]) <= 1.5 * (px * prev.scale[1]);
            let next = Level {
                shape: [if halve_z { prev.shape[0].div_ceil(2) } else { prev.shape[0] }, prev.shape[1].div_ceil(2), prev.shape[2].div_ceil(2)],
                scale: [prev.scale[0] * if halve_z { 2.0 } else { 1.0 }, prev.scale[1] * 2.0, prev.scale[2] * 2.0],
            };
            self.build_level::<T>(levels.len(), &prev, &next, halve_z)?;
            levels.push(next);
        }
        for (i, l) in levels.iter().enumerate() {
            self.write_zarray(i, l)?;
        }
        self.write_attrs(&levels, px, dz)?;
        Ok(self.root)
    }

    fn build_level<T: Sample>(&self, index: usize, prev: &Level, next: &Level, halve_z: bool) -> IoResult<()> {
        let c = self.options.chunks;
        let zf = if halve_z { 2 } else { 1 };
        for channel in 0..self.options.channels.len() {
            for cz in 0..next.shape[0].div_ceil(c[0]) {
                for cy in 0..next.shape[1].div_ceil(c[1]) {
                    for cx in 0..next.shape[2].div_ceil(c[2]) {
                        let origin = [cz * c[0], cy * c[1], cx * c[2]];
                        let size = [c[0].min(next.shape[0] - origin[0]), c[1].min(next.shape[1] - origin[1]), c[2].min(next.shape[2] - origin[2])];
                        let src_origin = [origin[0] * zf, origin[1] * 2, origin[2] * 2];
                        let src_size = [(size[0] * zf).min(prev.shape[0] - src_origin[0]), (size[1] * 2).min(prev.shape[1] - src_origin[1]), (size[2] * 2).min(prev.shape[2] - src_origin[2])];
                        let src = self.read_box::<T>(index - 1, channel, prev, src_origin, src_size)?;
                        let reduced = reduce::<T>(&src, src_size, size, zf);
                        write_chunk(&self.chunk_path(index, channel, [cz, cy, cx]), &reduced, size, c)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// A box of a level already on disk (missing chunks read as zeros).
    fn read_box<T: Sample>(&self, level: usize, channel: usize, l: &Level, origin: Zyx, size: Zyx) -> IoResult<Vec<T>> {
        let c = self.options.chunks;
        let mut out = vec![T::default(); size[0] * size[1] * size[2]];
        for cz in origin[0] / c[0]..=(origin[0] + size[0] - 1) / c[0] {
            for cy in origin[1] / c[1]..=(origin[1] + size[1] - 1) / c[1] {
                for cx in origin[2] / c[2]..=(origin[2] + size[2] - 1) / c[2] {
                    let path = self.chunk_path(level, channel, [cz, cy, cx]);
                    let Ok(bytes) = fs::read(&path) else { continue };
                    let values = T::from_bytes(&bytes);
                    let at = [cz * c[0], cy * c[1], cx * c[2]];
                    let lo = [origin[0].max(at[0]), origin[1].max(at[1]), origin[2].max(at[2])];
                    let hi = [(origin[0] + size[0]).min(at[0] + c[0]).min(l.shape[0]), (origin[1] + size[1]).min(at[1] + c[1]).min(l.shape[1]), (origin[2] + size[2]).min(at[2] + c[2]).min(l.shape[2])];
                    for z in lo[0]..hi[0] {
                        for y in lo[1]..hi[1] {
                            let from = ((z - at[0]) * c[1] + (y - at[1])) * c[2] + (lo[2] - at[2]);
                            let to = ((z - origin[0]) * size[1] + (y - origin[1])) * size[2] + (lo[2] - origin[2]);
                            let n = hi[2] - lo[2];
                            if from + n <= values.len() {
                                out[to..to + n].copy_from_slice(&values[from..from + n]);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn write_zarray(&self, level: usize, l: &Level) -> IoResult<()> {
        let c = self.options.chunks;
        let n = self.options.channels.len();
        let (shape, chunks) = if self.options.shape[0] > 1 { (json!([n, l.shape[0], l.shape[1], l.shape[2]]), json!([1, c[0], c[1], c[2]])) } else { (json!([n, l.shape[1], l.shape[2]]), json!([1, c[1], c[2]])) };
        let meta = json!({
            "zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": self.sample.dtype(), "compressor": null,
            "fill_value": 0, "order": "C", "filters": null, "dimension_separator": "/",
        });
        fs::create_dir_all(self.root.join(level.to_string()))?;
        fs::write(self.root.join(level.to_string()).join(".zarray"), serde_json::to_string_pretty(&meta)?)?;
        Ok(())
    }

    fn write_attrs(&self, levels: &[Level], px: f64, dz: f64) -> IoResult<()> {
        let volume = self.options.shape[0] > 1;
        let unit = self.options.pixel_size_um.is_some().then_some("micrometer");
        let space = |name: &str| match unit {
            Some(u) => json!({"name": name, "type": "space", "unit": u}),
            None => json!({"name": name, "type": "space"}),
        };
        let mut axes = vec![json!({"name": "c", "type": "channel"})];
        if volume {
            axes.push(space("z"));
        }
        axes.push(space("y"));
        axes.push(space("x"));
        let datasets: Vec<_> = levels
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let scale = if volume { json!([1.0, dz * l.scale[0], px * l.scale[1], px * l.scale[2]]) } else { json!([1.0, px * l.scale[1], px * l.scale[2]]) };
                json!({"path": i.to_string(), "coordinateTransformations": [{"type": "scale", "scale": scale}]})
            })
            .collect();
        let channels: Vec<_> = self.options.channels.iter().map(|c| json!({"label": c, "active": true})).collect();
        let attrs = json!({
            "multiscales": [{"version": "0.4", "name": self.options.name, "axes": axes, "datasets": datasets}],
            "omero": {"channels": channels},
        });
        fs::write(self.root.join(".zattrs"), serde_json::to_string_pretty(&attrs)?)?;
        Ok(())
    }
}

/// Writes `data` of `size` as a full chunk of `chunk` (padded with zeros at the image's edge, as
/// Zarr's chunks are always whole).
fn write_chunk<T: Sample>(path: &Path, data: &[T], size: Zyx, chunk: Zyx) -> IoResult<()> {
    if size == chunk {
        return write_raw(path, data);
    }
    let mut full = vec![T::default(); chunk[0] * chunk[1] * chunk[2]];
    for z in 0..size[0] {
        for y in 0..size[1] {
            let from = (z * size[1] + y) * size[2];
            let to = (z * chunk[1] + y) * chunk[2];
            full[to..to + size[2]].copy_from_slice(&data[from..from + size[2]]);
        }
    }
    write_raw(path, &full)
}

fn write_raw<T: Sample>(path: &Path, values: &[T]) -> IoResult<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut bytes = Vec::with_capacity(values.len() * 4);
    T::to_bytes(values, &mut bytes);
    fs::write(path, bytes)?;
    Ok(())
}

/// `src` of `src_size` reduced to `size`, halving y and x and dividing z by `zf`.
fn reduce<T: Sample>(src: &[T], src_size: Zyx, size: Zyx, zf: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(size[0] * size[1] * size[2]);
    let mut covered = Vec::with_capacity(8);
    for z in 0..size[0] {
        for y in 0..size[1] {
            for x in 0..size[2] {
                covered.clear();
                for dz in 0..zf {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (zz, yy, xx) = (z * zf + dz, y * 2 + dy, x * 2 + dx);
                            if zz < src_size[0] && yy < src_size[1] && xx < src_size[2] {
                                covered.push(src[(zz * src_size[1] + yy) * src_size[2] + xx]);
                            }
                        }
                    }
                }
                out.push(if covered.is_empty() { T::default() } else { T::reduce(&covered) });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use lumen_core::tiling::schedule;

    use super::*;
    use crate::pyramid::Pyramid;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-writer-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn roundtrip(shape: Zyx, chunks: Zyx) {
        let root = temp(&format!("{}-{}", shape[0], chunks[1]));
        let n = shape[0] * shape[1] * shape[2];
        let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let options = WriterOptions { name: "t".into(), channels: vec!["a".into(), "b".into()], shape, chunks, pixel_size_um: Some(0.5), z_spacing_um: Some(0.5) };
        let w = ZarrWriter::create::<f32>(&root, options).unwrap();
        for b in schedule(shape, chunks, [0; 3]) {
            for c in 0..2 {
                let part: Vec<f32> = lumen_core::tiling::crop(&values, shape, b.origin, b.size).into_iter().map(|v| v + c as f32).collect();
                w.write(c, b.origin, b.size, &part).unwrap();
            }
        }
        let path = w.finish::<f32>().unwrap();
        let py = Pyramid::open(&path).unwrap();
        assert_eq!((py.depth(), py.height(), py.width()), (shape[0], shape[1], shape[2]));
        assert_eq!(py.channels.len(), 2);
        assert_eq!(py.pixel_size_um, Some(0.5));
        assert!(py.levels.len() > 1, "coarser levels are built");
        let got = py.read_block(0, 1, [0, 0, 0], shape).unwrap();
        assert_eq!(got, values.iter().map(|v| v + 1.0).collect::<Vec<_>>());
        fs::remove_dir_all(&path).unwrap();
    }

    #[test]
    fn a_2d_store_reads_back_exactly() {
        roundtrip([1, 150, 100], [1, 64, 64]);
    }

    #[test]
    fn a_3d_store_reads_back_exactly() {
        roundtrip([20, 90, 70], [8, 32, 32]);
    }

    #[test]
    fn an_existing_path_is_never_replaced_and_blocks_stay_on_the_grid() {
        let root = temp("exists");
        let options = WriterOptions { name: "t".into(), channels: vec!["a".into()], shape: [1, 10, 10], chunks: [1, 4, 4], pixel_size_um: None, z_spacing_um: None };
        let w = ZarrWriter::create::<u32>(&root, options.clone()).unwrap();
        assert!(ZarrWriter::create::<u32>(&root, options).is_err());
        assert!(w.write(0, [0, 1, 0], [1, 4, 4], &[0u32; 16]).is_err());
        assert!(w.write(0, [0, 0, 0], [1, 4, 4], &[0u32; 3]).is_err());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn labels_can_be_remapped_after_writing() {
        let root = temp("remap");
        let options = WriterOptions { name: "t".into(), channels: vec!["a".into()], shape: [1, 4, 4], chunks: [1, 4, 4], pixel_size_um: None, z_spacing_um: None };
        let w = ZarrWriter::create::<u32>(&root, options).unwrap();
        let data: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        w.write(0, [0, 0, 0], [1, 4, 4], &data).unwrap();
        w.remap_labels(&[0, 7, 7, 9]).unwrap();
        let path = w.finish::<u32>().unwrap();
        let py = Pyramid::open(&path).unwrap();
        assert_eq!(&py.read_block(0, 0, [0, 0, 0], [1, 1, 4]).unwrap(), &[0.0, 7.0, 7.0, 9.0]);
        fs::remove_dir_all(&path).unwrap();
    }
}

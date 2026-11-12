//! Your own operations over an image of any size, built from lucem's core methods.
//!
//! lucem reads the image in blocks (the next ones are read on other threads while your function
//! runs) and uploads each block once, so your function only ever sees one block, already on the
//! device, and can chain [`lucem_ops`] calls on it. Each block is read with a halo of context and
//! only the part it owns is kept, so the result equals the function on the whole image for any
//! function that looks at most `halo` pixels away.
//!
//! - [`measure`]: one row per block.
//! - [`transform`]: an image in, an image of the same shape out, written to OME-Zarr.
//! - [`segment`]: labels per block, written to OME-Zarr with objects cut by block borders joined,
//!   and a table with one row per object.
//!
//! A volume is walked plane by plane by default, where each plane is `[h, w]`, or in blocks with
//! [`Walk::Volume`], where each is `[d, h, w]`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::{check_allocation, DeviceArray};
use lucem_core::tiling::{crop, schedule};
use lucem_core::{Block, Shape, Zyx};
use lucem_io::{BlockLoader, IoError, LoadedBlock, Pyramid, Sample, WriterOptions, ZarrWriter};
use thiserror::Error;

use crate::eval::{at_downsample, evaluate, halo};
use crate::ops::Value;
use crate::spec::PipelineSpec;
use crate::stitch::LabelStitcher;
use crate::table::Table;

/// Default side of a block when the image's chunks are not used.
pub const DEFAULT_BLOCK_SIDE: usize = 1024;

#[derive(Debug, Error)]
pub enum RunError {
    #[error(transparent)]
    Io(#[from] IoError),
    #[error("{0}")]
    Function(String),
    #[error("{0}")]
    Invalid(String),
    #[error("cancelled")]
    Cancelled,
}

/// How a volume is walked (a 2D image has only planes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Walk {
    /// Each plane by itself: the function gets `[h, w]`.
    #[default]
    Planes,
    /// In 3D blocks: the function gets `[d, h, w]`.
    Volume,
}

/// Called with blocks done and blocks in all.
pub type ProgressFn = Arc<dyn Fn(usize, usize) + Send + Sync>;

/// How to walk an image.
#[derive(Clone)]
pub struct RunOptions {
    pub level: usize,
    /// Channels to process; all of them when `None`.
    pub channels: Option<Vec<usize>>,
    pub walk: Walk,
    /// Side of a block in y and x; the image's tile size when `None` (at least 256).
    pub block: Option<usize>,
    /// Depth of a block in [`Walk::Volume`]; the image's chunk depth when `None` (at least 32).
    pub block_z: Option<usize>,
    /// Pixels of context on each side, in y and x.
    pub halo: usize,
    /// Planes of context on each side in [`Walk::Volume`]; `halo` when `None`.
    pub halo_z: Option<usize>,
    /// Blocks read ahead of the one being processed.
    pub read_ahead: usize,
    pub threads: usize,
    pub cancel: Option<Arc<AtomicBool>>,
    pub progress: Option<ProgressFn>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self { level: 0, channels: None, walk: Walk::Planes, block: None, block_z: None, halo: 0, halo_z: None, read_ahead: 4, threads: 4, cancel: None, progress: None }
    }
}

/// Where a block sits, for the function that receives it.
#[derive(Debug, Clone)]
pub struct BlockContext {
    pub index: usize,
    pub total: usize,
    pub level: usize,
    pub channel: usize,
    pub block: Block,
    /// Level-0 pixels per pixel of this level.
    pub downsample: f64,
    /// Size of a pixel of this level (µm), if known.
    pub pixel_size_um: Option<f64>,
    /// Distance between planes of this level (µm), if known.
    pub z_spacing_um: Option<f64>,
}

impl RunOptions {
    /// The grid and the blocks of a walk.
    pub fn blocks(&self, py: &Pyramid) -> Result<(Zyx, Vec<Block>), RunError> {
        let l = py.levels.get(self.level).ok_or_else(|| RunError::Invalid(format!("no level {} (the image has {})", self.level, py.levels.len())))?;
        let dims = [l.depth, l.height, l.width];
        let (by, bx) = match self.block {
            Some(b) if b > 0 => (b, b),
            Some(_) => return Err(RunError::Invalid("the block side must be positive".into())),
            None => (l.tile_h.max(256), l.tile_w.max(256)),
        };
        let (grid, halo) = if self.walk == Walk::Volume && l.depth > 1 {
            ([self.block_z.unwrap_or(l.tile_d.max(32)).max(1), by, bx], [self.halo_z.unwrap_or(self.halo), self.halo, self.halo])
        } else {
            ([1, by, bx], [0, self.halo, self.halo])
        };
        Ok((grid, schedule(dims, grid, halo)))
    }

    fn channel_list(&self, py: &Pyramid) -> Result<Vec<usize>, RunError> {
        let list = self.channels.clone().unwrap_or_else(|| (0..py.channels.len()).collect());
        match list.iter().find(|c| **c >= py.channels.len()) {
            Some(c) => Err(RunError::Invalid(format!("no channel {c} (the image has {})", py.channels.len()))),
            None if list.is_empty() => Err(RunError::Invalid("no channels to process".into())),
            None => Ok(list),
        }
    }

    fn context(&self, py: &Pyramid, index: usize, total: usize, channel: usize, block: Block) -> BlockContext {
        let l = &py.levels[self.level];
        BlockContext {
            index,
            total,
            level: self.level,
            channel,
            block,
            downsample: l.downsample,
            pixel_size_um: py.pixel_size_um.map(|p| p * l.downsample),
            z_spacing_um: py.z_spacing_um.map(|z| z * l.z_downsample),
        }
    }

    fn check_cancel(&self) -> Result<(), RunError> {
        if self.cancel.as_ref().is_some_and(|c| c.load(Ordering::Acquire)) { Err(RunError::Cancelled) } else { Ok(()) }
    }

    fn report(&self, done: usize, total: usize) {
        if let Some(p) = &self.progress {
            p(done, total);
        }
    }

    fn writer_options(&self, py: &Pyramid, channels: Vec<String>, grid: Zyx, shape: Zyx) -> WriterOptions {
        let l = &py.levels[self.level];
        WriterOptions {
            name: py.name.clone(),
            channels,
            shape,
            chunks: [grid[0].min(shape[0]), grid[1].min(shape[1]), grid[2].min(shape[2])],
            pixel_size_um: py.pixel_size_um.map(|p| p * l.downsample),
            z_spacing_um: py.z_spacing_um.map(|z| z * l.z_downsample),
        }
    }
}

/// Uploads the read part of a block: one transfer, within the memory budget.
fn upload(client: &Client, loaded: LoadedBlock) -> Result<DeviceArray<f32>, RunError> {
    check_allocation(client, "a block", (loaded.data.len() * 4) as u64).map_err(|e| RunError::Function(e.to_string()))?;
    DeviceArray::from_host(client, Shape::new(loaded.block.read_dims()), loaded.data).map_err(|e| RunError::Function(e.to_string()))
}

/// Downloads a result and keeps the part the block owns.
fn owned<T: lucem_core::cubecl::prelude::CubeElement + Copy>(client: &Client, out: &DeviceArray<T>, block: &Block, what: &str) -> Result<Vec<T>, RunError> {
    if out.dims() != block.read_dims().as_slice() {
        return Err(RunError::Function(format!("{what} returned shape {:?} for a block of {:?}", out.dims(), block.read_dims())));
    }
    Ok(crop(&out.to_host(client), block.read_size, block.inner_offset(), block.size))
}

/// Walks the image, giving each block on the device to `visit`.
fn walk(client: &Client, py: &Arc<Pyramid>, options: &RunOptions, channels: Vec<usize>, blocks: Vec<Block>, mut visit: impl FnMut(DeviceArray<f32>, BlockContext) -> Result<(), RunError>) -> Result<(), RunError> {
    let loader = BlockLoader::new(py.clone(), options.level, channels, blocks, options.read_ahead, options.threads);
    let total = loader.total();
    for (index, loaded) in loader.enumerate() {
        options.check_cancel()?;
        let loaded = loaded?;
        let context = options.context(py, index, total, loaded.channel, loaded.block);
        visit(upload(client, loaded)?, context)?;
        options.report(index + 1, total);
    }
    Ok(())
}

/// Calls `f` on every block and gathers what it returns into a table, one row per block and
/// channel (with `channel`, `z`, `y`, `x` of the block).
pub fn measure<F>(client: &Client, py: &Arc<Pyramid>, options: &RunOptions, mut f: F) -> Result<Table, RunError>
where
    F: FnMut(&Client, &DeviceArray<f32>, &BlockContext) -> Result<Vec<(String, f64)>, String>,
{
    let channels = options.channel_list(py)?;
    let (_, blocks) = options.blocks(py)?;
    let mut table = Table::new();
    walk(client, py, options, channels, blocks, |array, ctx| {
        let values = f(client, &array, &ctx).map_err(RunError::Function)?;
        let mut row = vec![("channel".to_string(), ctx.channel as f64), ("z".to_string(), ctx.block.origin[0] as f64), ("y".to_string(), ctx.block.origin[1] as f64), ("x".to_string(), ctx.block.origin[2] as f64)];
        row.extend(values);
        table.push_row(&row);
        Ok(())
    })?;
    Ok(table)
}

/// Writes `f` of every block to an OME-Zarr store at `output` (which must not exist), channel by
/// channel. `f` returns an array of the shape it was given.
pub fn transform<F>(client: &Client, py: &Arc<Pyramid>, options: &RunOptions, output: &Path, mut f: F) -> Result<PathBuf, RunError>
where
    F: FnMut(&Client, &DeviceArray<f32>, &BlockContext) -> Result<DeviceArray<f32>, String>,
{
    let channels = options.channel_list(py)?;
    let (grid, blocks) = options.blocks(py)?;
    let names = channels.iter().map(|c| py.channels[*c].label.clone()).collect();
    let l = &py.levels[options.level];
    let writer = ZarrWriter::create::<f32>(output, options.writer_options(py, names, grid, [l.depth, l.height, l.width]))?;
    let order = channels.clone();
    walk(client, py, options, channels, blocks, |array, ctx| {
        let result = f(client, &array, &ctx).map_err(RunError::Function)?;
        let kept = owned(client, &result, &ctx.block, "the function")?;
        let channel = order.iter().position(|c| *c == ctx.channel).expect("a channel of the walk");
        writer.write(channel, ctx.block.origin, ctx.block.size, &kept)?;
        Ok(())
    })?;
    Ok(writer.finish::<f32>()?)
}

/// Writes labels from `f` of every block of one channel to an OME-Zarr store at `output` (which
/// must not exist), joining objects cut by block borders, and returns one row per object. Labels
/// need not be numbered consistently between blocks. In [`Walk::Planes`] each plane's objects are
/// separate; in [`Walk::Volume`] objects are joined along z too.
pub fn segment<F>(client: &Client, py: &Arc<Pyramid>, options: &RunOptions, output: &Path, mut f: F) -> Result<Table, RunError>
where
    F: FnMut(&Client, &DeviceArray<f32>, &BlockContext) -> Result<DeviceArray<u32>, String>,
{
    let channels = options.channel_list(py)?;
    if channels.len() != 1 {
        return Err(RunError::Invalid("segment runs on one channel: set `channels` to it".into()));
    }
    let (grid, blocks) = options.blocks(py)?;
    let l = &py.levels[options.level];
    let name = py.channels[channels[0]].label.clone();
    let writer = ZarrWriter::create::<u32>(output, options.writer_options(py, vec![name], grid, [l.depth, l.height, l.width]))?;
    let mut stitcher = LabelStitcher::new(grid, options.walk == Walk::Volume);
    walk(client, py, options, channels, blocks, |array, ctx| {
        let labels = f(client, &array, &ctx).map_err(RunError::Function)?;
        let kept = owned(client, &labels, &ctx.block, "the function")?;
        let intensity = owned(client, &array, &ctx.block, "the image")?;
        let ids = stitcher.add(&ctx.block, &kept, Some(&intensity));
        writer.write(0, ctx.block.origin, ctx.block.size, &ids)?;
        Ok(())
    })?;
    let done = stitcher.finish();
    writer.remap_labels(&done.map)?;
    writer.finish::<u32>()?;
    Ok(done.table)
}

/// What a headless run wrote.
#[derive(Debug, Clone)]
pub struct RunSummary {
    pub output: PathBuf,
    /// The spec, written next to the output.
    pub spec: PathBuf,
    pub blocks: usize,
    pub warnings: Vec<String>,
    /// Objects found (labels and tables).
    pub objects: Option<usize>,
}

/// Runs a pipeline spec over an image, plane by plane: its target's result is written to `output`
/// (OME-Zarr for images, masks and labels; CSV for tables), and the spec next to it.
pub fn run_spec(client: &Client, py: &Arc<Pyramid>, spec: &PipelineSpec, output: &Path, mut options: RunOptions) -> Result<RunSummary, RunError> {
    spec.validate().map_err(RunError::Invalid)?;
    if output.exists() {
        return Err(RunError::Invalid(format!("{} already exists", output.display())));
    }
    options.level = spec.level;
    options.walk = Walk::Planes;
    let mut sources: Vec<usize> = spec.sources.values().copied().collect();
    sources.sort_unstable();
    sources.dedup();
    options.channels = Some(sources.clone());
    let channels = options.channel_list(py)?;
    let l = py.levels.get(spec.level).ok_or_else(|| RunError::Invalid(format!("no level {}", spec.level)))?.clone();
    let plan = at_downsample(&spec.plan(), l.downsample);
    options.halo = options.halo.max(halo(&plan, &spec.target));
    let (grid, blocks) = options.blocks(py)?;
    let n_blocks = blocks.len();
    let mut warnings = Vec::new();
    if plan.values().any(|n| n.op.as_ref().is_some_and(|o| o.is_global())) {
        warnings.push("a global threshold (Otsu) was computed per block: its value can differ between blocks".to_string());
    }
    let shape = [l.depth, l.height, l.width];
    let mut sink = Sink::Unset;
    let loader = BlockLoader::new(py.clone(), spec.level, channels.clone(), blocks, options.read_ahead.max(channels.len()), options.threads);
    let total = n_blocks;
    let mut group: Vec<LoadedBlock> = Vec::with_capacity(channels.len());
    let mut done_blocks = 0usize;
    for loaded in loader {
        options.check_cancel()?;
        group.push(loaded?);
        if group.len() < channels.len() {
            continue;
        }
        let block = group[0].block;
        let mut arrays = std::collections::HashMap::new();
        for g in group.drain(..) {
            let (lo, hi) = if spec.normalize { py.channels[g.channel].range } else { (0.0, 1.0) };
            let span = (hi - lo).abs().max(1e-12);
            let channel = g.channel;
            let scaled = LoadedBlock { data: g.data.into_iter().map(|v| (v - lo) / span).collect(), ..g };
            arrays.insert(channel, upload(client, scaled)?);
        }
        let mut source = |id: &str| -> Result<Value, String> {
            let c = spec.sources.get(id).ok_or_else(|| format!("no source {id:?}"))?;
            Ok(Value::Image(arrays[c].clone()))
        };
        let value = evaluate(client, &plan, &spec.target, &mut source).map_err(RunError::Function)?;
        sink.accept(client, py, &options, output, spec, grid, shape, &block, &arrays, value)?;
        done_blocks += 1;
        options.report(done_blocks, total);
    }
    let objects = sink.finish(output)?;
    let spec_path = spec_path_for(output);
    std::fs::write(&spec_path, serde_json::to_string_pretty(spec).map_err(|e| RunError::Invalid(e.to_string()))?).map_err(IoError::from)?;
    Ok(RunSummary { output: output.to_path_buf(), spec: spec_path, blocks: n_blocks, warnings, objects })
}

/// `<output>.spec.json`.
fn spec_path_for(output: &Path) -> PathBuf {
    let mut s = output.as_os_str().to_owned();
    s.push(".spec.json");
    PathBuf::from(s)
}

/// Where a run's target goes, decided by the type of its first result.
enum Sink {
    Unset,
    Image(ZarrWriter),
    Mask(ZarrWriter),
    Labels(ZarrWriter, Box<LabelStitcher>),
    Table(Table),
}

impl Sink {
    #[allow(clippy::too_many_arguments)]
    fn accept(&mut self, client: &Client, py: &Pyramid, options: &RunOptions, output: &Path, spec: &PipelineSpec, grid: Zyx, shape: Zyx, block: &Block, arrays: &std::collections::HashMap<usize, DeviceArray<f32>>, value: Value) -> Result<(), RunError> {
        if matches!(self, Sink::Unset) {
            let names = vec![spec.target.clone()];
            *self = match &value {
                Value::Image(_) => Sink::Image(ZarrWriter::create::<f32>(output, options.writer_options(py, names, grid, shape))?),
                Value::Mask(_) => Sink::Mask(ZarrWriter::create::<u32>(output, options.writer_options(py, names, grid, shape))?),
                Value::Labels(_) => Sink::Labels(ZarrWriter::create::<u32>(output, options.writer_options(py, names, grid, shape))?, Box::new(LabelStitcher::new(grid, false))),
                Value::Table(_) => Sink::Table(Table::new()),
            };
        }
        match (self, value) {
            (Sink::Image(w), Value::Image(a)) => put(w, block, &owned(client, &a, block, "the pipeline")?),
            (Sink::Mask(w), Value::Mask(a)) => put(w, block, &owned(client, &a, block, "the pipeline")?),
            (Sink::Labels(w, s), Value::Labels(a)) => {
                let kept = owned(client, &a, block, "the pipeline")?;
                let intensity = arrays.values().next().map(|i| owned(client, i, block, "the image")).transpose()?;
                let ids = s.add(block, &kept, intensity.as_deref());
                put(w, block, &ids)
            }
            (Sink::Table(t), Value::Table(rows)) => {
                let inner = block.inner_offset();
                for r in rows {
                    // An object belongs to the block owning its centroid, so it is counted once
                    let dims = block.read_dims();
                    let c: Vec<usize> = r.centroid.iter().map(|v| v.max(0.0) as usize).collect();
                    let (cy, cx) = if dims.len() == 2 && c.len() == 2 { (c[0], c[1]) } else { continue };
                    if cy < inner[1] || cx < inner[2] || cy >= inner[1] + block.size[1] || cx >= inner[2] + block.size[2] {
                        continue;
                    }
                    t.push_row(&[
                        ("z".into(), block.origin[0] as f64),
                        ("centroid_y".into(), block.read_origin[1] as f64 + r.centroid[0]),
                        ("centroid_x".into(), block.read_origin[2] as f64 + r.centroid[1]),
                        ("area".into(), r.area as f64),
                        ("eccentricity".into(), r.eccentricity),
                        ("perimeter".into(), r.perimeter),
                        ("mean_intensity".into(), r.mean_intensity),
                        ("min_intensity".into(), r.min_intensity),
                        ("max_intensity".into(), r.max_intensity),
                    ]);
                }
                Ok(())
            }
            _ => Err(RunError::Function("the pipeline's result changed type between blocks".into())),
        }
    }

    fn finish(self, output: &Path) -> Result<Option<usize>, RunError> {
        Ok(match self {
            Sink::Unset => return Err(RunError::Invalid("the image has no blocks".into())),
            Sink::Image(w) => {
                w.finish::<f32>()?;
                None
            }
            Sink::Mask(w) => {
                w.finish::<u32>()?;
                None
            }
            Sink::Labels(w, s) => {
                let done = s.finish();
                w.remap_labels(&done.map)?;
                w.finish::<u32>()?;
                Some(done.table.rows())
            }
            Sink::Table(t) => {
                t.write_csv(output).map_err(IoError::from)?;
                Some(t.rows())
            }
        })
    }
}

fn put<T: Sample>(writer: &ZarrWriter, block: &Block, data: &[T]) -> Result<(), RunError> {
    Ok(writer.write(0, block.origin, block.size, data)?)
}

#[cfg(test)]
mod tests {
    use lucem_core::ComputeTarget;
    use lucem_io::{ChannelInfo, Pyramid};
    use lucem_ops::filter::gaussian::{gaussian_blur, GaussianBlur};

    use super::*;

    fn image(d: usize, h: usize, w: usize) -> Arc<Pyramid> {
        let info = vec![ChannelInfo { label: "a".into(), color: [255, 255, 255], window: (0.0, 1.0), range: (0.0, 1.0), active: true }];
        let values: Vec<f32> = (0..d * h * w).map(|i| (((i % w) as f32 * 0.31).sin() + ((i / w) as f32 * 0.17).cos()) * 0.25 + 0.5).collect();
        Arc::new(Pyramid::from_volume("t", info, vec![values], [d, h, w], None, None, "float32").unwrap())
    }

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lucem-custom-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    /// The blocked result equals the function on the whole image.
    #[test]
    fn a_function_with_a_halo_matches_the_whole_image() {
        for target in ComputeTarget::for_tests() {
            let client = target.client().unwrap();
            let py = image(1, 300, 310);
            let blur = |c: &Client, a: &DeviceArray<f32>, _: &BlockContext| gaussian_blur(c, a, &[2.0, 2.0], &GaussianBlur::default()).map_err(|e| e.to_string());
            let out = temp(&format!("blur-{target}"));
            let options = RunOptions { block: Some(128), halo: 16, ..RunOptions::default() };
            let path = transform(&client, &py, &options, &out, blur).unwrap();
            let got = Pyramid::open(&path).unwrap().read_block(0, 0, [0, 0, 0], [1, 300, 310]).unwrap();
            let whole = DeviceArray::from_host(&client, Shape::new([300, 310]), py.read_block(0, 0, [0, 0, 0], [1, 300, 310]).unwrap()).unwrap();
            let expected = gaussian_blur(&client, &whole, &[2.0, 2.0], &GaussianBlur::default()).unwrap().to_host(&client);
            let worst = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(worst < 1e-3, "{target}: worst difference {worst}");
            std::fs::remove_dir_all(&path).unwrap();
        }
    }

    #[test]
    fn measure_gives_a_row_per_block_and_cancel_stops_a_walk() {
        for target in ComputeTarget::for_tests() {
            let client = target.client().unwrap();
            let py = image(3, 64, 64);
            let options = RunOptions { block: Some(32), ..RunOptions::default() };
            let table = measure(&client, &py, &options, |c, a, _| Ok(vec![("mean".into(), a.to_host(c).iter().sum::<f32>() as f64 / a.len() as f64)])).unwrap();
            assert_eq!(table.rows(), 3 * 4);
            let cancelled = RunOptions { block: Some(32), cancel: Some(Arc::new(AtomicBool::new(true))), ..RunOptions::default() };
            assert!(matches!(measure(&client, &py, &cancelled, |_, _, _| Ok(vec![])), Err(RunError::Cancelled)));
        }
    }

    const SPEC: &str = r#"{
        "sources": {"image": 0},
        "nodes": {
            "blur": {"op": {"op": "gaussian_blur", "sigma": 1.0}, "inputs": ["image"]},
            "mask": {"op": {"op": "threshold", "value": 0.5}, "inputs": ["blur"]},
            "objects": {"op": {"op": "label"}, "inputs": ["mask"]}
        },
        "target": "objects"
    }"#;

    #[test]
    fn a_spec_runs_headless_and_is_written_next_to_its_output() {
        for target in ComputeTarget::for_tests() {
            let client = target.client().unwrap();
            let py = image(1, 200, 220);
            let spec: PipelineSpec = serde_json::from_str(SPEC).unwrap();
            let out = temp(&format!("spec-{target}"));
            let done = run_spec(&client, &py, &spec, &out, RunOptions { block: Some(64), ..RunOptions::default() }).unwrap();
            assert!(done.objects.unwrap() > 0);
            assert!(done.spec.is_file());
            let labels = Pyramid::open(&done.output).unwrap().read_block(0, 0, [0, 0, 0], [1, 200, 220]).unwrap();
            assert_eq!(labels.iter().map(|v| *v as usize).max().unwrap(), done.objects.unwrap());
            assert!(run_spec(&client, &py, &spec, &out, RunOptions::default()).is_err(), "never written over");
            std::fs::remove_dir_all(&out).unwrap();
            std::fs::remove_file(&done.spec).unwrap();
        }
    }
}

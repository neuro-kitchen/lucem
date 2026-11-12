//! What lucem does without a window, once: list the operations, run a pipeline over an image,
//! run a session's steps on an image and write its results. `lucem ops`, `lucem run`,
//! `lucem session run` and the agent's tools ([`crate::server`]) all call these.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lucem_core::compute::ComputeTask;
use lucem_core::cubecl::prelude::Client;
use lucem_core::ComputeTarget;
use lucem_hist::output::{self, RegionShape};
use lucem_hist::session::{self, Action, RunResult, Session};
use lucem_io::Pyramid;
use lucem_pipeline::{catalog, run_spec, Op, PipelineSpec, RunOptions};
use serde_json::{json, Value};

/// Runs a closure with a compute client.
struct Job<F>(F);

impl<T, F: FnOnce(&Client) -> T> ComputeTask for Job<F> {
    type Output = T;
    fn run(self, client: Client) -> T {
        (self.0)(&client)
    }
}

/// The runtime named by `name` (`wgpu`, `cpu`, `cuda`, `hip`), else `LUCEM_RUNTIME`'s, else the
/// first one compiled in.
pub fn target(name: Option<&str>) -> Result<ComputeTarget, String> {
    match name {
        Some(r) => ComputeTarget::parse(r).and_then(ComputeTarget::checked).map_err(|e| e.to_string()),
        None => ComputeTarget::from_env().map_err(|e| e.to_string()),
    }
}

// ---- operations --------------------------------------------------------------------------------

/// What the schema says of each operation: its description and its parameters'.
fn descriptions() -> Vec<(String, String, serde_json::Map<String, Value>)> {
    let schema = serde_json::to_value(rmcp::schemars::schema_for!(Op)).unwrap_or(Value::Null);
    let variants = schema.get("oneOf").and_then(Value::as_array).cloned().unwrap_or_default();
    variants
        .iter()
        .filter_map(|v| {
            let props = v.get("properties")?.as_object()?;
            let tag = props.get("op").and_then(|o| o.get("const").or_else(|| o.get("enum").and_then(|e| e.get(0)))).and_then(Value::as_str)?.to_string();
            let about = v.get("description").and_then(Value::as_str).unwrap_or("").to_string();
            let params = props.iter().filter(|(k, _)| *k != "op").map(|(k, p)| (k.clone(), p.get("description").cloned().unwrap_or(Value::Null))).collect();
            Some((tag, about, params))
        })
        .collect()
}

/// Every pipeline operation: `[{op, title, family, inputs, output, params, about}]`.
pub fn operations() -> Value {
    let docs = descriptions();
    let ops: Vec<Value> = catalog()
        .iter()
        .map(|op| {
            let (about, param_docs) = docs.iter().find(|(t, _, _)| t == op.tag()).map(|(_, a, p)| (a.clone(), p.clone())).unwrap_or_default();
            let defaults = serde_json::to_value(op).unwrap_or(Value::Null);
            let params: Vec<Value> = defaults
                .as_object()
                .map(|o| o.iter().filter(|(k, _)| *k != "op").map(|(k, v)| json!({"name": k, "default": v, "about": param_docs.get(k).cloned().unwrap_or(Value::Null)})).collect())
                .unwrap_or_default();
            json!({
                "op": op.tag(),
                "title": op.title(),
                "family": op.family(),
                "inputs": op.inputs().iter().map(|(t, n)| json!({"name": n, "type": t.name()})).collect::<Vec<_>>(),
                "output": op.output().name(),
                "params": params,
                "about": about,
            })
        })
        .collect();
    Value::Array(ops)
}

// ---- images ------------------------------------------------------------------------------------

/// An image's description: size, channels, pixel size, levels.
pub fn describe(path: &Path) -> Result<Value, String> {
    let p = Pyramid::open(path).map_err(|e| e.to_string())?;
    Ok(json!({
        "name": p.name,
        "file": path.file_name().map(|n| n.to_string_lossy().into_owned()),
        "width": p.width(),
        "height": p.height(),
        "depth": p.depth(),
        "dtype": p.dtype.to_string(),
        "channels": p.channels.iter().map(|c| c.label.clone()).collect::<Vec<_>>(),
        "pixel_size_um": p.pixel_size_um,
        "levels": p.levels.len(),
    }))
}

// ---- pipelines ---------------------------------------------------------------------------------

/// What a pipeline run wrote.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PipelineDone {
    pub output: PathBuf,
    pub blocks: usize,
    pub objects: Option<usize>,
    pub warnings: Vec<String>,
    pub spec: PathBuf,
}

/// Runs `spec` over the whole of `image`, writing `output` (OME-Zarr), block by block.
pub fn run_pipeline(target: &ComputeTarget, spec: &PipelineSpec, image: &Path, output: &Path, block: Option<usize>, progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>) -> Result<PipelineDone, String> {
    let image = Arc::new(Pyramid::open(image).map_err(|e| e.to_string())?);
    let output = output.to_path_buf();
    target
        .run(Job(move |client: &Client| {
            let options = RunOptions { block, progress, ..RunOptions::default() };
            let done = run_spec(client, &image, spec, &output, options).map_err(|e| e.to_string())?;
            Ok(PipelineDone { output: done.output, blocks: done.blocks, objects: done.objects, warnings: done.warnings, spec: done.spec })
        }))
        .map_err(|e| e.to_string())?
}

// ---- sessions ----------------------------------------------------------------------------------

/// `recipe` run on the image at `path` through the engine (the app's steps).
pub fn analyse(target: &ComputeTarget, path: &Path, recipe: &[&Action], progress: &mut dyn FnMut(&str)) -> Result<(Pyramid, RunResult), String> {
    let p = Pyramid::open(path).map_err(|e| e.to_string())?;
    let r = target.run(Job(|client: &Client| session::run(client, &p, recipe, progress))).map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
    Ok((p, r))
}

/// Writes a run's results to the image's output store (inside its OME-Zarr when writable, else
/// `out_dir/<image>.lucem.zarr`): labels, Parquet tables, GeoParquet shapes, provenance.
pub fn write_results(path: &Path, p: &Pyramid, r: &RunResult, s: &Session, run: &str, out_dir: Option<&Path>) -> Result<PathBuf, String> {
    let um = p.pixel_size_um.ok_or("the image has no pixel size (µm)")?;
    let name = path.file_name().map_or_else(|| "image".to_string(), |n| n.to_string_lossy().into_owned());
    let fallback = out_dir.map(Path::to_path_buf).or_else(|| path.parent().map(Path::to_path_buf)).unwrap_or_default();
    let store = output::store_for(path, &name, &fallback);
    if store != path {
        output::ensure_group(&store).map_err(|e| e.to_string())?;
    }
    // Each object's region: the tissue region holding it
    let region_of = |o: &lucem_hist::HistObject| r.tissue.iter().position(|t| lucem_hist::polygon::contains(t, o.centroid)).map(|i| format!("tissue{}", i + 1));
    let cell_regions: Vec<Option<String>> = r.cells.iter().map(region_of).collect();
    let tile_regions: Vec<Option<String>> = r.tiles.iter().map(region_of).collect();
    if !r.cells.is_empty() {
        output::write_objects(&store, "cells", &r.cells, &cell_regions, um).map_err(|e| e.to_string())?;
        output::write_labels(&store, "cells", &r.cells, false, (p.width(), p.height()), Some(um)).map_err(|e| e.to_string())?;
        output::write_labels(&store, "nuclei", &r.cells, true, (p.width(), p.height()), Some(um)).map_err(|e| e.to_string())?;
    }
    if !r.tiles.is_empty() {
        output::write_objects(&store, "tiles", &r.tiles, &tile_regions, um).map_err(|e| e.to_string())?;
    }
    let ids: Vec<String> = (1..=r.tissue.len()).map(|i| format!("tissue{i}")).collect();
    let regions: Vec<RegionShape> = r.tissue.iter().zip(&ids).map(|(o, id)| RegionShape { id, class: "Tissue", outline: o }).collect();
    output::write_shapes(&store, &[r.cells.clone(), r.tiles.clone()].concat(), &regions).map_err(|e| e.to_string())?;
    let provenance = json!({
        "lucem": env!("CARGO_PKG_VERSION"),
        "run": run,
        "session": s,
        "notes": r.notes,
    });
    output::write_provenance(&store, path, &provenance).map_err(|e| e.to_string())?;
    Ok(store)
}

/// One image through a session: its steps, then its results written. Returns the store, the
/// numbers of cells and tiles, and what each step did.
pub fn run_session_on(target: &ComputeTarget, path: &Path, s: &Session, run: &str, out_dir: Option<&Path>, progress: &mut dyn FnMut(&str)) -> Result<(PathBuf, usize, usize, Vec<String>), String> {
    let recipe = s.recipe();
    let (p, r) = analyse(target, path, &recipe, progress)?;
    let store = write_results(path, &p, &r, s, run, out_dir)?;
    Ok((store, r.cells.len(), r.tiles.len(), r.notes))
}

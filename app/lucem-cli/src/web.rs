//! What `lucem serve --web` adds for browser pages: pipelines run on the tiles a page asks for
//! (the server has the GPU; the page only draws), with the operators' halo read around each tile.

use std::sync::Arc;

use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::DeviceArray;
use lucem_core::{ComputeTarget, Shape};
use lucem_io::Pyramid;
use lucem_pipeline::{at_downsample, evaluate, halo, PipelineSpec, Value};
use lucem_stream::engine::TileProcessor;

/// Pipelines on tiles, on one compute device of the server's.
pub struct PipelineTiles {
    client: Client,
    runtime: String,
}

impl PipelineTiles {
    /// On `target`'s default device; `None` (and why) when it has none.
    pub fn open(target: ComputeTarget) -> Result<Arc<Self>, String> {
        let client = target.client().map_err(|e| e.to_string())?;
        Ok(Arc::new(Self { client, runtime: target.name().to_string() }))
    }

    /// The runtime it runs on, for pages (`wgpu`, `cuda`, `cpu`).
    pub fn runtime(&self) -> &str {
        &self.runtime
    }
}

impl TileProcessor for PipelineTiles {
    fn tile(&self, image: &Pyramid, spec_json: &str, level: usize, z: usize, tx: usize, ty: usize) -> Result<(usize, usize, Vec<f32>), String> {
        let spec: PipelineSpec = serde_json::from_str(spec_json).map_err(|e| format!("the pipeline: {e}"))?;
        spec.validate()?;
        let l = image.levels.get(level).ok_or_else(|| format!("no level {level}"))?.clone();
        let (x0, y0) = (tx * l.tile_w, ty * l.tile_h);
        if x0 >= l.width || y0 >= l.height || z >= l.depth {
            return Err(format!("tile {tx},{ty} (plane {z}) is outside level {level}"));
        }
        let (w, h) = ((l.width - x0).min(l.tile_w), (l.height - y0).min(l.tile_h));
        // The operators' reach, in this level's pixels, read around the tile (clipped to the image)
        let plan = at_downsample(&spec.plan(), l.downsample);
        let reach = halo(&plan, &spec.target);
        let (rx0, ry0) = (x0.saturating_sub(reach), y0.saturating_sub(reach));
        let (rx1, ry1) = ((x0 + w + reach).min(l.width), (y0 + h + reach).min(l.height));
        let (rw, rh) = (rx1 - rx0, ry1 - ry0);
        let client = &self.client;
        let value = evaluate(client, &plan, &spec.target, &mut |id: &str| {
            let channel = *spec.sources.get(id).ok_or_else(|| format!("{id} is neither a source nor a node"))?;
            let mut values = image.read_region_at(level, channel, z, rx0, ry0, rw, rh).map_err(|e| e.to_string())?;
            // Raw values to 0–1 by the type's range, as the app does
            let top = image.channels.get(channel).map_or(1.0, |c| c.range.1);
            if spec.normalize && top > 1.0 {
                let k = 1.0 / top;
                values.iter_mut().for_each(|v| *v *= k);
            }
            DeviceArray::from_host(client, Shape::new([rh, rw]), values).map(Value::Image).map_err(|e| e.to_string())
        })?;
        let region = value.to_host(client).ok_or("a table is not a tile: its rows are a plugin call")?;
        if region.len() != rw * rh {
            return Err(format!("the pipeline gave {} values for a region of {rw} × {rh}", region.len()));
        }
        // The tile itself, out of the region read with its halo
        let (ox, oy) = (x0 - rx0, y0 - ry0);
        let mut out = Vec::with_capacity(w * h);
        for y in oy..oy + h {
            out.extend_from_slice(&region[y * rw + ox..y * rw + ox + w]);
        }
        Ok((w, h, out))
    }
}

/// Whole-image pipeline runs, as jobs a page (or a computer) starts and follows: the server's
/// `pipeline` plugin. `run {path, output, spec}` starts one (writing under the data folder, for a
/// client that may write) and answers `{job}`; `job {job}` answers its progress or its end.
#[derive(Default)]
pub struct PipelinePlugin {
    jobs: std::sync::Mutex<Vec<Job>>,
}

struct Job {
    who: String,
    done: Arc<std::sync::atomic::AtomicUsize>,
    total: Arc<std::sync::atomic::AtomicUsize>,
    result: Arc<std::sync::Mutex<Option<Result<serde_json::Value, String>>>>,
}

impl lucem_stream::plugin::ServerPlugin for PipelinePlugin {
    fn id(&self) -> &'static str {
        "pipeline"
    }

    fn call(&self, method: &str, body: serde_json::Value, context: &lucem_stream::plugin::PluginContext) -> Result<serde_json::Value, String> {
        use std::sync::atomic::Ordering;
        let text = |key: &str| body.get(key).and_then(|v| v.as_str()).map(str::to_string).ok_or_else(|| format!("{method}: {key} is missing"));
        match method {
            "run" => {
                if !context.may_write {
                    return Err("this client may read only: a run writes its result".into());
                }
                let image = context.resolve(&text("path")?)?;
                let output = context.resolve(&text("output")?)?;
                let spec: PipelineSpec = serde_json::from_value(body.get("spec").cloned().ok_or("run: spec is missing")?).map_err(|e| format!("run: the spec: {e}"))?;
                let target = lucem_mcp::work::target(None)?;
                let (done, total) = (Arc::new(std::sync::atomic::AtomicUsize::new(0)), Arc::new(std::sync::atomic::AtomicUsize::new(0)));
                let result = Arc::new(std::sync::Mutex::new(None));
                let (d, t, r) = (done.clone(), total.clone(), result.clone());
                let progress: Arc<dyn Fn(usize, usize) + Send + Sync> = Arc::new(move |a, b| {
                    d.store(a, Ordering::Relaxed);
                    t.store(b, Ordering::Relaxed);
                });
                std::thread::Builder::new()
                    .name("lucem-pipeline-job".into())
                    .spawn(move || {
                        let out = lucem_mcp::work::run_pipeline(&target, &spec, &image, &output, None, Some(progress)).map(|done| serde_json::to_value(done).unwrap_or_default());
                        *r.lock().unwrap_or_else(|e| e.into_inner()) = Some(out);
                    })
                    .map_err(|e| e.to_string())?;
                let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
                jobs.push(Job { who: context.peer.clone(), done, total, result });
                Ok(serde_json::json!({ "job": jobs.len() - 1 }))
            }
            "job" => {
                let id = body.get("job").and_then(|v| v.as_u64()).ok_or("job: which job?")? as usize;
                let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
                // A client follows only its own jobs
                let job = jobs.get(id).filter(|j| j.who == context.peer).ok_or_else(|| format!("no job {id}"))?;
                let result = job.result.lock().unwrap_or_else(|e| e.into_inner()).clone();
                Ok(match result {
                    None => serde_json::json!({ "running": true, "done": job.done.load(Ordering::Relaxed), "total": job.total.load(Ordering::Relaxed) }),
                    Some(Ok(v)) => serde_json::json!({ "running": false, "result": v }),
                    Some(Err(e)) => serde_json::json!({ "running": false, "error": e }),
                })
            }
            other => Err(format!("the pipeline plugin has no method {other:?}")),
        }
    }
}

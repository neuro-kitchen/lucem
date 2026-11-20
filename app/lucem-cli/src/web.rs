//! What `lucem serve --web` adds for browser pages: pipelines run on the tiles a page asks for
//! (the server has the GPU; the page only draws), with the operators' halo read around each tile.

use std::sync::{Arc, Mutex};

use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::DeviceArray;
use lucem_core::{ComputeTarget, Shape};
use lucem_io::Pyramid;
use lucem_pipeline::{align_region, at_downsample, crop_and_download, evaluate_cached, halo, EvalCache, EvalTimings, PipelineSpec, Value};
use lucem_stream::engine::TileProcessor;

/// Pipelines on tiles, on one compute device of the server's.
///
/// Keeps an LRU device cache ([`EvalCache`]) of uploaded source tiles and intermediate node
/// outputs so parameter edits or op toggles reuse the prefix directly on the GPU, and crops
/// halo margins on the device before downloading only the requested tile.
pub struct PipelineTiles {
    client: Client,
    runtime: String,
    cache: Arc<EvalCache>,
    last_timings: Mutex<EvalTimings>,
}

impl PipelineTiles {
    /// On `target`'s default device; `None` (and why) when it has none.
    pub fn open(target: ComputeTarget) -> Result<Arc<Self>, String> {
        let client = target.client().map_err(|e| e.to_string())?;
        Ok(Arc::new(Self {
            client,
            runtime: target.name().to_string(),
            cache: Arc::new(EvalCache::default()),
            last_timings: Mutex::new(EvalTimings::default()),
        }))
    }

    /// The runtime it runs on, for pages (`wgpu`, `cuda`, `cpu`).
    pub fn runtime(&self) -> &str {
        &self.runtime
    }

    /// Where the last tile's time went.
    #[allow(dead_code)]
    pub fn last_timings(&self) -> EvalTimings {
        self.last_timings.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

struct ScriptTileSpec {
    path: String,
    source_channel: usize,
    input_name: String,
    params: Vec<(String, f64)>,
    enabled: bool,
    halo: usize,
    normalize: bool,
}

fn parse_script_tile(spec_json: &str) -> Option<ScriptTileSpec> {
    let v: serde_json::Value = serde_json::from_str(spec_json).ok()?;
    let n = v.get("script_node")?;
    let path = n.get("path")?.as_str()?.to_string();
    let source_channel = n.get("source_channel")?.as_u64()? as usize;
    let input_name = n.get("input_name").and_then(|s| s.as_str()).unwrap_or("image").to_string();
    let enabled = n.get("enabled").and_then(|b| b.as_bool()).unwrap_or(true);
    let halo = n.get("halo").and_then(|h| h.as_u64()).unwrap_or(0) as usize;
    let normalize = n.get("normalize").and_then(|b| b.as_bool()).unwrap_or(true);
    let mut params = Vec::new();
    if let Some(arr) = n.get("params").and_then(|p| p.as_array()) {
        for item in arr {
            if let Some(pair) = item.as_array()
                && let (Some(k), Some(val)) = (pair.first().and_then(|s| s.as_str()), pair.get(1).and_then(|f| f.as_f64()))
            {
                params.push((k.to_string(), val));
            }
        }
    }
    Some(ScriptTileSpec { path, source_channel, input_name, params, enabled, halo, normalize })
}

fn parse_pep723_deps(src: &str) -> Option<Vec<String>> {
    let mut in_block = false;
    let mut block = String::new();
    for line in src.lines() {
        let t = line.trim();
        if t == "# /// script" {
            in_block = true;
            continue;
        }
        if in_block {
            if t == "# ///" {
                break;
            }
            if let Some(rest) = t.strip_prefix('#') {
                block.push_str(rest.trim());
                block.push('\n');
            }
        }
    }
    if !in_block {
        return None;
    }
    let mut deps = Vec::new();
    if let Some(pos) = block.find("dependencies")
        && let Some(open) = block[pos..].find('[')
        && let Some(close) = block[pos + open..].find(']')
    {
        let inner = &block[pos + open + 1..pos + open + close];
        for item in inner.split(',') {
            let s = item.trim().trim_matches('"').trim_matches('\'');
            if !s.is_empty() {
                deps.push(s.to_string());
            }
        }
    }
    if !deps.iter().any(|d| d.starts_with("numpy")) {
        deps.push("numpy".to_string());
    }
    Some(deps)
}

const SERVER_WORKER_PY: &str = r#"import contextlib, hashlib, importlib.util, inspect, json, sys, traceback, types
import numpy as np

class _LucemCustomShim:
    @staticmethod
    def transform(*_a, **_kw):
        def _dec(fn):
            fn._lucem_op = True
            return fn
        return _dec

class _LucemShim:
    custom = _LucemCustomShim()
    @staticmethod
    def op(*_a, **_kw):
        def _dec(fn):
            fn._lucem_op = True
            return fn
        return _dec
    @staticmethod
    def transform(*_a, **_kw):
        def _dec(fn):
            fn._lucem_op = True
            return fn
        return _dec
    @staticmethod
    def input(*_a, **_kw):
        return lambda fn: fn
    @staticmethod
    def param(*_a, **_kw):
        return lambda fn: fn

if "lucem" not in sys.modules:
    _m = types.ModuleType("lucem")
    _m.op = _LucemShim.op
    _m.transform = _LucemShim.transform
    _m.input = _LucemShim.input
    _m.param = _LucemShim.param
    _m.custom = _LucemShim.custom
    sys.modules["lucem"] = _m

_CACHE = {}

def _load_mod(script_path):
    with open(script_path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    cached = _CACHE.get(script_path)
    if cached is not None and cached[0] == digest:
        return cached[1]
    mod_name = "lucem_srv_op_" + digest[:12]
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[script_path] = (digest, mod)
    return mod

def _find_entry(mod):
    for _, v in vars(mod).items():
        if callable(v) and getattr(v, "_lucem_op", False):
            return v
    return getattr(mod, "run", None)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        arr = np.memmap(req["in_path"], dtype=np.float32, mode="r", shape=(int(req["rh"]), int(req["rw"])))
        mod = _load_mod(req["script_path"])
        fn = _find_entry(mod)
        if fn is None:
            raise RuntimeError("script defines neither an @lucem.op function nor run(...)")
        kwargs = {req["in_name"]: arr}
        for k, v in req["params"]:
            kwargs[k] = float(v)
        sig = inspect.signature(fn)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_varkw:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        with contextlib.redirect_stdout(sys.stderr):
            res = fn(**kwargs)
        if isinstance(res, dict):
            res = next(iter(res.values()))
        out = np.ascontiguousarray(np.asarray(res, dtype=np.float32))
        out.tofile(req["out_path"])
        sys.stdout.write('{"ok":true}\n')
        sys.stdout.flush()
    except Exception:
        tb = traceback.format_exc().strip().splitlines()
        msg = "\n".join(tb[-8:])
        sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
        sys.stdout.flush()
"#;

struct ServerPythonWorker {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
}

impl Drop for ServerPythonWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl ServerPythonWorker {
    fn spawn(deps: Option<&[String]>) -> Result<Self, String> {
        let worker_dir = if std::path::Path::new("/dev/shm").is_dir() {
            std::path::PathBuf::from("/dev/shm/lucem-serve-scripts")
        } else {
            std::env::temp_dir().join("lucem-serve-scripts")
        };
        std::fs::create_dir_all(&worker_dir).map_err(|e| e.to_string())?;
        let worker_path = worker_dir.join(format!("worker-{}.py", std::process::id()));
        if !worker_path.exists() {
            std::fs::write(&worker_path, SERVER_WORKER_PY).map_err(|e| e.to_string())?;
        }
        let mut cmd = if let Some(deps) = deps {
            let mut c = std::process::Command::new("uv");
            c.args(["run", "--no-project", "--quiet"]);
            for dep in deps {
                c.args(["--with", dep]);
            }
            c.args(["python", "-u"]);
            c
        } else {
            let py = std::env::var("LUCEM_PYTHON").ok().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "python3".into());
            let mut c = std::process::Command::new(py);
            c.arg("-u");
            c
        };
        let mut child = cmd
            .current_dir(&worker_dir)
            .arg(&worker_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| format!("could not spawn warm Python worker: {e}"))?;
        let stdin = child.stdin.take().ok_or("worker stdin unavailable")?;
        let stdout = std::io::BufReader::new(child.stdout.take().ok_or("worker stdout unavailable")?);
        Ok(Self { child, stdin, stdout })
    }

    fn call(&mut self, req_json: &str) -> Result<(), String> {
        use std::io::{BufRead as _, Write as _};
        writeln!(self.stdin, "{req_json}").map_err(|e| format!("worker write failed: {e}"))?;
        self.stdin.flush().map_err(|e| format!("worker flush failed: {e}"))?;
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).map_err(|e| format!("worker read failed: {e}"))?;
        if n == 0 {
            return Err("warm Python worker exited unexpectedly".into());
        }
        let resp: serde_json::Value = serde_json::from_str(line.trim()).map_err(|e| format!("worker reply parse error: {e} ({line})"))?;
        if resp.get("ok").and_then(|v| v.as_bool()) == Some(true) {
            Ok(())
        } else {
            Err(resp.get("error").and_then(|v| v.as_str()).unwrap_or("Python script error").to_string())
        }
    }
}

fn server_worker_pool() -> &'static Mutex<std::collections::HashMap<String, ServerPythonWorker>> {
    static POOL: std::sync::OnceLock<Mutex<std::collections::HashMap<String, ServerPythonWorker>>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn run_script_tile(image: &Pyramid, spec: &ScriptTileSpec, level: usize, z: usize, tx: usize, ty: usize) -> Result<(usize, usize, Vec<f32>), String> {
    let l = image.levels.get(level).ok_or_else(|| format!("no level {level}"))?.clone();
    let (x0, y0) = (tx * l.tile_w, ty * l.tile_h);
    if x0 >= l.width || y0 >= l.height || z >= l.depth {
        return Err(format!("tile {tx},{ty} (plane {z}) is outside level {level}"));
    }
    let (w, h) = ((l.width - x0).min(l.tile_w), (l.height - y0).min(l.tile_h));
    let reach = if spec.enabled { spec.halo } else { 0 };
    let rx0 = x0.saturating_sub(reach);
    let ry0 = y0.saturating_sub(reach);
    let rw = (x0 + w + reach).min(l.width) - rx0;
    let rh = (y0 + h + reach).min(l.height) - ry0;
    let mut values = image.read_region_at(level, spec.source_channel, z, rx0, ry0, rw, rh).map_err(|e| e.to_string())?;
    let top = image.channels.get(spec.source_channel).map_or(1.0, |c| c.range.1);
    if spec.normalize && top > 1.0 {
        let k = 1.0 / top;
        values.iter_mut().for_each(|v| *v *= k);
    }
    if !spec.enabled {
        return Ok((w, h, values));
    }
    // RAM-backed /dev/shm on Linux for zero-disk-I/O transfer to the persistent warm worker
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let base = if std::path::Path::new("/dev/shm").is_dir() {
        std::path::PathBuf::from("/dev/shm/lucem-serve-scripts")
    } else {
        std::env::temp_dir().join("lucem-serve-scripts")
    };
    let dir = base.join(format!("tile-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let in_path = dir.join("in.bin");
    let out_path = dir.join("out.bin");
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&in_path, &raw_bytes).map_err(|e| e.to_string())?;
    let script_src = std::fs::read_to_string(&spec.path).map_err(|e| format!("{}: {e}", spec.path))?;
    let deps = parse_pep723_deps(&script_src);
    let pool_key = match &deps {
        Some(d) => format!("uv:{}", d.join(",")),
        None => std::env::var("LUCEM_PYTHON").ok().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "python3".into()),
    };
    let req_json = serde_json::json!({
        "script_path": spec.path,
        "in_path": in_path.to_string_lossy(),
        "out_path": out_path.to_string_lossy(),
        "rh": rh,
        "rw": rw,
        "in_name": spec.input_name,
        "params": spec.params,
    })
    .to_string();
    let call_res = {
        let mut pool = server_worker_pool().lock().unwrap_or_else(|e| e.into_inner());
        if !pool.contains_key(&pool_key) {
            let w = ServerPythonWorker::spawn(deps.as_deref())?;
            pool.insert(pool_key.clone(), w);
        }
        let first = pool.get_mut(&pool_key).expect("worker present").call(&req_json);
        match first {
            Ok(()) => Ok(()),
            Err(e) if e.contains("worker") => {
                // Worker exited or pipe broke: respawn once transparently
                pool.remove(&pool_key);
                let mut fresh = ServerPythonWorker::spawn(deps.as_deref())?;
                let retry = fresh.call(&req_json);
                if retry.is_ok() {
                    pool.insert(pool_key, fresh);
                }
                retry
            }
            Err(e) => Err(e),
        }
    };
    if let Err(e) = call_res {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(e);
    }
    let out_bytes = std::fs::read(&out_path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&dir);
    let full: Vec<f32> = out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    if full.len() != rw * rh {
        return Err(format!("script returned {} pixels, expected {} ({rw}×{rh})", full.len(), rw * rh));
    }
    // Crop halo back to the requested w x h tile
    let ox = x0 - rx0;
    let oy = y0 - ry0;
    let mut cropped = Vec::with_capacity(w * h);
    for y in 0..h {
        let start = (oy + y) * rw + ox;
        cropped.extend_from_slice(&full[start..start + w]);
    }
    Ok((w, h, cropped))
}

impl TileProcessor for PipelineTiles {
    fn tile(&self, image: &Pyramid, spec_json: &str, level: usize, z: usize, tx: usize, ty: usize) -> Result<(usize, usize, Vec<f32>), String> {
        if let Some(spec) = parse_script_tile(spec_json) {
            return run_script_tile(image, &spec, level, z, tx, ty);
        }
        let spec: PipelineSpec = serde_json::from_str(spec_json).map_err(|e| format!("the pipeline: {e}"))?;
        spec.validate()?;
        let l = image.levels.get(level).ok_or_else(|| format!("no level {level}"))?.clone();
        let (x0, y0) = (tx * l.tile_w, ty * l.tile_h);
        if x0 >= l.width || y0 >= l.height || z >= l.depth {
            return Err(format!("tile {tx},{ty} (plane {z}) is outside level {level}"));
        }
        let (w, h) = ((l.width - x0).min(l.tile_w), (l.height - y0).min(l.tile_h));
        // The operators' reach, in this level's pixels, rounded out to READ_GRID so small halo
        // changes from editing a parameter reuse the already-uploaded raw region and prefix nodes.
        let plan = at_downsample(&spec.plan(), l.downsample);
        let reach = halo(&plan, &spec.target);
        let (rx0, ry0, rw, rh) = align_region(
            x0.saturating_sub(reach),
            y0.saturating_sub(reach),
            (x0 + w + reach).min(l.width),
            (y0 + h + reach).min(l.height),
            l.width,
            l.height,
        );
        let image_id = image.path.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| format!("{:p}", image as *const Pyramid));
        let region_tag = format!("{image_id}|L{level}|Z{z}|{rx0},{ry0},{rw},{rh}|norm:{}", spec.normalize);
        let client = &self.client;
        let (value, mut timings) = evaluate_cached(client, &plan, &spec.target, &region_tag, &self.cache, &mut |id: &str| {
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
        // Crop halo on the device so only the requested w × h tile crosses back to the host.
        let (out, readback_ms) = crop_and_download(client, &value, rx0, ry0, rw, rh, x0, y0, w, h)
            .map_err(|e| if e.contains("table") { "a table is not a tile: its rows are a plugin call".to_string() } else { e })?;
        timings.readback_ms = readback_ms;
        if std::env::var_os("LUCEM_PREVIEW_TIMING").is_some() {
            eprintln!(
                "[lucem-tile] L{level} ({tx},{ty}) {w}x{h} (read {rw}x{rh}): src={:.2}ms compute={:.2}ms readback={:.2}ms hits={} fused={} reused={}",
                timings.source_ms, timings.compute_ms, timings.readback_ms, timings.cache_hits, timings.fused_runs, timings.input_reused
            );
        }
        *self.last_timings.lock().unwrap_or_else(|e| e.into_inner()) = timings;
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

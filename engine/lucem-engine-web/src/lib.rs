//! lucem's engine in a browser worker: what `lucem serve` is to a page served by a server, this is
//! to a page that runs alone (GitHub Pages). It runs in a dedicated web worker as a module of its
//! own, built without threads, so the browser objects it holds (the picked `File`s) stay on its
//! one thread while the page draws on its own threads. (As dsp-kitchen's `dspk-engine-web`.)
//!
//! - **Files.** The page hands over the `File`s the user picked, by their path inside the folder
//!   picked; nothing is read until asked: an OME-Zarr chunk is read when a tile needs it, with
//!   the worker's synchronous reader (`FileReaderSync`: workers only), through
//!   [`lucem_io::ZarrStore`]; a plain image (PNG, TIFF, JPEG) is read whole once.
//! - **Requests.** The page sends the same protobuf requests it sends a server (list, describe,
//!   tile) and gets the same answers; the page's side is `lucem_stream::web_client`
//!   (`start_worker`).
//! - **Processing** (operators on tiles, histology, models) needs a server or the desktop app for
//!   now: running operators here needs reading results back from WebGPU asynchronously, which
//!   lucem-core does not do yet. The answer says so, and the page shows where it runs.
#![cfg(target_family = "wasm")]

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::DeviceArray;
use lucem_core::{ComputeTarget, Shape};
use lucem_io::{Pyramid, ZarrStore};
use lucem_pipeline::{align_region, at_downsample, crop_and_download_async, evaluate_cached, halo, EvalCache, PipelineSpec, Value};
use lucem_stream::wire;
use prost::Message as _;
use wasm_bindgen::prelude::*;

/// Why ML model plugins / server-only plugins do not run in the standalone browser worker.
const NO_PLUGINS: &str = "ML model plugins and server scripts require the desktop app (`lucem app`) or `lucem serve --web`";

/// A picked file, read whole with the worker's synchronous reader.
struct FileBytes(web_sys::File);

// SAFETY: this module is built without threads (no shared memory, no atomics): its one thread is
// the worker's, so the browser object is never touched from another thread.
unsafe impl Send for FileBytes {}
unsafe impl Sync for FileBytes {}

impl FileBytes {
    fn read(&self) -> std::io::Result<Vec<u8>> {
        let fail = |e: JsValue| std::io::Error::other(format!("{}: {e:?}", self.0.name()));
        let buffer = web_sys::FileReaderSync::new().map_err(fail)?.read_as_array_buffer(&self.0).map_err(fail)?;
        Ok(js_sys::Uint8Array::new(&buffer).to_vec())
    }
}

/// The picked files under one OME-Zarr folder, as a Zarr store (keys relative to it).
struct PickedStore(HashMap<String, FileBytes>);

impl ZarrStore for PickedStore {
    fn get(&self, key: &str) -> std::io::Result<Option<Vec<u8>>> {
        self.0.get(key).map(FileBytes::read).transpose()
    }
}

/// The site's samples (`dist/samples/`, next to `engine/`), read a key at a time with a
/// synchronous request (a worker may make one): only the chunks a tile needs are fetched.
struct SiteStore(String);

impl ZarrStore for SiteStore {
    fn get(&self, key: &str) -> std::io::Result<Option<Vec<u8>>> {
        let fail = |e: JsValue| std::io::Error::other(format!("{}/{key}: {e:?}", self.0));
        let xhr = web_sys::XmlHttpRequest::new().map_err(fail)?;
        xhr.open_with_async("GET", &format!("{}/{key}", self.0), false).map_err(fail)?;
        xhr.set_response_type(web_sys::XmlHttpRequestResponseType::Arraybuffer);
        xhr.send().map_err(fail)?;
        match xhr.status().map_err(fail)? {
            200 => Ok(Some(js_sys::Uint8Array::new(&xhr.response().map_err(fail)?).to_vec())),
            404 => Ok(None),
            s => Err(std::io::Error::other(format!("{}/{key}: HTTP {s}", self.0))),
        }
    }
}

/// The site's samples (`samples/index.json`), read once.
fn site_samples() -> Vec<String> {
    SiteStore("../samples".into()).get("index.json").ok().flatten().and_then(|b| serde_json::from_slice::<Vec<String>>(&b).ok()).unwrap_or_default()
}

/// Whether a picked path's first component names an image the engine opens: an OME-Zarr folder
/// (`.zarr`), or a plain image file.
fn image_kind(name: &str) -> Option<bool> {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".zarr") {
        return Some(true);
    }
    [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"].iter().any(|e| lower.ends_with(e)).then_some(false)
}

/// Runs the module's static constructors (CubeCL's IR registries, through pliron and inventory):
/// once, called by the worker right after loading. A module that exports `_initialize` is a
/// "reactor" to the linker; without it the linker wraps every export to run them again on each
/// call, which cost seconds per request (found in dsp-kitchen's engine, 29 s of 31 while opening a
/// 19,754-file recording, 2026-11-18; lucem's links the same CubeCL).
#[unsafe(no_mangle)]
pub extern "C" fn _initialize() {
    unsafe extern "C" {
        fn __wasm_call_ctors();
    }
    // SAFETY: called once, before anything else in the module runs
    unsafe { __wasm_call_ctors() }
}

/// The engine of one page: the files it was given, opened images, and the browser's WebGPU compute client.
struct Inner {
    /// The files picked, by their path ("<folder picked>/<path inside>", or a file's name): a map,
    /// so adding an OME-Zarr's thousands of chunk files is not quadratic, and a folder's files are
    /// one range of it.
    files: BTreeMap<String, web_sys::File>,
    /// Opened images, by path.
    open: HashMap<String, Arc<Pyramid>>,
    /// Browser WebGPU CubeCL client once opened by [`Engine::start`].
    client: Option<Client>,
    /// LRU cache of device-resident pipeline inputs and intermediates.
    cache: Arc<EvalCache>,
    /// Why GPU compute is unavailable, if WebGPU did not open.
    no_compute: Option<String>,
}

impl Default for Inner {
    fn default() -> Self {
        Self {
            files: BTreeMap::new(),
            open: HashMap::new(),
            client: None,
            cache: Arc::new(EvalCache::default()),
            no_compute: Some("WebGPU not opened yet".into()),
        }
    }
}

impl Inner {
    /// The images among the files picked: each OME-Zarr folder (at any depth, once) and each
    /// plain image file.
    fn list(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for path in self.files.keys() {
            let parts: Vec<&str> = path.split('/').collect();
            // The first component naming an image is the image
            for i in 0..parts.len() {
                match image_kind(parts[i]) {
                    Some(true) if i + 1 < parts.len() => {
                        let folder = parts[..=i].join("/");
                        if !out.contains(&folder) {
                            out.push(folder);
                        }
                        break;
                    }
                    Some(false) if i + 1 == parts.len() => {
                        out.push(path.clone());
                        break;
                    }
                    _ => {}
                }
            }
        }
        out.sort();
        out.extend(site_samples().into_iter().map(|s| format!("samples/{s}")));
        out
    }

    fn open(&mut self, path: &str) -> Result<Arc<Pyramid>, String> {
        if let Some(p) = self.open.get(path) {
            return Ok(p.clone());
        }
        // The site's samples: read from it, chunk by chunk
        if let Some(sample) = path.strip_prefix("samples/") {
            if sample.contains("..") || sample.starts_with('.') {
                return Err("that path is not allowed".into());
            }
            let name = sample.trim_end_matches(".ome.zarr").to_string();
            let p = Arc::new(Pyramid::open_store(&name, Arc::new(SiteStore(format!("../samples/{sample}")))).map_err(|e| format!("{path}: {e}"))?);
            self.open.insert(path.to_string(), p.clone());
            return Ok(p);
        }
        let name = path.rsplit('/').next().unwrap_or(path).to_string();
        let pyramid = match image_kind(&name) {
            Some(true) => {
                let prefix = format!("{}/", path.trim_end_matches('/'));
                let files: HashMap<String, FileBytes> = self.files.range(prefix.clone()..).map_while(|(p, f)| p.strip_prefix(&prefix).map(|key| (key.to_string(), FileBytes(f.clone())))).collect();
                if files.is_empty() {
                    return Err(format!("{path} was not among the folders picked"));
                }
                Pyramid::open_store(&name, Arc::new(PickedStore(files))).map_err(|e| format!("{path}: {e}"))?
            }
            Some(false) => {
                let file = self.files.get(path).map(|f| FileBytes(f.clone())).ok_or_else(|| format!("{path} was not among the files picked"))?;
                let bytes = file.read().map_err(|e| e.to_string())?;
                Pyramid::from_image_bytes(&name, &bytes).map_err(|e| format!("{path}: {e}"))?
            }
            None => return Err(format!("{path} is not an image lucem opens here (an OME-Zarr folder, PNG, TIFF, JPEG)")),
        };
        let pyramid = Arc::new(pyramid);
        self.open.insert(path.to_string(), pyramid.clone());
        Ok(pyramid)
    }

    async fn answer_kind(inner: &RefCell<Inner>, request: wire::Request) -> Result<wire::response::Kind, String> {
        use wire::request::Kind as Ask;
        use wire::response::Kind as Say;
        Ok(match request.kind.ok_or("an empty request")? {
            Ask::List(_) => Say::Images(wire::ImageList { paths: inner.borrow().list() }),
            Ask::Info(i) => Say::Info(wire::info_to_wire(&lucem_stream::protocol::describe(inner.borrow_mut().open(&i.path)?.as_ref()))),
            Ask::Tile(t) => {
                if let Some(proc) = t.processing.as_ref() {
                    let spec_json = proc.spec_json.as_str();
                    let (p, client, cache, no_compute) = {
                        let mut me = inner.borrow_mut();
                        let p = me.open(&t.path)?;
                        (p, me.client.clone(), me.cache.clone(), me.no_compute.clone())
                    };
                    if let Some(why) = lucem_core::compute::browser_gpu_lost() {
                        return Err(format!("the browser took its GPU back ({why}): reload the page to process again"));
                    }
                    let Some(client) = client else {
                        return Err(format!("processing needs WebGPU: {}", no_compute.unwrap_or_else(|| "no WebGPU adapter".into())));
                    };
                    let (w, h, data) = run_pipeline_tile_async(&client, &cache, &p, spec_json, t.level as usize, t.z as usize, t.tx as usize, t.ty as usize).await?;
                    return Ok(Say::Tile(wire::encode_tile(w, h, &data, "float32")));
                }
                let p = inner.borrow_mut().open(&t.path)?;
                let tile = p.read_tile_at(t.level as usize, t.channel as usize, t.z as usize, t.tx as usize, t.ty as usize).map_err(|e| e.to_string())?;
                Say::Tile(wire::encode_tile(tile.width, tile.height, &tile.data, &p.dtype))
            }
            Ask::Plane(q) => {
                let p = inner.borrow_mut().open(&q.path)?;
                let axis = [lucem_io::PlaneAxis::Z, lucem_io::PlaneAxis::Y, lucem_io::PlaneAxis::X].get(q.axis as usize).copied().ok_or("no such axis")?;
                let plane = p.read_plane(q.level as usize, q.channel as usize, axis, q.index as usize).map_err(|e| e.to_string())?;
                Say::Tile(wire::encode_tile(plane.width, plane.height, &plane.data, &p.dtype))
            }
            Ask::Plugin(_) => return Err(NO_PLUGINS.into()),
        })
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_pipeline_tile_async(
    client: &Client,
    cache: &EvalCache,
    image: &Pyramid,
    spec_json: &str,
    level: usize,
    z: usize,
    tx: usize,
    ty: usize,
) -> Result<(usize, usize, Vec<f32>), String> {
    if spec_json.contains("\"script_node\"") {
        return Err("Python/Rust script nodes require the desktop app (`lucem app`) or `lucem serve --web`".into());
    }
    let spec: PipelineSpec = serde_json::from_str(spec_json).map_err(|e| format!("the pipeline: {e}"))?;
    spec.validate()?;
    let l = image.levels.get(level).ok_or_else(|| format!("no level {level}"))?.clone();
    let (x0, y0) = (tx * l.tile_w, ty * l.tile_h);
    if x0 >= l.width || y0 >= l.height || z >= l.depth {
        return Err(format!("tile {tx},{ty} (plane {z}) is outside level {level}"));
    }
    let (w, h) = ((l.width - x0).min(l.tile_w), (l.height - y0).min(l.tile_h));
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
    let (value, _) = evaluate_cached(client, &plan, &spec.target, &region_tag, cache, &mut |id: &str| {
        let channel = *spec.sources.get(id).ok_or_else(|| format!("{id} is neither a source nor a node"))?;
        let mut values = image.read_region_at(level, channel, z, rx0, ry0, rw, rh).map_err(|e| e.to_string())?;
        let top = image.channels.get(channel).map_or(1.0, |c| c.range.1);
        if spec.normalize && top > 1.0 {
            let k = 1.0 / top;
            values.iter_mut().for_each(|v| *v *= k);
        }
        DeviceArray::from_host(client, Shape::new([rh, rw]), values).map(Value::Image).map_err(|e| e.to_string())
    })?;
    let (out, _) = crop_and_download_async(client, &value, rx0, ry0, rw, rh, x0, y0, w, h).await?;
    Ok((w, h, out))
}

/// The engine, as the worker script holds it (`engine-worker.js`); it answers one request at a
/// time (the script queues them).
#[wasm_bindgen]
pub struct Engine {
    inner: Rc<RefCell<Inner>>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Engine {
        std::panic::set_hook(Box::new(|info| console_error(&info.to_string())));
        Engine { inner: Rc::new(RefCell::new(Inner::default())) }
    }

    /// Opens the browser's WebGPU device for computing. Resolves to `""` when ops can run, else why not.
    pub fn start(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let why = match lucem_core::compute::open_browser_gpu().await {
                Ok(()) => match ComputeTarget::Wgpu.client() {
                    Ok(client) => {
                        let mut me = inner.borrow_mut();
                        me.client = Some(client);
                        me.no_compute = None;
                        String::new()
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        inner.borrow_mut().no_compute = Some(msg.clone());
                        msg
                    }
                },
                Err(e) => {
                    inner.borrow_mut().no_compute = Some(e.clone());
                    e
                }
            };
            Ok(JsValue::from_str(&why))
        })
    }

    /// Files the user picked, as `{path, file}`, added to earlier ones (a file of the same path
    /// replaces the earlier one; images opened before are read again from the new files).
    pub fn set_files(&self, files: js_sys::Array) {
        let mut inner = self.inner.borrow_mut();
        for entry in files.iter() {
            let path = js_sys::Reflect::get(&entry, &"path".into()).ok().and_then(|p| p.as_string());
            let file = js_sys::Reflect::get(&entry, &"file".into()).ok().and_then(|f| f.dyn_into::<web_sys::File>().ok());
            if let (Some(path), Some(file)) = (path, file) {
                inner.files.insert(path, file);
            }
        }
        inner.open.clear();
        inner.cache.clear();
    }

    /// Answers a protobuf `Request` with a protobuf `Response` (an error is a `Response` too).
    pub fn answer(&self, request: Vec<u8>) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let response = match wire::Request::decode(request.as_slice()) {
                Ok(request) => match Inner::answer_kind(&inner, request).await {
                    Ok(kind) => wire::Response { kind: Some(kind) },
                    Err(reason) => wire::error(reason),
                },
                Err(e) => wire::error(format!("a request that could not be read: {e}")),
            };
            let bytes: JsValue = js_sys::Uint8Array::from(response.encode_to_vec().as_slice()).into();
            Ok(bytes)
        })
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
}

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

use std::collections::HashMap;
use std::sync::Arc;

use lucem_io::{Pyramid, ZarrStore};
use lucem_stream::wire;
use prost::Message as _;
use wasm_bindgen::prelude::*;

/// Why processing does not run in the page's own engine.
const NO_PROCESSING: &str = "processing needs the container or the desktop app (this page views images only)";

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

/// The engine of one page: the files it was given and the images it opened.
#[derive(Default)]
struct Inner {
    /// The files picked, by their path ("<folder picked>/<path inside>", or a file's name).
    files: Vec<(String, web_sys::File)>,
    /// Opened images, by path.
    open: HashMap<String, Arc<Pyramid>>,
}

impl Inner {
    /// The images among the files picked: each OME-Zarr folder (at any depth, once) and each
    /// plain image file.
    fn list(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (path, _) in &self.files {
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
                let files: HashMap<String, FileBytes> = self.files.iter().filter_map(|(p, f)| p.strip_prefix(&prefix).map(|key| (key.to_string(), FileBytes(f.clone())))).collect();
                if files.is_empty() {
                    return Err(format!("{path} was not among the folders picked"));
                }
                Pyramid::open_store(&name, Arc::new(PickedStore(files))).map_err(|e| format!("{path}: {e}"))?
            }
            Some(false) => {
                let file = self.files.iter().find(|(p, _)| p == path).map(|(_, f)| FileBytes(f.clone())).ok_or_else(|| format!("{path} was not among the files picked"))?;
                let bytes = file.read().map_err(|e| e.to_string())?;
                Pyramid::from_image_bytes(&name, &bytes).map_err(|e| format!("{path}: {e}"))?
            }
            None => return Err(format!("{path} is not an image lucem opens here (an OME-Zarr folder, PNG, TIFF, JPEG)")),
        };
        let pyramid = Arc::new(pyramid);
        self.open.insert(path.to_string(), pyramid.clone());
        Ok(pyramid)
    }

    fn answer(&mut self, request: wire::Request) -> wire::Response {
        use wire::request::Kind as Ask;
        use wire::response::Kind as Say;
        let answer = || -> Result<Say, String> {
            Ok(match request.kind.ok_or("an empty request")? {
                Ask::List(_) => Say::Images(wire::ImageList { paths: self.list() }),
                Ask::Info(i) => Say::Info(wire::info_to_wire(&lucem_stream::protocol::describe(self.open(&i.path)?.as_ref()))),
                Ask::Tile(t) => {
                    if t.processing.is_some() {
                        return Err(NO_PROCESSING.into());
                    }
                    let p = self.open(&t.path)?;
                    let tile = p.read_tile_at(t.level as usize, t.channel as usize, t.z as usize, t.tx as usize, t.ty as usize).map_err(|e| e.to_string())?;
                    Say::Tile(wire::encode_tile(tile.width, tile.height, &tile.data, &p.dtype))
                }
                Ask::Plane(q) => {
                    let p = self.open(&q.path)?;
                    let axis = [lucem_io::PlaneAxis::Z, lucem_io::PlaneAxis::Y, lucem_io::PlaneAxis::X].get(q.axis as usize).copied().ok_or("no such axis")?;
                    let plane = p.read_plane(q.level as usize, q.channel as usize, axis, q.index as usize).map_err(|e| e.to_string())?;
                    Say::Tile(wire::encode_tile(plane.width, plane.height, &plane.data, &p.dtype))
                }
                Ask::Plugin(_) => return Err(NO_PROCESSING.into()),
            })
        };
        match answer() {
            Ok(kind) => wire::Response { kind: Some(kind) },
            Err(reason) => wire::error(reason),
        }
    }
}

/// The engine, as the worker script holds it (`engine-worker.js`); it answers one request at a
/// time (the script queues them).
#[wasm_bindgen]
pub struct Engine {
    inner: std::cell::RefCell<Inner>,
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
        Engine { inner: std::cell::RefCell::new(Inner::default()) }
    }

    /// Why processing cannot run here ("" when it can): it cannot yet.
    pub fn start(&self) -> String {
        NO_PROCESSING.to_string()
    }

    /// Files the user picked, as `{path, file}`, added to earlier ones (a file of the same path
    /// replaces the earlier one; images opened before are read again from the new files).
    pub fn set_files(&self, files: js_sys::Array) {
        let mut inner = self.inner.borrow_mut();
        for entry in files.iter() {
            let path = js_sys::Reflect::get(&entry, &"path".into()).ok().and_then(|p| p.as_string());
            let file = js_sys::Reflect::get(&entry, &"file".into()).ok().and_then(|f| f.dyn_into::<web_sys::File>().ok());
            if let (Some(path), Some(file)) = (path, file) {
                inner.files.retain(|(p, _)| *p != path);
                inner.files.push((path, file));
            }
        }
        inner.open.clear();
    }

    /// Answers a protobuf `Request` with a protobuf `Response` (an error is a `Response` too).
    pub fn answer(&self, request: Vec<u8>) -> js_sys::Promise {
        let response = match wire::Request::decode(request.as_slice()) {
            Ok(request) => self.inner.borrow_mut().answer(request),
            Err(e) => wire::error(format!("a request that could not be read: {e}")),
        };
        let bytes: JsValue = js_sys::Uint8Array::from(response.encode_to_vec().as_slice()).into();
        js_sys::Promise::resolve(&bytes)
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
}

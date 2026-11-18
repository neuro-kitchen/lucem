//! What a server answers, without its transport: the data folder's images (checked paths, kept
//! open) and each request's answer. QUIC ([`crate::server`]) calls it today; the browser endpoint
//! (WebSocket) calls the same, so both answer alike.

use std::collections::{HashMap, VecDeque};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use lucem_io::Pyramid;

use crate::protocol::{Request, Response};

/// Images kept open.
const OPEN_IMAGES: usize = 64;
/// How deep the listing looks under the data folder.
const LIST_DEPTH: usize = 4;

type OpenImages = (HashMap<PathBuf, Arc<Pyramid>>, VecDeque<PathBuf>);

pub struct Images {
    pub(crate) root: PathBuf,
    /// The app's samples (`lucem samples`), offered as `samples/<id>.ome.zarr` next to the data.
    samples: Option<PathBuf>,
    open: Mutex<OpenImages>,
}

/// Whether the path names an image the server may open.
fn is_image_name(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    name.ends_with(".zarr") || name.ends_with(".tif") || name.ends_with(".tiff")
}

impl Images {
    /// The images under `root` (already canonical).
    pub fn new(root: PathBuf) -> Self {
        Self { root, samples: None, open: Mutex::new((HashMap::new(), VecDeque::new())) }
    }

    /// Offers the samples in `dir` (`lucem samples`) as `samples/…`, so a page opens them as
    /// files instead of generating them.
    pub fn with_samples(mut self, dir: Option<PathBuf>) -> Self {
        self.samples = dir.and_then(|d| d.canonicalize().ok());
        self
    }

    /// The data folder.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The samples folder, when samples are offered.
    pub fn samples(&self) -> Option<&Path> {
        self.samples.as_deref()
    }

    /// `relative`, checked: inside the data folder, nothing hidden, an image.
    pub fn resolve(&self, relative: &str) -> Result<PathBuf, String> {
        let path = Path::new(relative);
        let plain = path.components().all(|c| matches!(c, Component::Normal(n) if !n.to_string_lossy().starts_with('.')));
        if relative.is_empty() || !plain {
            return Err("that path is not allowed".into());
        }
        if !is_image_name(path) {
            return Err("that is not an image".into());
        }
        // The samples, when offered (the same checks: plain names, inside their folder)
        let (base, rest) = match (self.samples.as_ref(), path.strip_prefix("samples")) {
            (Some(dir), Ok(rest)) if !rest.as_os_str().is_empty() => (dir, rest),
            _ => (&self.root, path),
        };
        let full = base.join(rest).canonicalize().map_err(|_| "no such image".to_string())?;
        if !full.starts_with(base) {
            return Err("that path is not allowed".into());
        }
        Ok(full)
    }

    pub fn open(&self, relative: &str) -> Result<Arc<Pyramid>, String> {
        let full = self.resolve(relative)?;
        if let Some(p) = self.open.lock().map_err(|e| e.to_string())?.0.get(&full) {
            return Ok(p.clone());
        }
        let pyramid = Arc::new(Pyramid::open(&full).map_err(|e| e.to_string())?);
        let mut open = self.open.lock().map_err(|e| e.to_string())?;
        if open.0.insert(full.clone(), pyramid.clone()).is_none() {
            open.1.push_back(full);
        }
        while open.1.len() > OPEN_IMAGES {
            if let Some(old) = open.1.pop_front() {
                open.0.remove(&old);
            }
        }
        Ok(pyramid)
    }

    pub fn list(&self) -> Vec<String> {
        let mut out = Vec::new();
        walk(&self.root, &self.root, 0, &mut out);
        out.sort();
        if let Some(dir) = &self.samples {
            let mut samples = Vec::new();
            walk(dir, dir, 0, &mut samples);
            samples.sort();
            out.extend(samples.into_iter().map(|s| format!("samples/{s}")));
        }
        out
    }
}

fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let path = e.path();
        if e.file_name().to_string_lossy().starts_with('.') || e.file_type().map_or(true, |t| t.is_symlink()) {
            continue;
        }
        if is_image_name(&path) {
            if let Ok(r) = path.strip_prefix(root) {
                out.push(r.to_string_lossy().into_owned());
            }
        } else if path.is_dir() && depth < LIST_DEPTH {
            walk(root, &path, depth + 1, out);
        }
    }
}

pub use crate::protocol::describe;

/// The answer to `request` (blocking: reads files, runs plugins): the response, and a tile's
/// values to follow it.
pub fn work(images: &Images, request: Request, plugins: &[Arc<dyn crate::plugin::ServerPlugin>], context: &crate::plugin::PluginContext) -> Result<(Response, Option<Vec<f32>>), String> {
    Ok(match request {
        Request::Plugin { plugin, method, body } => {
            let p = plugins.iter().find(|p| p.id() == plugin).ok_or_else(|| format!("this server has no plugin {plugin:?}"))?;
            (Response::Plugin { body: p.call(&method, body, context)? }, None)
        }
        Request::Pair { .. } => return Err("pairing is answered before this".into()),
        Request::List => (Response::Images { paths: images.list() }, None),
        Request::Info { path } => (Response::Info { info: describe(images.open(&path)?.as_ref()) }, None),
        Request::Tile { path, level, channel, z, tx, ty } => {
            let p = images.open(&path)?;
            if channel >= p.channels.len() {
                return Err(format!("no channel {channel}"));
            }
            let t = p.read_tile_at(level, channel, z, tx, ty).map_err(|e| e.to_string())?;
            (Response::Tile { width: t.width, height: t.height }, Some(t.data))
        }
    })
}


/// Runs a pipeline on one tile on the server (a page asked for a processed tile): registered by
/// the program that serves (`lucem serve`), which has the compute devices; this crate has none.
pub trait TileProcessor: Send + Sync {
    /// Tile `(tx, ty)` of plane `z` at `level` of `image`, processed by the pipeline in
    /// `spec_json` (`lucem_pipeline::PipelineSpec`), its halo read around it: `(width, height,
    /// values)`, the tile's size.
    fn tile(&self, image: &Pyramid, spec_json: &str, level: usize, z: usize, tx: usize, ty: usize) -> Result<(usize, usize, Vec<f32>), String>;
}

/// The answer to a page's request (the browser protocol, [`crate::wire`]), as [`work`] answers a
/// computer's: the same checks, the same images, the same plugins.
pub fn answer_wire(images: &Images, plugins: &[Arc<dyn crate::plugin::ServerPlugin>], processor: Option<&dyn TileProcessor>, request: crate::wire::Request, context: &crate::plugin::PluginContext) -> crate::wire::Response {
    use crate::wire::{request::Kind as Ask, response::Kind as Say};
    let answer = || -> Result<Say, String> {
        Ok(match request.kind.ok_or("an empty request")? {
            Ask::List(_) => Say::Images(crate::wire::ImageList { paths: images.list() }),
            Ask::Info(i) => Say::Info(crate::wire::info_to_wire(&describe(images.open(&i.path)?.as_ref()))),
            Ask::Tile(t) => {
                let p = images.open(&t.path)?;
                let (level, channel, z, tx, ty) = (t.level as usize, t.channel as usize, t.z as usize, t.tx as usize, t.ty as usize);
                if channel >= p.channels.len() {
                    return Err(format!("no channel {channel}"));
                }
                match t.processing {
                    None => {
                        let tile = p.read_tile_at(level, channel, z, tx, ty).map_err(|e| e.to_string())?;
                        Say::Tile(crate::wire::encode_tile(tile.width, tile.height, &tile.data, &p.dtype))
                    }
                    Some(processing) => {
                        let processor = processor.ok_or("this server does not process tiles")?;
                        let (w, h, values) = processor.tile(&p, &processing.spec_json, level, z, tx, ty)?;
                        Say::Tile(crate::wire::encode_tile(w, h, &values, "float32"))
                    }
                }
            }
            Ask::Plane(q) => {
                let p = images.open(&q.path)?;
                if q.channel as usize >= p.channels.len() {
                    return Err(format!("no channel {}", q.channel));
                }
                let axis = match q.axis {
                    0 => lucem_io::PlaneAxis::Z,
                    1 => lucem_io::PlaneAxis::Y,
                    2 => lucem_io::PlaneAxis::X,
                    a => return Err(format!("no axis {a} (0: XY, 1: XZ, 2: YZ)")),
                };
                let plane = p.read_plane(q.level as usize, q.channel as usize, axis, q.index as usize).map_err(|e| e.to_string())?;
                Say::Tile(crate::wire::encode_tile(plane.width, plane.height, &plane.data, &p.dtype))
            }
            Ask::Plugin(call) => {
                let p = plugins.iter().find(|p| p.id() == call.plugin).ok_or_else(|| format!("this server has no plugin {:?}", call.plugin))?;
                let body: serde_json::Value = if call.body.is_empty() { serde_json::Value::Null } else { serde_json::from_slice(&call.body).map_err(|e| format!("the call's body: {e}"))? };
                let reply = p.call(&call.method, body, context)?;
                Say::Plugin(crate::wire::PluginReply { body: serde_json::to_vec(&reply).map_err(|e| e.to_string())? })
            }
        })
    };
    match answer() {
        Ok(kind) => crate::wire::Response { kind: Some(kind) },
        Err(reason) => crate::wire::error(reason),
    }
}

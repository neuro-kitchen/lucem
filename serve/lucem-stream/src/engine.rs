//! What a server answers, without its transport: the data folder's images (checked paths, kept
//! open) and each request's answer. QUIC ([`crate::server`]) calls it today; the browser endpoint
//! (WebSocket) calls the same, so both answer alike.

use std::collections::{HashMap, VecDeque};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use lucem_io::Pyramid;

use crate::protocol::{ImageInfo, Request, Response};

/// Images kept open.
const OPEN_IMAGES: usize = 64;
/// How deep the listing looks under the data folder.
const LIST_DEPTH: usize = 4;

type OpenImages = (HashMap<PathBuf, Arc<Pyramid>>, VecDeque<PathBuf>);

pub struct Images {
    pub(crate) root: PathBuf,
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
        Self { root, open: Mutex::new((HashMap::new(), VecDeque::new())) }
    }

    /// The data folder.
    pub fn root(&self) -> &Path {
        &self.root
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
        let full = self.root.join(path).canonicalize().map_err(|_| "no such image".to_string())?;
        if !full.starts_with(&self.root) {
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

pub fn describe(p: &Pyramid) -> ImageInfo {
    ImageInfo {
        name: p.name.clone(),
        dtype: p.dtype.clone(),
        channels: p.channels.iter().map(|c| c.label.clone()).collect(),
        pixel_size_um: p.pixel_size_um,
        z_spacing_um: p.z_spacing_um,
        levels: p.levels.iter().map(|l| [l.depth, l.height, l.width, l.tile_d, l.tile_h, l.tile_w]).collect(),
        display: p.channels.iter().map(|c| crate::protocol::ChannelDisplay { color: c.color, window: c.window, range: c.range, active: c.active }).collect(),
    }
}

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


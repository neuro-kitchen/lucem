//! Display tiles of a multiscale image, made on worker threads.
//!
//! The viewer asks for the tiles it can see ([`TileService::request`]), nearest the centre first;
//! a new request replaces every tile still waiting (so a pan or zoom never waits behind tiles that
//! went off screen). A worker reads each visible channel's raw tile (kept in a byte-budgeted LRU
//! cache, so a display change does not read the disk again), blends the channels in their colours
//! and windows ([`composite_bgra`]) and sends the finished image back on a channel.
//!
//! Each request carries a revision (`rev`): the viewer keeps a tile until a newer revision of it
//! arrives, so the screen never goes blank while it refines.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use gpui_kit::RenderImage;
use image::{Frame, ImageBuffer, Rgba};
use lumen_io::Pyramid;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::tonemap::ColormapLut;

/// How one channel is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChannelDisplay {
    pub visible: bool,
    /// Linear RGB, 0–1.
    pub color: [f32; 3],
    /// Raw values mapped to black and to full colour.
    pub min: f32,
    pub max: f32,
    /// Display gamma (1 = linear).
    pub gamma: f32,
}

impl ChannelDisplay {
    pub fn new(color: [u8; 3], window: (f32, f32)) -> Self {
        Self { visible: true, color: [color[0] as f32 / 255.0, color[1] as f32 / 255.0, color[2] as f32 / 255.0], min: window.0, max: window.1, gamma: 1.0 }
    }
}

/// A display tile: `(level, tx, ty)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    pub level: usize,
    pub tx: usize,
    pub ty: usize,
}

/// A finished display tile.
#[derive(Clone)]
pub struct TileImage {
    pub key: TileKey,
    pub rev: u64,
    pub width: usize,
    pub height: usize,
    pub image: Arc<RenderImage>,
}

/// What came back for a key.
pub enum TileResult {
    Ready(TileImage),
    Failed(TileKey, String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RawKey {
    level: usize,
    channel: usize,
    tx: usize,
    ty: usize,
}

/// Raw tiles by key, least recently used dropped first once over `budget` bytes.
struct RawCache {
    map: HashMap<RawKey, (Arc<Vec<f32>>, usize, usize, u64)>,
    bytes: usize,
    budget: usize,
    clock: u64,
}

impl RawCache {
    fn get(&mut self, k: &RawKey) -> Option<(Arc<Vec<f32>>, usize, usize)> {
        self.clock += 1;
        let clock = self.clock;
        self.map.get_mut(k).map(|e| {
            e.3 = clock;
            (e.0.clone(), e.1, e.2)
        })
    }

    fn put(&mut self, k: RawKey, data: Arc<Vec<f32>>, w: usize, h: usize) {
        self.clock += 1;
        self.bytes += data.len() * 4;
        if let Some(old) = self.map.insert(k, (data, w, h, self.clock)) {
            self.bytes -= old.0.len() * 4;
        }
        while self.bytes > self.budget && self.map.len() > 1 {
            let Some((&oldest, _)) = self.map.iter().min_by_key(|(_, e)| e.3) else { break };
            if let Some(e) = self.map.remove(&oldest) {
                self.bytes -= e.0.len() * 4;
            }
        }
    }
}

struct Request {
    rev: u64,
    displays: Arc<Vec<ChannelDisplay>>,
    keys: VecDeque<TileKey>,
}

struct Shared {
    pyramid: Arc<Pyramid>,
    queue: Mutex<Option<Request>>,
    ready: Condvar,
    raw: Mutex<RawCache>,
    closed: Mutex<bool>,
}

/// Worker threads making display tiles of one pyramid.
pub struct TileService {
    shared: Arc<Shared>,
}

/// Default budget of the raw tile cache (bytes).
pub const RAW_BUDGET: usize = 768 * 1024 * 1024;

impl TileService {
    /// Starts `threads` workers; finished tiles arrive on the returned receiver.
    pub fn new(pyramid: Arc<Pyramid>, threads: usize) -> (Self, async_channel::Receiver<TileResult>) {
        let (tx, rx) = async_channel::unbounded();
        let shared = Arc::new(Shared {
            pyramid,
            queue: Mutex::new(None),
            ready: Condvar::new(),
            raw: Mutex::new(RawCache { map: HashMap::new(), bytes: 0, budget: RAW_BUDGET, clock: 0 }),
            closed: Mutex::new(false),
        });
        for i in 0..threads.max(1) {
            let (shared, tx) = (shared.clone(), tx.clone());
            thread::Builder::new().name(format!("lumen-tiles-{i}")).spawn(move || worker(shared, tx)).expect("failed to spawn a tile worker");
        }
        (Self { shared }, rx)
    }

    /// Threads a viewer uses by default (leaves a core for the UI).
    pub fn default_threads() -> usize {
        thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(1).clamp(1, 8)
    }

    pub fn pyramid(&self) -> &Arc<Pyramid> {
        &self.shared.pyramid
    }

    /// Replaces the waiting tiles with `keys` (made in this order) at revision `rev`.
    pub fn request(&self, rev: u64, displays: Arc<Vec<ChannelDisplay>>, keys: Vec<TileKey>) {
        let mut q = self.shared.queue.lock().expect("tile queue");
        *q = Some(Request { rev, displays, keys: keys.into() });
        self.shared.ready.notify_all();
    }

    /// Raw values of every channel at pixel `(x, y)` of `level`, from tiles already read (`None`
    /// where the tile is not in the cache; never reads the disk).
    pub fn sample(&self, level: usize, x: usize, y: usize) -> Vec<Option<f32>> {
        let py = &self.shared.pyramid;
        let Some(l) = py.levels.get(level) else { return Vec::new() };
        if x >= l.width || y >= l.height {
            return vec![None; py.channels.len()];
        }
        let (tx, ty) = (x / l.tile_w, y / l.tile_h);
        let Ok(mut raw) = self.shared.raw.lock() else { return Vec::new() };
        (0..py.channels.len())
            .map(|c| {
                raw.get(&RawKey { level, channel: c, tx, ty }).and_then(|(d, w, _)| d.get((y - ty * l.tile_h) * w + (x - tx * l.tile_w)).copied())
            })
            .collect()
    }

    /// A region of one channel at `level` (that level's pixels, clipped), from the raw cache;
    /// tiles not cached yet are read and kept.
    pub fn read_region(&self, level: usize, channel: usize, x0: usize, y0: usize, w: usize, h: usize) -> Result<Vec<f32>, String> {
        let py = &self.shared.pyramid;
        let l = py.levels.get(level).ok_or_else(|| format!("no level {level}"))?.clone();
        let mut out = vec![0.0f32; w * h];
        let (x1, y1) = ((x0 + w).min(l.width), (y0 + h).min(l.height));
        if x0 >= x1 || y0 >= y1 {
            return Ok(out);
        }
        for ty in y0 / l.tile_h..=(y1 - 1) / l.tile_h {
            for tx in x0 / l.tile_w..=(x1 - 1) / l.tile_w {
                let rk = RawKey { level, channel, tx, ty };
                let hit = self.shared.raw.lock().map_err(|e| e.to_string())?.get(&rk);
                let (data, tw, th) = match hit {
                    Some(t) => t,
                    None => {
                        let t = py.read_tile(level, channel, tx, ty).map_err(|e| e.to_string())?;
                        let d = Arc::new(t.data);
                        self.shared.raw.lock().map_err(|e| e.to_string())?.put(rk, d.clone(), t.width, t.height);
                        (d, t.width, t.height)
                    }
                };
                let (ox, oy) = (tx * l.tile_w, ty * l.tile_h);
                let (ax0, ax1) = (x0.max(ox), x1.min(ox + tw));
                let (ay0, ay1) = (y0.max(oy), y1.min(oy + th));
                for y in ay0..ay1 {
                    let src = (y - oy) * tw + (ax0 - ox);
                    let dst = (y - y0) * w + (ax0 - x0);
                    out[dst..dst + (ax1 - ax0)].copy_from_slice(&data[src..src + (ax1 - ax0)]);
                }
            }
        }
        Ok(out)
    }

    /// Bytes held by the raw tile cache.
    pub fn cached_bytes(&self) -> usize {
        self.shared.raw.lock().map(|r| r.bytes).unwrap_or(0)
    }
}

impl Drop for TileService {
    fn drop(&mut self) {
        if let Ok(mut c) = self.shared.closed.lock() {
            *c = true;
        }
        self.shared.ready.notify_all();
    }
}

fn worker(shared: Arc<Shared>, tx: async_channel::Sender<TileResult>) {
    loop {
        let job = {
            let mut q = shared.queue.lock().expect("tile queue");
            loop {
                if *shared.closed.lock().expect("closed flag") {
                    return;
                }
                if let Some(r) = q.as_mut() {
                    if let Some(k) = r.keys.pop_front() {
                        break (k, r.rev, r.displays.clone());
                    }
                }
                q = shared.ready.wait(q).expect("tile queue");
            }
        };
        let (key, rev, displays) = job;
        let result = match make_tile(&shared, key, &displays) {
            Ok((w, h, bgra)) => TileResult::Ready(TileImage { key, rev, width: w, height: h, image: bgra_image(w, h, bgra) }),
            Err(e) => TileResult::Failed(key, e),
        };
        if tx.send_blocking(result).is_err() {
            return;
        }
    }
}

fn make_tile(shared: &Shared, key: TileKey, displays: &[ChannelDisplay]) -> Result<(usize, usize, Vec<u8>), String> {
    let py = &shared.pyramid;
    let mut planes: Vec<(Arc<Vec<f32>>, ChannelDisplay)> = Vec::new();
    let (mut w, mut h) = (0, 0);
    for (c, d) in displays.iter().enumerate().take(py.channels.len()) {
        if !d.visible {
            continue;
        }
        let rk = RawKey { level: key.level, channel: c, tx: key.tx, ty: key.ty };
        let cached = shared.raw.lock().map_err(|e| e.to_string())?.get(&rk);
        let (data, tw, th) = match cached {
            Some(hit) => hit,
            None => {
                let t = py.read_tile(key.level, c, key.tx, key.ty).map_err(|e| e.to_string())?;
                let data = Arc::new(t.data);
                shared.raw.lock().map_err(|e| e.to_string())?.put(rk, data.clone(), t.width, t.height);
                (data, t.width, t.height)
            }
        };
        (w, h) = (tw, th);
        planes.push((data, *d));
    }
    if planes.is_empty() {
        let l = &py.levels[key.level];
        w = (l.width - key.tx * l.tile_w).min(l.tile_w);
        h = (l.height - key.ty * l.tile_h).min(l.tile_h);
    }
    let refs: Vec<(&[f32], ChannelDisplay)> = planes.iter().map(|(d, c)| (d.as_slice(), *c)).collect();
    Ok((w, h, composite_bgra(&refs, w * h)))
}

/// Size of the per-channel transfer table.
const LUT: usize = 1024;

/// Blends channels additively in their colours: each raw value goes through its window and
/// gamma to 0–1, times its colour. BGRA8, opaque.
pub fn composite_bgra(planes: &[(&[f32], ChannelDisplay)], n: usize) -> Vec<u8> {
    let mut acc = vec![[0.0f32; 3]; n];
    for (data, d) in planes {
        let span = (d.max - d.min).abs().max(1e-12);
        let inv_gamma = 1.0 / d.gamma.max(0.05);
        let table: Vec<f32> = (0..LUT).map(|i| (i as f32 / (LUT - 1) as f32).powf(inv_gamma)).collect();
        let scale = (LUT - 1) as f32 / span;
        for (a, &v) in acc.iter_mut().zip(data.iter()) {
            let i = ((v - d.min) * scale).clamp(0.0, (LUT - 1) as f32).round() as usize;
            let t = table[i];
            a[0] += d.color[0] * t;
            a[1] += d.color[1] * t;
            a[2] += d.color[2] * t;
        }
    }
    let mut out = vec![0u8; n * 4];
    for (px, a) in out.chunks_exact_mut(4).zip(acc.iter()) {
        px[0] = (a[2].min(1.0) * 255.0).round() as u8;
        px[1] = (a[1].min(1.0) * 255.0).round() as u8;
        px[2] = (a[0].min(1.0) * 255.0).round() as u8;
        px[3] = 255;
    }
    out
}

/// How a pipeline result is drawn over the image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerStyle {
    /// Intensities through a window and a colour map, at `alpha`.
    Intensity { min: f32, max: f32, lut: ColormapLut, alpha: f32 },
    /// Non-zero pixels in one colour.
    Mask { color: [u8; 3], alpha: f32 },
    /// Each label in its own colour, 0 clear.
    Labels { alpha: f32 },
}

/// A result layer as an image (BGRA8, straight alpha).
pub fn render_layer(values: &[f32], width: usize, height: usize, style: LayerStyle) -> Arc<RenderImage> {
    let n = width * height;
    let mut out = vec![0u8; n * 4];
    match style {
        LayerStyle::Intensity { min, max, lut, alpha } => {
            let span = (max - min).abs().max(1e-12);
            let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
            for (px, &v) in out.chunks_exact_mut(4).zip(values.iter()) {
                let [r, g, b] = lut.sample_rgb(((v - min) / span).clamp(0.0, 1.0));
                px.copy_from_slice(&[b, g, r, a]);
            }
        }
        LayerStyle::Mask { color, alpha } => {
            let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
            for (px, &v) in out.chunks_exact_mut(4).zip(values.iter()) {
                if v != 0.0 {
                    px.copy_from_slice(&[color[2], color[1], color[0], a]);
                }
            }
        }
        LayerStyle::Labels { alpha } => {
            let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
            for (px, &v) in out.chunks_exact_mut(4).zip(values.iter()) {
                let id = v as u32;
                if id != 0 {
                    let [r, g, b] = label_color(id);
                    px.copy_from_slice(&[b, g, r, a]);
                }
            }
        }
    }
    bgra_image(width, height, out)
}

/// A bright, distinct colour per label id.
pub fn label_color(id: u32) -> [u8; 3] {
    let h = (id.wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
    let (r, g, b) = hsv(h, 0.75, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn hsv(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// Wraps BGRA8 bytes as a GPUI image.
pub fn bgra_image(width: usize, height: usize, bgra: Vec<u8>) -> Arc<RenderImage> {
    let (w, h) = (width.max(1) as u32, height.max(1) as u32);
    let buf: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(w, h, bgra).unwrap_or_else(|| ImageBuffer::from_pixel(1, 1, Rgba([0, 0, 0, 0])));
    Arc::new(RenderImage::new(SmallVec::from_elem(Frame::new(buf), 1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_channels_blend_in_their_colours() {
        let a = [0.0f32, 100.0];
        let b = [100.0f32, 0.0];
        let red = ChannelDisplay::new([255, 0, 0], (0.0, 100.0));
        let blue = ChannelDisplay::new([0, 0, 255], (0.0, 100.0));
        let out = composite_bgra(&[(&a, red), (&b, blue)], 2);
        assert_eq!(&out[0..4], &[255, 0, 0, 255], "pixel 0: blue only (BGRA)");
        assert_eq!(&out[4..8], &[0, 0, 255, 255], "pixel 1: red only");
    }

    #[test]
    fn a_service_makes_the_tiles_it_is_asked_for() {
        let img = lumen_io::generate_phantom(lumen_io::PhantomKind::SiemensStarTarget, 700, 600);
        let py = Arc::new(Pyramid::from_loaded(img));
        let displays = Arc::new(py.channels.iter().map(|c| ChannelDisplay::new(c.color, c.window)).collect::<Vec<_>>());
        let (svc, rx) = TileService::new(py, 2);
        let keys = vec![TileKey { level: 0, tx: 0, ty: 0 }, TileKey { level: 0, tx: 1, ty: 1 }];
        svc.request(1, displays, keys);
        let mut got = 0;
        while got < 2 {
            match rx.recv_blocking().unwrap() {
                TileResult::Ready(t) => {
                    assert_eq!(t.rev, 1);
                    got += 1;
                }
                TileResult::Failed(_, e) => panic!("{e}"),
            }
        }
    }
}

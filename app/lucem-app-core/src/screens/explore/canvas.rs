//! The image canvas: the shown image drawn from its pyramid, tile by tile, with the pipeline's
//! results, the annotations, the tools at work, rulers, a scale bar and a readout.
//!
//! - **Tiles.** Each frame asks the tile service for the tiles it can see at the level the zoom
//!   needs (sharp on this display), nearest the centre first, and for the coarsest level whole (a
//!   few tiles), so something is always on screen: coarser tiles are drawn first and the finer
//!   ones cover them as they arrive. A display change (a channel's colour or window) bumps a
//!   revision; old tiles stay until their new version replaces them. Tiles far away are dropped
//!   (and their textures freed) past [`TILE_CAP`].
//! - **Camera.** `zoom` is screen px per level-0 px, `center` the level-0 point at the middle.
//!   The wheel zooms at the pointer; a two-finger swipe, Space+drag, the middle button or the
//!   Hand tool move the image.
//! - **Tools** turn pointer input into commands (an annotation added, moved, deleted), so every
//!   stroke is undoable and saved.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use gpui_kit::component::{h_flex, v_flex};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    canvas, div, fill, point, px, size, App, AppContext as _, Bounds, Context, Corners, CursorStyle, Entity, FocusHandle, Hsla, InteractiveElement as _, IntoElement, KeyDownEvent, KeyUpEvent,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, PathBuilder, Pixels, Render, RenderImage, ScrollDelta, ScrollEvent, Styled as _, Subscription, Task, Window,
};
use lucem_io::Pyramid;
use lucem_view::{ChannelDisplay, TileImage, TileKey, TileResult, TileService};

use crate::data::results::ResultRequest;
use crate::data::store::{AppEvent, Store};
use crate::domain::annotation::{Annotation, Shape};
use crate::domain::command::Command;
use crate::domain::geometry;
use crate::domain::project::ChannelDoc;
use crate::domain::tool::{step_brush, Overlays, Tool};
use crate::ui::theme::{self, color};
use crate::ui::widgets::format_length;

/// Most display tiles kept.
const TILE_CAP: usize = 384;
const MIN_ZOOM: f64 = 1.0 / 512.0;
const MAX_ZOOM: f64 = 64.0;

/// What was clicked for the segmentation model (level-0 px).
#[derive(Debug, Clone, Copy)]
enum SamPrompt {
    /// A point on the object (`true`) or off it.
    Point([f64; 2], bool),
    /// A box around it.
    Box([f64; 2], [f64; 2]),
}

/// What the pointer is doing.
#[derive(Debug, Clone)]
enum Gesture {
    Pan { mouse: (f32, f32), center: (f64, f64) },
    /// Rectangle, ellipse or ruler from `a` to `b` (level-0 px).
    Drag { tool: Tool, a: [f64; 2], b: [f64; 2] },
    /// Lasso, brush or eraser path.
    Path { tool: Tool, points: Vec<[f64; 2]>, radius: f64 },
    /// Moving the selected annotations.
    Move { start: [f64; 2], now: [f64; 2], moved: bool },
    /// The wand painting: a mask grown from each stamp over similar pixels.
    Wand(Box<Wand>),
}

/// A wand stroke: the visible region of a level, its visible channels in display units (0–1),
/// the painted mask and its bounds.
#[derive(Debug, Clone)]
struct Wand {
    level: usize,
    ds: f64,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    channels: Vec<Vec<f32>>,
    mask: Vec<u8>,
    /// `x0, y0, x1, y1` (exclusive) of what is painted, region px.
    bbox: Option<(usize, usize, usize, usize)>,
    /// Brush radius, level-0 px.
    radius: f64,
    last: [f64; 2],
    tolerance: f32,
    color: [u8; 3],
    overlay: Option<([f64; 4], Arc<RenderImage>)>,
}

impl Wand {
    /// Paints the pixels within the brush that are connected to its centre and look like it.
    fn stamp(&mut self, p: [f64; 2]) {
        let (cx, cy) = (p[0] / self.ds - self.x0 as f64, p[1] / self.ds - self.y0 as f64);
        if cx < 0.0 || cy < 0.0 || cx >= self.w as f64 || cy >= self.h as f64 {
            return;
        }
        let (sx, sy) = (cx as usize, cy as usize);
        let r = (self.radius / self.ds).max(1.0);
        let seed: Vec<f32> = self.channels.iter().map(|c| c[sy * self.w + sx]).collect();
        let tol = self.tolerance;
        let (w, h) = (self.w as isize, self.h as isize);
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![(sx as isize, sy as isize)];
        seen.insert((sx as isize, sy as isize));
        let (mut bx0, mut by0, mut bx1, mut by1) = self.bbox.unwrap_or((usize::MAX, usize::MAX, 0, 0));
        while let Some((x, y)) = stack.pop() {
            let i = (y * w + x) as usize;
            self.mask[i] = 1;
            (bx0, by0, bx1, by1) = (bx0.min(x as usize), by0.min(y as usize), bx1.max(x as usize + 1), by1.max(y as usize + 1));
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= h || seen.contains(&(nx, ny)) {
                    continue;
                }
                let (ddx, ddy) = (nx as f64 + 0.5 - cx, ny as f64 + 0.5 - cy);
                if ddx * ddx + ddy * ddy > r * r {
                    continue;
                }
                let j = (ny * w + nx) as usize;
                if self.channels.iter().zip(&seed).all(|(c, s)| (c[j] - s).abs() <= tol) {
                    seen.insert((nx, ny));
                    stack.push((nx, ny));
                }
            }
        }
        if bx1 > bx0 {
            self.bbox = Some((bx0, by0, bx1, by1));
        }
        self.last = p;
    }

    /// Stamps along the segment from the last stamp to `p`.
    fn stroke_to(&mut self, p: [f64; 2]) {
        let step = (self.radius * 0.5).max(self.ds);
        let (a, d) = (self.last, (p[0] - self.last[0]).hypot(p[1] - self.last[1]));
        if d < step {
            return;
        }
        let n = (d / step).ceil() as usize;
        for k in 1..=n {
            let t = k as f64 / n as f64;
            self.stamp([a[0] + t * (p[0] - a[0]), a[1] + t * (p[1] - a[1])]);
        }
        self.refresh_overlay();
    }

    /// The painted part as an image over the canvas.
    fn refresh_overlay(&mut self) {
        let Some((bx0, by0, bx1, by1)) = self.bbox else { return };
        let (bw, bh) = (bx1 - bx0, by1 - by0);
        let mut crop = Vec::with_capacity(bw * bh);
        for y in by0..by1 {
            crop.extend(self.mask[y * self.w + bx0..y * self.w + bx1].iter().map(|&v| v as f32));
        }
        let img = lucem_view::render_layer(&crop, bw, bh, lucem_view::LayerStyle::Mask { color: self.color, alpha: 0.5 });
        let ds = self.ds;
        self.overlay = Some(([(self.x0 + bx0) as f64 * ds, (self.y0 + by0) as f64 * ds, bw as f64 * ds, bh as f64 * ds], img));
    }

    /// The painted regions as level-0 polygons.
    fn polygons(&self) -> Vec<Vec<[f64; 2]>> {
        let Some((bx0, by0, bx1, by1)) = self.bbox else { return Vec::new() };
        let (bw, bh) = (bx1 - bx0, by1 - by0);
        let mut crop = Vec::with_capacity(bw * bh);
        for y in by0..by1 {
            crop.extend_from_slice(&self.mask[y * self.w + bx0..y * self.w + bx1]);
        }
        geometry::mask_to_polygons(&crop, bw, bh, 4)
            .into_iter()
            .map(|p| geometry::simplify(&p, 0.6).into_iter().map(|[x, y]| [(x + (self.x0 + bx0) as f64) * self.ds, (y + (self.y0 + by0) as f64) * self.ds]).collect())
            .collect()
    }
}

struct Viewer {
    image: String,
    pyramid: Arc<Pyramid>,
    service: TileService,
    tiles: HashMap<TileKey, TileImage>,
    failed: HashSet<TileKey>,
    rev: u64,
    displays: Arc<Vec<ChannelDisplay>>,
    display_docs: Vec<ChannelDoc>,
    requested: Vec<TileKey>,
    requested_rev: u64,
    _pump: Task<()>,
}

/// A closed brush stroke and what it encloses as one outline (level-0 px): the largest piece of
/// [`traced`] with the inside filled.
fn filled_outline(points: &[[f64; 2]], radius: f64) -> Option<Vec<[f64; 2]>> {
    let area = |p: &Vec<[f64; 2]>| p.iter().zip(p.iter().cycle().skip(1)).map(|(a, b)| a[0] * b[1] - b[0] * a[1]).sum::<f64>().abs();
    traced(points, radius, true).into_iter().max_by(|a, b| area(a).total_cmp(&area(b)))
}

/// The outlines (level-0 px) of what a brush stroke paints, with the region inside its path too
/// when `fill`: rasterized at about 4 px across the brush, then traced.
fn traced(points: &[[f64; 2]], radius: f64, fill: bool) -> Vec<Vec<[f64; 2]>> {
    if points.is_empty() {
        return Vec::new();
    }
    let (x0, y0, x1, y1) = points.iter().fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, p| (b.0.min(p[0]), b.1.min(p[1]), b.2.max(p[0]), b.3.max(p[1])));
    let (ox, oy) = (x0 - radius - 2.0, y0 - radius - 2.0);
    let (bw, bh) = (x1 - x0 + 2.0 * radius + 4.0, y1 - y0 + 2.0 * radius + 4.0);
    // About 4 raster px across the brush, at most 2048 px a side
    let ds = (radius / 4.0).max(bw.max(bh) / 2048.0).max(1e-3);
    let (w, h) = (((bw / ds).ceil() as usize).max(1), ((bh / ds).ceil() as usize).max(1));
    let moved: Vec<[f64; 2]> = points.iter().map(|p| [p[0] - ox, p[1] - oy]).collect();
    let ann = |shape: Shape| Annotation { id: String::new(), image: String::new(), shape, class: "fill".into(), note: String::new(), z: None };
    let (stroke, region) = (ann(Shape::Stroke { points: moved.clone(), radius, erase: false }), ann(Shape::Freehand { points: moved }));
    let shapes: Vec<&Annotation> = if fill { vec![&region, &stroke] } else { vec![&stroke] };
    let mask: Vec<u8> = crate::domain::annotation::rasterize(&shapes, None, 0, 0, w, h, ds).into_iter().map(|v| u8::from(v != 0)).collect();
    geometry::mask_to_polygons(&mask, w, h, 4).into_iter().map(|p| geometry::simplify(&p, 0.6).into_iter().map(|[x, y]| [x * ds + ox, y * ds + oy]).collect::<Vec<_>>()).filter(|p| p.len() >= 3).collect()
}

/// Opacity of painted strokes.
const PAINT_ALPHA: f32 = 0.45;
/// Longest side of a paint raster (px).
const PAINT_MAX_SIDE: f64 = 4096.0;

/// The brush and eraser strokes of the shown plane over a region, one colour per class.
struct PaintLayer {
    /// Level-0 rectangle it covers, and level-0 px per raster px.
    rect: [f64; 4],
    ds: f64,
    /// Image, plane and annotations revision it was made for.
    key: String,
    image: Option<Arc<RenderImage>>,
}

/// The strokes (`Shape::Stroke`) of each class over `rect` (level-0) at `ds` level-0 px per pixel,
/// as BGRA with straight alpha: brush strokes fill, eraser strokes of any class clear, in drawing
/// order. Returns the rectangle the raster covers and the image (`None` when nothing is painted).
/// `classes`: name, colour and opacity (times `alpha`) of each class drawn.
fn paint_raster(strokes: &[Annotation], classes: &[(String, [u8; 3], f32)], rect: [f64; 4], ds: f64, alpha: f32) -> ([f64; 4], Option<Arc<RenderImage>>) {
    let (w, h) = (((rect[2] / ds).ceil() as usize).max(1), ((rect[3] / ds).ceil() as usize).max(1));
    // Shifted so the region starts at 0 (the rasterizer's region is in unsigned pixels)
    let shifted: Vec<Annotation> = strokes.iter().map(|a| Annotation { shape: a.shape.translated(-rect[0], -rect[1]), ..a.clone() }).collect();
    let mut out = vec![0u8; w * h * 4];
    let mut any = false;
    for (name, color, opacity) in classes {
        let a8 = ((alpha * opacity).clamp(0.0, 1.0) * 255.0) as u8;
        if !shifted.iter().any(|a| &a.class == name && matches!(a.shape, Shape::Stroke { erase: false, .. })) {
            continue;
        }
        let list: Vec<&Annotation> = shifted.iter().filter(|a| &a.class == name || matches!(a.shape, Shape::Stroke { erase: true, .. })).collect();
        let mask = crate::domain::annotation::rasterize(&list, None, 0, 0, w, h, ds);
        for (i, &m) in mask.iter().enumerate() {
            if m != 0 {
                out[i * 4..i * 4 + 4].copy_from_slice(&[color[2], color[1], color[0], a8]);
                any = true;
            }
        }
    }
    let covered = [rect[0], rect[1], w as f64 * ds, h as f64 * ds];
    (covered, any.then(|| lucem_view::bgra_image(w, h, out)))
}

pub struct ImageCanvas {
    store: Entity<Store>,
    pub focus: FocusHandle,
    viewer: Option<Viewer>,
    pub zoom: f64,
    pub center: (f64, f64),
    /// The plane shown, level-0 planes (0 for a 2D image): the z of the store's 3D cursor.
    pub z: usize,
    /// Screen px on the left covered by the floating tool bar (Fit leaves them free).
    pub inset_left: f32,
    fit_pending: bool,
    /// The plane's bounds in the window (set while painting).
    bounds: Rc<Cell<Bounds<Pixels>>>,
    scale: f32,
    gesture: Option<Gesture>,
    /// Corners of the polygon being drawn.
    polygon: Vec<[f64; 2]>,
    /// Pointer, plane-local px.
    mouse: Option<(f32, f32)>,
    space: bool,
    /// Textures to free at the next paint.
    drops: Vec<Arc<RenderImage>>,
    /// The pending result refresh (replaced by each camera move: one timer, not one per move).
    refresh: Option<Task<()>>,
    /// The Threshold tool's live mask over the view, and what it was made for.
    threshold: Option<([f64; 4], Arc<RenderImage>)>,
    threshold_for: String,
    threshold_task: Option<Task<()>>,
    /// Brush and eraser strokes of the shown plane as one raster per class, so overlapping parts
    /// of a stroke are not drawn twice (translucent paths double their alpha where they overlap,
    /// and their joins spike at sharp turns).
    paint: Option<PaintLayer>,
    paint_want: String,
    paint_task: Option<Task<()>>,
    /// Histology cells of the shown image as a raster (nuclei filled, cell outlines).
    cells: Option<PaintLayer>,
    cells_want: String,
    cells_task: Option<Task<()>>,
    /// The stroke being painted, as a raster.
    live: Option<([f64; 4], Arc<RenderImage>)>,
    /// Outlines of selected brush strokes (traced from their paint), by id and version.
    outlines: std::cell::RefCell<HashMap<String, Vec<Vec<[f64; 2]>>>>,
    /// An image shown whatever is selected (`None`: the selected image).
    pub pinned: Option<String>,
    /// Only this channel is drawn (a cell of the Channels view).
    pub channel: Option<usize>,
    /// No plane strip, footer or scale bar (a cell of a grid).
    pub compact: bool,
    /// Its own footer (position, values, level, zoom); off in Explore, whose bottom bar shows the
    /// focused view's.
    pub footer: bool,
    /// This view's own overlays; `None`: the shared ones (the tool state's).
    pub overlays: Option<Overlays>,
    /// Asks for the pipeline's results when its camera moves (the focused view does; the others
    /// draw what it gets).
    pub results_source: bool,
    /// Threads reading its tiles.
    pub threads: usize,
    /// The camera is being set from a linked view: not sent back.
    syncing: bool,
    _subs: Vec<Subscription>,
}

/// What the bottom bar shows of a view.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CanvasStatus {
    /// The pointer's position (µm when the pixel size is known) and the values under it.
    pub position: Option<String>,
    pub values: Option<String>,
    /// The pyramid level drawn and its downsample.
    pub level: Option<String>,
    pub zoom: f64,
    pub cached_mb: usize,
}

/// What a canvas tells the views linked to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CanvasEvent {
    /// Its camera moved: screen px per level-0 px, and the level-0 point at the middle.
    Camera { zoom: f64, center: (f64, f64) },
}

impl gpui_kit::EventEmitter<CanvasEvent> for ImageCanvas {}

impl ImageCanvas {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            match e {
                AppEvent::ImageReady(_) => this.fit_pending = true,
                AppEvent::Cursor => {
                    let s = this.store.read(cx);
                    let z = this.viewer.as_ref().and_then(|v| s.cursor_of(&v.image)).map_or(0, |c| c[0]);
                    if z != this.z && this.viewer.is_some() {
                        this.z = z;
                        this.camera_changed(cx);
                    }
                }
                AppEvent::Document(crate::domain::command::Changed::Images) => this.fit_pending = this.viewer.as_ref().map(|v| v.image.clone()) != this.image_id(this.store.read(cx)),
                _ => {}
            }
            cx.notify();
        })];
        Self {
            store,
            focus: cx.focus_handle(),
            viewer: None,
            zoom: 1.0,
            center: (0.0, 0.0),
            z: 0,
            inset_left: 0.0,
            fit_pending: true,
            bounds: Rc::new(Cell::new(Bounds::default())),
            scale: 1.0,
            gesture: None,
            polygon: Vec::new(),
            mouse: None,
            space: false,
            drops: Vec::new(),
            refresh: None,
            threshold: None,
            threshold_for: String::new(),
            threshold_task: None,
            paint: None,
            cells: None,
            cells_want: String::new(),
            cells_task: None,
            paint_want: String::new(),
            paint_task: None,
            live: None,
            outlines: Default::default(),
            pinned: None,
            channel: None,
            compact: false,
            footer: true,
            overlays: None,
            results_source: true,
            threads: TileService::default_threads(),
            syncing: false,
            _subs: subs,
        }
    }

    // ------------------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------------------

    /// The image it shows: the pinned one while it is in the project, else the selected one.
    pub fn image_id(&self, s: &Store) -> Option<String> {
        self.pinned.clone().filter(|p| s.doc.image(p).is_some()).or_else(|| s.doc.active_image.clone())
    }

    /// The pointer's position and values, the level and the zoom (for the bottom bar).
    /// Changes an overlay: this view's own when it has them, else the shared ones.
    pub fn toggle_overlay(&mut self, f: impl FnOnce(&mut Overlays), cx: &mut Context<Self>) {
        match self.overlays.as_mut() {
            Some(o) => {
                f(o);
                cx.notify();
            }
            None => self.store.update(cx, |s, cx| {
                s.update_tool(
                    |t| {
                        let mut o = t.overlays();
                        f(&mut o);
                        t.set_overlays(o);
                    },
                    cx,
                )
            }),
        }
    }

    pub fn status(&self, cx: &App) -> CanvasStatus {
        let Some(v) = &self.viewer else { return CanvasStatus::default() };
        let um = v.pyramid.pixel_size_um;
        let names: Vec<String> = self.store.read(cx).doc.image(&v.image).map(|i| i.channels.iter().map(|c| c.label.clone()).collect()).unwrap_or_default();
        let (mut position, mut values) = (None, None);
        if let Some((p, _, vals)) = self.readout() {
            position = Some(match um {
                Some(u) => format!("x {:.1}  y {:.1} µm", p[0] * u, p[1] * u),
                None => format!("x {:.0}  y {:.0} px", p[0], p[1]),
            });
            let v: Vec<String> = vals.iter().enumerate().map(|(i, x)| format!("{} {}", names.get(i).map_or("", String::as_str), x.map_or("–".to_string(), |x| trim(x as f64)))).collect();
            values = (!v.is_empty()).then(|| v.join("  "));
        }
        let l = v.pyramid.level_for_zoom(self.zoom * self.scale as f64);
        CanvasStatus {
            position,
            values,
            // Plain: the scale shown ("1:4"), not the pyramid's internals
            level: Some({
                let ds = v.pyramid.levels[l].downsample;
                if (ds - ds.round()).abs() < 0.05 { format!("1:{}", ds.round() as i64) } else { format!("1:{ds:.1}") }
            }),
            zoom: self.zoom,
            cached_mb: v.service.cached_bytes() / (1024 * 1024),
        }
    }

    /// Takes the camera of a linked view (not sent back to it).
    pub fn set_camera(&mut self, zoom: f64, center: (f64, f64), cx: &mut Context<Self>) {
        if self.zoom == zoom && self.center == center {
            return;
        }
        self.zoom = zoom;
        self.center = center;
        self.fit_pending = false;
        self.syncing = true;
        self.camera_changed(cx);
        self.syncing = false;
    }

    fn view_size(&self) -> (f32, f32) {
        let b = self.bounds.get().size;
        (b.width.as_f32(), b.height.as_f32())
    }

    /// Plane-local px → level-0 px.
    pub fn to_image(&self, (x, y): (f32, f32)) -> [f64; 2] {
        let (w, h) = self.view_size();
        [self.center.0 + (x as f64 - w as f64 / 2.0) / self.zoom, self.center.1 + (y as f64 - h as f64 / 2.0) / self.zoom]
    }

    /// Level-0 px → plane-local px.
    pub fn to_screen(&self, [x, y]: [f64; 2]) -> (f32, f32) {
        let (w, h) = self.view_size();
        (((x - self.center.0) * self.zoom + w as f64 / 2.0) as f32, ((y - self.center.1) * self.zoom + h as f64 / 2.0) as f32)
    }

    fn local(&self, p: gpui_kit::Point<Pixels>) -> (f32, f32) {
        let o = self.bounds.get().origin;
        ((p.x - o.x).as_f32(), (p.y - o.y).as_f32())
    }

    /// The visible level-0 rectangle `x, y, w, h`.
    pub fn visible(&self) -> [f64; 4] {
        let (w, h) = self.view_size();
        let a = self.to_image((0.0, 0.0));
        [a[0], a[1], w as f64 / self.zoom, h as f64 / self.zoom]
    }

    pub fn zoom_at(&mut self, factor: f64, anchor: Option<(f32, f32)>, cx: &mut Context<Self>) {
        let (w, h) = self.view_size();
        let anchor = anchor.unwrap_or((w / 2.0, h / 2.0));
        let before = self.to_image(anchor);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let after = self.to_image(anchor);
        self.center = (self.center.0 + before[0] - after[0], self.center.1 + before[1] - after[1]);
        self.camera_changed(cx);
    }

    /// The whole image in view.
    pub fn fit(&mut self, cx: &mut Context<Self>) {
        let (w, h) = self.view_size();
        let Some(v) = &self.viewer else { return };
        if w < 10.0 || h < 10.0 {
            self.fit_pending = true;
            return;
        }
        self.fit_pending = false;
        let (iw, ih) = (v.pyramid.width() as f64, v.pyramid.height() as f64);
        let inset = self.inset_left as f64;
        self.zoom = ((w as f64 - 48.0 - inset) / iw).min((h as f64 - 48.0) / ih).clamp(MIN_ZOOM, MAX_ZOOM);
        // The image sits in the part the tool bar leaves free
        self.center = (iw / 2.0 - inset / 2.0 / self.zoom, ih / 2.0);
        self.camera_changed(cx);
    }

    /// One image pixel per screen pixel (of the display).
    pub fn actual_size(&mut self, cx: &mut Context<Self>) {
        let f = (1.0 / self.scale as f64) / self.zoom;
        self.zoom_at(f, None, cx);
    }

    /// `self.z` at `level`.
    fn plane_at(&self, level: usize) -> usize {
        let Some(v) = &self.viewer else { return 0 };
        let l = &v.pyramid.levels[level];
        ((self.z as f64 / l.z_downsample) as usize).min(l.depth - 1)
    }

    /// Shows plane `z` (level-0 planes).
    pub fn set_z(&mut self, z: usize, cx: &mut Context<Self>) {
        let Some(v) = &self.viewer else { return };
        let z = z.min(v.pyramid.depth() - 1);
        let id = v.image.clone();
        if z != self.z {
            self.z = z;
            self.camera_changed(cx);
            // The other views follow the cursor
            self.store.update(cx, |s, cx| {
                if let Some(mut at) = s.cursor_of(&id) {
                    at[0] = z;
                    s.set_cursor_of(&id, at, cx);
                }
            });
        }
    }

    /// The camera moved: results for the new view follow once it rests (200 ms).
    fn camera_changed(&mut self, cx: &mut Context<Self>) {
        cx.notify();
        if !self.syncing {
            cx.emit(CanvasEvent::Camera { zoom: self.zoom, center: self.center });
        }
        if self.results_source {
            self.refresh = Some(cx.spawn(async move |this, cx| {
                cx.background_executor().timer(Duration::from_millis(200)).await;
                let _ = this.update(cx, |c, cx| c.request_results(cx));
            }));
        }
    }

    /// The Threshold tool's mask over the visible region, recomputed (off the UI thread) when
    /// the settings or the view change.
    fn sync_threshold(&mut self, cx: &mut Context<Self>) {
        let t = self.store.read(cx).tool.clone();
        let Some(v) = &self.viewer else { return };
        if t.tool != Tool::Threshold {
            self.threshold = None;
            self.threshold_for.clear();
            return;
        }
        let py = v.pyramid.clone();
        let rect = self.visible();
        // The finest level whose visible region fits 1024 px a side
        let level = (0..py.levels.len()).find(|&l| (rect[2] / py.levels[l].downsample).max(rect[3] / py.levels[l].downsample) <= 1024.0).unwrap_or(py.coarsest());
        let plane = self.plane_at(level);
        let key = format!("{level}/{plane}/{:.0}/{:.0}/{:.0}/{:.0}/{}/{}/{}/{}", rect[0], rect[1], rect[2], rect[3], t.threshold_channel, t.threshold_value, t.threshold_above, t.threshold_sigma);
        if key == self.threshold_for {
            return;
        }
        self.threshold_for = key;
        let color = self.store.read(cx).doc.class(&t.class).map_or([0, 220, 255], |c| c.color);
        let task = cx.background_spawn(async move {
            let l = py.levels[level].clone();
            let ds = l.downsample;
            let x0 = (rect[0] / ds).floor().max(0.0) as usize;
            let y0 = (rect[1] / ds).floor().max(0.0) as usize;
            let x1 = (((rect[0] + rect[2]) / ds).ceil().max(0.0) as usize).min(l.width);
            let y1 = (((rect[1] + rect[3]) / ds).ceil().max(0.0) as usize).min(l.height);
            if x1 <= x0 || y1 <= y0 {
                return None;
            }
            let (w, h) = (x1 - x0, y1 - y0);
            let mut v = py.read_region_at(level, t.threshold_channel, plane, x0, y0, w, h).ok()?;
            geometry::blur(&mut v, w, h, t.threshold_sigma / ds);
            let mask: Vec<f32> = v.iter().map(|&x| if (x >= t.threshold_value) == t.threshold_above { 1.0 } else { 0.0 }).collect();
            let img = lucem_view::render_layer(&mask, w, h, lucem_view::LayerStyle::Mask { color, alpha: 0.5 });
            Some(([x0 as f64 * ds, y0 as f64 * ds, w as f64 * ds, h as f64 * ds], img))
        });
        self.threshold_task = Some(cx.spawn(async move |this, cx| {
            let r = task.await;
            let _ = this.update(cx, |c, cx| {
                if let Some(old) = std::mem::replace(&mut c.threshold, r) {
                    c.drops.push(old.1);
                }
                cx.notify();
            });
        }));
    }

    /// The paint raster of the visible region, made again (off the UI thread) when the strokes,
    /// the plane or the image change, when the view leaves what it covers, or when the zoom
    /// changes by more than half. It covers twice the view, so panning rarely needs a new one.
    /// Keeps the cell raster for the view (twice the view, at about screen resolution), made in
    /// the background when the view or the cells change.
    fn sync_cells(&mut self, cx: &mut Context<Self>) {
        let Some(v) = &self.viewer else {
            self.cells = None;
            return;
        };
        let s = self.store.read(cx);
        let ov = self.overlays.unwrap_or_else(|| s.tool.overlays());
        let cells: Vec<lucem_hist::HistObject> = if s.hist.show && ov.annotations { s.hist.cells.get(&v.image).into_iter().chain(s.hist.tiles.get(&v.image)).flatten().cloned().collect() } else { Vec::new() };
        let key = format!("{}/{}/{}/{}", v.image, s.hist.rev, ov.opacity, s.sel.objects.join(","));
        // Selected objects, by their index in the list drawn (cells, then tiles)
        let n_cells = s.hist.cells.get(&v.image).map_or(0, Vec::len);
        let selected: std::collections::HashSet<usize> = cells.iter().enumerate().filter(|(i, o)| s.sel.objects.contains(&format!("{}{}", if *i < n_cells { "c" } else { "t" }, o.id))).map(|(i, _)| i).collect();
        let (iw, ih) = (v.pyramid.width() as f64, v.pyramid.height() as f64);
        let view = self.visible();
        let ds_view = 1.0 / self.zoom.max(1e-9);
        let (vx0, vy0, vx1, vy1) = (view[0].max(0.0), view[1].max(0.0), (view[0] + view[2]).min(iw), (view[1] + view[3]).min(ih));
        if let Some(p) = &self.cells {
            let covers = p.image.is_none() || (p.rect[0] <= vx0 && p.rect[1] <= vy0 && p.rect[0] + p.rect[2] >= vx1 && p.rect[1] + p.rect[3] >= vy1);
            let ratio = p.ds / ds_view;
            if p.key == key && covers && (0.67..=1.5).contains(&ratio) {
                return;
            }
        }
        let x0 = (view[0] - view[2] * 0.5).clamp(0.0, iw);
        let y0 = (view[1] - view[3] * 0.5).clamp(0.0, ih);
        let x1 = (view[0] + view[2] * 1.5).clamp(0.0, iw);
        let y1 = (view[1] + view[3] * 1.5).clamp(0.0, ih);
        let rect = [x0, y0, (x1 - x0).max(1.0), (y1 - y0).max(1.0)];
        let ds = ds_view.max(rect[2].max(rect[3]) / PAINT_MAX_SIDE);
        let want = format!("{key}/{:.0}/{:.0}/{:.0}/{:.0}/{ds:.5}", rect[0], rect[1], rect[2], rect[3]);
        if self.cells_want == want {
            return;
        }
        self.cells_want = want;
        if cells.is_empty() {
            if let Some(old) = self.cells.replace(PaintLayer { rect, ds, key, image: None }).and_then(|p| p.image) {
                self.drops.push(old);
            }
            return;
        }
        let color = s.hist_color();
        let alpha = ov.opacity;
        let work = cx.background_spawn(async move { crate::data::hist::cells_raster(&cells, &color, rect, ds, alpha, &selected) });
        self.cells_task = Some(cx.spawn(async move |this, cx| {
            let image = work.await;
            let _ = this.update(cx, |c, cx| {
                if let Some(old) = c.cells.replace(PaintLayer { rect, ds, key, image }).and_then(|p| p.image) {
                    c.drops.push(old);
                }
                cx.notify();
            });
        }));
    }

    fn sync_paint(&mut self, cx: &mut Context<Self>) {
        let Some(v) = &self.viewer else {
            self.paint = None;
            return;
        };
        let s = self.store.read(cx);
        let ov = self.overlays.unwrap_or_else(|| s.tool.overlays());
        let (shown, alpha) = (ov.annotations, PAINT_ALPHA * ov.opacity);
        let key = format!("{}/{}/{}/{shown}/{alpha:.3}", v.image, self.z, s.annotations_rev);
        let (iw, ih) = (v.pyramid.width() as f64, v.pyramid.height() as f64);
        let view = self.visible();
        let ds_view = 1.0 / self.zoom.max(1e-9);
        let (vx0, vy0, vx1, vy1) = (view[0].max(0.0), view[1].max(0.0), (view[0] + view[2]).min(iw), (view[1] + view[3]).min(ih));
        if let Some(p) = &self.paint {
            let covers = p.image.is_none() || (p.rect[0] <= vx0 && p.rect[1] <= vy0 && p.rect[0] + p.rect[2] >= vx1 && p.rect[1] + p.rect[3] >= vy1);
            let ratio = p.ds / ds_view;
            if p.key == key && covers && (0.67..=1.5).contains(&ratio) {
                return;
            }
        }
        // Twice the view, inside the image
        let x0 = (view[0] - view[2] * 0.5).clamp(0.0, iw);
        let y0 = (view[1] - view[3] * 0.5).clamp(0.0, ih);
        let x1 = (view[0] + view[2] * 1.5).clamp(0.0, iw);
        let y1 = (view[1] + view[3] * 1.5).clamp(0.0, ih);
        let rect = [x0, y0, (x1 - x0).max(1.0), (y1 - y0).max(1.0)];
        let ds = ds_view.max(rect[2].max(rect[3]) / PAINT_MAX_SIDE);
        let want = format!("{key}/{:.0}/{:.0}/{:.0}/{:.0}/{ds:.5}", rect[0], rect[1], rect[2], rect[3]);
        if self.paint_want == want {
            return;
        }
        self.paint_want = want;
        let z = self.z;
        let strokes: Vec<Annotation> = if shown { s.doc.annotations_of(&v.image).filter(|a| a.on_plane(z) && matches!(a.shape, Shape::Stroke { .. })).cloned().collect() } else { Vec::new() };
        let classes: Vec<(String, [u8; 3], f32)> = s.doc.classes.iter().filter(|c| c.visible).map(|c| (c.name.clone(), c.color, c.opacity)).collect();
        if strokes.is_empty() {
            if let Some(old) = self.paint.replace(PaintLayer { rect, ds, key, image: None }).and_then(|p| p.image) {
                self.drops.push(old);
            }
            return;
        }
        let work = cx.background_spawn(async move { paint_raster(&strokes, &classes, rect, ds, alpha) });
        self.paint_task = Some(cx.spawn(async move |this, cx| {
            let (rect, image) = work.await;
            let _ = this.update(cx, |c, cx| {
                if let Some(old) = c.paint.replace(PaintLayer { rect, ds, key, image }).and_then(|p| p.image) {
                    c.drops.push(old);
                }
                cx.notify();
            });
        }));
    }

    /// The brush or eraser stroke in progress as a raster at screen resolution (an eraser shows
    /// as a dark wash until it is applied).
    fn refresh_live(&mut self, cx: &App) {
        let Some(Gesture::Path { tool, points, radius }) = &self.gesture else { return };
        let (x0, y0, x1, y1) = points.iter().fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, p| (b.0.min(p[0]), b.1.min(p[1]), b.2.max(p[0]), b.3.max(p[1])));
        let r = *radius;
        let rect = [x0 - r - 1.0, y0 - r - 1.0, x1 - x0 + 2.0 * r + 2.0, y1 - y0 + 2.0 * r + 2.0];
        let ds = (1.0 / self.zoom.max(1e-9)).max(rect[2].max(rect[3]) / PAINT_MAX_SIDE);
        let s = self.store.read(cx);
        let erase = *tool == Tool::Eraser;
        let color = if erase { [0, 0, 0] } else { s.doc.class(&s.tool.class).map_or([255, 255, 255], |c| c.color) };
        let stroke = Annotation { id: String::new(), image: String::new(), shape: Shape::Stroke { points: points.clone(), radius: r, erase: false }, class: "live".into(), note: String::new(), z: None };
        let (covered, image) = paint_raster(&[stroke], &[("live".to_string(), color, 1.0)], rect, ds, PAINT_ALPHA);
        if let Some(old) = std::mem::replace(&mut self.live, image.map(|i| (covered, i))) {
            self.drops.push(old.1);
        }
    }

    /// Sends the visible region (its shown channels as RGB, at the finest level that fits 1024 px a
    /// side) and the prompt to the segmentation model; the regions it outlines become annotations.
    fn segment(&mut self, prompt: SamPrompt, cx: &mut Context<Self>) {
        use std::hash::{Hash, Hasher};
        let Some(v) = &self.viewer else { return };
        let py = v.pyramid.clone();
        let (iw, ih) = (py.width() as f64, py.height() as f64);
        let rect = self.visible();
        let (rx0, ry0, rx1, ry1) = (rect[0].max(0.0), rect[1].max(0.0), (rect[0] + rect[2]).min(iw), (rect[1] + rect[3]).min(ih));
        if rx1 <= rx0 || ry1 <= ry0 {
            return;
        }
        let level = (0..py.levels.len()).find(|&l| (rx1 - rx0).max(ry1 - ry0) / py.levels[l].downsample <= 1024.0).unwrap_or(py.coarsest());
        let l = py.levels[level].clone();
        let ds = l.downsample;
        let (x0, y0) = ((rx0 / ds).floor() as usize, (ry0 / ds).floor() as usize);
        let w = (((rx1 - rx0) / ds).ceil() as usize).max(1).min(l.width.saturating_sub(x0));
        let h = (((ry1 - ry0) / ds).ceil() as usize).max(1).min(l.height.saturating_sub(y0));
        if w < 8 || h < 8 {
            return;
        }
        let plane = self.plane_at(level);
        let displays = v.displays.clone();
        let mut planes = Vec::new();
        for (c, d) in displays.iter().enumerate().filter(|(_, d)| d.visible) {
            match v.service.read_region(level, plane, c, x0, y0, w, h) {
                Ok(p) => planes.push((p, *d)),
                Err(e) => return self.store.update(cx, |s, cx| s.set_status(format!("Segment: {e}"), cx)),
            }
        }
        if planes.is_empty() {
            return self.store.update(cx, |s, cx| s.set_status("Segment: no channel is shown", cx));
        }
        let refs: Vec<(&[f32], ChannelDisplay)> = planes.iter().map(|(p, d)| (p.as_slice(), *d)).collect();
        let bgra = lucem_view::composite_bgra(&refs, w * h);
        let rgb: Vec<u8> = bgra.as_chunks::<4>().0.iter().flat_map(|p| [p[2], p[1], p[0]]).collect();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (&v.image, level, plane, x0, y0, w, h, format!("{:?}", displays)).hash(&mut hasher);
        let to = |p: [f64; 2]| (((p[0] / ds) - x0 as f64) as f32, ((p[1] / ds) - y0 as f64) as f32);
        let prompts = vec![match prompt {
            SamPrompt::Point(p, positive) => {
                let (x, y) = to(p);
                lucem_ml::Prompt::Point { x, y, positive }
            }
            SamPrompt::Box(a, b) => {
                let ((ax, ay), (bx, by)) = (to(a), to(b));
                lucem_ml::Prompt::Box { x0: ax.min(bx), y0: ay.min(by), x1: ax.max(bx), y1: ay.max(by) }
            }
        }];
        let job = crate::data::ml::SegmentJob {
            image: v.image.clone(),
            z: self.plane_of_drawing(),
            class: self.store.read(cx).tool.class.clone(),
            origin: [x0 as f64 * ds, y0 as f64 * ds],
            ds,
            width: w as u32,
            height: h as u32,
            rgb,
            prompts,
            key: format!("{:016x}", hasher.finish()),
        };
        // A volume, with "Through z": the object is followed on the planes of this level
        let through = (self.store.read(cx).tool.sam_through_z && py.is_volume()).then(|| crate::data::ml::ZRun {
            reader: v.service.region_reader(),
            level,
            plane,
            depth: l.depth,
            z_downsample: l.z_downsample,
            x0,
            y0,
            displays: displays.iter().enumerate().filter(|(_, d)| d.visible).map(|(c, d)| (c, *d)).collect(),
        });
        self.store.update(cx, |s, cx| s.segment(job, through, cx));
    }

    /// Starts a wand stroke at `p`: reads the visible region of the visible channels.
    fn start_wand(&mut self, p: [f64; 2], cx: &mut Context<Self>) -> Option<Wand> {
        let v = self.viewer.as_ref()?;
        let py = v.pyramid.clone();
        let level = py.level_for_zoom(self.zoom * self.scale as f64);
        let l = py.levels[level].clone();
        let ds = l.downsample;
        let rect = self.visible();
        let x0 = (rect[0] / ds).floor().max(0.0) as usize;
        let y0 = (rect[1] / ds).floor().max(0.0) as usize;
        let x1 = (((rect[0] + rect[2]) / ds).ceil().max(0.0) as usize).min(l.width);
        let y1 = (((rect[1] + rect[3]) / ds).ceil().max(0.0) as usize).min(l.height);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let (w, h) = (x1 - x0, y1 - y0);
        let plane = self.plane_at(level);
        let mut channels = Vec::new();
        for (c, d) in v.display_docs.iter().enumerate().filter(|(_, d)| d.visible) {
            let mut raw = v.service.read_region(level, plane, c, x0, y0, w, h).ok()?;
            let span = (d.max - d.min).max(1e-12);
            raw.iter_mut().for_each(|x| *x = ((*x - d.min) / span).clamp(0.0, 1.0));
            channels.push(raw);
        }
        if channels.is_empty() {
            return None;
        }
        let s = self.store.read(cx);
        let color = s.doc.class(&s.tool.class).map_or([255, 255, 255], |c| c.color);
        let mut wand = Wand { level, ds, x0, y0, w, h, channels, mask: vec![0; w * h], bbox: None, radius: self.brush_radius(cx), last: p, tolerance: s.tool.wand_tolerance, color, overlay: None };
        let _ = wand.level;
        wand.stamp(p);
        wand.refresh_overlay();
        Some(wand)
    }

    /// Asks the store for the results of what it shows.
    pub fn request_results(&mut self, cx: &mut Context<Self>) {
        let Some(v) = &self.viewer else { return };
        let req = ResultRequest { image: v.image.clone(), rect: self.visible(), zoom: self.zoom * self.scale as f64, z: self.z };
        self.store.update(cx, |s, cx| s.request_results(req, cx));
    }

    /// Centres the view on level-0 point `p` (keeps the zoom).
    pub fn reveal(&mut self, p: [f64; 2], cx: &mut Context<Self>) {
        self.center = (p[0], p[1]);
        self.camera_changed(cx);
    }

    // ------------------------------------------------------------------------
    // Tiles
    // ------------------------------------------------------------------------

    /// Follows the store: the shown image (a new viewer) and its channel display.
    fn sync_viewer(&mut self, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let active = self.image_id(s);
        let pyramid = active.as_ref().and_then(|id| s.pyramids.get(id).cloned());
        let mut docs: Vec<ChannelDoc> = active.as_ref().and_then(|id| s.doc.image(id)).map(|i| i.channels.clone()).unwrap_or_default();
        // A channel shown processed is drawn by its result, not raw (non-destructive processing)
        if let Some(id) = &active {
            for c in s.processed_channels(id).keys() {
                if let Some(d) = docs.get_mut(*c) {
                    d.visible = false;
                }
            }
        }
        // A cell of the Channels view draws its one channel
        if let Some(only) = self.channel {
            for (i, d) in docs.iter_mut().enumerate() {
                d.visible = i == only;
            }
        }
        match (&active, pyramid) {
            (Some(id), Some(py)) => {
                let same = self.viewer.as_ref().is_some_and(|v| &v.image == id && Arc::ptr_eq(&v.pyramid, &py));
                if !same {
                    if let Some(old) = self.viewer.take() {
                        self.drops.extend(old.tiles.into_values().map(|t| t.image));
                    }
                    let (service, rx) = TileService::new(py.clone(), self.threads);
                    let pump = cx.spawn(async move |this, cx| {
                        while let Ok(first) = rx.recv().await {
                            let mut batch = vec![first];
                            while let Ok(more) = rx.try_recv() {
                                batch.push(more);
                            }
                            if this.update(cx, |c, cx| c.on_tiles(batch, cx)).is_err() {
                                break;
                            }
                        }
                    });
                    self.viewer = Some(Viewer {
                        image: id.clone(),
                        pyramid: py,
                        service,
                        tiles: HashMap::new(),
                        failed: HashSet::new(),
                        rev: 0,
                        displays: Arc::new(Vec::new()),
                        display_docs: Vec::new(),
                        requested: Vec::new(),
                        requested_rev: u64::MAX,
                        _pump: pump,
                    });
                    self.polygon.clear();
                    self.gesture = None;
                    self.fit_pending = true;
                    self.z = self.store.read(cx).cursor_of(id).map_or(0, |c| c[0]);
                }
                let v = self.viewer.as_mut().expect("just set");
                if v.display_docs != docs {
                    v.display_docs = docs.clone();
                    v.rev += 1;
                    v.displays = Arc::new(docs.iter().map(|c| ChannelDisplay { visible: c.visible, color: [c.color[0] as f32 / 255.0, c.color[1] as f32 / 255.0, c.color[2] as f32 / 255.0], min: c.min, max: c.max, gamma: c.gamma }).collect());
                }
            }
            _ => {
                if let Some(old) = self.viewer.take() {
                    self.drops.extend(old.tiles.into_values().map(|t| t.image));
                }
            }
        }
    }

    fn on_tiles(&mut self, batch: Vec<TileResult>, cx: &mut Context<Self>) {
        let Some(v) = self.viewer.as_mut() else { return };
        for r in batch {
            match r {
                TileResult::Ready(t) => {
                    let keep = v.tiles.get(&t.key).is_none_or(|old| old.rev <= t.rev);
                    if keep {
                        if let Some(old) = v.tiles.insert(t.key, t) {
                            self.drops.push(old.image);
                        }
                    } else {
                        self.drops.push(t.image);
                    }
                }
                TileResult::Failed(k, e) => {
                    if v.failed.insert(k) && v.failed.len() == 1 {
                        let msg = format!("Some tiles could not be read: {e}");
                        self.store.update(cx, |s, cx| s.set_status(msg, cx));
                    }
                }
            }
        }
        cx.notify();
    }

    /// Asks for the tiles in view, evicts far ones; returns what to draw (coarse first).
    fn plan_tiles(&mut self) -> Vec<(Bounds<Pixels>, Arc<RenderImage>)> {
        let (vw, vh) = self.view_size();
        let rect = self.visible();
        let (zoom, scale, center, z0) = (self.zoom, self.scale as f64, self.center, self.z);
        let Some(v) = self.viewer.as_mut() else { return Vec::new() };
        let py = v.pyramid.clone();
        let target = py.level_for_zoom(zoom * scale);
        let coarsest = py.coarsest();
        let plane = |level: usize| {
            let l = &py.levels[level];
            ((z0 as f64 / l.z_downsample) as usize).min(l.depth - 1)
        };
        let tiles_in = |level: usize| -> Vec<TileKey> {
            let l = &py.levels[level];
            let ds = l.downsample;
            let z = plane(level);
            let (tw, th) = (l.tile_w as f64 * ds, l.tile_h as f64 * ds);
            let tx0 = (rect[0] / tw).floor().max(0.0) as usize;
            let ty0 = (rect[1] / th).floor().max(0.0) as usize;
            let tx1 = (((rect[0] + rect[2]) / tw).ceil().max(0.0) as usize).min(l.tiles_x());
            let ty1 = (((rect[1] + rect[3]) / th).ceil().max(0.0) as usize).min(l.tiles_y());
            let mut out = Vec::new();
            for ty in ty0..ty1 {
                for tx in tx0..tx1 {
                    out.push(TileKey { level, z, tx, ty });
                }
            }
            out
        };
        let dist = |k: &TileKey| {
            let l = &py.levels[k.level];
            let ds = l.downsample;
            let (mx, my) = ((k.tx as f64 + 0.5) * l.tile_w as f64 * ds, (k.ty as f64 + 0.5) * l.tile_h as f64 * ds);
            (mx - center.0).hypot(my - center.1)
        };
        // What to ask for: the coarsest level (whole, few tiles) first, then the target level
        let mut want: Vec<TileKey> = Vec::new();
        if coarsest != target {
            let l = &py.levels[coarsest];
            for ty in 0..l.tiles_y() {
                for tx in 0..l.tiles_x() {
                    want.push(TileKey { level: coarsest, z: plane(coarsest), tx, ty });
                }
            }
        }
        let mut fine = tiles_in(target);
        fine.sort_by(|a, b| dist(a).total_cmp(&dist(b)));
        want.extend(fine.iter().copied());
        let rev = v.rev;
        let missing: Vec<TileKey> = want.iter().copied().filter(|k| !v.failed.contains(k) && v.tiles.get(k).is_none_or(|t| t.rev < rev)).collect();
        if missing != v.requested || v.requested_rev != rev {
            v.service.request(rev, v.displays.clone(), missing.clone());
            v.requested = missing;
            v.requested_rev = rev;
        }
        // Evict: never the coarsest level or what is in view at the target level
        if v.tiles.len() > TILE_CAP {
            let keep: HashSet<TileKey> = fine.iter().copied().collect();
            let mut extra: Vec<TileKey> = v.tiles.keys().copied().filter(|k| k.level != coarsest && !keep.contains(k)).collect();
            extra.sort_by(|a, b| {
                let la = (a.level as i64 - target as i64).abs();
                let lb = (b.level as i64 - target as i64).abs();
                lb.cmp(&la).then(dist(b).total_cmp(&dist(a)))
            });
            let n = v.tiles.len() - TILE_CAP;
            for k in extra.into_iter().take(n) {
                if let Some(t) = v.tiles.remove(&k) {
                    self.drops.push(t.image);
                }
            }
        }
        // Draw: from the coarsest level down to the target, what is in view
        let mut draw = Vec::new();
        let v = self.viewer.as_ref().expect("checked");
        for level in (target..=coarsest).rev() {
            for k in tiles_in(level) {
                let Some(t) = v.tiles.get(&k) else { continue };
                let l = &py.levels[level];
                let ds = l.downsample;
                let x0 = (k.tx * l.tile_w) as f64 * ds;
                let y0 = (k.ty * l.tile_h) as f64 * ds;
                let (sx, sy) = (((x0 - center.0) * zoom + vw as f64 / 2.0) as f32, ((y0 - center.1) * zoom + vh as f64 / 2.0) as f32);
                let (sw, sh) = ((t.width as f64 * ds * zoom) as f32, (t.height as f64 * ds * zoom) as f32);
                // Half a pixel of overlap hides seams between neighbours
                // Plane-local: the paint adds where the plane is now
                let b = Bounds { origin: point(px(sx - 0.25), px(sy - 0.25)), size: size(px(sw + 0.5), px(sh + 0.5)) };
                draw.push((b, t.image.clone()));
            }
        }
        draw
    }

    /// Pyramid level and raw values under the pointer.
    fn readout(&self) -> Option<([f64; 2], usize, Vec<Option<f32>>)> {
        let v = self.viewer.as_ref()?;
        let m = self.mouse?;
        let p = self.to_image(m);
        if p[0] < 0.0 || p[1] < 0.0 || p[0] >= v.pyramid.width() as f64 || p[1] >= v.pyramid.height() as f64 {
            return Some((p, 0, Vec::new()));
        }
        let level = v.pyramid.level_for_zoom(self.zoom * self.scale as f64);
        let ds = v.pyramid.levels[level].downsample;
        Some((p, level, v.service.sample(level, self.plane_at(level), (p[0] / ds) as usize, (p[1] / ds) as usize)))
    }

    // ------------------------------------------------------------------------
    // Tools
    // ------------------------------------------------------------------------

    fn tool(&self, cx: &App) -> Tool {
        self.store.read(cx).tool.tool
    }

    /// The brush radius in level-0 px.
    fn brush_radius(&self, cx: &App) -> f64 {
        let t = &self.store.read(cx).tool;
        if t.brush_in_image { t.brush_image_px } else { t.brush_px as f64 / self.zoom }
    }

    fn add_annotation(&mut self, shape: Shape, cx: &mut Context<Self>) {
        let Some(image) = self.viewer.as_ref().map(|v| v.image.clone()) else { return };
        let class = if matches!(shape, Shape::Line { .. }) { "Measurement".to_string() } else { self.store.read(cx).tool.class.clone() };
        let z = self.plane_of_drawing();
        let a = Annotation { id: String::new(), image, shape, class, note: String::new(), z };
        self.store.update(cx, |s, cx| {
            if s.execute(Command::AddAnnotation { annotation: a }, cx).is_ok()
                && let Some(last) = s.doc.annotations.last().map(|a| a.id.clone()) {
                    s.select([last], cx);
                }
        });
    }

    /// The outline(s) of what a brush stroke painted (level-0 px), traced once per version.
    fn stroke_outlines(&self, id: &str, points: &[[f64; 2]], radius: f64) -> Vec<Vec<[f64; 2]>> {
        let key = format!("{id}/{}/{radius}/{:?}", points.len(), points.last());
        if let Some(o) = self.outlines.borrow().get(&key) {
            return o.clone();
        }
        let o = traced(points, radius, false);
        let mut cache = self.outlines.borrow_mut();
        if cache.len() > 64 {
            cache.clear();
        }
        cache.insert(key, o.clone());
        o
    }

    /// The plane new annotations belong to: the shown plane of a volume, none on a 2D image.
    fn plane_of_drawing(&self) -> Option<usize> {
        self.viewer.as_ref().filter(|v| v.pyramid.is_volume()).map(|_| self.z)
    }

    /// The topmost annotation of the shown image at `p`.
    fn hit(&self, p: [f64; 2], cx: &App) -> Option<String> {
        let image = self.viewer.as_ref()?.image.clone();
        let s = self.store.read(cx);
        let tol = 6.0 / self.zoom;
        s.doc.annotations.iter().rev().filter(|a| a.image == image && a.on_plane(self.z) && s.doc.class(&a.class).is_none_or(|c| c.visible)).find(|a| a.shape.hit(p[0], p[1], tol)).map(|a| a.id.clone())
    }

    fn close_polygon(&mut self, cx: &mut Context<Self>) {
        let pts = std::mem::take(&mut self.polygon);
        if pts.len() >= 3 {
            self.add_annotation(Shape::Polygon { points: pts }, cx);
        }
        cx.notify();
    }

    fn on_down(&mut self, e: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.focus, cx);
        if self.viewer.is_none() {
            return;
        }
        let local = self.local(e.position);
        let p = self.to_image(local);
        let tool = self.tool(cx);
        if e.button == MouseButton::Middle || self.space || tool == Tool::Hand {
            self.gesture = Some(Gesture::Pan { mouse: local, center: self.center });
            return;
        }
        match tool {
            Tool::Zoom => self.zoom_at(if e.modifiers.alt { 0.5 } else { 2.0 }, Some(local), cx),
            Tool::Probe => {
                // The 3D cursor to the pixel clicked (the other views follow)
                if let Some(id) = self.viewer.as_ref().map(|v| v.image.clone()) {
                    let at = [self.z, p[1].max(0.0) as usize, p[0].max(0.0) as usize];
                    self.store.update(cx, |s, cx| s.set_cursor_of(&id, at, cx));
                }
                self.gesture = Some(Gesture::Pan { mouse: local, center: self.center });
            }
            Tool::Select => {
                match self.hit(p, cx) {
                    Some(id) => {
                        let sel = self.store.read(cx).sel.shapes.clone();
                        if e.modifiers.shift {
                            let mut s = sel.clone();
                            if !s.remove(&id) {
                                s.insert(id);
                            }
                            self.store.update(cx, |st, cx| st.select(s, cx));
                        } else if !sel.contains(&id) {
                            self.store.update(cx, |st, cx| st.select([id], cx));
                        }
                        self.gesture = Some(Gesture::Move { start: p, now: p, moved: false });
                    }
                    None => {
                        // A detected cell (or tile) under the pointer: select it
                        let object = self.store.read(cx).object_at(p);
                        match object {
                            Some(key) => self.store.update(cx, |st, cx| {
                                if !e.modifiers.shift {
                                    st.select([], cx);
                                }
                                st.select_object(key, e.modifiers.shift, cx)
                            }),
                            None if !e.modifiers.shift => self.store.update(cx, |st, cx| {
                                st.select([], cx);
                                st.sel.objects.clear();
                                cx.notify();
                            }),
                            None => {}
                        }
                        self.gesture = Some(Gesture::Pan { mouse: local, center: self.center });
                    }
                }
            }
            Tool::Rect | Tool::Ellipse | Tool::Ruler | Tool::Segment | Tool::Marquee => self.gesture = Some(Gesture::Drag { tool, a: p, b: p }),
            Tool::Lasso => self.gesture = Some(Gesture::Path { tool, points: vec![p], radius: 0.0 }),
            Tool::Brush | Tool::Eraser => {
                let radius = self.brush_radius(cx);
                self.gesture = Some(Gesture::Path { tool, points: vec![p], radius });
                self.refresh_live(cx);
            }
            Tool::Wand => {
                if let Some(w) = self.start_wand(p, cx) {
                    self.gesture = Some(Gesture::Wand(Box::new(w)));
                }
            }
            Tool::Threshold => self.gesture = Some(Gesture::Pan { mouse: local, center: self.center }),
            Tool::Polygon => {
                if e.click_count >= 2 {
                    self.close_polygon(cx);
                } else {
                    // A click on the first corner closes it too
                    let near_first = self.polygon.first().is_some_and(|f| {
                        let s = self.to_screen(*f);
                        (s.0 - local.0).hypot(s.1 - local.1) < 8.0
                    });
                    if near_first && self.polygon.len() >= 3 {
                        self.close_polygon(cx);
                    } else {
                        self.polygon.push(p);
                    }
                }
            }
            Tool::Point => self.add_annotation(Shape::Point { x: p[0], y: p[1] }, cx),
            Tool::Hand => {}
        }
        cx.notify();
    }

    fn on_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        let local = self.local(e.position);
        self.mouse = Some(local);
        let p = self.to_image(local);
        if e.pressed_button.is_none() && self.gesture.is_some() {
            // The button went up outside the plane
            self.gesture = None;
        }
        let zoom = self.zoom;
        let mut repaint_live = false;
        match &mut self.gesture {
            Some(Gesture::Pan { mouse, center }) => {
                self.center = (center.0 - (local.0 - mouse.0) as f64 / zoom, center.1 - (local.1 - mouse.1) as f64 / zoom);
                self.camera_changed(cx);
                return;
            }
            Some(Gesture::Drag { b, .. }) => *b = p,
            // Shift: a straight line from where the stroke started
            Some(Gesture::Path { points, tool, .. }) if e.modifiers.shift && *tool != Tool::Lasso => {
                points.truncate(1);
                points.push(p);
                repaint_live = true;
            }
            Some(Gesture::Path { points, radius, tool }) => {
                let last = *points.last().expect("starts with one point");
                // Points every 2 screen px (fine enough on screen, few enough to draw fast)
                let step = if *tool == Tool::Lasso { 2.0 / zoom } else { (2.0 / zoom).max(*radius * 0.1) };
                if (p[0] - last[0]).hypot(p[1] - last[1]) >= step {
                    points.push(p);
                    repaint_live = *tool != Tool::Lasso;
                }
            }
            Some(Gesture::Wand(w)) => {
                let old = w.overlay.as_ref().map(|o| o.1.clone());
                w.stroke_to(p);
                if let (Some(o), Some(n)) = (old, w.overlay.as_ref())
                    && !Arc::ptr_eq(&o, &n.1) {
                        self.drops.push(o);
                    }
            }
            Some(Gesture::Move { start, now, moved }) => {
                *now = p;
                if ((now[0] - start[0]) * zoom).hypot((now[1] - start[1]) * zoom) > 3.0 {
                    *moved = true;
                }
            }
            None => {}
        }
        if repaint_live {
            self.refresh_live(cx);
        }
        cx.notify();
    }

    fn on_up(&mut self, e: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        let Some(g) = self.gesture.take() else { return };
        // Segment: a click is a point (Alt: not this), a drag a box
        if let Gesture::Drag { tool: Tool::Segment, a, b } = &g {
            let tiny = ((b[0] - a[0]) * self.zoom).abs() < 4.0 && ((b[1] - a[1]) * self.zoom).abs() < 4.0;
            let prompt = if tiny { SamPrompt::Point(*a, !e.modifiers.alt) } else { SamPrompt::Box(*a, *b) };
            self.segment(prompt, cx);
            cx.notify();
            return;
        }
        if let Some(l) = self.live.take() {
            self.drops.push(l.1);
        }
        match g {
            Gesture::Pan { .. } => {}
            Gesture::Drag { tool, a, b } => {
                let (x0, y0, x1, y1) = (a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1]));
                let tiny = (x1 - x0) * self.zoom < 3.0 && (y1 - y0) * self.zoom < 3.0;
                if tool == Tool::Marquee {
                    // A box selects what is inside it; a click (no box) clears the selection
                    let add = e.modifiers.shift;
                    self.store.update(cx, |s, cx| s.select_in_rect(if tiny { None } else { Some([x0, y0, x1, y1]) }, add, cx));
                } else if !tiny {
                    let shape = match tool {
                        Tool::Rect => Shape::Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 },
                        Tool::Ellipse => Shape::Ellipse { cx: (x0 + x1) / 2.0, cy: (y0 + y1) / 2.0, rx: (x1 - x0) / 2.0, ry: (y1 - y0) / 2.0 },
                        _ => Shape::Line { x0: a[0], y0: a[1], x1: b[0], y1: b[1] },
                    };
                    self.add_annotation(shape, cx);
                }
            }
            Gesture::Path { tool, points, radius } => match tool {
                Tool::Lasso if points.len() >= 3 => self.add_annotation(Shape::Freehand { points }, cx),
                Tool::Brush | Tool::Eraser => {
                    // A stroke ending near its start also fills what it encloses (when asked)
                    let closed = points.len() >= 8 && (points[0][0] - points[points.len() - 1][0]).hypot(points[0][1] - points[points.len() - 1][1]) <= radius * 2.0;
                    let filled = (tool == Tool::Brush && closed && self.store.read(cx).tool.brush_fill).then(|| filled_outline(&points, radius)).flatten();
                    if let Some(outline) = filled {
                        // One region: the stroke and what it encloses
                        self.add_annotation(Shape::Polygon { points: outline }, cx);
                    } else {
                        self.add_annotation(Shape::Stroke { points, radius, erase: tool == Tool::Eraser }, cx);
                    }
                }
                _ => {}
            },
            Gesture::Wand(w) => {
                if let Some(o) = &w.overlay {
                    self.drops.push(o.1.clone());
                }
                let shapes: Vec<Shape> = w.polygons().into_iter().filter(|p| p.len() >= 3).map(|points| Shape::Polygon { points }).collect();
                if let Some(image) = self.viewer.as_ref().map(|v| v.image.clone()) {
                    let class = self.store.read(cx).tool.class.clone();
                    let z = self.plane_of_drawing();
                    let annotations: Vec<Annotation> = shapes.into_iter().map(|shape| Annotation { id: String::new(), image: image.clone(), shape, class: class.clone(), note: String::new(), z }).collect();
                    if !annotations.is_empty() {
                        self.store.update(cx, |s, cx| {
                            let _ = s.execute(Command::AddAnnotations { annotations }, cx);
                        });
                    }
                }
            }
            Gesture::Move { start, now, moved } => {
                if moved {
                    let ids: Vec<String> = self.store.read(cx).sel.shapes.iter().cloned().collect();
                    let (dx, dy) = (now[0] - start[0], now[1] - start[1]);
                    self.store.update(cx, |s, cx| {
                        let _ = s.execute(Command::MoveAnnotations { ids, dx, dy }, cx);
                    });
                }
            }
        }
        cx.notify();
    }

    fn on_wheel(&mut self, e: &ScrollEvent, _: &mut Window, cx: &mut Context<Self>) {
        let local = self.local(e.position);
        if e.modifiers.alt {
            // Alt + wheel moves through the planes
            let dy = match e.delta {
                ScrollDelta::Lines(d) => d.y as f64,
                ScrollDelta::Pixels(d) => d.y.as_f32() as f64,
            };
            let step = if dy > 0.0 { -1 } else if dy < 0.0 { 1 } else { 0 };
            self.set_z((self.z as i64 + step).max(0) as usize, cx);
            cx.stop_propagation();
            return;
        }
        match e.delta {
            ScrollDelta::Lines(d) => {
                self.zoom_at(1.25f64.powf(-d.y as f64), Some(local), cx);
            }
            ScrollDelta::Pixels(d) => {
                let (dx, dy) = (d.x.as_f32() as f64, d.y.as_f32() as f64);
                if e.modifiers.control || e.modifiers.platform {
                    // Pinch (and Ctrl + swipe): zoom at the pointer
                    self.zoom_at((-dy * 0.01).exp(), Some(local), cx);
                } else {
                    self.center = (self.center.0 - dx / self.zoom, self.center.1 - dy / self.zoom);
                    self.camera_changed(cx);
                }
            }
        }
        cx.stop_propagation();
    }

    fn on_key(&mut self, e: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        let k = &e.keystroke;
        let cmd = k.modifiers.control || k.modifiers.platform;
        let key = k.key.as_str();
        match key {
            "space" => self.space = true,
            "escape" => {
                if !self.polygon.is_empty() || self.gesture.is_some() {
                    self.polygon.clear();
                    self.gesture = None;
                } else {
                    self.store.update(cx, |s, cx| s.select([], cx));
                }
            }
            "enter" => self.close_polygon(cx),
            "backspace" if !self.polygon.is_empty() => {
                self.polygon.pop();
            }
            "delete" | "backspace" => self.store.update(cx, |s, cx| s.delete_selection(cx)),
            "pageup" => self.set_z(self.z.saturating_sub(1), cx),
            "pagedown" => self.set_z(self.z + 1, cx),
            // Overlay toggles: A annotations, Shift+F fill, Shift+N names
            "a" if !cmd && !k.modifiers.alt && !k.modifiers.shift => self.toggle_overlay(|o| o.annotations = !o.annotations, cx),
            "f" if !cmd && k.modifiers.shift => self.toggle_overlay(|o| o.fill = !o.fill, cx),
            "n" if !cmd && k.modifiers.shift => self.toggle_overlay(|o| o.names = !o.names, cx),
            "0" if cmd => self.fit(cx),
            "1" if cmd => self.actual_size(cx),
            "=" | "+" => self.zoom_at(1.5, None, cx),
            "-" => self.zoom_at(1.0 / 1.5, None, cx),
            "f" if !cmd => self.fit(cx),
            "[" | "]" => {
                let dir = if key == "]" { 1.0 } else { -1.0 };
                self.store.update(cx, |s, cx| {
                    s.update_tool(
                        |t| {
                            if t.brush_in_image {
                                t.brush_image_px = (t.brush_image_px * if dir > 0.0 { 1.25 } else { 0.8 }).clamp(1.0, 5000.0);
                            } else {
                                t.brush_px = step_brush(t.brush_px, dir);
                            }
                        },
                        cx,
                    )
                });
            }
            _ if !cmd && !k.modifiers.alt => match Tool::from_key(key) {
                Some(t) => {
                    self.polygon.clear();
                    self.store.update(cx, |s, cx| s.set_tool(t, cx));
                }
                None => return,
            },
            _ => return,
        }
        cx.stop_propagation();
        cx.notify();
    }

    fn on_key_up(&mut self, e: &KeyUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        if e.keystroke.key == "space" {
            self.space = false;
            cx.notify();
        }
    }

    // ------------------------------------------------------------------------
    // Drawing
    // ------------------------------------------------------------------------

    /// Annotations (and the shape being drawn) as screen paths.
    fn annotation_paths(&self, cx: &App) -> Vec<Stroke> {
        let Some(v) = &self.viewer else { return Vec::new() };
        let s = self.store.read(cx);
        let zoom = self.zoom as f32;
        let mut out = Vec::new();
        let delta = match &self.gesture {
            Some(Gesture::Move { start, now, moved: true }) => (now[0] - start[0], now[1] - start[1]),
            _ => (0.0, 0.0),
        };
        let rect = self.visible();
        let ov = self.overlays.unwrap_or_else(|| s.tool.overlays());
        let (shown, filled, opacity) = (ov.annotations, ov.fill, ov.opacity);
        for a in s.doc.annotations.iter().filter(|a| shown && a.image == v.image && a.on_plane(self.z)) {
            let Some(class) = s.doc.class(&a.class) else { continue };
            if !class.visible {
                continue;
            }
            let selected = s.sel.shapes.contains(&a.id);
            // Brush and eraser strokes are in the paint raster; a selected one shows the outline
            // of what it painted
            if matches!(a.shape, Shape::Stroke { .. }) && !selected {
                continue;
            }
            let shape = if selected && delta != (0.0, 0.0) { a.shape.translated(delta.0, delta.1) } else { a.shape.clone() };
            let (bx0, by0, bx1, by1) = shape.bounds();
            if bx1 < rect[0] || by1 < rect[1] || bx0 > rect[0] + rect[2] || by0 > rect[1] + rect[3] {
                continue;
            }
            let c = theme::rgb8(class.color).opacity((opacity * class.opacity).max(0.15));
            if let Shape::Stroke { points, radius, erase: false } = &shape {
                for outline in self.stroke_outlines(&a.id, points, *radius) {
                    let pts: Vec<(f32, f32)> = outline.iter().map(|p| self.to_screen(*p)).collect();
                    out.push(Stroke { points: geometry::thin(&pts, 1.5), closed: true, fill: None, width: 2.0, color: c, selected: true, dot: false });
                }
                continue;
            }
            let mut st = stroke_of(&shape, c, selected, zoom, |p| self.to_screen(p));
            if let Some(Fill::Area(alpha)) = st.fill {
                st.fill = filled.then_some(Fill::Area(alpha * opacity));
            }
            out.push(st);
        }
        // In progress
        let live = theme::rgb8(s.doc.class(&s.tool.class).map_or([255, 255, 255], |c| c.color));
        match &self.gesture {
            Some(Gesture::Drag { tool, a, b }) => {
                let (x0, y0, x1, y1) = (a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1]));
                let shape = match tool {
                    Tool::Rect | Tool::Segment | Tool::Marquee => Shape::Rect { x: x0, y: y0, w: (x1 - x0).max(1e-6), h: (y1 - y0).max(1e-6) },
                    Tool::Ellipse => Shape::Ellipse { cx: (x0 + x1) / 2.0, cy: (y0 + y1) / 2.0, rx: (x1 - x0).max(1e-6) / 2.0, ry: (y1 - y0).max(1e-6) / 2.0 },
                    _ => Shape::Line { x0: a[0], y0: a[1], x1: b[0], y1: b[1] },
                };
                out.push(stroke_of(&shape, if matches!(tool, Tool::Ruler | Tool::Marquee) { gpui_kit::white() } else { live }, true, zoom, |p| self.to_screen(p)));
            }
            // Brush and eraser strokes in progress are the live raster
            Some(Gesture::Path { tool: Tool::Lasso, points, .. }) => {
                let st = stroke_of(&Shape::Polygon { points: points.clone() }, live, false, zoom, |p| self.to_screen(p));
                out.push(st);
            }
            _ => {}
        }
        if !self.polygon.is_empty() {
            let mut pts: Vec<(f32, f32)> = self.polygon.iter().map(|p| self.to_screen(*p)).collect();
            if let Some(m) = self.mouse {
                pts.push(m);
            }
            out.push(Stroke { points: pts, closed: false, fill: None, width: 1.5, color: live, selected: true, dot: false });
        }
        out
    }

    /// Labels over the plane: ruler lengths, the selected region's area.
    fn labels(&self, cx: &App) -> Vec<(f32, f32, String)> {
        let Some(v) = &self.viewer else { return Vec::new() };
        let s = self.store.read(cx);
        let um = v.pyramid.pixel_size_um;
        let (w, h) = self.view_size();
        let mut out = Vec::new();
        let mut push = |p: (f32, f32), text: String| {
            if p.0 > -50.0 && p.1 > -20.0 && p.0 < w + 50.0 && p.1 < h + 20.0 {
                out.push((p.0, p.1, text));
            }
        };
        for a in s.doc.annotations.iter().filter(|a| a.image == v.image && a.on_plane(self.z)) {
            match &a.shape {
                Shape::Line { x0, y0, x1, y1 } => {
                    let len = a.shape.length().unwrap_or(0.0);
                    push(self.to_screen([(x0 + x1) / 2.0, (y0 + y1) / 2.0]), format_length(len, um));
                }
                shape if s.sel.shapes.contains(&a.id) && shape.is_region() => {
                    let (x0, y0, _, _) = shape.bounds();
                    let (sx, sy) = self.to_screen([x0, y0]);
                    push((sx, sy - 20.0), crate::ui::widgets::format_area(shape.area().unwrap_or(0.0), um));
                }
                _ => {}
            }
        }
        // Names (class, or the note) at the middle of each region and point
        let ov = self.overlays.unwrap_or_else(|| s.tool.overlays());
        if ov.names && ov.annotations {
            for a in s.doc.annotations.iter().filter(|a| a.image == v.image && a.on_plane(self.z)) {
                if matches!(a.shape, Shape::Line { .. } | Shape::Stroke { erase: true, .. }) || s.doc.class(&a.class).is_some_and(|c| !c.visible) {
                    continue;
                }
                let (x0, y0, x1, y1) = a.shape.bounds();
                let (sx, sy) = self.to_screen([(x0 + x1) / 2.0, (y0 + y1) / 2.0]);
                push((sx - 20.0, sy), if a.note.is_empty() { a.class.clone() } else { a.note.clone() });
            }
        }
        if let Some(Gesture::Drag { tool: Tool::Ruler, a, b }) = &self.gesture {
            let len = (b[0] - a[0]).hypot(b[1] - a[1]);
            push(self.to_screen([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]), format_length(len, um));
        }
        out
    }

    fn cursor(&self, cx: &App) -> CursorStyle {
        match (&self.gesture, self.tool(cx)) {
            (Some(Gesture::Pan { .. }), _) => CursorStyle::ClosedHand,
            _ if self.space => CursorStyle::OpenHand,
            (_, Tool::Hand) => CursorStyle::OpenHand,
            (_, Tool::Select) => CursorStyle::Arrow,
            (_, Tool::Zoom) => CursorStyle::PointingHand,
            _ => CursorStyle::Crosshair,
        }
    }

    /// Ruler ticks along one axis: (screen offset, label) every step, for `len` px of plane
    /// starting at level-0 coordinate `start`.
    fn ticks(&self, start: f64, len: f32, um: Option<f64>) -> Vec<(f32, String)> {
        let unit = um.unwrap_or(1.0);
        // A step of about 80 screen px, rounded to 1 / 2 / 5 × 10^k units
        let raw = 80.0 / self.zoom * unit;
        let mag = 10f64.powf(raw.log10().floor());
        let step = [1.0, 2.0, 5.0, 10.0].iter().map(|m| m * mag).find(|s| *s >= raw).unwrap_or(10.0 * mag);
        let first = (start * unit / step).floor() * step;
        let mut out = Vec::new();
        let mut v = first;
        while out.len() < 200 {
            let screen = ((v / unit - start) * self.zoom) as f32;
            if screen > len {
                break;
            }
            if screen >= 0.0 {
                out.push((screen, trim(v)));
            }
            v += step;
        }
        out
    }

    /// A scale bar of about 120 screen px: (px, label).
    fn scale_bar(&self, um: Option<f64>) -> (f32, String) {
        let unit = um.unwrap_or(1.0);
        let raw = 120.0 / self.zoom * unit;
        let mag = 10f64.powf(raw.log10().floor());
        let len = [1.0, 2.0, 5.0, 10.0].iter().map(|m| m * mag).rev().find(|s| *s <= raw).unwrap_or(mag);
        let px_len = (len / unit * self.zoom) as f32;
        let label = match um {
            Some(_) if len >= 1000.0 => format!("{} mm", trim(len / 1000.0)),
            Some(_) => format!("{} µm", trim(len)),
            None => format!("{} px", trim(len)),
        };
        (px_len, label)
    }
}

/// A path to draw.
#[derive(Clone)]
struct Stroke {
    points: Vec<(f32, f32)>,
    closed: bool,
    /// Fill of a region / brush stroke width (screen px).
    fill: Option<Fill>,
    width: f32,
    color: Hsla,
    selected: bool,
    /// A point marker.
    dot: bool,
}

#[derive(Clone, Copy)]
enum Fill {
    Area(f32),
}

fn stroke_of(shape: &Shape, color: Hsla, selected: bool, _zoom: f32, to_screen: impl Fn([f64; 2]) -> (f32, f32)) -> Stroke {
    // At screen resolution: a long stroke seen zoomed out is a few points to draw
    let points: Vec<(f32, f32)> = geometry::thin(&shape.outline().into_iter().map(&to_screen).collect::<Vec<_>>(), 1.5);
    match shape {
        Shape::Point { .. } => Stroke { points, closed: false, fill: None, width: 1.5, color, selected, dot: true },
        Shape::Line { .. } => Stroke { points, closed: false, fill: None, width: 1.5, color, selected, dot: false },
        // A selected stroke: its path (the paint itself is a raster)
        Shape::Stroke { .. } => Stroke { points, closed: false, fill: None, width: 1.5, color, selected, dot: false },
        // Faint fills (≤ 15 %): the image under a region stays readable; the selection is
        // its thicker outline and handles (`simultaneous-contrast`)
        _ => Stroke { points, closed: true, fill: Some(Fill::Area(if selected { 0.14 } else { 0.10 })), width: if selected { 2.0 } else { 1.5 }, color, selected, dot: false },
    }
}

/// `120` rather than `120.0`; short decimals.
fn trim(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{v:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn paint_stroke(window: &mut Window, o: gpui_kit::Point<Pixels>, s: &Stroke) {
    let at = |p: &(f32, f32)| point(o.x + px(p.0), o.y + px(p.1));
    let pts: Vec<gpui_kit::Point<Pixels>> = s.points.iter().map(at).collect();
    if s.dot {
        if let Some(c) = pts.first() {
            let r = if s.selected { 6.0 } else { 4.5 };
            let mut b = PathBuilder::fill();
            b.add_polygon(&circle(*c, r + 1.5), true);
            if let Ok(p) = b.build() {
                window.paint_path(p, gpui_kit::black().opacity(0.7));
            }
            let mut b = PathBuilder::fill();
            b.add_polygon(&circle(*c, r), true);
            if let Ok(p) = b.build() {
                window.paint_path(p, s.color);
            }
        }
        return;
    }
    if pts.len() < 2 {
        return;
    }
    match s.fill {
        Some(Fill::Area(alpha)) if pts.len() >= 3 => {
            let mut b = PathBuilder::fill();
            b.add_polygon(&pts, true);
            if let Ok(p) = b.build() {
                window.paint_path(p, s.color.opacity(alpha));
            }
        }
        _ => {}
    }
    // A dark halo under the line keeps it visible on bright tissue
    for (w, c) in [(s.width + 2.0, gpui_kit::black().opacity(0.55)), (s.width, s.color)] {
        let mut b = PathBuilder::stroke(px(w));
        b.add_polygon(&pts, s.closed);
        if let Ok(p) = b.build() {
            window.paint_path(p, c);
        }
    }
    if s.selected && s.fill.is_some() {
        for p in pts.iter().step_by((pts.len() / 24).max(1)) {
            let r = 3.0;
            window.paint_quad(fill(Bounds { origin: point(p.x - px(r), p.y - px(r)), size: size(px(2.0 * r), px(2.0 * r)) }, gpui_kit::white()));
        }
    }
}

fn circle(c: gpui_kit::Point<Pixels>, r: f32) -> Vec<gpui_kit::Point<Pixels>> {
    let n = 48;
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32 * std::f32::consts::TAU;
            point(c.x + px(r * t.cos()), c.y + px(r * t.sin()))
        })
        .collect()
}

impl Render for ImageCanvas {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.scale = window.scale_factor();
        self.sync_viewer(cx);
        if self.fit_pending && self.viewer.is_some() && self.view_size().0 > 10.0 {
            self.fit(cx);
        }
        let tiles = self.plan_tiles();
        self.sync_threshold(cx);
        self.sync_paint(cx);
        self.sync_cells(cx);
        let s = self.store.read(cx);
        let mut tool = s.tool.clone();
        if let Some(o) = self.overlays {
            tool.set_overlays(o);
        }
        let image = self.viewer.as_ref().map(|v| v.image.clone());
        let um = self.viewer.as_ref().and_then(|v| v.pyramid.pixel_size_um);
        // Result layers of the shown image
        let mut overlays: Vec<([f64; 4], Arc<RenderImage>)> = Vec::new();
        if tool.show_results {
            for n in s.doc.graph.views() {
                let visible = matches!(&n.kind, crate::domain::graph::NodeKind::View { style, .. } if style.visible);
                if let Some(r) = s.results.get(&n.id).filter(|r| visible && Some(&r.image) == image.as_ref())
                    && let Some(l) = &r.layer {
                        overlays.push((r.rect, l.clone()));
                    }
            }
        }
        // Histology: class map, density map, then cells and tiles
        if let Some(img) = image.as_ref().filter(|_| s.hist.show_maps && tool.show_results) {
            if let Some(m) = s.hist.class_maps.get(img) {
                overlays.push((m.rect, m.image.clone()));
            }
            if let Some(d) = s.hist.density_maps.get(img) {
                overlays.push((d.rect, d.image.clone()));
            }
        }
        if let Some(p) = &self.cells
            && let Some(img) = &p.image {
                overlays.push((p.rect, img.clone()));
            }
        if let Some(p) = &self.paint
            && let Some(img) = &p.image {
                overlays.push((p.rect, img.clone()));
            }
        if let Some(l) = &self.live {
            overlays.push((l.0, l.1.clone()));
        }
        if let Some(t) = &self.threshold {
            overlays.push((t.0, t.1.clone()));
        }
        if let Some(Gesture::Wand(w)) = &self.gesture
            && let Some(o) = &w.overlay {
                overlays.push((o.0, o.1.clone()));
            }
        let strokes = self.annotation_paths(cx);
        let labels = self.labels(cx);
        let (zoom, center) = (self.zoom, self.center);
        let (vw, vh) = self.view_size();
        // The brush outline at true size
        let brush = match (tool.tool.uses_brush(), self.mouse, &self.gesture) {
            (true, Some(m), None | Some(Gesture::Path { .. })) => Some((m, (self.brush_radius(cx) * zoom) as f32)),
            _ => None,
        };
        let cursor = self.cursor(cx);
        // Where the image lies (plane-local px), for its frame and shadow
        let frame = self.viewer.as_ref().map(|v| {
            let (a, b) = (self.to_screen([0.0, 0.0]), self.to_screen([v.pyramid.width() as f64, v.pyramid.height() as f64]));
            (a.0, a.1, b.0 - a.0, b.1 - a.1)
        });
        // The 3D cursor as a crosshair (x red, y green)
        let crosshair = if tool.crosshair {
            self.viewer.as_ref().and_then(|v| s.cursor_of(&v.image)).map(|c| self.to_screen([c[2] as f64 + 0.5, c[1] as f64 + 0.5]))
        } else {
            None
        };
        let drops = std::mem::take(&mut self.drops);
        let bounds_cell = self.bounds.clone();
        let plane = canvas(
            move |bounds, window, _| {
                // The plane moved or changed size (a side panel opened or closed, a pane was
                // added): everything was laid out for the old bounds, so render again
                if bounds_cell.get() != bounds {
                    bounds_cell.set(bounds);
                    window.refresh();
                }
            },
            move |bounds, _, window, _| {
                for img in drops {
                    let _ = window.drop_image(img);
                }
                let o = bounds.origin;
                window.paint_quad(fill(bounds, color::canvas()));
                window.with_content_mask(Some(gpui_kit::ContentMask { bounds }), |window| {
                    // The image as a document on the pasteboard: a soft shadow and a black page,
                    // so dark edges of a fluorescence image still read as its edge
                    if let Some((fx, fy, fw, fh)) = frame {
                        for (grow, alpha) in [(10.0, 0.10), (6.0, 0.14), (3.0, 0.22)] {
                            window.paint_quad(fill(Bounds { origin: point(o.x + px(fx - grow + 2.0), o.y + px(fy - grow + 4.0)), size: size(px(fw + 2.0 * grow), px(fh + 2.0 * grow)) }, gpui_kit::black().opacity(alpha)));
                        }
                        window.paint_quad(fill(Bounds { origin: point(o.x + px(fx), o.y + px(fy)), size: size(px(fw), px(fh)) }, gpui_kit::black()));
                    }
                    for (b, img) in tiles {
                        let b = Bounds { origin: point(o.x + b.origin.x, o.y + b.origin.y), size: b.size };
                        let _ = window.paint_image(bounds, b, Corners::default(), img, 0, false);
                    }
                    for (r, img) in overlays {
                        let sx = ((r[0] - center.0) * zoom + vw as f64 / 2.0) as f32;
                        let sy = ((r[1] - center.1) * zoom + vh as f64 / 2.0) as f32;
                        let b = Bounds { origin: point(o.x + px(sx), o.y + px(sy)), size: size(px((r[2] * zoom) as f32), px((r[3] * zoom) as f32)) };
                        let _ = window.paint_image(bounds, b, Corners::default(), img, 0, false);
                    }
                    for st in &strokes {
                        paint_stroke(window, o, st);
                    }
                    if let Some((cx_, cy_)) = crosshair {
                        let (bw, bh) = (bounds.size.width, bounds.size.height);
                        window.paint_quad(fill(Bounds { origin: point(o.x + px(cx_), o.y), size: size(px(1.), bh) }, theme::rgb8([235, 92, 92]).opacity(0.85)));
                        window.paint_quad(fill(Bounds { origin: point(o.x, o.y + px(cy_)), size: size(bw, px(1.)) }, theme::rgb8([92, 200, 120]).opacity(0.85)));
                    }
                    if let Some((m, r)) = brush {
                        let c = point(o.x + px(m.0), o.y + px(m.1));
                        for (w, col) in [(3.0, gpui_kit::black().opacity(0.6)), (1.0, gpui_kit::white())] {
                            let mut b = PathBuilder::stroke(px(w));
                            b.add_polygon(&circle(c, r.max(1.5)), true);
                            if let Ok(p) = b.build() {
                                window.paint_path(p, col);
                            }
                        }
                    }
                });
            },
        )
        .size_full();

        let label_elems = labels.into_iter().map(|(x, y, text)| {
            div().absolute().left(px(x + 6.0)).top(px(y - 9.0)).px(px(4.)).rounded(px(3.)).bg(gpui_kit::black().opacity(0.65)).text_xs().text_color(gpui_kit::white()).child(text)
        });
        let empty = image.is_none().then(|| {
            div().absolute().size_full().flex().items_center().justify_center().text_sm().text_color(color::text_muted()).child(if s.opening.is_empty() { "No image".to_string() } else { format!("Opening {}…", s.opening.join(", ")) })
        });
        let (bar_px, bar_label) = self.scale_bar(um);
        let scale_bar = (image.is_some() && !self.compact).then(|| {
            v_flex()
                .absolute()
                .left(px(theme::LG))
                .bottom(px(theme::LG))
                .gap(px(2.))
                .p(px(theme::XS))
                .rounded(px(4.))
                .bg(gpui_kit::black().opacity(0.45))
                .child(div().text_xs().text_color(gpui_kit::white()).child(bar_label))
                .child(div().w(px(bar_px)).h(px(3.)).bg(gpui_kit::white()))
        });
        let plane_el = div()
            .id("image-plane")
            .relative()
            .flex_1()
            .min_w_0()
            .h_full()
            .overflow_hidden()
            .cursor(cursor)
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_down))
            .on_mouse_down(MouseButton::Middle, cx.listener(Self::on_down))
            .on_mouse_move(cx.listener(Self::on_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_up))
            .on_mouse_up(MouseButton::Middle, cx.listener(Self::on_up))
            .on_scroll(cx.listener(Self::on_wheel))
            .child(plane)
            .children(label_elems)
            .children(empty)
            .children(scale_bar);

        // Rulers
        let rulers = tool.rulers && image.is_some();
        let top = self.to_image((0.0, 0.0));
        let ruler = |horizontal: bool, ticks: Vec<(f32, String)>, mouse: Option<f32>| {
            let ticks2 = ticks.clone();
            let ruler_canvas = canvas(
                |_, _, _| {},
                move |bounds, _, window, _| {
                    window.paint_quad(fill(bounds, color::chrome()));
                    let o = bounds.origin;
                    for (t, _) in &ticks2 {
                        let b = if horizontal {
                            Bounds { origin: point(o.x + px(*t), o.y + px(theme::RULER - 8.0)), size: size(px(1.), px(8.)) }
                        } else {
                            Bounds { origin: point(o.x + px(theme::RULER - 8.0), o.y + px(*t)), size: size(px(8.), px(1.)) }
                        };
                        window.paint_quad(fill(b, color::text_muted()));
                    }
                    if let Some(m) = mouse {
                        let b = if horizontal { Bounds { origin: point(o.x + px(m), o.y), size: size(px(1.), px(theme::RULER)) } } else { Bounds { origin: point(o.x, o.y + px(m)), size: size(px(theme::RULER), px(1.)) } };
                        window.paint_quad(fill(b, color::accent_hover()));
                    }
                },
            )
            .absolute()
            .size_full();
            let labels = ticks.into_iter().map(move |(t, l)| {
                if horizontal {
                    div().absolute().left(px(t + 3.0)).top(px(1.)).whitespace_nowrap().text_size(px(10.)).text_color(color::text_muted()).child(l)
                } else {
                    div().absolute().top(px(t + 2.0)).left(px(1.)).whitespace_nowrap().text_size(px(9.)).text_color(color::text_muted()).child(l)
                }
            });
            div().relative().overflow_hidden().flex_none().when(horizontal, |d| d.h(px(theme::RULER)).flex_1()).when(!horizontal, |d| d.w(px(theme::RULER)).h_full()).border_color(color::border()).when(horizontal, |d| d.border_b_1()).when(!horizontal, |d| d.border_r_1()).child(ruler_canvas).children(labels)
        };
        let unit = if um.is_some() { "µm" } else { "px" };
        let body = if rulers {
            v_flex()
                .size_full()
                .child(
                    h_flex()
                        .flex_none()
                        .w_full()
                        .child(div().flex_none().size(px(theme::RULER)).bg(color::chrome()).border_r_1().border_b_1().border_color(color::border()).flex().items_center().justify_center().text_size(px(9.)).text_color(color::text_muted()).child(unit))
                        .child(ruler(true, self.ticks(top[0], vw, um), self.mouse.map(|m| m.0))),
                )
                .child(h_flex().flex_1().min_h_0().w_full().child(ruler(false, self.ticks(top[1], vh, um), self.mouse.map(|m| m.1))).child(plane_el))
                .into_any_element()
        } else {
            div().size_full().flex().child(plane_el).into_any_element()
        };

        // Readout
        let readout = self.readout();
        let level_text = self.viewer.as_ref().map(|v| {
            let l = v.pyramid.level_for_zoom(zoom * self.scale as f64);
            format!("{l}/{} · {}×", v.pyramid.levels.len() - 1, trim(v.pyramid.levels[l].downsample))
        });
        let names: Vec<String> = s.doc.active().map(|i| i.channels.iter().map(|c| c.label.clone()).collect()).unwrap_or_default();
        let mut info = Vec::new();
        if let Some((p, _, values)) = &readout {
            let pos = match um {
                Some(u) => format!("x {:.1}  y {:.1} µm", p[0] * u, p[1] * u),
                None => format!("x {:.0}  y {:.0} px", p[0], p[1]),
            };
            info.push(pos);
            let vals: Vec<String> = values.iter().enumerate().map(|(i, v)| format!("{} {}", names.get(i).map_or("", String::as_str), v.map_or("–".to_string(), |v| trim(v as f64)))).collect();
            if !vals.is_empty() {
                info.push(vals.join("  "));
            }
        }
        let cached = self.viewer.as_ref().map_or(0, |v| v.service.cached_bytes()) / (1024 * 1024);
        let depth = self.viewer.as_ref().map_or(1, |v| v.pyramid.depth());
        let z_now = self.z;
        let me = cx.entity().downgrade();
        let z_strip = (depth > 1 && !self.compact).then(|| {
            h_flex()
                .flex_none()
                .h(px(26.))
                .px(px(theme::SM))
                .gap(px(theme::SM))
                .border_t_1()
                .border_color(color::border())
                .bg(color::chrome())
                .child(div().flex_none().text_xs().text_color(color::text_muted()).child("z"))
                .child(div().flex_1().child(crate::ui::widgets::axis_slider("plane-z".to_string(), z_now, depth, move |at, _, cx| {
                    let _ = me.update(cx, |c, cx| c.set_z(at, cx));
                })))
                .child(div().flex_none().min_w(px(64.)).text_xs().font_family("monospace").text_color(color::text_muted()).child(format!("{z_now} / {}", depth - 1)))
        });
        let footer = h_flex()
            .flex_none()
            .h(px(22.))
            .px(px(theme::SM))
            .gap(px(theme::LG))
            .border_t_1()
            .border_color(color::border())
            .bg(color::chrome())
            .text_xs()
            .text_color(color::text_muted())
            .font_family("monospace")
            .children(info.into_iter().map(|t| div().child(t)))
            .child(div().flex_1())
            .when(s.computing, |d| d.child(div().text_color(color::warning()).child("computing results…")))
            .children(level_text.map(|t| div().child(format!("level {t}"))))
            .child({
                let me = cx.entity().downgrade();
                crate::ui::widgets::scrub_label("plane-zoom", format!("{:.1} %", zoom * 100.0), move |d, _, cx| {
                    let _ = me.update(cx, |c, cx| c.zoom_at(if d > 0.0 { 1.05 } else { 1.0 / 1.05 }, None, cx));
                })
            })
            .when(cached > 0, |d| d.child(div().child(format!("{cached} MB cached"))));

        v_flex()
            .id("image-canvas")
            .key_context("ImageCanvas")
            .track_focus(&self.focus)
            .size_full()
            .bg(color::canvas())
            .on_key_down(cx.listener(Self::on_key))
            .on_key_up(cx.listener(Self::on_key_up))
            .on_mouse_down(MouseButton::Left, cx.listener(|this, _: &MouseDownEvent, window, cx| window.focus(&this.focus, cx)))
            .child(div().flex_1().min_h_0().w_full().child(body))
            .children(z_strip)
            .when(!self.compact && self.footer, |d| d.child(footer))
    }
}

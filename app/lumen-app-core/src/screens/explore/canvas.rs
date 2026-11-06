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
    canvas, div, fill, point, px, size, App, Bounds, Context, Corners, CursorStyle, Entity, FocusHandle, Hsla, InteractiveElement as _, IntoElement, KeyDownEvent, KeyUpEvent,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, PathBuilder, Pixels, Render, RenderImage, ScrollDelta, ScrollEvent, Styled as _, Subscription, Task, Window,
};
use lumen_io::Pyramid;
use lumen_view::{ChannelDisplay, TileImage, TileKey, TileResult, TileService};

use crate::data::results::ResultRequest;
use crate::data::store::{AppEvent, Store};
use crate::domain::annotation::{Annotation, Shape};
use crate::domain::command::Command;
use crate::domain::project::ChannelDoc;
use crate::domain::tool::{step_brush, Tool};
use crate::ui::theme::{self, color};
use crate::ui::widgets::format_length;

/// Most display tiles kept.
const TILE_CAP: usize = 384;
const MIN_ZOOM: f64 = 1.0 / 512.0;
const MAX_ZOOM: f64 = 64.0;

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

pub struct ImageCanvas {
    store: Entity<Store>,
    pub focus: FocusHandle,
    viewer: Option<Viewer>,
    pub zoom: f64,
    pub center: (f64, f64),
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
    refresh_token: u64,
    _subs: Vec<Subscription>,
}

impl ImageCanvas {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            match e {
                AppEvent::ImageReady(_) => this.fit_pending = true,
                AppEvent::Document(crate::domain::command::Changed::Images) => this.fit_pending = this.viewer.as_ref().map(|v| v.image.clone()) != this.store.read(cx).doc.active_image,
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
            fit_pending: true,
            bounds: Rc::new(Cell::new(Bounds::default())),
            scale: 1.0,
            gesture: None,
            polygon: Vec::new(),
            mouse: None,
            space: false,
            drops: Vec::new(),
            refresh_token: 0,
            _subs: subs,
        }
    }

    // ------------------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------------------

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
        self.zoom = ((w as f64 - 48.0) / iw).min((h as f64 - 48.0) / ih).clamp(MIN_ZOOM, MAX_ZOOM);
        self.center = (iw / 2.0, ih / 2.0);
        self.camera_changed(cx);
    }

    /// One image pixel per screen pixel (of the display).
    pub fn actual_size(&mut self, cx: &mut Context<Self>) {
        let f = (1.0 / self.scale as f64) / self.zoom;
        self.zoom_at(f, None, cx);
    }

    /// The camera moved: results for the new view follow once it rests (200 ms).
    fn camera_changed(&mut self, cx: &mut Context<Self>) {
        cx.notify();
        self.refresh_token += 1;
        let token = self.refresh_token;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(Duration::from_millis(200)).await;
            let _ = this.update(cx, |c, cx| {
                if c.refresh_token == token {
                    c.request_results(cx);
                }
            });
        })
        .detach();
    }

    fn request_results(&mut self, cx: &mut Context<Self>) {
        let Some(v) = &self.viewer else { return };
        let req = ResultRequest { image: v.image.clone(), rect: self.visible(), zoom: self.zoom * self.scale as f64 };
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
        let active = s.doc.active_image.clone();
        let pyramid = active.as_ref().and_then(|id| s.pyramids.get(id).cloned());
        let docs: Vec<ChannelDoc> = s.doc.active().map(|i| i.channels.clone()).unwrap_or_default();
        match (&active, pyramid) {
            (Some(id), Some(py)) => {
                let same = self.viewer.as_ref().is_some_and(|v| &v.image == id && Arc::ptr_eq(&v.pyramid, &py));
                if !same {
                    if let Some(old) = self.viewer.take() {
                        self.drops.extend(old.tiles.into_values().map(|t| t.image));
                    }
                    let (service, rx) = TileService::new(py.clone(), TileService::default_threads());
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
        let o = self.bounds.get().origin;
        let rect = self.visible();
        let (zoom, scale, center) = (self.zoom, self.scale as f64, self.center);
        let Some(v) = self.viewer.as_mut() else { return Vec::new() };
        let py = v.pyramid.clone();
        let target = py.level_for_zoom(zoom * scale);
        let coarsest = py.coarsest();
        let tiles_in = |level: usize| -> Vec<TileKey> {
            let l = &py.levels[level];
            let ds = l.downsample;
            let (tw, th) = (l.tile_w as f64 * ds, l.tile_h as f64 * ds);
            let tx0 = (rect[0] / tw).floor().max(0.0) as usize;
            let ty0 = (rect[1] / th).floor().max(0.0) as usize;
            let tx1 = (((rect[0] + rect[2]) / tw).ceil().max(0.0) as usize).min(l.tiles_x());
            let ty1 = (((rect[1] + rect[3]) / th).ceil().max(0.0) as usize).min(l.tiles_y());
            let mut out = Vec::new();
            for ty in ty0..ty1 {
                for tx in tx0..tx1 {
                    out.push(TileKey { level, tx, ty });
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
                    want.push(TileKey { level: coarsest, tx, ty });
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
                let b = Bounds { origin: point(o.x + px(sx - 0.25), o.y + px(sy - 0.25)), size: size(px(sw + 0.5), px(sh + 0.5)) };
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
        Some((p, level, v.service.sample(level, (p[0] / ds) as usize, (p[1] / ds) as usize)))
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
        let a = Annotation { id: String::new(), image, shape, class, note: String::new() };
        self.store.update(cx, |s, cx| {
            if s.execute(Command::AddAnnotation { annotation: a }, cx).is_ok() {
                if let Some(last) = s.doc.annotations.last().map(|a| a.id.clone()) {
                    s.select([last], cx);
                }
            }
        });
    }

    /// The topmost annotation of the shown image at `p`.
    fn hit(&self, p: [f64; 2], cx: &App) -> Option<String> {
        let image = self.viewer.as_ref()?.image.clone();
        let s = self.store.read(cx);
        let tol = 6.0 / self.zoom;
        s.doc.annotations.iter().rev().filter(|a| a.image == image && s.doc.class(&a.class).is_none_or(|c| c.visible)).find(|a| a.shape.hit(p[0], p[1], tol)).map(|a| a.id.clone())
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
            Tool::Select | Tool::Probe => {
                match self.hit(p, cx) {
                    Some(id) => {
                        let sel = self.store.read(cx).selection.clone();
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
                        if !e.modifiers.shift {
                            self.store.update(cx, |st, cx| st.select([], cx));
                        }
                        self.gesture = Some(Gesture::Pan { mouse: local, center: self.center });
                    }
                }
            }
            Tool::Rect | Tool::Ellipse | Tool::Ruler => self.gesture = Some(Gesture::Drag { tool, a: p, b: p }),
            Tool::Lasso => self.gesture = Some(Gesture::Path { tool, points: vec![p], radius: 0.0 }),
            Tool::Brush | Tool::Eraser => {
                let radius = self.brush_radius(cx);
                self.gesture = Some(Gesture::Path { tool, points: vec![p], radius });
            }
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
        match &mut self.gesture {
            Some(Gesture::Pan { mouse, center }) => {
                self.center = (center.0 - (local.0 - mouse.0) as f64 / zoom, center.1 - (local.1 - mouse.1) as f64 / zoom);
                self.camera_changed(cx);
                return;
            }
            Some(Gesture::Drag { b, .. }) => *b = p,
            Some(Gesture::Path { points, radius, tool }) => {
                let last = *points.last().expect("starts with one point");
                let step = if *tool == Tool::Lasso { 2.0 / zoom } else { (*radius * 0.25).max(1.0 / zoom) };
                if (p[0] - last[0]).hypot(p[1] - last[1]) >= step {
                    points.push(p);
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
        cx.notify();
    }

    fn on_up(&mut self, _: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        let Some(g) = self.gesture.take() else { return };
        match g {
            Gesture::Pan { .. } => {}
            Gesture::Drag { tool, a, b } => {
                let (x0, y0, x1, y1) = (a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1]));
                let tiny = (x1 - x0) * self.zoom < 3.0 && (y1 - y0) * self.zoom < 3.0;
                if !tiny {
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
                Tool::Brush | Tool::Eraser => self.add_annotation(Shape::Stroke { points, radius, erase: tool == Tool::Eraser }, cx),
                _ => {}
            },
            Gesture::Move { start, now, moved } => {
                if moved {
                    let ids: Vec<String> = self.store.read(cx).selection.iter().cloned().collect();
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
        for a in s.doc.annotations.iter().filter(|a| a.image == v.image) {
            let Some(class) = s.doc.class(&a.class) else { continue };
            if !class.visible {
                continue;
            }
            let selected = s.selection.contains(&a.id);
            let shape = if selected && delta != (0.0, 0.0) { a.shape.translated(delta.0, delta.1) } else { a.shape.clone() };
            let (bx0, by0, bx1, by1) = shape.bounds();
            if bx1 < rect[0] || by1 < rect[1] || bx0 > rect[0] + rect[2] || by0 > rect[1] + rect[3] {
                continue;
            }
            let c = theme::rgb8(class.color);
            out.push(stroke_of(&shape, c, selected, zoom, |p| self.to_screen(p)));
        }
        // In progress
        let live = theme::rgb8(s.doc.class(&s.tool.class).map_or([255, 255, 255], |c| c.color));
        match &self.gesture {
            Some(Gesture::Drag { tool, a, b }) => {
                let (x0, y0, x1, y1) = (a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1]));
                let shape = match tool {
                    Tool::Rect => Shape::Rect { x: x0, y: y0, w: (x1 - x0).max(1e-6), h: (y1 - y0).max(1e-6) },
                    Tool::Ellipse => Shape::Ellipse { cx: (x0 + x1) / 2.0, cy: (y0 + y1) / 2.0, rx: (x1 - x0).max(1e-6) / 2.0, ry: (y1 - y0).max(1e-6) / 2.0 },
                    _ => Shape::Line { x0: a[0], y0: a[1], x1: b[0], y1: b[1] },
                };
                out.push(stroke_of(&shape, if *tool == Tool::Ruler { gpui_kit::white() } else { live }, true, zoom, |p| self.to_screen(p)));
            }
            Some(Gesture::Path { tool, points, radius }) => {
                let shape = if *tool == Tool::Lasso { Shape::Polygon { points: points.clone() } } else { Shape::Stroke { points: points.clone(), radius: *radius, erase: *tool == Tool::Eraser } };
                let mut st = stroke_of(&shape, live, false, zoom, |p| self.to_screen(p));
                st.closed = *tool != Tool::Lasso && st.closed;
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
        for a in s.doc.annotations.iter().filter(|a| a.image == v.image) {
            match &a.shape {
                Shape::Line { x0, y0, x1, y1 } => {
                    let len = a.shape.length().unwrap_or(0.0);
                    push(self.to_screen([(x0 + x1) / 2.0, (y0 + y1) / 2.0]), format_length(len, um));
                }
                shape if s.selection.contains(&a.id) && shape.is_region() => {
                    let (x0, y0, _, _) = shape.bounds();
                    let (sx, sy) = self.to_screen([x0, y0]);
                    push((sx, sy - 20.0), crate::ui::widgets::format_area(shape.area().unwrap_or(0.0), um));
                }
                _ => {}
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
    /// A brush stroke: its width on screen and whether it erases.
    Brush(f32, bool),
}

fn stroke_of(shape: &Shape, color: Hsla, selected: bool, zoom: f32, to_screen: impl Fn([f64; 2]) -> (f32, f32)) -> Stroke {
    let points: Vec<(f32, f32)> = shape.outline().into_iter().map(&to_screen).collect();
    match shape {
        Shape::Point { .. } => Stroke { points, closed: false, fill: None, width: 1.5, color, selected, dot: true },
        Shape::Line { .. } => Stroke { points, closed: false, fill: None, width: 1.5, color, selected, dot: false },
        Shape::Stroke { radius, erase, .. } => Stroke { points, closed: false, fill: Some(Fill::Brush((*radius as f32 * 2.0 * zoom).max(1.0), *erase)), width: 1.0, color, selected, dot: false },
        _ => Stroke { points, closed: true, fill: Some(Fill::Area(if selected { 0.22 } else { 0.12 })), width: if selected { 2.0 } else { 1.5 }, color, selected, dot: false },
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
        Some(Fill::Brush(width, erase)) => {
            let mut b = PathBuilder::stroke(px(width));
            b.add_polygon(&pts, false);
            if let Ok(p) = b.build() {
                window.paint_path(p, if erase { gpui_kit::black().opacity(0.45) } else { s.color.opacity(0.45) });
            }
            return;
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
        let drops = std::mem::take(&mut self.drops);
        let s = self.store.read(cx);
        let tool = s.tool.clone();
        let image = self.viewer.as_ref().map(|v| v.image.clone());
        let um = self.viewer.as_ref().and_then(|v| v.pyramid.pixel_size_um);
        // Result layers of the shown image
        let mut overlays: Vec<([f64; 4], Arc<RenderImage>)> = Vec::new();
        if tool.show_results {
            for n in s.doc.graph.views() {
                let visible = matches!(&n.kind, crate::domain::graph::NodeKind::View { style, .. } if style.visible);
                if let Some(r) = s.results.get(&n.id).filter(|r| visible && Some(&r.image) == image.as_ref()) {
                    if let Some(l) = &r.layer {
                        overlays.push((r.rect, l.clone()));
                    }
                }
            }
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
        let bounds_cell = self.bounds.clone();
        let plane = canvas(
            move |bounds, _, _| bounds_cell.set(bounds),
            move |bounds, _, window, _| {
                for img in drops {
                    let _ = window.drop_image(img);
                }
                let o = bounds.origin;
                window.paint_quad(fill(bounds, color::canvas()));
                window.with_content_mask(Some(gpui_kit::ContentMask { bounds }), |window| {
                    for (b, img) in tiles {
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
        let scale_bar = image.is_some().then(|| {
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
                    div().absolute().left(px(t + 3.0)).top(px(1.)).text_size(px(10.)).text_color(color::text_muted()).child(l)
                } else {
                    div().absolute().top(px(t + 2.0)).left(px(2.)).text_size(px(10.)).text_color(color::text_muted()).child(l)
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
            format!("{} {}/{}", v.pyramid.levels[l].path, l, v.pyramid.levels.len() - 1)
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
            .child(div().child(format!("{:.1} %", zoom * 100.0)))
            .child(div().child(format!("{cached} MB cached")));

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
            .child(footer)
    }
}

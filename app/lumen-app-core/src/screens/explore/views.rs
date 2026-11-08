//! The views of an image beyond a single plane, chosen and combined as dsp-kitchen's time views:
//! - [`OrthoView`] ("2D"): the XY, XZ and YZ planes together, each with its own slider and a cursor
//!   linked between them, for an image with depth;
//! - [`VolumeView`] ("3D"): the volume rendered on the device (`lumen_view::render_volume`), at a
//!   coarse level while it turns and a finer one when it rests.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{canvas, div, fill, point, px, size, AppContext as _, Bounds, Context, Corners, Entity, InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, Pixels, Render, RenderImage, ScrollDelta, ScrollEvent, Styled as _, Subscription, Task, Window};
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;
use lumen_io::{PlaneAxis, Pyramid};
use lumen_view::{bgra_image, composite_bgra, render_volume, ChannelDisplay, OrbitCamera, VolumeMode};

use crate::data::store::{AppEvent, Store};
use crate::ui::theme::{self, color};
use crate::ui::widgets::axis_slider;

/// Longest side of a plane drawn in the three-plane view.
const PLANE_SIDE: usize = 1024;
/// Longest side of the volume while it turns and when it rests.
const VOLUME_MOVING: usize = 96;
const VOLUME_RESTING: usize = 256;
/// Longest side of the image rendered while turning.
const IMAGE_MOVING: usize = 320;
const IMAGE_RESTING: usize = 1024;

fn displays_of(store: &Store) -> Vec<ChannelDisplay> {
    store
        .doc
        .active()
        .map(|i| i.channels.iter().map(|c| ChannelDisplay { visible: c.visible, color: [c.color[0] as f32 / 255.0, c.color[1] as f32 / 255.0, c.color[2] as f32 / 255.0], min: c.min, max: c.max, gamma: c.gamma }).collect())
        .unwrap_or_default()
}

/// The finest level whose longest side fits `side` (else the coarsest).
fn level_under(py: &Pyramid, side: usize) -> usize {
    (0..py.levels.len()).find(|&l| py.levels[l].width.max(py.levels[l].height).max(py.levels[l].depth) <= side).unwrap_or(py.coarsest())
}

/// The three planes of the view, in order: XY, XZ, YZ.
const AXES: [(PlaneAxis, usize, &str); 3] = [(PlaneAxis::Z, 0, "XY"), (PlaneAxis::Y, 1, "XZ"), (PlaneAxis::X, 2, "YZ")];

struct PlaneState {
    image: Option<(Arc<RenderImage>, usize, usize)>,
    key: String,
    task: Option<Task<()>>,
}

/// XY, XZ and YZ planes with linked cursors.
pub struct OrthoView {
    store: Entity<Store>,
    /// `[z, y, x]`, level-0 pixels.
    position: [usize; 3],
    image: Option<String>,
    planes: [PlaneState; 3],
    /// Where each plane's image was drawn (window pixels).
    drawn: [Rc<Cell<Bounds<Pixels>>>; 3],
    drops: Vec<Arc<RenderImage>>,
    _subs: Vec<Subscription>,
}

impl OrthoView {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify())];
        let plane = || PlaneState { image: None, key: String::new(), task: None };
        Self { store, position: [0; 3], image: None, planes: [plane(), plane(), plane()], drawn: std::array::from_fn(|_| Rc::new(Cell::new(Bounds::default()))), drops: Vec::new(), _subs: subs }
    }

    fn pyramid(&self, cx: &Context<Self>) -> Option<Arc<Pyramid>> {
        self.store.read(cx).active_pyramid().cloned()
    }

    /// Starts each plane whose position, level or display changed.
    fn sync(&mut self, cx: &mut Context<Self>) {
        let Some(py) = self.pyramid(cx) else { return };
        let id = self.store.read(cx).doc.active_image.clone();
        if id != self.image {
            self.image = id;
            self.position = [py.depth() / 2, py.height() / 2, py.width() / 2];
        }
        let displays = displays_of(self.store.read(cx));
        let level = level_under(&py, PLANE_SIDE);
        let l = py.levels[level].clone();
        let at = [((self.position[0] as f64 / l.z_downsample) as usize).min(l.depth - 1), ((self.position[1] as f64 / l.downsample) as usize).min(l.height - 1), ((self.position[2] as f64 / l.downsample) as usize).min(l.width - 1)];
        for (i, (axis, a, _)) in AXES.into_iter().enumerate() {
            let key = format!("{:?}/{level}/{}/{:?}", self.image, at[a], displays);
            if self.planes[i].key == key {
                continue;
            }
            self.planes[i].key = key;
            let (py, displays, index) = (py.clone(), displays.clone(), at[a]);
            let work = cx.background_spawn(async move {
                let mut planes = Vec::new();
                let (mut w, mut h) = (0, 0);
                for (c, d) in displays.iter().enumerate().filter(|(_, d)| d.visible) {
                    let p = py.read_plane(level, c, axis, index).ok()?;
                    (w, h) = (p.width, p.height);
                    planes.push((p.data, *d));
                }
                if planes.is_empty() {
                    return None;
                }
                let refs: Vec<(&[f32], ChannelDisplay)> = planes.iter().map(|(v, d)| (v.as_slice(), *d)).collect();
                Some((bgra_image(w, h, composite_bgra(&refs, w * h)), w, h))
            });
            self.planes[i].task = Some(cx.spawn(async move |this, cx| {
                let done = work.await;
                let _ = this.update(cx, |v, cx| {
                    if let Some(old) = std::mem::replace(&mut v.planes[i].image, done) {
                        v.drops.push(old.0);
                    }
                    cx.notify();
                });
            }));
        }
    }

    /// Moves the cursor to a point of plane `i`, given as fractions of its image.
    fn point_at(&mut self, i: usize, fx: f64, fy: f64, cx: &mut Context<Self>) {
        let Some(py) = self.pyramid(cx) else { return };
        let (d, h, w) = (py.depth(), py.height(), py.width());
        let clip = |f: f64, n: usize| ((f.clamp(0.0, 1.0) * n as f64) as usize).min(n - 1);
        match i {
            0 => (self.position[2], self.position[1]) = (clip(fx, w), clip(fy, h)),
            1 => (self.position[2], self.position[0]) = (clip(fx, w), clip(fy, d)),
            _ => (self.position[1], self.position[0]) = (clip(fx, h), clip(fy, d)),
        }
        cx.notify();
    }

    fn pane(&self, i: usize, py: &Arc<Pyramid>, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let (_, axis, title) = AXES[i];
        let (d, h, w) = (py.depth(), py.height(), py.width());
        let n = [d, h, w][axis];
        let value = self.position[axis].min(n - 1);
        let (px_um, dz_um) = (py.pixel_size_um.unwrap_or(1.0), py.z_spacing_um.or(py.pixel_size_um).unwrap_or(1.0));
        // Physical size of what is drawn, so voxels deeper than wide are not squashed
        let (phys_w, phys_h) = match i {
            0 => (w as f64 * px_um, h as f64 * px_um),
            1 => (w as f64 * px_um, d as f64 * dz_um),
            _ => (h as f64 * px_um, d as f64 * dz_um),
        };
        let (u, v) = match i {
            0 => (self.position[2] as f64 / w as f64, self.position[1] as f64 / h as f64),
            1 => (self.position[2] as f64 / w as f64, self.position[0] as f64 / d as f64),
            _ => (self.position[1] as f64 / h as f64, self.position[0] as f64 / d as f64),
        };
        let image = self.planes[i].image.as_ref().map(|p| p.0.clone());
        let drawn = self.drawn[i].clone();
        let drawn_paint = drawn.clone();
        let plane = canvas(
            |_, _, _| {},
            move |bounds, _, window, _| {
                window.paint_quad(fill(bounds, color::canvas()));
                let scale = (bounds.size.width.as_f32() as f64 / phys_w).min(bounds.size.height.as_f32() as f64 / phys_h);
                let (sw, sh) = ((phys_w * scale) as f32, (phys_h * scale) as f32);
                let at = point(bounds.origin.x + px((bounds.size.width.as_f32() - sw) / 2.0), bounds.origin.y + px((bounds.size.height.as_f32() - sh) / 2.0));
                let b = Bounds { origin: at, size: size(px(sw), px(sh)) };
                drawn_paint.set(b);
                if let Some(img) = image {
                    let _ = window.paint_image(bounds, b, Corners::default(), img, 0, false);
                }
                let (cx_, cy_) = (b.origin.x + px(sw * u as f32), b.origin.y + px(sh * v as f32));
                let line = color::accent_hover();
                window.paint_quad(fill(Bounds { origin: point(cx_, b.origin.y), size: size(px(1.), px(sh)) }, line));
                window.paint_quad(fill(Bounds { origin: point(b.origin.x, cy_), size: size(px(sw), px(1.)) }, line));
            },
        )
        .size_full();
        let me = cx.entity().downgrade();
        let on_slide = move |at: usize, _: &mut Window, cx: &mut gpui_kit::App| {
            let _ = me.update(cx, |v, cx| {
                v.position[axis] = at;
                cx.notify();
            });
        };
        let hit = |e: gpui_kit::Point<Pixels>, b: Bounds<Pixels>| (((e.x - b.origin.x).as_f32() / b.size.width.as_f32().max(1.0)) as f64, ((e.y - b.origin.y).as_f32() / b.size.height.as_f32().max(1.0)) as f64);
        let (down, moved) = (drawn.clone(), drawn);
        v_flex()
            .flex_1()
            .min_w_0()
            .min_h_0()
            .border_1()
            .border_color(color::border())
            .child(
                h_flex()
                    .flex_none()
                    .h(px(26.))
                    .px(px(theme::SM))
                    .gap(px(theme::SM))
                    .bg(color::chrome())
                    .child(div().flex_none().w(px(22.)).text_xs().text_color(color::text_muted()).child(title))
                    .child(div().flex_1().child(axis_slider(format!("ortho-{i}"), value, n, on_slide)))
                    .child(div().flex_none().min_w(px(56.)).text_xs().font_family("monospace").text_color(color::text_muted()).child(format!("{value}/{}", n - 1))),
            )
            .child(
                div()
                    .id(("ortho-plane", i))
                    .flex_1()
                    .min_h_0()
                    .relative()
                    .on_mouse_down(MouseButton::Left, cx.listener(move |v, e: &MouseDownEvent, _, cx| {
                        let (fx, fy) = hit(e.position, down.get());
                        v.point_at(i, fx, fy, cx);
                    }))
                    .on_mouse_move(cx.listener(move |v, e: &MouseMoveEvent, _, cx| {
                        if e.pressed_button == Some(MouseButton::Left) {
                            let (fx, fy) = hit(e.position, moved.get());
                            v.point_at(i, fx, fy, cx);
                        }
                    }))
                    .on_scroll(cx.listener(move |v, e: &ScrollEvent, _, cx| {
                        let dy = match e.delta {
                            ScrollDelta::Lines(d) => d.y as f64,
                            ScrollDelta::Pixels(d) => d.y.as_f32() as f64 / 20.0,
                        };
                        let step = if dy > 0.0 { -1 } else if dy < 0.0 { 1 } else { 0 };
                        v.position[axis] = (v.position[axis] as i64 + step).clamp(0, n as i64 - 1) as usize;
                        cx.notify();
                    }))
                    .child(plane),
            )
    }
}

impl Render for OrthoView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        for img in std::mem::take(&mut self.drops) {
            let _ = window.drop_image(img);
        }
        self.sync(cx);
        let Some(py) = self.pyramid(cx) else { return v_flex().size_full().into_any_element() };
        if !py.is_volume() {
            return v_flex().size_full().items_center().justify_center().text_sm().text_color(color::text_muted()).child("This image has no depth: the three-plane view needs a z axis").into_any_element();
        }
        let info = format!("x {}  y {}  z {}", self.position[2], self.position[1], self.position[0]);
        v_flex()
            .size_full()
            .child(h_flex().flex_1().min_h_0().child(self.pane(0, &py, cx)).child(self.pane(2, &py, cx)))
            .child(h_flex().flex_1().min_h_0().child(self.pane(1, &py, cx)).child(v_flex().flex_1().items_center().justify_center().text_sm().font_family("monospace").text_color(color::text_muted()).child(info)))
            .into_any_element()
    }
}

/// Volumes on the device, by image, level and channel (only the last image's are kept).
type VolumeCache = Arc<Mutex<HashMap<(String, usize, usize), Arc<DeviceArray<f32>>>>>;

/// The volume rendered, turned by dragging and zoomed by the wheel.
pub struct VolumeView {
    store: Entity<Store>,
    camera: OrbitCamera,
    mode: VolumeMode,
    volumes: VolumeCache,
    image: Option<Arc<RenderImage>>,
    drops: Vec<Arc<RenderImage>>,
    bounds: Rc<Cell<Bounds<Pixels>>>,
    drag: Option<(f32, f32, OrbitCamera)>,
    busy: bool,
    /// A frame to make when the current one is done (`true`: a fast one).
    queued: Option<bool>,
    rendered_key: String,
    settle: Option<Task<()>>,
    /// Timing of the last frame.
    last: Option<(usize, f64)>,
    error: Option<String>,
    _subs: Vec<Subscription>,
}

impl VolumeView {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify())];
        Self { store, camera: OrbitCamera::default(), mode: VolumeMode::MaximumIntensity, volumes: Arc::new(Mutex::new(HashMap::new())), image: None, drops: Vec::new(), bounds: Rc::new(Cell::new(Bounds::default())), drag: None, busy: false, queued: None, rendered_key: String::new(), settle: None, last: None, error: None, _subs: subs }
    }

    /// A fast frame now, a sharp one once the view has rested.
    fn moved(&mut self, cx: &mut Context<Self>) {
        self.request(true, cx);
        self.settle = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(Duration::from_millis(220)).await;
            let _ = this.update(cx, |v, cx| v.request(false, cx));
        }));
    }

    fn request(&mut self, fast: bool, cx: &mut Context<Self>) {
        if self.busy {
            self.queued = Some(fast);
            return;
        }
        let s = self.store.read(cx);
        let Some(device) = s.device.clone() else {
            self.store.update(cx, |s, cx| s.ensure_device(cx));
            return;
        };
        let Some(py) = s.active_pyramid().cloned() else { return };
        let Some(id) = s.doc.active_image.clone() else { return };
        if !py.is_volume() {
            return;
        }
        let displays = displays_of(s);
        let size = self.bounds.get().size;
        let (bw, bh) = (size.width.as_f32() as usize, size.height.as_f32() as usize);
        if bw < 16 || bh < 16 {
            return;
        }
        let longest = if fast { IMAGE_MOVING } else { IMAGE_RESTING };
        let scale = (longest as f32 / bw.max(bh) as f32).min(1.0);
        let (iw, ih) = (((bw as f32 * scale) as usize).max(2), ((bh as f32 * scale) as usize).max(2));
        let level = level_under(&py, if fast { VOLUME_MOVING } else { VOLUME_RESTING });
        let (camera, mode, volumes) = (self.camera, self.mode, self.volumes.clone());
        self.busy = true;
        let work = cx.background_spawn(async move {
            let t0 = Instant::now();
            let client = &device.client;
            let l = py.levels[level].clone();
            let (px_um, dz_um) = (py.pixel_size_um.unwrap_or(1.0) * l.downsample, py.z_spacing_um.or(py.pixel_size_um).unwrap_or(1.0) * l.z_downsample);
            let spacing = [dz_um as f32, px_um as f32, px_um as f32];
            let dims = [l.depth, l.height, l.width];
            let mut planes = Vec::new();
            for (c, d) in displays.iter().enumerate().filter(|(_, d)| d.visible) {
                let key = (id.clone(), level, c);
                let cached = volumes.lock().map_err(|e| e.to_string())?.get(&key).cloned();
                let volume = match cached {
                    Some(v) => v,
                    None => {
                        let values = py.read_block(level, c, [0, 0, 0], dims).map_err(|e| e.to_string())?;
                        let v = Arc::new(DeviceArray::from_host(client, Shape::new(dims), values).map_err(|e| e.to_string())?);
                        let mut cache = volumes.lock().map_err(|e| e.to_string())?;
                        cache.retain(|k, _| k.0 == id);
                        cache.insert(key, v.clone());
                        v
                    }
                };
                let out = render_volume(client, &volume, dims, spacing, camera, (iw, ih), (d.min, d.max), mode)?;
                planes.push((out.to_host(client), ChannelDisplay { min: 0.0, max: 1.0, ..*d }));
            }
            if planes.is_empty() {
                return Err("No channel is visible".to_string());
            }
            let refs: Vec<(&[f32], ChannelDisplay)> = planes.iter().map(|(v, d)| (v.as_slice(), *d)).collect();
            let image = bgra_image(iw, ih, composite_bgra(&refs, iw * ih));
            Ok::<_, String>((image, l.width.max(l.height).max(l.depth), t0.elapsed().as_secs_f64() * 1000.0))
        });
        cx.spawn(async move |this, cx| {
            let done = work.await;
            let _ = this.update(cx, |v, cx| {
                v.busy = false;
                match done {
                    Ok((image, side, ms)) => {
                        if let Some(old) = v.image.replace(image) {
                            v.drops.push(old);
                        }
                        v.last = Some((side, ms));
                        v.error = None;
                    }
                    Err(e) => v.error = Some(e),
                }
                if let Some(fast) = v.queued.take() {
                    v.request(fast, cx);
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn on_down(&mut self, e: &MouseDownEvent, _: &mut Window, _: &mut Context<Self>) {
        self.drag = Some((e.position.x.as_f32(), e.position.y.as_f32(), self.camera));
    }

    fn on_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if e.pressed_button.is_none() {
            self.drag = None;
            return;
        }
        let Some((x0, y0, start)) = self.drag else { return };
        self.camera.yaw = start.yaw + (e.position.x.as_f32() - x0) * 0.01;
        self.camera.pitch = (start.pitch + (e.position.y.as_f32() - y0) * 0.01).clamp(-1.5, 1.5);
        self.moved(cx);
    }

    fn on_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.drag = None;
    }

    fn on_wheel(&mut self, e: &ScrollEvent, _: &mut Window, cx: &mut Context<Self>) {
        let dy = match e.delta {
            ScrollDelta::Lines(d) => d.y * 40.0,
            ScrollDelta::Pixels(d) => d.y.as_f32(),
        };
        self.camera.zoom = (self.camera.zoom * (dy * 0.003).exp()).clamp(0.2, 32.0);
        self.moved(cx);
    }
}

impl Render for VolumeView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        for img in std::mem::take(&mut self.drops) {
            let _ = window.drop_image(img);
        }
        let s = self.store.read(cx);
        let Some(py) = s.active_pyramid().cloned() else { return v_flex().size_full().into_any_element() };
        if !py.is_volume() {
            return v_flex().size_full().items_center().justify_center().text_sm().text_color(color::text_muted()).child("This image has no depth: the 3D view needs a z axis").into_any_element();
        }
        // A frame when what is shown changed (image, channels, size of the pane, a device arriving)
        let b = self.bounds.get().size;
        let key = format!("{:?}/{:?}/{}/{}/{}", s.doc.active_image, displays_of(s), b.width.as_f32() as i32, b.height.as_f32() as i32, s.device.is_some());
        if key != self.rendered_key && b.width.as_f32() > 16.0 {
            self.rendered_key = key;
            self.request(false, cx);
        }
        let image = self.image.clone();
        let bounds_cell = self.bounds.clone();
        let picture = canvas(
            move |bounds, _, _| bounds_cell.set(bounds),
            move |bounds, _, window, _| {
                window.paint_quad(fill(bounds, gpui_kit::black()));
                if let Some(img) = image {
                    let _ = window.paint_image(bounds, bounds, Corners::default(), img, 0, false);
                }
            },
        )
        .size_full();
        let me = cx.entity().downgrade();
        let mode_button = |id: &'static str, label: &'static str, mode: VolumeMode, on: bool, me: gpui_kit::WeakEntity<Self>| {
            Button::new(id).ghost().xsmall().label(label).selected(on).on_click(move |_, _, cx| {
                let _ = me.update(cx, |v, cx| {
                    v.mode = mode;
                    v.request(false, cx);
                });
            })
        };
        let status = match (&self.error, self.last) {
            (Some(e), _) => e.clone(),
            (None, Some((side, ms))) => format!("{side} voxels a side · {ms:.0} ms"),
            _ => "Rendering…".to_string(),
        };
        let (reset, mode) = (me.clone(), self.mode);
        v_flex()
            .size_full()
            .child(
                h_flex()
                    .flex_none()
                    .h(px(28.))
                    .px(px(theme::SM))
                    .gap(px(theme::XS))
                    .bg(color::chrome())
                    .border_b_1()
                    .border_color(color::border())
                    .child(mode_button("volume-mip", "Maximum", VolumeMode::MaximumIntensity, mode == VolumeMode::MaximumIntensity, me.clone()))
                    .child(mode_button("volume-composite", "Composite", VolumeMode::Composite, mode == VolumeMode::Composite, me))
                    .child(Button::new("volume-reset").ghost().xsmall().icon(Icon::new(IconName::RotateCcw)).tooltip("Reset the view").on_click(move |_, _, cx| {
                        let _ = reset.update(cx, |v, cx| {
                            v.camera = OrbitCamera::default();
                            v.request(false, cx);
                        });
                    }))
                    .child(div().flex_1())
                    .child(div().text_xs().font_family("monospace").text_color(color::text_muted()).child(status)),
            )
            .child(
                div()
                    .id("volume-picture")
                    .flex_1()
                    .min_h_0()
                    .relative()
                    .on_mouse_down(MouseButton::Left, cx.listener(Self::on_down))
                    .on_mouse_move(cx.listener(Self::on_move))
                    .on_mouse_up(MouseButton::Left, cx.listener(Self::on_up))
                    .on_scroll(cx.listener(Self::on_wheel))
                    .child(picture)
                    .when(self.image.is_none(), |d| d.child(div().absolute().size_full().flex().items_center().justify_center().text_sm().text_color(color::text_muted()).child("Rendering…"))),
            )
            .into_any_element()
    }
}

//! The views of an image beyond a single plane, chosen and combined as dsp-kitchen's time views:
//! - [`OrthoView`] ("2D"): the XY, XZ and YZ planes together, crossing at the store's 3D cursor;
//!   dragging in a plane moves the cursor, the wheel moves through the plane under the pointer,
//!   and the annotations show on all three planes;
//! - [`VolumeView`] ("3D"): the volume rendered on the device (`lucem_view::render_volume_rgba`,
//!   every channel and every annotation class in one launch), at a coarse level while it turns
//!   and a finer one when it rests, with the box outline and the cursor drawn over it.
//!
//! Axis colours everywhere: x red, y green, z blue.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    canvas, div, fill, point, px, size, AppContext as _, Bounds, Context, Corners, Entity, Hsla, InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _,
    PathBuilder, Pixels, Render, RenderImage, ScrollDelta, ScrollEvent, Styled as _, Subscription, Task, Window,
};
use lucem_core::memory::DeviceArray;
use lucem_core::Shape;
use lucem_io::{PlaneAxis, Pyramid};
use lucem_view::{bgra_image, composite_bgra, project_point, render_volume_rgba, unpack_bgra, ChannelDisplay, OrbitCamera, VolumeChannel, VolumeMode, VolumeQuality};

use crate::data::labels::{plane_overlay, volume_labels, ClassColor};
use crate::data::store::{AppEvent, Store};
use crate::domain::annotation::Annotation;
use crate::ui::theme::{self, color};
use crate::ui::widgets::stepper;

/// Longest side of a plane drawn in the three-plane view.
const PLANE_SIDE: usize = 1024;
/// Longest side of the volume while it turns and when it rests.
const VOLUME_MOVING: usize = 96;
const VOLUME_RESTING: usize = 256;
/// Longest side of the image rendered while turning; at rest, 2 px per voxel up to the cap.
const IMAGE_MOVING: usize = 320;
const IMAGE_RESTING: usize = 1024;
/// Opacity of annotations over the planes.
const OVERLAY_ALPHA: f32 = 0.45;
/// Wheel pixels per plane.
const WHEEL_STEP: f32 = 24.0;

/// Axis colours: x, y, z.
const AXIS_X: [u8; 3] = [235, 92, 92];
const AXIS_Y: [u8; 3] = [92, 200, 120];
const AXIS_Z: [u8; 3] = [92, 156, 255];

/// The colour of axis `a` of `[z, y, x]`.
fn axis_color(a: usize) -> Hsla {
    theme::rgb8([AXIS_Z, AXIS_Y, AXIS_X][a])
}

fn displays_of(store: &Store) -> Vec<ChannelDisplay> {
    store
        .doc
        .active()
        .map(|i| i.channels.iter().map(|c| ChannelDisplay { visible: c.visible, color: [c.color[0] as f32 / 255.0, c.color[1] as f32 / 255.0, c.color[2] as f32 / 255.0], min: c.min, max: c.max, gamma: c.gamma }).collect())
        .unwrap_or_default()
}

/// The annotations of the shown image whose class is visible, and those classes.
fn annotations_of(store: &Store) -> (Vec<Annotation>, Vec<ClassColor>) {
    let Some(id) = store.doc.active_image.as_deref() else { return (Vec::new(), Vec::new()) };
    let classes: Vec<ClassColor> = store.doc.classes.iter().filter(|c| c.visible).map(|c| (c.name.clone(), c.color)).collect();
    let anns = store.doc.annotations_of(id).filter(|a| classes.iter().any(|(n, _)| *n == a.class)).cloned().collect();
    (anns, classes)
}

/// The finest level whose longest side fits `side` (else the coarsest).
fn level_under(py: &Pyramid, side: usize) -> usize {
    (0..py.levels.len()).find(|&l| py.levels[l].width.max(py.levels[l].height).max(py.levels[l].depth) <= side).unwrap_or(py.coarsest())
}

/// Paints a line with a dark halo.
fn paint_line(window: &mut Window, a: gpui_kit::Point<Pixels>, b: gpui_kit::Point<Pixels>, width: f32, c: Hsla) {
    for (w, col) in [(width + 2.0, gpui_kit::black().opacity(0.5)), (width, c)] {
        let mut path = PathBuilder::stroke(px(w));
        path.add_polygon(&[a, b], false);
        if let Ok(p) = path.build() {
            window.paint_path(p, col);
        }
    }
}

// ----------------------------------------------------------------------------
// 2D: three planes
// ----------------------------------------------------------------------------

/// The three planes of the view, in order: XY, XZ, YZ, each with the axis it is across (index into
/// `[z, y, x]`) and its two in-plane axes (horizontal, vertical).
const AXES: [(PlaneAxis, usize, &str, usize, usize); 3] = [(PlaneAxis::Z, 0, "XY", 2, 1), (PlaneAxis::Y, 1, "XZ", 2, 0), (PlaneAxis::X, 2, "YZ", 1, 0)];

#[derive(Default)]
struct PlaneState {
    image: Option<Arc<RenderImage>>,
    key: String,
    task: Option<Task<()>>,
    /// The annotations on this plane.
    overlay: Option<Arc<RenderImage>>,
    overlay_key: String,
    overlay_task: Option<Task<()>>,
}

/// XY, XZ and YZ planes crossing at the 3D cursor.
pub struct OrthoView {
    store: Entity<Store>,
    planes: [PlaneState; 3],
    /// Where each plane's image was drawn (window pixels).
    drawn: [Rc<Cell<Bounds<Pixels>>>; 3],
    /// Wheel pixels not yet turned into a plane step.
    wheel: f32,
    drops: Vec<Arc<RenderImage>>,
    _subs: Vec<Subscription>,
}

impl OrthoView {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify())];
        Self { store, planes: Default::default(), drawn: std::array::from_fn(|_| Rc::new(Cell::new(Bounds::default()))), wheel: 0.0, drops: Vec::new(), _subs: subs }
    }

    /// Starts each plane (and its annotations) whose position, level, display or annotations
    /// changed.
    fn sync(&mut self, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let Some(py) = s.active_pyramid().cloned() else { return };
        let Some(cursor) = s.cursor() else { return };
        if !py.is_volume() {
            return;
        }
        let image = s.doc.active_image.clone();
        let displays = displays_of(s);
        let (anns, classes) = annotations_of(s);
        let rev = s.annotations_rev;
        let level = level_under(&py, PLANE_SIDE);
        let l = py.levels[level].clone();
        let at = [((cursor[0] as f64 / l.z_downsample) as usize).min(l.depth - 1), ((cursor[1] as f64 / l.downsample) as usize).min(l.height - 1), ((cursor[2] as f64 / l.downsample) as usize).min(l.width - 1)];
        for (i, (axis, a, _, _, _)) in AXES.into_iter().enumerate() {
            let key = format!("{image:?}/{level}/{}/{displays:?}", at[a]);
            if self.planes[i].key != key {
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
                    Some(bgra_image(w, h, composite_bgra(&refs, w * h)))
                });
                self.planes[i].task = Some(cx.spawn(async move |this, cx| {
                    let done = work.await;
                    let _ = this.update(cx, |v, cx| {
                        if let Some(old) = std::mem::replace(&mut v.planes[i].image, done) {
                            v.drops.push(old);
                        }
                        cx.notify();
                    });
                }));
            }
            let overlay_key = format!("{image:?}/{level}/{}/{rev}", at[a]);
            if self.planes[i].overlay_key != overlay_key {
                self.planes[i].overlay_key = overlay_key;
                let (anns, classes, l, index) = (anns.clone(), classes.clone(), l.clone(), at[a]);
                let work = cx.background_spawn(async move { plane_overlay(&anns, &classes, &l, axis, index, OVERLAY_ALPHA).map(|(bgra, w, h)| bgra_image(w, h, bgra)) });
                self.planes[i].overlay_task = Some(cx.spawn(async move |this, cx| {
                    let done = work.await;
                    let _ = this.update(cx, |v, cx| {
                        if let Some(old) = std::mem::replace(&mut v.planes[i].overlay, done) {
                            v.drops.push(old);
                        }
                        cx.notify();
                    });
                }));
            }
        }
    }

    /// Moves the cursor to a point of plane `i`, given as fractions of its image.
    fn point_at(&mut self, i: usize, fx: f64, fy: f64, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let (Some(py), Some(mut at)) = (s.active_pyramid().cloned(), s.cursor()) else { return };
        let dims = [py.depth(), py.height(), py.width()];
        let (_, _, _, ha, va) = AXES[i];
        let clip = |f: f64, n: usize| ((f.clamp(0.0, 1.0) * n as f64) as usize).min(n.saturating_sub(1));
        at[ha] = clip(fx, dims[ha]);
        at[va] = clip(fy, dims[va]);
        self.store.update(cx, |s, cx| s.set_cursor(at, cx));
    }

    /// The wheel over plane `i`: through the planes along its axis.
    fn on_wheel(&mut self, i: usize, e: &ScrollEvent, cx: &mut Context<Self>) {
        let dy = match e.delta {
            ScrollDelta::Lines(d) => d.y * WHEEL_STEP,
            ScrollDelta::Pixels(d) => d.y.as_f32(),
        };
        self.wheel += dy;
        let steps = (self.wheel / WHEEL_STEP).trunc();
        if steps != 0.0 {
            self.wheel -= steps * WHEEL_STEP;
            let axis = AXES[i].1;
            self.store.update(cx, |s, cx| s.nudge_cursor(axis, -steps as i64, cx));
        }
        cx.stop_propagation();
    }

    fn pane(&self, i: usize, py: &Arc<Pyramid>, cursor: [usize; 3], cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let (_, axis, title, ha, va) = AXES[i];
        let dims = [py.depth(), py.height(), py.width()];
        let (px_um, dz_um) = (py.pixel_size_um.unwrap_or(1.0), py.z_spacing_um.or(py.pixel_size_um).unwrap_or(1.0));
        let unit = [dz_um, px_um, px_um];
        // Physical size of what is drawn, so voxels deeper than wide are not squashed
        let (phys_w, phys_h) = (dims[ha] as f64 * unit[ha], dims[va] as f64 * unit[va]);
        let (u, v) = ((cursor[ha] as f64 + 0.5) / dims[ha] as f64, (cursor[va] as f64 + 0.5) / dims[va] as f64);
        let (h_color, v_color) = (axis_color(ha), axis_color(va));
        let image = self.planes[i].image.clone();
        let overlay = self.planes[i].overlay.clone();
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
                if let Some(img) = overlay {
                    let _ = window.paint_image(bounds, b, Corners::default(), img, 0, false);
                }
                // The cursor: the line across the horizontal axis has that axis' colour
                let (cx_, cy_) = (b.origin.x + px(sw * u as f32), b.origin.y + px(sh * v as f32));
                window.paint_quad(fill(Bounds { origin: point(cx_, b.origin.y), size: size(px(1.), px(sh)) }, h_color));
                window.paint_quad(fill(Bounds { origin: point(b.origin.x, cy_), size: size(px(sw), px(1.)) }, v_color));
            },
        )
        .size_full();
        let hit = |e: gpui_kit::Point<Pixels>, b: Bounds<Pixels>| (((e.x - b.origin.x).as_f32() / b.size.width.as_f32().max(1.0)) as f64, ((e.y - b.origin.y).as_f32() / b.size.height.as_f32().max(1.0)) as f64);
        let (down, moved) = (drawn.clone(), drawn);
        let names = ["z", "y", "x"];
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
                    .child(div().flex_none().text_xs().font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(axis_color(axis)).child(title))
                    .child(div().flex_1())
                    .child(div().flex_none().text_xs().font_family("monospace").text_color(color::text_muted()).child(format!("{} {} / {}", names[axis], cursor[axis], dims[axis] - 1))),
            )
            .child(
                div()
                    .id(("ortho-plane", i))
                    .flex_1()
                    .min_h_0()
                    .relative()
                    .cursor(gpui_kit::CursorStyle::Crosshair)
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
                    .on_scroll(cx.listener(move |v, e: &ScrollEvent, _, cx| v.on_wheel(i, e, cx)))
                    .child(plane),
            )
    }

    /// The fourth cell: the cursor, one stepper per axis.
    fn readout(&self, py: &Arc<Pyramid>, cursor: [usize; 3], cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let um = [py.z_spacing_um.or(py.pixel_size_um), py.pixel_size_um, py.pixel_size_um];
        let row = |a: usize, name: &'static str| {
            let store = self.store.clone();
            let position = match um[a] {
                Some(u) => format!("{:.1} µm", cursor[a] as f64 * u),
                None => String::new(),
            };
            h_flex()
                .gap(px(theme::SM))
                .child(div().w(px(14.)).text_sm().font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(axis_color(a)).child(name))
                .child(stepper(&format!("cursor-{name}"), cursor[a].to_string(), move |d, _, cx| store.update(cx, |s, cx| s.nudge_cursor(a, d as i64, cx))))
                .child(div().w(px(72.)).text_xs().font_family("monospace").text_color(color::text_muted()).child(position))
        };
        let _ = cx;
        v_flex()
            .flex_1()
            .min_w_0()
            .min_h_0()
            .items_center()
            .justify_center()
            .gap(px(theme::SM))
            .border_1()
            .border_color(color::border())
            .child(div().text_xs().font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(color::text_muted()).child("CURSOR"))
            .child(row(2, "x"))
            .child(row(1, "y"))
            .child(row(0, "z"))
            .child(div().max_w(px(240.)).text_xs().text_center().text_color(color::text_muted()).child("Drag in a plane to move the cursor; the wheel moves through the plane under the pointer."))
    }
}

impl Render for OrthoView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        for img in std::mem::take(&mut self.drops) {
            let _ = window.drop_image(img);
        }
        self.sync(cx);
        let s = self.store.read(cx);
        let (Some(py), Some(cursor)) = (s.active_pyramid().cloned(), s.cursor()) else { return v_flex().size_full().into_any_element() };
        if !py.is_volume() {
            return v_flex().size_full().items_center().justify_center().text_sm().text_color(color::text_muted()).child("This image has no depth: the three-plane view needs a z axis").into_any_element();
        }
        // Rows are plain flex rows: `h_flex` centres its children on the cross axis, which leaves
        // the planes 0 px tall; a plain row stretches them
        v_flex()
            .size_full()
            .child(div().flex().flex_row().flex_1().min_h_0().child(self.pane(0, &py, cursor, cx)).child(self.pane(2, &py, cursor, cx)))
            .child(div().flex().flex_row().flex_1().min_h_0().child(self.pane(1, &py, cursor, cx)).child(self.readout(&py, cursor, cx)))
            .into_any_element()
    }
}

// ----------------------------------------------------------------------------
// 3D: the volume
// ----------------------------------------------------------------------------

/// What lives on the device for the 3D view: the image's channels at one level, and the
/// annotation classes rasterized at that level (only the last of each is kept).
/// `(image, level)` → every channel, stacked.
type ImageVolume = ((String, usize), Arc<DeviceArray<f32>>);
/// `(image, level, annotations revision)` → every class with annotations, and their colours.
type LabelVolume = ((String, usize, u64), Option<Arc<DeviceArray<f32>>>, Vec<[u8; 3]>);

#[derive(Default)]
struct DeviceVolumes {
    image: Option<ImageVolume>,
    labels: Option<LabelVolume>,
    /// The device they live on (the store's device generation).
    generation: u64,
}

/// A rendered frame and how it was made (to draw the outline and the cursor over it).
struct Frame {
    image: Arc<RenderImage>,
    camera: OrbitCamera,
    size: (usize, usize),
    dims: [usize; 3],
    spacing: [f32; 3],
}

/// Timing of the last frame: voxels a side, device (render and read back), whole frame (ms).
#[derive(Clone, Copy)]
struct Timing {
    side: usize,
    device_ms: f64,
    total_ms: f64,
}

/// The volume rendered, turned by dragging and zoomed by the wheel.
pub struct VolumeView {
    store: Entity<Store>,
    camera: OrbitCamera,
    mode: VolumeMode,
    /// Annotation classes rendered as channels.
    show_labels: bool,
    volumes: Arc<Mutex<DeviceVolumes>>,
    frame: Option<Frame>,
    drops: Vec<Arc<RenderImage>>,
    bounds: Rc<Cell<Bounds<Pixels>>>,
    drag: Option<(f32, f32, OrbitCamera)>,
    busy: bool,
    /// A frame to make when the current one is done (`true`: a fast one).
    queued: Option<bool>,
    rendered_key: String,
    settle: Option<Task<()>>,
    last: Option<Timing>,
    error: Option<String>,
    _subs: Vec<Subscription>,
}

impl VolumeView {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify())];
        Self {
            store,
            camera: OrbitCamera::default(),
            mode: VolumeMode::MaximumIntensity,
            show_labels: true,
            volumes: Arc::new(Mutex::new(DeviceVolumes::default())),
            frame: None,
            drops: Vec::new(),
            bounds: Rc::new(Cell::new(Bounds::default())),
            drag: None,
            busy: false,
            queued: None,
            rendered_key: String::new(),
            settle: None,
            last: None,
            error: None,
            _subs: subs,
        }
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
            // A sharp frame asked for while busy is never downgraded by a fast one
            self.queued = Some(fast && self.queued != Some(false));
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
        let (anns, classes) = if self.show_labels { annotations_of(s) } else { (Vec::new(), Vec::new()) };
        let rev = if self.show_labels { s.annotations_rev } else { u64::MAX };
        let size = self.bounds.get().size;
        let (bw, bh) = (size.width.as_f32() as usize, size.height.as_f32() as usize);
        if bw < 16 || bh < 16 {
            return;
        }
        let level = level_under(&py, if fast { VOLUME_MOVING } else { VOLUME_RESTING });
        let l = py.levels[level].clone();
        let side = l.width.max(l.height).max(l.depth);
        // No more than 2 image px per voxel: finer only costs time
        let longest = if fast { IMAGE_MOVING } else { (2 * side).clamp(256, IMAGE_RESTING) };
        let scale = (longest as f32 / bw.max(bh) as f32).min(1.0);
        let (iw, ih) = (((bw as f32 * scale) as usize).max(2), ((bh as f32 * scale) as usize).max(2));
        let (camera, mode, volumes) = (self.camera, self.mode, self.volumes.clone());
        let generation = s.device_generation;
        let quality = if fast { VolumeQuality::Fast } else { VolumeQuality::Fine };
        self.busy = true;
        let work = cx.background_spawn(async move {
            let t0 = Instant::now();
            let client = &device.client;
            let (px_um, dz_um) = (py.pixel_size_um.unwrap_or(1.0) * l.downsample, py.z_spacing_um.or(py.pixel_size_um).unwrap_or(1.0) * l.z_downsample);
            let spacing = [dz_um as f32, px_um as f32, px_um as f32];
            let dims = [l.depth, l.height, l.width];
            let err = |e: &dyn std::fmt::Display| e.to_string();
            // The channels on the device, read and uploaded once per image and level
            // Volumes uploaded to a device used before a switch are not on this one
            {
                let mut v = volumes.lock().map_err(|e| err(&e))?;
                if v.generation != generation {
                    *v = DeviceVolumes { generation, ..Default::default() };
                }
            }
            let image_key = (id.clone(), level);
            let cached = volumes.lock().map_err(|e| err(&e))?.image.as_ref().filter(|(k, _)| *k == image_key).map(|(_, v)| v.clone());
            let volume = match cached {
                Some(v) => v,
                None => {
                    let mut values = Vec::new();
                    for c in 0..displays.len() {
                        values.extend(py.read_block(level, c, [0, 0, 0], dims).map_err(|e| err(&e))?);
                    }
                    let v = Arc::new(DeviceArray::from_host(client, Shape::new([displays.len() * dims[0], dims[1], dims[2]]), values).map_err(|e| err(&e))?);
                    volumes.lock().map_err(|e| err(&e))?.image = Some((image_key, v.clone()));
                    v
                }
            };
            // The annotation classes, rasterized again when the annotations change
            let labels_key = (id.clone(), level, rev);
            let cached = volumes.lock().map_err(|e| err(&e))?.labels.as_ref().filter(|(k, _, _)| *k == labels_key).map(|(_, v, c)| (v.clone(), c.clone()));
            let (labels, label_colors) = match cached {
                Some(l) => l,
                None => {
                    let (values, colors) = volume_labels(&anns, &classes, &l);
                    let v = if colors.is_empty() { None } else { Some(Arc::new(DeviceArray::from_host(client, Shape::new([colors.len() * dims[0], dims[1], dims[2]]), values).map_err(|e| err(&e))?)) };
                    volumes.lock().map_err(|e| err(&e))?.labels = Some((labels_key, v.clone(), colors.clone()));
                    (v, colors)
                }
            };
            let mut channels: Vec<VolumeChannel> = displays.iter().map(|d| VolumeChannel { visible: d.visible, window: (d.min, d.max), color: d.color, gamma: d.gamma }).collect();
            if labels.is_some() {
                channels.extend(label_colors.iter().map(|c| VolumeChannel { visible: true, window: (0.0, 1.0), color: [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0], gamma: 1.0 }));
            }
            if !channels.iter().any(|c| c.visible) {
                return Err("No channel is visible".to_string());
            }
            let t_device = Instant::now();
            let out = render_volume_rgba(client, &volume, labels.as_deref(), dims, spacing, camera, (iw, ih), &channels, mode, quality)?;
            let pixels = out.to_host(client);
            let device_ms = t_device.elapsed().as_secs_f64() * 1000.0;
            let image = bgra_image(iw, ih, unpack_bgra(&pixels));
            let frame = Frame { image, camera, size: (iw, ih), dims, spacing };
            Ok::<_, String>((frame, Timing { side, device_ms, total_ms: t0.elapsed().as_secs_f64() * 1000.0 }))
        });
        cx.spawn(async move |this, cx| {
            let done = work.await;
            let _ = this.update(cx, |v, cx| {
                v.busy = false;
                match done {
                    Ok((frame, timing)) => {
                        if let Some(old) = v.frame.replace(frame) {
                            v.drops.push(old.image);
                        }
                        v.last = Some(timing);
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
        let dims0 = [py.depth(), py.height(), py.width()];
        let cursor = s.cursor();
        // A frame when what is shown changed (image, channels, annotations, size of the pane, a
        // device arriving)
        let b = self.bounds.get().size;
        let rev = if self.show_labels { s.annotations_rev } else { u64::MAX };
        let key = format!("{:?}/{:?}/{rev}/{}/{}/{:?}", s.doc.active_image, displays_of(s), b.width.as_f32() as i32, b.height.as_f32() as i32, s.device.as_ref().map(|_| s.device_generation));
        if key != self.rendered_key && b.width.as_f32() > 16.0 {
            self.rendered_key = key;
            self.request(false, cx);
        }
        // The box outline and the cursor, projected with the frame's own camera
        let overlay = self.frame.as_ref().map(|f| {
            let corner = |z: f32, y: f32, x: f32| project_point(f.dims, f.spacing, f.camera, f.size, [z, y, x]);
            let corners: Vec<[f32; 2]> = (0..8).map(|i| corner((i >> 2 & 1) as f32, (i >> 1 & 1) as f32, (i & 1) as f32)).collect();
            let frac = cursor.map(|c| [(c[0] as f32 + 0.5) / dims0[0] as f32, (c[1] as f32 + 0.5) / dims0[1] as f32, (c[2] as f32 + 0.5) / dims0[2] as f32]);
            let cur = frac.map(|q| corner(q[0], q[1], q[2]));
            // The three planes through the cursor, each in the colour of the axis it is across
            let planes: Vec<(Vec<[f32; 2]>, [u8; 3])> = frac
                .map(|q| {
                    let sq = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
                    vec![
                        (sq.iter().map(|&(a, b)| corner(q[0], a, b)).collect(), AXIS_Z),
                        (sq.iter().map(|&(a, b)| corner(a, q[1], b)).collect(), AXIS_Y),
                        (sq.iter().map(|&(a, b)| corner(a, b, q[2])).collect(), AXIS_X),
                    ]
                })
                .unwrap_or_default();
            (corners, cur, f.size, planes)
        });
        let image = self.frame.as_ref().map(|f| f.image.clone());
        let bounds_cell = self.bounds.clone();
        let picture = canvas(
            move |bounds, window, _| {
                // The pane changed size: render again so the frame is made for it
                if bounds_cell.get() != bounds {
                    bounds_cell.set(bounds);
                    window.refresh();
                }
            },
            move |bounds, _, window, _| {
                window.paint_quad(fill(bounds, gpui_kit::black()));
                if let Some(img) = image {
                    let _ = window.paint_image(bounds, bounds, Corners::default(), img, 0, false);
                }
                let Some((corners, cur, (iw, ih), planes)) = overlay else { return };
                let to = |p: [f32; 2]| point(bounds.origin.x + px(p[0] / iw as f32 * bounds.size.width.as_f32()), bounds.origin.y + px(p[1] / ih as f32 * bounds.size.height.as_f32()));
                // The 12 edges; the three from voxel (0, 0, 0) in the axis colours
                for i in 0..8usize {
                    for bit in [1usize, 2, 4] {
                        let j = i | bit;
                        if j == i {
                            continue;
                        }
                        let (c, w) = if i == 0 {
                            (match bit {
                                1 => theme::rgb8(AXIS_X),
                                2 => theme::rgb8(AXIS_Y),
                                _ => theme::rgb8(AXIS_Z),
                            }, 1.5)
                        } else {
                            (gpui_kit::white().opacity(0.25), 1.0)
                        };
                        let mut path = PathBuilder::stroke(px(w));
                        path.add_polygon(&[to(corners[i]), to(corners[j])], false);
                        if let Ok(p) = path.build() {
                            window.paint_path(p, c);
                        }
                    }
                }
                for (quad, col) in &planes {
                    let pts: Vec<_> = quad.iter().map(|p| to(*p)).collect();
                    let mut path = PathBuilder::stroke(px(1.));
                    path.add_polygon(&pts, true);
                    if let Ok(p) = path.build() {
                        window.paint_path(p, theme::rgb8(*col).opacity(0.7));
                    }
                }
                if let Some(c) = cur {
                    let c = to(c);
                    let r = px(6.);
                    paint_line(window, point(c.x - r, c.y), point(c.x + r, c.y), 1.5, gpui_kit::white());
                    paint_line(window, point(c.x, c.y - r), point(c.x, c.y + r), 1.5, gpui_kit::white());
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
            (None, Some(t)) => format!("{} voxels a side · device {:.0} ms · frame {:.0} ms", t.side, t.device_ms, t.total_ms),
            _ => "Rendering…".to_string(),
        };
        let (reset, labels, mode, show_labels) = (me.clone(), me.clone(), self.mode, self.show_labels);
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
                    .child(Button::new("volume-labels").ghost().xsmall().label("Annotations").tooltip("Show the annotation classes in the volume").selected(show_labels).on_click(move |_, _, cx| {
                        let _ = labels.update(cx, |v, cx| {
                            v.show_labels = !v.show_labels;
                            v.request(false, cx);
                        });
                    }))
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
                    .when(self.frame.is_none(), |d| d.child(div().absolute().size_full().flex().items_center().justify_center().text_sm().text_color(color::text_muted()).child("Rendering…"))),
            )
            .into_any_element()
    }
}

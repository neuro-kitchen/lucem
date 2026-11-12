//! The Navigator (Affinity's): the whole image small, with the part the focused view shows
//! outlined; a click or drag moves the view there, and the zoom is stepped below it.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Sizable as _};
use gpui_kit::{
    canvas, div, fill, point, px, size, AppContext as _, Bounds, Context, Corners, Entity, InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, ParentElement as _, PathBuilder,
    Pixels, Render, RenderImage, Styled as _, Subscription, Task, WeakEntity, Window,
};
use lucem_view::{bgra_image, composite_bgra, ChannelDisplay};

use crate::data::store::{AppEvent, Store};
use crate::screens::explore::canvas::ImageCanvas;
use crate::ui::theme::{self, color};
use crate::ui::widgets::stepper;

/// Longest side of the thumbnail's level.
const THUMB_SIDE: usize = 512;
/// Tallest the thumbnail is drawn (px).
const THUMB_MAX_H: f32 = 220.;

pub struct Navigator {
    store: Entity<Store>,
    /// The view it follows (Explore's focused plane).
    pub canvas: Option<WeakEntity<ImageCanvas>>,
    thumb: Option<Arc<RenderImage>>,
    key: String,
    task: Option<Task<()>>,
    drops: Vec<Arc<RenderImage>>,
    drawn: Rc<Cell<Bounds<Pixels>>>,
    _subs: Vec<Subscription>,
}

impl Navigator {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify())];
        Self { store, canvas: None, thumb: None, key: String::new(), task: None, drops: Vec::new(), drawn: Rc::new(Cell::new(Bounds::default())), _subs: subs }
    }

    /// The thumbnail of the image the view shows, at its plane, made again when that changes.
    fn sync(&mut self, canvas: &Entity<ImageCanvas>, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let c = canvas.read(cx);
        let Some(id) = c.image_id(s) else { return };
        let Some(py) = s.pyramids.get(&id).cloned() else { return };
        let Some(img) = s.doc.image(&id) else { return };
        let displays: Vec<ChannelDisplay> = img.channels.iter().map(|c| ChannelDisplay { visible: c.visible, color: [c.color[0] as f32 / 255.0, c.color[1] as f32 / 255.0, c.color[2] as f32 / 255.0], min: c.min, max: c.max, gamma: c.gamma }).collect();
        let level = (0..py.levels.len()).find(|&l| py.levels[l].width.max(py.levels[l].height) <= THUMB_SIDE).unwrap_or(py.coarsest());
        let l = py.levels[level].clone();
        let plane = ((c.z as f64 / l.z_downsample) as usize).min(l.depth - 1);
        let key = format!("{id}/{level}/{plane}/{displays:?}");
        if key == self.key {
            return;
        }
        self.key = key;
        let work = cx.background_spawn(async move {
            let mut planes = Vec::new();
            for (ch, d) in displays.iter().enumerate().filter(|(_, d)| d.visible) {
                planes.push((py.read_region_at(level, ch, plane, 0, 0, l.width, l.height).ok()?, *d));
            }
            let refs: Vec<(&[f32], ChannelDisplay)> = planes.iter().map(|(v, d)| (v.as_slice(), *d)).collect();
            Some(bgra_image(l.width, l.height, composite_bgra(&refs, l.width * l.height)))
        });
        self.task = Some(cx.spawn(async move |this, cx| {
            let done = work.await;
            let _ = this.update(cx, |n, cx| {
                if let Some(old) = std::mem::replace(&mut n.thumb, done) {
                    n.drops.push(old);
                }
                cx.notify();
            });
        }));
    }

    /// Centres the view on the point at fractions `(fx, fy)` of the thumbnail.
    fn go(&self, fx: f64, fy: f64, cx: &mut Context<Self>) {
        let Some(canvas) = self.canvas.as_ref().and_then(|c| c.upgrade()) else { return };
        let s = self.store.read(cx);
        let Some(py) = canvas.read(cx).image_id(s).and_then(|id| s.pyramids.get(&id).cloned()) else { return };
        let p = [fx.clamp(0.0, 1.0) * py.width() as f64, fy.clamp(0.0, 1.0) * py.height() as f64];
        canvas.update(cx, |c, cx| c.reveal(p, cx));
    }
}

impl Render for Navigator {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        for img in std::mem::take(&mut self.drops) {
            let _ = window.drop_image(img);
        }
        let Some(view) = self.canvas.as_ref().and_then(|c| c.upgrade()) else {
            return v_flex().p(px(theme::MD)).text_sm().text_color(color::text_muted()).child("No view").into_any_element();
        };
        self.sync(&view, cx);
        let s = self.store.read(cx);
        let c = view.read(cx);
        let Some(py) = c.image_id(s).and_then(|id| s.pyramids.get(&id).cloned()) else {
            return v_flex().p(px(theme::MD)).text_sm().text_color(color::text_muted()).child("No image").into_any_element();
        };
        let (iw, ih) = (py.width() as f64, py.height() as f64);
        let shown = c.visible();
        let zoom = c.zoom;
        let thumb = self.thumb.clone();
        let drawn = self.drawn.clone();
        let picture = canvas(
            |_, _, _| {},
            move |bounds, _, window, _| {
                let scale = (bounds.size.width.as_f32() as f64 / iw).min(bounds.size.height.as_f32() as f64 / ih);
                let (w, h) = ((iw * scale) as f32, (ih * scale) as f32);
                let o = point(bounds.origin.x + px((bounds.size.width.as_f32() - w) / 2.0), bounds.origin.y + px((bounds.size.height.as_f32() - h) / 2.0));
                let b = Bounds { origin: o, size: size(px(w), px(h)) };
                drawn.set(b);
                window.paint_quad(fill(b, gpui_kit::black()));
                if let Some(img) = thumb {
                    let _ = window.paint_image(bounds, b, Corners::default(), img, 0, false);
                }
                // What the view shows, clipped to the image
                let (x0, y0) = ((shown[0].max(0.0) * scale) as f32, (shown[1].max(0.0) * scale) as f32);
                let (x1, y1) = ((((shown[0] + shown[2]).min(iw)) * scale) as f32, (((shown[1] + shown[3]).min(ih)) * scale) as f32);
                if x1 > x0 && y1 > y0 {
                    let pts = [point(o.x + px(x0), o.y + px(y0)), point(o.x + px(x1), o.y + px(y0)), point(o.x + px(x1), o.y + px(y1)), point(o.x + px(x0), o.y + px(y1))];
                    for (width, col) in [(3.0, gpui_kit::black().opacity(0.6)), (1.5, theme::rgb8([235, 92, 92]))] {
                        let mut path = PathBuilder::stroke(px(width));
                        path.add_polygon(&pts, true);
                        if let Ok(p) = path.build() {
                            window.paint_path(p, col);
                        }
                    }
                }
            },
        )
        .size_full();
        let at = |e: gpui_kit::Point<Pixels>, b: Bounds<Pixels>| (((e.x - b.origin.x).as_f32() / b.size.width.as_f32().max(1.0)) as f64, ((e.y - b.origin.y).as_f32() / b.size.height.as_f32().max(1.0)) as f64);
        let (d1, d2) = (self.drawn.clone(), self.drawn.clone());
        let thumb_h = ((ih / iw) as f32 * 272.0).clamp(80.0, THUMB_MAX_H);
        let (zc, fc) = (view.clone(), view.clone());
        v_flex()
            .size_full()
            .p(px(theme::SM))
            .gap(px(theme::SM))
            .child(
                div()
                    .id("navigator-thumb")
                    .w_full()
                    .h(px(thumb_h))
                    .cursor(gpui_kit::CursorStyle::Crosshair)
                    .on_mouse_down(MouseButton::Left, cx.listener(move |n, e: &MouseDownEvent, _, cx| {
                        let (fx, fy) = at(e.position, d1.get());
                        n.go(fx, fy, cx);
                    }))
                    .on_mouse_move(cx.listener(move |n, e: &MouseMoveEvent, _, cx| {
                        if e.pressed_button == Some(MouseButton::Left) {
                            let (fx, fy) = at(e.position, d2.get());
                            n.go(fx, fy, cx);
                        }
                    }))
                    .child(picture),
            )
            .child(
                h_flex()
                    .gap(px(theme::SM))
                    .child(div().text_xs().text_color(color::text_muted()).child("Zoom"))
                    .child(stepper("navigator-zoom", format!("{:.0} %", zoom * 100.0), move |d, _, cx| zc.update(cx, |c, cx| c.zoom_at(if d > 0.0 { 1.25 } else { 0.8 }, None, cx))))
                    .child(div().flex_1())
                    .child(Button::new("navigator-fit").ghost().xsmall().label("Fit").on_click(move |_, _, cx| fc.update(cx, |c, cx| c.fit(cx)))),
            )
            .into_any_element()
    }
}

//! A colour wheel (Affinity style): a hue ring around a saturation / value square, with swatches
//! of the usual fluorescence colours. Dragging on either emits [`WheelEvent::Changing`] on every
//! move (for a live preview) and [`WheelEvent::Done`] on release (one undo step).

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use gpui_kit::component::{h_flex, v_flex};
use gpui_kit::{
    canvas, div, point, px, size, Bounds, Context, Corners, EventEmitter, InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _,
    PathBuilder, Pixels, Render, RenderImage, StatefulInteractiveElement as _, Styled as _, Window,
};
use lucem_view::bgra_image;

use super::theme::{self, color};

/// Side of the wheel (px).
const SIDE: f32 = 168.;
/// Width of the hue ring (px).
const RING: f32 = 16.;
/// Swatches under the wheel.
pub const SWATCHES: [[u8; 3]; 10] = [[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 128, 0], [255, 0, 0], [255, 0, 255], [128, 0, 255], [255, 255, 255], [64, 128, 255]];

pub enum WheelEvent {
    Changing([u8; 3]),
    Done([u8; 3]),
}

#[derive(Clone, Copy, PartialEq)]
enum Grab {
    Ring,
    Square,
}

pub struct ColorWheel {
    /// Hue (0–1), saturation, value.
    h: f32,
    s: f32,
    v: f32,
    ring: Arc<RenderImage>,
    square: Arc<RenderImage>,
    square_hue: f32,
    grab: Option<Grab>,
    bounds: Rc<Cell<Bounds<Pixels>>>,
}

impl EventEmitter<WheelEvent> for ColorWheel {}

impl ColorWheel {
    pub fn new(rgb: [u8; 3]) -> Self {
        let (h, s, v) = rgb_to_hsv(rgb);
        Self { h, s, v, ring: ring_image(), square: square_image(h), square_hue: h, grab: None, bounds: Rc::new(Cell::new(Bounds::default())) }
    }

    pub fn set(&mut self, rgb: [u8; 3], cx: &mut Context<Self>) {
        if self.grab.is_some() {
            return;
        }
        let (h, s, v) = rgb_to_hsv(rgb);
        (self.h, self.s, self.v) = (h, s, v);
        cx.notify();
    }

    pub fn rgb(&self) -> [u8; 3] {
        hsv_to_rgb(self.h, self.s, self.v)
    }

    /// Geometry: (centre, outer radius, square half side) in local px.
    fn geom() -> ((f32, f32), f32, f32) {
        let r = SIDE / 2.0;
        let inner = r - RING - 6.0;
        ((r, r), r, inner / std::f32::consts::SQRT_2)
    }

    fn pick(&mut self, local: (f32, f32), start: bool) -> bool {
        let ((cx0, cy0), r, half) = Self::geom();
        let (dx, dy) = (local.0 - cx0, local.1 - cy0);
        if start {
            let d = dx.hypot(dy);
            self.grab = if d >= r - RING - 2.0 && d <= r + 2.0 {
                Some(Grab::Ring)
            } else if dx.abs() <= half && dy.abs() <= half {
                Some(Grab::Square)
            } else {
                None
            };
        }
        match self.grab {
            Some(Grab::Ring) => {
                self.h = (dy.atan2(dx) / std::f32::consts::TAU).rem_euclid(1.0);
                true
            }
            Some(Grab::Square) => {
                self.s = ((dx + half) / (2.0 * half)).clamp(0.0, 1.0);
                self.v = (1.0 - (dy + half) / (2.0 * half)).clamp(0.0, 1.0);
                true
            }
            None => false,
        }
    }

    fn local(&self, p: gpui_kit::Point<Pixels>) -> (f32, f32) {
        let o = self.bounds.get().origin;
        ((p.x - o.x).as_f32(), (p.y - o.y).as_f32())
    }

    fn on_down(&mut self, e: &MouseDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.pick(self.local(e.position), true) {
            cx.emit(WheelEvent::Changing(self.rgb()));
            cx.notify();
        }
    }

    fn on_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.grab.is_some() && e.pressed_button == Some(MouseButton::Left) && self.pick(self.local(e.position), false) {
            cx.emit(WheelEvent::Changing(self.rgb()));
            cx.notify();
        }
    }

    fn on_up(&mut self, _: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.grab.take().is_some() {
            cx.emit(WheelEvent::Done(self.rgb()));
            cx.notify();
        }
    }
}

impl Render for ColorWheel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if (self.square_hue - self.h).abs() > 1e-4 {
            self.square = square_image(self.h);
            self.square_hue = self.h;
        }
        let (ring, square) = (self.ring.clone(), self.square.clone());
        let ((cx0, cy0), r, half) = Self::geom();
        let (h, s, v) = (self.h, self.s, self.v);
        let cell = self.bounds.clone();
        let wheel = canvas(
            move |b, _, _| cell.set(b),
            move |b, _, window, _| {
                let o = b.origin;
                let full = Bounds { origin: o, size: size(px(SIDE), px(SIDE)) };
                let _ = window.paint_image(full, full, Corners::default(), ring, 0, false);
                let sq = Bounds { origin: point(o.x + px(cx0 - half), o.y + px(cy0 - half)), size: size(px(2.0 * half), px(2.0 * half)) };
                let _ = window.paint_image(sq, sq, Corners::default(), square, 0, false);
                // Markers: on the ring at the hue, in the square at (s, v)
                let a = h * std::f32::consts::TAU;
                let rm = r - RING / 2.0;
                let hm = point(o.x + px(cx0 + rm * a.cos()), o.y + px(cy0 + rm * a.sin()));
                let sm = point(o.x + px(cx0 - half + s * 2.0 * half), o.y + px(cy0 - half + (1.0 - v) * 2.0 * half));
                for c in [hm, sm] {
                    for (w, col) in [(3.0, gpui_kit::black()), (1.5, gpui_kit::white())] {
                        let mut p = PathBuilder::stroke(px(w));
                        let pts: Vec<_> = (0..24).map(|i| {
                            let t = i as f32 / 24.0 * std::f32::consts::TAU;
                            point(c.x + px(6.0 * t.cos()), c.y + px(6.0 * t.sin()))
                        }).collect();
                        p.add_polygon(&pts, true);
                        if let Ok(p) = p.build() {
                            window.paint_path(p, col);
                        }
                    }
                }
            },
        )
        .size_full();
        let current = self.rgb();
        let swatches = SWATCHES.iter().enumerate().map(|(i, c)| {
            let c = *c;
            div()
                .id(("wheel-swatch", i))
                .size(px(16.))
                .rounded(px(3.))
                .border_1()
                .border_color(if c == current { color::text() } else { color::border() })
                .bg(theme::rgb8(c))
                .cursor_pointer()
                .on_click(cx.listener(move |w, _, _, cx| {
                    (w.h, w.s, w.v) = rgb_to_hsv(c);
                    cx.emit(WheelEvent::Done(c));
                    cx.notify();
                }))
        });
        v_flex()
            .gap(px(theme::SM))
            .child(
                div()
                    .id("color-wheel")
                    .size(px(SIDE))
                    .cursor_crosshair()
                    .on_mouse_down(MouseButton::Left, cx.listener(Self::on_down))
                    .on_mouse_move(cx.listener(Self::on_move))
                    .on_mouse_up(MouseButton::Left, cx.listener(Self::on_up))
                    .child(wheel),
            )
            .child(h_flex().gap(px(theme::XS)).flex_wrap().w(px(SIDE)).children(swatches))
            .child(
                h_flex()
                    .gap(px(theme::SM))
                    .child(div().size(px(16.)).rounded(px(3.)).bg(theme::rgb8(current)))
                    .child(div().text_xs().font_family("monospace").text_color(color::text_muted()).child(format!("#{:02X}{:02X}{:02X}", current[0], current[1], current[2]))),
            )
    }
}

/// The hue ring (transparent elsewhere).
fn ring_image() -> Arc<RenderImage> {
    let n = SIDE as usize * 2;
    let (c, r) = (n as f32 / 2.0, n as f32 / 2.0);
    let inner = r - RING * 2.0;
    let mut px_ = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let (dx, dy) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
            let d = dx.hypot(dy);
            // Soft edges (1 px)
            let a = ((r - d).clamp(0.0, 1.0)).min((d - inner).clamp(0.0, 1.0));
            if a > 0.0 {
                let h = (dy.atan2(dx) / std::f32::consts::TAU).rem_euclid(1.0);
                let [rr, g, b] = hsv_to_rgb(h, 1.0, 1.0);
                let i = (y * n + x) * 4;
                px_[i..i + 4].copy_from_slice(&[b, g, rr, (a * 255.0) as u8]);
            }
        }
    }
    bgra_image(n, n, px_)
}

/// Saturation (x) by value (y, top bright) at hue `h`.
fn square_image(h: f32) -> Arc<RenderImage> {
    let n = 96;
    let mut px_ = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let [r, g, b] = hsv_to_rgb(h, x as f32 / (n - 1) as f32, 1.0 - y as f32 / (n - 1) as f32);
            let i = (y * n + x) * 4;
            px_[i..i + 4].copy_from_slice(&[b, g, r, 255]);
        }
    }
    bgra_image(n, n, px_)
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [(r * 255.0).round() as u8, (g * 255.0).round() as u8, (b * 255.0).round() as u8]
}

pub fn rgb_to_hsv([r, g, b]: [u8; 3]) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };
    (h, if max == 0.0 { 0.0 } else { d / max }, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsv_round_trips() {
        for c in SWATCHES {
            let (h, s, v) = rgb_to_hsv(c);
            assert_eq!(hsv_to_rgb(h, s, v), c);
        }
    }
}

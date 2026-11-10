//! The Display tab (brightness & contrast): per channel, its histogram with the black
//! and white points as markers to drag, its colour on a colour wheel, gamma, visibility and an
//! automatic window. A drag shows live and lands as one undo step.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::assets::IconName;
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::{h_flex, v_flex, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    canvas, div, fill, point, px, size, AppContext as _, Bounds, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, ParentElement as _, Pixels, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::data::store::{AppEvent, ChannelHistogram, Store};
use crate::domain::command::Command;
use crate::domain::project::ChannelDoc;
use crate::ui::color_wheel::{ColorWheel, WheelEvent};
use crate::ui::theme::{self, color};
use crate::ui::widgets::{row, stepper, swatch};

/// Height of a histogram (px).
const HIST_H: f32 = 64.;

#[derive(Clone, Copy, PartialEq)]
enum Marker {
    Min,
    Max,
}

pub struct DisplayPanel {
    store: Entity<Store>,
    /// Histogram bounds per channel (set while painting).
    bounds: Vec<Rc<Cell<Bounds<Pixels>>>>,
    drag: Option<(usize, Marker)>,
    wheel: Option<(usize, Entity<ColorWheel>, Subscription)>,
    /// Channels whose card is folded to its header.
    folded: std::collections::HashSet<usize>,
    _subs: Vec<Subscription>,
}

impl DisplayPanel {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::ImageReady(_)) {
                cx.notify();
            }
        })];
        Self { store, bounds: Vec::new(), drag: None, wheel: None, folded: Default::default(), _subs: subs }
    }

    fn channel(&self, c: usize, cx: &Context<Self>) -> Option<(String, ChannelDoc)> {
        let s = self.store.read(cx);
        let img = s.doc.active()?;
        Some((img.id.clone(), img.channels.get(c)?.clone()))
    }

    /// The histogram's value range for channel `c`.
    fn range(&self, c: usize, cx: &Context<Self>) -> (f32, f32) {
        let s = self.store.read(cx);
        let id = s.doc.active_image.clone().unwrap_or_default();
        match s.histograms.get(&id).and_then(|h| h.get(c)) {
            Some(h) => {
                // Include the current window, so markers outside the data stay reachable
                let ch = s.doc.active().and_then(|i| i.channels.get(c));
                (h.lo.min(ch.map_or(h.lo, |d| d.min)), h.hi.max(ch.map_or(h.hi, |d| d.max)))
            }
            None => s.active_pyramid().and_then(|p| p.channels.get(c)).map_or((0.0, 1.0), |i| i.range),
        }
    }

    fn value_at(&self, c: usize, x: Pixels, cx: &Context<Self>) -> Option<f32> {
        let b = self.bounds.get(c)?.get();
        let t = ((x - b.origin.x).as_f32() / b.size.width.as_f32().max(1.0)).clamp(0.0, 1.0);
        let (lo, hi) = self.range(c, cx);
        Some(lo + t * (hi - lo))
    }

    fn on_hist_down(&mut self, c: usize, e: &MouseDownEvent, cx: &mut Context<Self>) {
        let Some(v) = self.value_at(c, e.position.x, cx) else { return };
        let Some((_, ch)) = self.channel(c, cx) else { return };
        // The nearer marker follows the pointer
        let m = if (v - ch.min).abs() <= (v - ch.max).abs() { Marker::Min } else { Marker::Max };
        self.drag = Some((c, m));
        self.drag_to(e.position.x, cx);
    }

    fn drag_to(&mut self, x: Pixels, cx: &mut Context<Self>) {
        let Some((c, m)) = self.drag else { return };
        let Some(v) = self.value_at(c, x, cx) else { return };
        let Some((img, mut ch)) = self.channel(c, cx) else { return };
        let (lo, hi) = self.range(c, cx);
        let eps = ((hi - lo) * 0.002).max(1e-6);
        match m {
            Marker::Min => ch.min = v.min(ch.max - eps),
            Marker::Max => ch.max = v.max(ch.min + eps),
        }
        self.store.update(cx, |s, cx| s.set_channel_live(&img, c, ch, cx));
    }

    fn on_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.drag.is_some() {
            if e.pressed_button == Some(MouseButton::Left) {
                self.drag_to(e.position.x, cx);
            } else {
                self.finish(cx);
            }
        }
    }

    fn finish(&mut self, cx: &mut Context<Self>) {
        if self.drag.take().is_some() {
            self.store.update(cx, |s, cx| s.commit_live(cx));
        }
    }

    fn toggle_wheel(&mut self, c: usize, cx: &mut Context<Self>) {
        if self.wheel.as_ref().is_some_and(|(w, _, _)| *w == c) {
            self.wheel = None;
            cx.notify();
            return;
        }
        let Some((_, ch)) = self.channel(c, cx) else { return };
        let wheel = cx.new(|_| ColorWheel::new(ch.color));
        let sub = cx.subscribe(&wheel, move |this, _, e: &WheelEvent, cx| {
            let Some((img, mut ch)) = this.channel(c, cx) else { return };
            match e {
                WheelEvent::Changing(rgb) => {
                    ch.color = *rgb;
                    this.store.update(cx, |s, cx| s.set_channel_live(&img, c, ch, cx));
                }
                WheelEvent::Done(rgb) => {
                    ch.color = *rgb;
                    this.store.update(cx, |s, cx| {
                        s.set_channel_live(&img, c, ch, cx);
                        s.commit_live(cx);
                    });
                }
            }
        });
        self.wheel = Some((c, wheel, sub));
        cx.notify();
    }

    fn histogram(&self, c: usize, h: Option<ChannelHistogram>, ch: &ChannelDoc, (lo, hi): (f32, f32), cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let cell = self.bounds[c].clone();
        let col = theme::rgb8(ch.color);
        let (min, max) = (ch.min, ch.max);
        let plot = canvas(
            move |b, _, _| cell.set(b),
            move |b, _, window, _| {
                let o = b.origin;
                let (w, hh) = (b.size.width.as_f32(), b.size.height.as_f32());
                window.paint_quad(fill(b, color::canvas()));
                let span = (hi - lo).max(1e-12);
                let xf = |v: f32| ((v - lo) / span * w).clamp(0.0, w);
                if let Some(h) = &h {
                    // Log counts: faint tails stay visible
                    let peak = h.bins.iter().map(|&c| (c as f32 + 1.0).ln()).fold(0.0, f32::max).max(1e-6);
                    let n = h.bins.len() as f32;
                    let hspan = (h.hi - h.lo).max(1e-12);
                    for (i, &count) in h.bins.iter().enumerate() {
                        if count == 0 {
                            continue;
                        }
                        let v0 = h.lo + i as f32 / n * hspan;
                        let v1 = h.lo + (i as f32 + 1.0) / n * hspan;
                        let (x0, x1) = (xf(v0), xf(v1).max(xf(v0) + 1.0));
                        let bh = (count as f32 + 1.0).ln() / peak * (hh - 4.0);
                        window.paint_quad(fill(Bounds { origin: point(o.x + px(x0), o.y + px(hh - bh)), size: size(px(x1 - x0), px(bh)) }, col.opacity(0.75)));
                    }
                }
                // Outside the window: shaded
                let (mx0, mx1) = (xf(min), xf(max));
                window.paint_quad(fill(Bounds { origin: o, size: size(px(mx0), px(hh)) }, gpui_kit::black().opacity(0.45)));
                window.paint_quad(fill(Bounds { origin: point(o.x + px(mx1), o.y), size: size(px((w - mx1).max(0.0)), px(hh)) }, gpui_kit::black().opacity(0.45)));
                for x in [mx0, mx1] {
                    window.paint_quad(fill(Bounds { origin: point(o.x + px(x - 1.0), o.y), size: size(px(2.), px(hh)) }, gpui_kit::white()));
                }
            },
        )
        .size_full();
        div()
            .id(SharedString::from(format!("hist-{c}")))
            .w_full()
            .h(px(HIST_H))
            .rounded(px(4.))
            .overflow_hidden()
            .border_1()
            .border_color(color::border())
            .cursor_ew_resize()
            .on_mouse_down(MouseButton::Left, cx.listener(move |this, e: &MouseDownEvent, _, cx| this.on_hist_down(c, e, cx)))
            .child(plot)
    }
}

fn trim(v: f32) -> String {
    if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else if v.abs() >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.4}")
    }
}

impl Render for DisplayPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let s = self.store.read(cx);
        let Some(img) = s.doc.active().cloned() else {
            return div().p(px(theme::MD)).text_sm().text_color(color::text_muted()).child("No image.").into_any_element();
        };
        let hists = s.histograms.get(&img.id).cloned();
        let py = s.pyramids.get(&img.id).cloned();
        while self.bounds.len() < img.channels.len() {
            self.bounds.push(Rc::new(Cell::new(Bounds::default())));
        }
        let mut list = v_flex().w_full().gap(px(theme::SM));
        // The image
        if let Some(py) = &py {
            list = list.child(
                v_flex()
                    .gap(px(theme::XS))
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).truncate().child(img.name.clone()))
                    .child(div().text_xs().text_color(color::text_muted()).child(format!(
                        "{} × {} px · {} · {} levels{}",
                        py.width(),
                        py.height(),
                        py.dtype,
                        py.levels.len(),
                        py.pixel_size_um.map_or(String::new(), |u| format!(" · {u:.4} µm/px"))
                    ))),
            );
        }
        for (c, ch) in img.channels.iter().enumerate() {
            let range = self.range(c, cx);
            let id = img.id.clone();
            let set = {
                let (store, id, ch) = (self.store.clone(), id.clone(), ch.clone());
                Rc::new(move |f: &dyn Fn(&mut ChannelDoc), cx: &mut gpui_kit::App| {
                    let mut d = ch.clone();
                    f(&mut d);
                    let id = id.clone();
                    store.update(cx, |s, cx| {
                        let _ = s.execute(Command::SetChannel { image: id, channel: c, display: d }, cx);
                    });
                })
            };
            let (s1, s2) = (set.clone(), set.clone());
            let auto = {
                let (store, id, ch, py) = (self.store.clone(), id.clone(), ch.clone(), py.clone());
                move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut gpui_kit::App| {
                    let Some(py) = &py else { return };
                    if let Ok((lo, hi)) = py.auto_window(c, 0.001, 0.998) {
                        let d = ChannelDoc { min: lo, max: hi.max(lo + 1e-6), ..ch.clone() };
                        let id = id.clone();
                        store.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetChannel { image: id, channel: c, display: d }, cx);
                        });
                    }
                }
            };
            let wheel_open = self.wheel.as_ref().filter(|(w, _, _)| *w == c).map(|(_, e, _)| e.clone());
            if let Some(w) = &wheel_open {
                let rgb = ch.color;
                w.update(cx, |w, cx| w.set(rgb, cx));
            }
            let hist = hists.as_ref().and_then(|h| h.get(c).cloned());
            let folded = self.folded.contains(&c);
            let fold = Button::new(SharedString::from(format!("ch-fold-{c}"))).ghost().xsmall().icon(gpui_kit::component::Icon::new(if folded { IconName::ChevronRight } else { IconName::ChevronDown })).on_click(cx.listener(move |this, _, _, cx| {
                if !this.folded.remove(&c) {
                    this.folded.insert(c);
                }
                cx.notify();
            }));
            let header = h_flex()
                            .gap(px(theme::SM))
                            .child(fold)
                            .child(Checkbox::new(SharedString::from(format!("ch-visible-{c}"))).checked(ch.visible).on_click(move |on: &bool, _, cx| {
                                let on = *on;
                                s1(&move |d: &mut ChannelDoc| d.visible = on, cx)
                            }))
                            .child(
                                div()
                                    .id(SharedString::from(format!("ch-color-{c}")))
                                    .cursor_pointer()
                                    .rounded(px(3.))
                                    .when(wheel_open.is_some(), |d| d.border_1().border_color(color::accent_hover()))
                                    .child(swatch(ch.color, 16.))
                                    .on_click(cx.listener(move |this, _, _, cx| this.toggle_wheel(c, cx))),
                            )
                            .child(div().flex_1().min_w_0().truncate().text_sm().font_weight(FontWeight::SEMIBOLD).child(ch.label.clone()))
                            .child(Button::new(SharedString::from(format!("ch-auto-{c}"))).ghost().xsmall().label("Auto").tooltip("Window from the 0.1–99.8 % percentiles").on_click(auto));
            // A card per channel; folded, only its header
            if folded {
                list = list.child(v_flex().w_full().p(px(theme::SM)).rounded(px(6.)).bg(color::raised()).child(header));
                continue;
            }
            list = list.child(
                v_flex()
                    .w_full()
                    .gap(px(theme::SM))
                    .p(px(theme::SM))
                    .rounded(px(6.))
                    .bg(color::raised())
                    .child(header)
                    .children(wheel_open.map(|w| div().pl(px(theme::LG)).child(w)))
                    .child(self.histogram(c, hist, ch, range, cx))
                    .child(
                        h_flex()
                            .text_xs()
                            .font_family("monospace")
                            .text_color(color::text_muted())
                            .child(div().child(format!("black {}", trim(ch.min))))
                            .child(div().flex_1())
                            .child(div().child(format!("white {}", trim(ch.max)))),
                    )
                    .child(row("Gamma", stepper(&format!("ch-gamma-{c}"), format!("{:.2}", ch.gamma), move |d, _, cx| s2(&move |x: &mut ChannelDoc| x.gamma = (x.gamma * if d > 0.0 { 1.1 } else { 1.0 / 1.1 }).clamp(0.1, 5.0), cx))))
            );
        }
        div()
            .id("display-panel")
            .size_full()
            .overflow_y_scroll()
            .p(px(theme::MD))
            .on_mouse_move(cx.listener(Self::on_move))
            .on_mouse_up(MouseButton::Left, cx.listener(|this, _: &MouseUpEvent, _, cx| this.finish(cx)))
            .child(list)
            .into_any_element()
    }
}

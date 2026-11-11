//! The editor at the top of the Layers panel (Affinity's / Procreate's colour-over-layers): it
//! edits whatever is selected in the tree under it, and shows only what that needs.
//!
//! - **A channel**: its colour (a swatch opening the colour wheel), the histogram right under it
//!   with the black and white points to drag, gamma; then its **processing**: the steps applied
//!   to it (dsp-kitchen's Processing card), Processed / Raw, and "+ Add step".
//! - **A shape class** (or one shape: its class): colour presets, opacity, fill.
//! - **A result**: colour map (or a mask's colour), opacity.
//! - **A processing step**: its settings.
//!
//! A drag shows live and lands as one undo step.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    canvas, div, fill, point, px, size, AnyElement, AppContext as _, Bounds, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, ParentElement as _, Pixels, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};
use lumen_pipeline::PortType;

use crate::data::results::COLORMAPS;
use crate::data::store::{AppEvent, ChannelHistogram, Store};
use crate::domain::annotation::ClassDoc;
use crate::domain::command::Command;
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::domain::project::ChannelDoc;
use crate::screens::pipeline::catalog::{family_icon, operations, summary, Template};
use crate::ui::color_wheel::{ColorWheel, WheelEvent};
use crate::ui::param_editor::{Commit, ParamEditor};
use crate::ui::theme::{self, color};
use crate::ui::widgets::{row, slider, swatch};

/// Height of a histogram (px).
const HIST_H: f32 = 64.;

/// Colours offered for a channel (the usual fluorescence and brightfield ones).
pub const CHANNEL_COLORS: [(&str, [u8; 3]); 7] = [("Red", [255, 0, 0]), ("Green", [0, 255, 0]), ("Blue", [0, 0, 255]), ("Cyan", [0, 255, 255]), ("Magenta", [255, 0, 255]), ("Yellow", [255, 255, 0]), ("White", [255, 255, 255])];

/// One row of colour presets that share the width (they narrow with the panel, never wrap),
/// the current one outlined; `pick` gets the colour.
fn preset_row(id: &str, presets: &[(&'static str, [u8; 3])], current: [u8; 3], pick: impl Fn([u8; 3], &mut gpui_kit::App) + 'static) -> gpui_kit::Div {
    let pick = Rc::new(pick);
    h_flex().w_full().gap(px(theme::XS)).children(presets.iter().map(|(name, rgb)| {
        let (pick, rgb, name) = (pick.clone(), *rgb, *name);
        div()
            .id(SharedString::from(format!("{id}-{name}")))
            .flex_1()
            .min_w(px(10.))
            .h(px(20.))
            .rounded(px(4.))
            .cursor_pointer()
            .bg(theme::rgb8(rgb))
            .border_2()
            .border_color(if rgb == current { color::accent_hover() } else { color::border() })
            .tooltip(move |w, cx| gpui_kit::component::tooltip::Tooltip::new(name).build(w, cx))
            .on_click(move |_, _, cx| pick(rgb, cx))
    }))
}

/// Colours offered for a shape class.
pub const CLASS_COLORS: [(&str, [u8; 3]); 8] = [("Yellow", [255, 196, 0]), ("Red", [230, 70, 70]), ("Green", [70, 190, 100]), ("Blue", [70, 130, 240]), ("Purple", [170, 90, 230]), ("Cyan", [0, 200, 220]), ("Orange", [240, 140, 40]), ("Grey", [150, 150, 150])];

#[derive(Clone, Copy, PartialEq)]
enum Marker {
    Min,
    Max,
}

/// What the editor edits.
#[derive(Clone, PartialEq)]
enum Target {
    Channel(usize),
    Class(String),
    View(String),
    Step(String),
    Nothing,
}

/// The selected layer: the last row picked in the tree, else the class of the one selected shape.
fn target(s: &Store) -> Target {
    if let Some(k) = s.sel.layers.last() {
        let (kind, rest) = k.split_once(':').unwrap_or((k.as_str(), ""));
        match kind {
            "channel" => {
                if let Ok(c) = rest.parse() {
                    return Target::Channel(c);
                }
            }
            "class" => return Target::Class(rest.to_string()),
            "view" => return Target::View(rest.to_string()),
            "node" => return Target::Step(rest.to_string()),
            _ => {}
        }
    }
    if s.sel.shapes.len() == 1
        && let Some(a) = s.doc.annotations.iter().find(|a| s.sel.shapes.contains(&a.id))
    {
        return Target::Class(a.class.clone());
    }
    Target::Nothing
}

/// The processing of channel `channel` of `image` (see `GraphDoc::channel_chain`).
pub fn chain_of(s: &Store, image: &str, channel: usize) -> (Vec<String>, Option<String>) {
    s.doc.graph.channel_chain(image, channel)
}

pub struct DisplayPanel {
    store: Entity<Store>,
    /// Histogram bounds per channel (set while painting).
    bounds: Vec<Rc<Cell<Bounds<Pixels>>>>,
    drag: Option<(usize, Marker)>,
    /// The colour wheel open, and for which layer key (`channel:2`, `class:Tumor`).
    wheel: Option<(String, Entity<ColorWheel>, Subscription)>,
    /// A slider being dragged: its key and value (committed on release, one undo step).
    sliding: Option<(String, f32)>,
    params: Entity<ParamEditor>,
    /// The channel editor's page (Affinity's Colour panel: tabs on top, one page at a time).
    page: ChannelPage,
    _subs: Vec<Subscription>,
}

/// The pages of a channel's editor.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ChannelPage {
    Colour,
    Levels,
}

/// A row of page tabs (Affinity's panel tabs): the chosen one bold and underlined.
fn page_tabs<T: Copy + PartialEq + 'static>(id: &'static str, pages: Vec<(T, String)>, current: T, pick: impl Fn(T, &mut gpui_kit::App) + 'static) -> impl IntoElement {
    let pick = Rc::new(pick);
    h_flex().gap(px(theme::MD)).border_b_1().border_color(color::border()).children(pages.into_iter().enumerate().map(|(i, (p, label))| {
        let (on, pick) = (p == current, pick.clone());
        div()
            .id((id, i))
            .pb(px(4.))
            .text_sm()
            .cursor_pointer()
            .border_b_2()
            .border_color(if on { color::accent() } else { gpui_kit::transparent_black() })
            .when(on, |d| d.font_weight(FontWeight::SEMIBOLD))
            .when(!on, |d| d.text_color(color::text_muted()).hover(|d| d.text_color(color::text())))
            .child(label.clone())
            .on_click(move |_, _, cx| pick(p, cx))
    }))
}

impl DisplayPanel {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if !matches!(e, AppEvent::Status) {
                cx.notify();
            }
        })];
        Self { store, bounds: Vec::new(), drag: None, wheel: None, sliding: None, params: cx.new(|_| ParamEditor::new()), page: ChannelPage::Colour, _subs: subs }
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

    /// Opens (or closes) the colour wheel for a channel or a class.
    fn toggle_wheel(&mut self, key: String, rgb: [u8; 3], cx: &mut Context<Self>) {
        if self.wheel.as_ref().is_some_and(|(k, _, _)| *k == key) {
            self.wheel = None;
            cx.notify();
            return;
        }
        let wheel = cx.new(|_| ColorWheel::new(rgb));
        let k = key.clone();
        let sub = cx.subscribe(&wheel, move |this, _, e: &WheelEvent, cx| {
            let (rgb, done) = match e {
                WheelEvent::Changing(c) => (*c, false),
                WheelEvent::Done(c) => (*c, true),
            };
            if let Some(c) = k.strip_prefix("channel:").and_then(|c| c.parse::<usize>().ok()) {
                let Some((img, mut ch)) = this.channel(c, cx) else { return };
                ch.color = rgb;
                this.store.update(cx, |s, cx| {
                    s.set_channel_live(&img, c, ch, cx);
                    if done {
                        s.commit_live(cx);
                    }
                });
            } else if let (Some(name), true) = (k.strip_prefix("class:"), done) {
                let name = name.to_string();
                this.store.update(cx, |s, cx| {
                    if let Some(class) = s.doc.class(&name).cloned() {
                        let _ = s.execute(Command::SetClass { class: ClassDoc { color: rgb, ..class } }, cx);
                    }
                });
            }
        });
        self.wheel = Some((key, wheel, sub));
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

    /// The value a slider shows: its drag while dragged, else the document's.
    fn shown(&self, key: &str, doc_value: f32) -> f32 {
        self.sliding.as_ref().filter(|(k, _)| k == key).map_or(doc_value, |(_, v)| *v)
    }

    // ------------------------------------------------------------------------
    // The editors
    // ------------------------------------------------------------------------

    fn channel_editor(&mut self, c: usize, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let Some(img) = s.doc.active().cloned() else { return div().into_any_element() };
        let Some(ch) = img.channels.get(c).cloned() else { return hint("That channel is gone.") };
        let hist = s.histograms.get(&img.id).and_then(|h| h.get(c).cloned());
        let py = s.pyramids.get(&img.id).cloned();
        let (steps, view) = chain_of(s, &img.id, c);
        let view_style = view.as_ref().and_then(|v| match s.doc.graph.node(v).map(|n| &n.kind) {
            Some(NodeKind::View { name, style }) => Some((name.clone(), style.clone())),
            _ => None,
        });
        let step_rows: Vec<(String, String, String, IconName, bool)> = steps
            .iter()
            .filter_map(|id| {
                let n = s.doc.graph.node(id)?;
                Some(match &n.kind {
                    NodeKind::Op { op, enabled } => (id.clone(), op.title().to_string(), summary(op), family_icon(op.family()), *enabled),
                    NodeKind::Script { path, .. } => (id.clone(), std::path::Path::new(path).file_name().map_or(path.clone(), |f| f.to_string_lossy().into_owned()), "script".into(), IconName::SquareTerminal, true),
                    NodeKind::Model { model, enabled, .. } => (id.clone(), model.clone(), "model".into(), IconName::BrainCircuit, *enabled),
                    _ => return None,
                })
            })
            .collect();
        let selected_steps: Vec<String> = s.sel.layers.clone();
        while self.bounds.len() <= c {
            self.bounds.push(Rc::new(Cell::new(Bounds::default())));
        }
        let range = self.range(c, cx);
        let (store, id) = (self.store.clone(), img.id.clone());
        let set = {
            let (store, id, ch) = (store.clone(), id.clone(), ch.clone());
            Rc::new(move |f: &dyn Fn(&mut ChannelDoc), cx: &mut gpui_kit::App| {
                let mut d = ch.clone();
                f(&mut d);
                let id = id.clone();
                store.update(cx, |s, cx| {
                    let _ = s.execute(Command::SetChannel { image: id, channel: c, display: d }, cx);
                });
            })
        };
        let s1 = set.clone();
        let auto = {
            let (store, id, ch, py) = (store.clone(), id.clone(), ch.clone(), py.clone());
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
        let key = format!("channel:{c}");
        // The Colour page shows the wheel (made for this channel when it is not)
        if self.page == ChannelPage::Colour && !self.wheel.as_ref().is_some_and(|(k, _, _)| *k == key) {
            self.toggle_wheel(key.clone(), ch.color, cx);
        }
        let wheel_open = self.wheel.as_ref().filter(|(k, _, _)| *k == key).map(|(_, e, _)| e.clone());
        if let Some(w) = &wheel_open {
            let rgb = ch.color;
            w.update(cx, |w, cx| w.set(rgb, cx));
        }
        let page = self.page;
        // The channel itself (shown, colour, name) above the pages
        let head = h_flex()
                    .gap(px(theme::SM))
                    .child(Checkbox::new(SharedString::from(format!("ch-visible-{c}"))).checked(ch.visible).tooltip("Shown").on_click(move |on: &bool, _, cx| {
                        let on = *on;
                        s1(&move |d: &mut ChannelDoc| d.visible = on, cx)
                    }))
                    .child(
                        div()
                            .id(SharedString::from(format!("ch-color-{c}")))
                            .cursor_pointer()
                            .rounded(px(3.))
                            .when(page == ChannelPage::Colour, |d| d.border_1().border_color(color::accent_hover()))
                            .child(swatch(ch.color, 20.))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.page = ChannelPage::Colour;
                                cx.notify();
                            })),
                    )
                    .child(div().flex_1().min_w_0().text_sm().font_weight(FontWeight::SEMIBOLD).child(ch.label.clone()));
        let me = cx.entity().downgrade();
        let tabs = page_tabs(
            "ch-page",
            vec![(ChannelPage::Colour, "Colour".into()), (ChannelPage::Levels, "Levels".into())],
            page,
            move |p, cx| {
                let _ = me.update(cx, |this, cx| {
                    this.page = p;
                    cx.notify();
                });
            },
        );
        // Colour: the wheel
        let colour = v_flex().gap(px(theme::SM)).children(wheel_open.map(|w| h_flex().w_full().justify_center().child(w)));
        // Levels: the histogram with its black and white points, gamma
        let levels = v_flex()
            .gap(px(theme::SM))
            .child(h_flex().child(div().flex_1()).child(Button::new(SharedString::from(format!("ch-auto-{c}"))).ghost().xsmall().label("Auto").tooltip("Black and white points from the 0.1–99.8 % percentiles").on_click(auto)))
            .child(self.histogram(c, hist, &ch, range, cx))
            .child(
                h_flex()
                    .text_xs()
                    .font_family("monospace")
                    .text_color(color::text_muted())
                    .child(div().child(format!("black {}", trim(ch.min))))
                    .child(div().flex_1())
                    .child(div().child(format!("white {}", trim(ch.max)))),
            )
            .child(row("Gamma", {
                let (store, id, ch) = (store.clone(), id.clone(), ch.clone());
                slider(format!("ch-gamma-{c}"), ch.gamma, (0.2, 3.0), |v| format!("{v:.2}"), move |v, done, _, cx| {
                    let d = ChannelDoc { gamma: v, ..ch.clone() };
                    let id = id.clone();
                    store.update(cx, |s, cx| {
                        s.set_channel_live(&id, c, d, cx);
                        if done {
                            s.commit_live(cx);
                        }
                    });
                })
            }));
        // Its processing (dsp-kitchen's card)
        let mut processing = v_flex().gap(px(theme::XS)).child(
            h_flex()
                .gap(px(theme::SM))
                .child(div().flex_1().text_sm().font_weight(FontWeight::SEMIBOLD).child("Processing"))
                .children(view_style.clone().zip(view.clone()).map(|((name, style), vid)| {
                    let (a, b) = (store.clone(), store.clone());
                    let (n1, n2, s1, s2, v1, v2) = (name.clone(), name, style.clone(), style.clone(), vid.clone(), vid);
                    h_flex()
                        .gap(px(2.))
                        .p(px(2.))
                        .rounded(px(6.))
                        .bg(color::control())
                        .child(Button::new("proc-processed").ghost().xsmall().label("Processed").selected(style.visible).on_click(move |_, _, cx| {
                            let style = ViewStyle { visible: true, ..s1.clone() };
                            let (id, name) = (v1.clone(), n1.clone());
                            a.update(cx, |s, cx| {
                                let _ = s.execute(Command::SetView { id, name, style }, cx);
                            });
                        }))
                        .child(Button::new("proc-raw").ghost().xsmall().label("Raw").selected(!style.visible).on_click(move |_, _, cx| {
                            let style = ViewStyle { visible: false, ..s2.clone() };
                            let (id, name) = (v2.clone(), n2.clone());
                            b.update(cx, |s, cx| {
                                let _ = s.execute(Command::SetView { id, name, style }, cx);
                            });
                        }))
                })),
        );
        if step_rows.is_empty() {
            processing = processing.child(div().text_xs().text_color(color::text_muted()).child("Raw: no step yet."));
        }
        for (i, (sid, title, sub, icon, on)) in step_rows.into_iter().enumerate() {
            let key = format!("node:{sid}");
            let selected = selected_steps.contains(&key);
            let (st_sel, st_on, st_del) = (store.clone(), store.clone(), store.clone());
            let (k2, id_on, id_del) = (key.clone(), sid.clone(), sid.clone());
            processing = processing.child(
                h_flex()
                    .id(SharedString::from(format!("proc-step-{i}")))
                    .h(px(28.))
                    .px(px(theme::XS))
                    .gap(px(theme::SM))
                    .rounded(px(4.))
                    .cursor_pointer()
                    .when(selected, |d| d.bg(color::accent().opacity(0.3)))
                    .when(!selected, |d| d.hover(|d| d.bg(color::control())))
                    .child(Icon::new(icon).small().text_color(if on { color::text() } else { color::text_muted() }))
                    .child(div().flex_1().min_w_0().truncate().text_sm().when(!on, |d| d.text_color(color::text_muted())).child(title))
                    .child(div().flex_none().max_w(px(110.)).truncate().text_xs().text_color(color::text_muted()).child(sub))
                    .child(Button::new(SharedString::from(format!("proc-on-{i}"))).ghost().xsmall().icon(Icon::new(if on { IconName::Eye } else { IconName::EyeOff })).tooltip(if on { "Turn off (passes its input through)" } else { "Turn on" }).on_click(move |_, _, cx| {
                        cx.stop_propagation();
                        let id = id_on.clone();
                        st_on.update(cx, |s, cx| {
                            let _ = s.execute(Command::EnableNode { id, enabled: !on }, cx);
                        });
                    }))
                    .child(Button::new(SharedString::from(format!("proc-del-{i}"))).ghost().xsmall().icon(Icon::new(IconName::Trash)).tooltip("Remove the step").on_click(move |_, _, cx| {
                        cx.stop_propagation();
                        let id = id_del.clone();
                        st_del.update(cx, |s, cx| s.remove_step(id, cx));
                    }))
                    .on_click(move |_, _, cx| st_sel.update(cx, |s, cx| s.select_layer(k2.clone(), false, cx))),
            );
        }
        let (st_add, img_add, view_add) = (store, img.id.clone(), view);
        processing = processing.child(Button::new("proc-add").outline().small().w_full().icon(Icon::new(IconName::Plus)).label("Add step").dropdown_menu(move |menu: PopupMenu, _, _| {
            let mut m = menu.scrollable(true).max_h(px(480.));
            for cat in operations() {
                m = m.label(cat.name);
                for it in cat.items {
                    let Template::Op(op) = it.template else { continue };
                    let (st, image, view) = (st_add.clone(), img_add.clone(), view_add.clone());
                    m = m.item(PopupMenuItem::new(it.label).icon(Icon::new(cat.icon)).on_click(move |_, _, cx| {
                        let (op, image, view) = (op.clone(), image.clone(), view.clone());
                        st.update(cx, |s, cx| {
                            let title = op.title();
                            if s.execute(Command::ApplyOp { op, view, image, channel: c }, cx).is_ok() {
                                s.set_status(format!("{title} added to this channel's processing"), cx);
                            }
                        });
                    }));
                }
            }
            m
        }));
        let body = match page {
            ChannelPage::Colour => colour.into_any_element(),
            ChannelPage::Levels => levels.into_any_element(),
        };
        // Processing is always there (first-class, not a tab), under the display pages
        v_flex()
            .gap(px(theme::MD))
            .child(head)
            .child(tabs)
            .child(body)
            .child(div().h(px(1.)).w_full().bg(color::border()))
            .child(processing)
            .into_any_element()
    }

    fn class_editor(&mut self, name: &str, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let Some(class) = s.doc.class(name).cloned() else { return hint("That class is gone.") };
        let count = s.doc.annotations.iter().filter(|a| a.class == class.name).count();
        let drawing = s.tool.class == class.name;
        let fill_on = s.tool.fill_annotations;
        let store = self.store.clone();
        let key = format!("class:{}", class.name);
        let wheel_open = self.wheel.as_ref().filter(|(k, _, _)| *k == key).map(|(_, e, _)| e.clone());
        let (wk, rgb) = (key.clone(), class.color);
        let st_pre = store.clone();
        let cls_pre = class.clone();
        let presets = h_flex()
            .w_full()
            .gap(px(theme::XS))
            .child(preset_row("class-preset", &CLASS_COLORS, class.color, move |rgb, cx| {
                let class = ClassDoc { color: rgb, ..cls_pre.clone() };
                st_pre.update(cx, |s, cx| {
                    let _ = s.execute(Command::SetClass { class }, cx);
                });
            }))
            .child(Button::new("class-custom").ghost().xsmall().icon(Icon::new(IconName::Palette)).tooltip("Any colour").selected(wheel_open.is_some()).on_click(cx.listener(move |this, _, _, cx| this.toggle_wheel(wk.clone(), rgb, cx))));
        let opacity_key = format!("{key}/opacity");
        let shown = self.shown(&opacity_key, class.opacity);
        let (st_op, cls_op, me) = (store.clone(), class.clone(), cx.entity().downgrade());
        let (st_vis, cls_vis, st_draw, name_draw, st_fill) = (store.clone(), class.clone(), store.clone(), class.name.clone(), store);
        v_flex()
            .gap(px(theme::MD))
            .child(
                h_flex()
                    .gap(px(theme::SM))
                    .child(Checkbox::new("class-visible").checked(class.visible).tooltip("Shown").on_click(move |on: &bool, _, cx| {
                        let class = ClassDoc { visible: *on, ..cls_vis.clone() };
                        st_vis.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetClass { class }, cx);
                        });
                    }))
                    .child(swatch(class.color, 20.))
                    .child(div().flex_1().min_w_0().text_sm().font_weight(FontWeight::SEMIBOLD).child(class.name.clone()))
                    .child(div().text_xs().text_color(color::text_muted()).child(format!("{count} shape(s)"))),
            )
            .child(presets)
            .children(wheel_open.map(|w| h_flex().w_full().justify_center().child(w)))
            .child(row(
                "Opacity",
                slider(opacity_key.clone(), shown, (0.0, 1.0), |v| format!("{:.0} %", v * 100.0), move |v, done, _, cx| {
                    let ok = opacity_key.clone();
                    let _ = me.update(cx, |p, cx| {
                        p.sliding = (!done).then(|| (ok, v));
                        cx.notify();
                    });
                    if done {
                        let class = ClassDoc { opacity: v, ..cls_op.clone() };
                        st_op.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetClass { class }, cx);
                        });
                    }
                }),
            ))
            .child(
                h_flex()
                    .gap(px(theme::SM))
                    .child(Checkbox::new("class-fill").checked(fill_on).label("Fill regions").on_click(move |_, _, cx| st_fill.update(cx, |s, cx| s.update_tool(|t| t.fill_annotations = !t.fill_annotations, cx))))
                    .child(div().flex_1())
                    .child(Button::new("class-draw").small().when(drawing, |b| b.ghost()).when(!drawing, |b| b.outline()).icon(Icon::new(IconName::Pencil)).label(if drawing { "Drawing with it" } else { "Draw with it" }).disabled(drawing).on_click(move |_, _, cx| {
                        let n = name_draw.clone();
                        st_draw.update(cx, |s, cx| s.update_tool(|t| t.class = n, cx));
                    })),
            )
            .into_any_element()
    }

    fn view_editor(&mut self, id: &str, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let Some(NodeKind::View { name, style }) = s.doc.graph.node(id).map(|n| n.kind.clone()) else { return hint("That result is gone.") };
        let port = s.doc.graph.input(id, 0).and_then(|i| s.doc.graph.node(i)).and_then(|n| n.kind.output());
        let store = self.store.clone();
        let set = {
            let (store, id, name, style) = (store.clone(), id.to_string(), name.clone(), style.clone());
            Rc::new(move |f: &dyn Fn(&mut ViewStyle), cx: &mut gpui_kit::App| {
                let mut st = style.clone();
                f(&mut st);
                let (id, name) = (id.clone(), name.clone());
                store.update(cx, |s, cx| {
                    let _ = s.execute(Command::SetView { id, name, style: st }, cx);
                });
            })
        };
        let colours: AnyElement = match port {
            Some(PortType::Mask) => {
                let mut row_ = h_flex().flex_wrap().gap(px(theme::XS));
                for (cname, rgb) in CLASS_COLORS {
                    let set = set.clone();
                    row_ = row_.child(
                        div()
                            .id(SharedString::from(format!("view-color-{cname}")))
                            .cursor_pointer()
                            .rounded(px(4.))
                            .p(px(2.))
                            .when(style.color == rgb, |d| d.border_1().border_color(color::accent_hover()))
                            .child(swatch(rgb, 18.))
                            .on_click(move |_, _, cx| set(&move |st: &mut ViewStyle| st.color = rgb, cx)),
                    );
                }
                row_.into_any_element()
            }
            Some(PortType::Image) => {
                let mut chips = h_flex().flex_wrap().gap(px(theme::XS));
                for m in COLORMAPS {
                    let set = set.clone();
                    let lut = crate::data::results::lut(m);
                    // A strip of the map, then its name
                    let strip = h_flex().w(px(48.)).h(px(10.)).rounded(px(2.)).overflow_hidden().children((0..12).map(move |i| {
                        let [r, g, b] = lut.sample_rgb(i as f32 / 11.0);
                        div().flex_1().h_full().bg(theme::rgb8([r, g, b]))
                    }));
                    chips = chips.child(
                        v_flex()
                            .id(SharedString::from(format!("view-lut-{m}")))
                            .gap(px(2.))
                            .p(px(4.))
                            .rounded(px(4.))
                            .cursor_pointer()
                            .when(style.colormap == m, |d| d.bg(color::accent().opacity(0.3)))
                            .when(style.colormap != m, |d| d.hover(|d| d.bg(color::control())))
                            .child(strip)
                            .child(div().text_xs().child(m))
                            .on_click(move |_, _, cx| set(&move |st: &mut ViewStyle| st.colormap = m.to_string(), cx)),
                    );
                }
                chips.into_any_element()
            }
            _ => div().text_xs().text_color(color::text_muted()).child("Labels: each object in its own colour.").into_any_element(),
        };
        let key = format!("view:{id}/opacity");
        let shown = self.shown(&key, style.opacity);
        let me = cx.entity().downgrade();
        let set_op = set.clone();
        let set_vis = set;
        v_flex()
            .gap(px(theme::MD))
            .child(
                h_flex()
                    .gap(px(theme::SM))
                    .child(Checkbox::new("view-visible").checked(style.visible).tooltip("Drawn over the image").on_click(move |on: &bool, _, cx| {
                        let on = *on;
                        set_vis(&move |st: &mut ViewStyle| st.visible = on, cx)
                    }))
                    .child(div().flex_1().min_w_0().text_sm().font_weight(FontWeight::SEMIBOLD).child(name))
                    .child(div().text_xs().text_color(color::text_muted()).child(port.map_or("—", PortType::name))),
            )
            .child(colours)
            .child(row(
                "Opacity",
                slider(key.clone(), shown, (0.0, 1.0), |v| format!("{:.0} %", v * 100.0), move |v, done, _, cx| {
                    let k = key.clone();
                    let _ = me.update(cx, |p, cx| {
                        p.sliding = (!done).then(|| (k, v));
                        cx.notify();
                    });
                    if done {
                        set_op(&move |st: &mut ViewStyle| st.opacity = v, cx);
                    }
                }),
            ))
            .into_any_element()
    }

    fn step_editor(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let Some(node) = s.doc.graph.node(id).cloned() else { return hint("That step is gone.") };
        let store = self.store.clone();
        match &node.kind {
            NodeKind::Op { op, enabled } => {
                let commit: Commit = {
                    let (store, id) = (store.clone(), id.to_string());
                    Rc::new(move |op, cx| store.update(cx, |s, cx| s.execute(Command::SetOp { id: id.clone(), op }, cx)).map(|_| ()).map_err(|e| e.to_string()))
                };
                self.params.update(cx, |p, cx| p.set_target(id, op, commit, window, cx));
                let (st, id2, on) = (store, id.to_string(), *enabled);
                v_flex()
                    .gap(px(theme::MD))
                    .child(h_flex().gap(px(theme::SM)).child(Icon::new(family_icon(op.family())).small()).child(div().flex_1().text_sm().font_weight(FontWeight::SEMIBOLD).child(op.title())))
                    .child(Checkbox::new("step-on").checked(on).label("On (off: passes its input through)").on_click(move |v: &bool, _, cx| {
                        let (id, enabled) = (id2.clone(), *v);
                        st.update(cx, |s, cx| {
                            let _ = s.execute(Command::EnableNode { id, enabled }, cx);
                        });
                    }))
                    .child(self.params.clone())
                    .into_any_element()
            }
            _ => {
                let st = store;
                v_flex()
                    .gap(px(theme::SM))
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(id.to_string()))
                    .child(Button::new("step-open-pipeline").outline().small().icon(Icon::new(IconName::Workflow)).label("Edit in Pipeline").on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_workspace(crate::navigation::route::Workspace::Pipeline, cx))))
                    .into_any_element()
            }
        }
    }
}

fn hint(text: &'static str) -> AnyElement {
    div().text_xs().text_color(color::text_muted()).child(text).into_any_element()
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
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let s = self.store.read(cx);
        if s.doc.active().is_none() {
            return div().into_any_element();
        }
        let t = target(s);
        // The wheel belongs to the layer it was opened for
        if let Some((k, _, _)) = &self.wheel {
            let still = match &t {
                Target::Channel(c) => *k == format!("channel:{c}"),
                Target::Class(n) => *k == format!("class:{n}"),
                _ => false,
            };
            if !still {
                self.wheel = None;
            }
        }
        let body = match t {
            Target::Channel(c) => self.channel_editor(c, cx),
            Target::Class(n) => self.class_editor(&n, cx),
            Target::View(id) => self.view_editor(&id, cx),
            Target::Step(id) => self.step_editor(&id, window, cx),
            Target::Nothing => hint("Select a channel, a shape class, a result or a step below to edit it here."),
        };
        div()
            .id("layer-editor")
            .w_full()
            .p(px(theme::MD))
            .on_mouse_move(cx.listener(Self::on_move))
            .on_mouse_up(MouseButton::Left, cx.listener(|this, _: &MouseUpEvent, _, cx| this.finish(cx)))
            .child(body)
            .into_any_element()
    }
}

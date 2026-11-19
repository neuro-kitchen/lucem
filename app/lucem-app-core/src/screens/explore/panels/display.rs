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
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    canvas, div, fill, point, px, size, AnyElement, AppContext as _, Bounds, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, ParentElement as _, PathBuilder, Pixels, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};
use lucem_pipeline::PortType;

use crate::data::results::COLORMAPS;
use crate::data::store::{AppEvent, ChannelHistogram, Store};
use crate::domain::annotation::ClassDoc;
use crate::domain::command::Command;
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::domain::project::ChannelDoc;
use crate::screens::pipeline::catalog::{family_icon, operations, summary, Template};
use crate::ui::color_wheel::{ColorWheel, WheelEvent};
use crate::ui::param_editor::{Commit, EditOps as _, ParamEditor};
use crate::ui::theme::{self, color};
use crate::ui::widgets::{row, scrub_label, slider, swatch};

/// Height of a histogram (px).
const HIST_H: f32 = 84.;

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
    /// Whether the Levels histogram uses a logarithmic count scale (`true`) or linear (`false`).
    hist_log: bool,
    /// User dismissed the goal-first starter cards on the empty Layers editor.
    starters_dismissed: bool,
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
        Self { store, bounds: Vec::new(), drag: None, wheel: None, sliding: None, params: cx.new(|_| ParamEditor::new()), page: ChannelPage::Colour, hist_log: true, starters_dismissed: false, _subs: subs }
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
        let (min, max, gamma) = (ch.min, ch.max, ch.gamma.max(0.05));
        let log_scale = self.hist_log;
        let plot = canvas(
            move |b, _, _| cell.set(b),
            move |b, _, window, _| {
                let o = b.origin;
                let (w, hh) = (b.size.width.as_f32(), b.size.height.as_f32());
                let rail_h = 10.0;
                let plot_h = (hh - rail_h).max(16.0);
                window.paint_quad(fill(b, color::canvas()));
                // Subtle horizontal reference grid lines (25%, 50%, 75%)
                for frac in [0.25, 0.5, 0.75] {
                    let gy = o.y + px(plot_h * frac);
                    window.paint_quad(fill(Bounds { origin: point(o.x, gy), size: size(px(w), px(1.)) }, gpui_kit::white().opacity(0.06)));
                }
                let span = (hi - lo).max(1e-12);
                let xf = |v: f32| ((v - lo) / span * w).clamp(0.0, w);
                let (mx0, mx1) = (xf(min), xf(max));
                if let Some(h) = &h {
                    let val = |count: u32| if log_scale { (count as f32 + 1.0).ln() } else { count as f32 };
                    let peak = h.bins.iter().map(|&cnt| val(cnt)).fold(0.0f32, f32::max).max(1e-6);
                    let n = h.bins.len() as f32;
                    let hspan = (h.hi - h.lo).max(1e-12);
                    for (i, &count) in h.bins.iter().enumerate() {
                        if count == 0 {
                            continue;
                        }
                        let v0 = h.lo + i as f32 / n * hspan;
                        let v1 = h.lo + (i as f32 + 1.0) / n * hspan;
                        let (x0, x1) = (xf(v0), xf(v1).max(xf(v0) + 1.0));
                        let bh = (val(count) / peak * (plot_h - 4.0)).max(1.0);
                        let inside = x1 >= mx0 && x0 <= mx1;
                        let alpha = if inside { 0.85 } else { 0.28 };
                        window.paint_quad(fill(Bounds { origin: point(o.x + px(x0), o.y + px(plot_h - bh)), size: size(px(x1 - x0), px(bh)) }, col.opacity(alpha)));
                    }
                }
                // Outside the window: subtle dark mask
                window.paint_quad(fill(Bounds { origin: o, size: size(px(mx0), px(plot_h)) }, gpui_kit::black().opacity(0.32)));
                window.paint_quad(fill(Bounds { origin: point(o.x + px(mx1), o.y), size: size(px((w - mx1).max(0.0)), px(plot_h)) }, gpui_kit::black().opacity(0.32)));
                // Gamma transfer curve across [mx0..mx1]
                let mut curve = PathBuilder::stroke(px(1.5));
                let steps = 24;
                for s in 0..=steps {
                    let t = s as f32 / steps as f32;
                    let x = mx0 + t * (mx1 - mx0);
                    let y = (1.0 - t.powf(1.0 / gamma)) * (plot_h - 4.0) + 2.0;
                    let pt = point(o.x + px(x), o.y + px(y));
                    if s == 0 {
                        curve.move_to(pt);
                    } else {
                        curve.line_to(pt);
                    }
                }
                if let Ok(path) = curve.build() {
                    window.paint_path(path, gpui_kit::white().opacity(0.85));
                }
                // Vertical window bounds + bottom rail
                window.paint_quad(fill(Bounds { origin: point(o.x, o.y + px(plot_h)), size: size(px(w), px(rail_h)) }, color::control()));
                for x in [mx0, mx1] {
                    window.paint_quad(fill(Bounds { origin: point(o.x + px(x - 0.5), o.y), size: size(px(1.5), px(plot_h)) }, gpui_kit::white().opacity(0.8)));
                }
                // Draggable black & white handle triangles on the bottom rail
                for (x, is_white) in [(mx0, false), (mx1, true)] {
                    let cx_p = x.clamp(5.0, (w - 5.0).max(5.0));
                    let tip = point(o.x + px(cx_p), o.y + px(plot_h + 1.0));
                    let bl = point(o.x + px(cx_p - 5.0), o.y + px(hh - 1.0));
                    let br = point(o.x + px(cx_p + 5.0), o.y + px(hh - 1.0));
                    let mut tri = PathBuilder::fill();
                    tri.add_polygon(&[tip, br, bl], true);
                    if let Ok(p) = tri.build() {
                        window.paint_path(p, if is_white { gpui_kit::white() } else { gpui_kit::black() });
                    }
                    let mut outline = PathBuilder::stroke(px(1.0));
                    outline.add_polygon(&[tip, br, bl], true);
                    if let Ok(p) = outline.build() {
                        window.paint_path(p, if is_white { gpui_kit::black().opacity(0.75) } else { gpui_kit::white().opacity(0.9) });
                    }
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
        let hex = format!("#{:02X}{:02X}{:02X}", ch.color[0], ch.color[1], ch.color[2]);
        // Clean channel header: colour dot + name + muted hex (visibility lives in the Layers tree)
        let head = h_flex()
            .gap(px(theme::SM))
            .items_center()
            .child(div().flex_none().size(px(10.)).rounded_full().bg(theme::rgb8(ch.color)))
            .child(div().flex_1().min_w_0().truncate().text_sm().font_weight(FontWeight::SEMIBOLD).child(ch.label.clone()))
            .child(div().flex_none().text_xs().font_family("monospace").text_color(color::text_muted()).child(hex));
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
        // Levels: Log/Lin toggle + Auto, upgraded histogram with gamma curve & handles, scrubbable black/white, gamma slider
        let hist_log = self.hist_log;
        let span = (range.1 - range.0).max(1e-6);
        let (st_lo, id_lo, ch_lo) = (store.clone(), id.clone(), ch.clone());
        let (st_hi, id_hi, ch_hi) = (store.clone(), id.clone(), ch.clone());
        let levels = v_flex()
            .gap(px(theme::SM))
            .child(
                h_flex()
                    .gap(px(theme::XS))
                    .items_center()
                    .child(
                        Button::new(SharedString::from(format!("ch-hist-scale-{c}")))
                            .ghost()
                            .xsmall()
                            .label(if hist_log { "Log" } else { "Lin" })
                            .tooltip("Toggle logarithmic / linear histogram count scale")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.hist_log = !this.hist_log;
                                cx.notify();
                            })),
                    )
                    .child(div().flex_1())
                    .child(Button::new(SharedString::from(format!("ch-auto-{c}"))).ghost().xsmall().label("Auto").tooltip("Black and white points from the 0.1–99.8 % percentiles").on_click(auto)),
            )
            .child(self.histogram(c, hist, &ch, range, cx))
            .child(
                h_flex()
                    .items_center()
                    .text_xs()
                    .text_color(color::text_muted())
                    .child(div().pr(px(2.)).child("Black"))
                    .child(scrub_label(format!("ch-black-{c}"), trim(ch.min), move |d, _, cx| {
                        let mut doc = ch_lo.clone();
                        doc.min = (doc.min + d * span * 0.01).clamp(range.0, doc.max - 1e-6);
                        let id = id_lo.clone();
                        st_lo.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetChannel { image: id, channel: c, display: doc }, cx);
                        });
                    }))
                    .child(div().flex_1())
                    .child(div().pr(px(2.)).child("White"))
                    .child(scrub_label(format!("ch-white-{c}"), trim(ch.max), move |d, _, cx| {
                        let mut doc = ch_hi.clone();
                        doc.max = (doc.max + d * span * 0.01).clamp(doc.min + 1e-6, range.1);
                        let id = id_hi.clone();
                        st_hi.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetChannel { image: id, channel: c, display: doc }, cx);
                        });
                    })),
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
        // M3 Expressive Split Button: left half opens the non-blocking Visual Strip in 1 click,
        // right chevron opens the categorized op menu for power users.
        let (st_add, img_add, view_add) = (store.clone(), img.id.clone(), view.clone());
        let menu_chevron = Button::new("proc-add").ghost().xsmall().icon(Icon::new(IconName::ChevronDown)).tooltip("Browse all operations by category").dropdown_menu(move |menu: PopupMenu, window, cx| {
            let mut m = menu;
            for cat in operations() {
                let (st, image, view) = (st_add.clone(), img_add.clone(), view_add.clone());
                let items: Vec<_> = cat.items.into_iter().filter_map(|it| match it.template {
                    Template::Op(op) => Some((it.label, op)),
                    _ => None,
                }).collect();
                if items.is_empty() {
                    continue;
                }
                let icon = cat.icon;
                m = m.submenu(cat.name, window, cx, move |mut sub, _, _| {
                    for (label, op) in &items {
                        let (st, image, view, op_click, op_hov) = (st.clone(), image.clone(), view.clone(), op.clone(), op.clone());
                        let st_menu = st.clone();
                        sub = sub.item(crate::ui::op_gallery::op_menu_item(
                            format!("disp-op-{}", op_hov.tag()),
                            label.clone(),
                            icon,
                            op_hov,
                            st_menu,
                            move |_, cx| {
                                let (op, image, view) = (op_click.clone(), image.clone(), view.clone());
                                st.update(cx, |s, cx| {
                                    let title = op.title();
                                    if s.execute(Command::ApplyOp { op, view, image, channel: c }, cx).is_ok() {
                                        s.set_status(format!("{title} added to this channel's processing"), cx);
                                    }
                                });
                            },
                        ));
                    }
                    sub
                });
            }
            m
        });
        let st_gal = store.clone();
        let split_add = h_flex()
            .items_center()
            .rounded(px(999.))
            .border_1()
            .border_color(color::border())
            .bg(color::control())
            .overflow_hidden()
            .child(
                Button::new("proc-gallery")
                    .ghost()
                    .xsmall()
                    .icon(Icon::new(IconName::Sparkles))
                    .label("Visual strip")
                    .tooltip("Open the live non-blocking Visual Effect Strip over the canvas")
                    .on_click(move |_, _, cx| st_gal.update(cx, |_, cx| cx.emit(crate::data::store::AppEvent::OpenOpGallery))),
            )
            .child(div().w(px(1.)).h(px(14.)).bg(color::border()))
            .child(menu_chevron);
        let proc_bar = h_flex()
            .gap(px(theme::XS))
            .child(split_add)
            .child(div().flex_1())
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
            }));
        // M3 Expressive "What's next?" tonal card with hover Before <-> After live preview
        let last_op = {
            let s = self.store.read(cx);
            steps.last().and_then(|sid| match s.doc.graph.node(sid).map(|n| &n.kind) {
                Some(NodeKind::Op { op, .. }) => Some(op.clone()),
                _ => None,
            })
        };
        let prompt = match &last_op {
            Some(op) => op.next_step_prompt(),
            None => "Start enhancing this channel?",
        };
        let suggested_tags: &[&str] = match &last_op {
            Some(op) => op.next_steps(),
            None => &["rolling_ball", "gaussian_blur", "li_threshold"],
        };
        let next_chips: Vec<AnyElement> = suggested_tags
            .iter()
            .filter_map(|tag| lucem_pipeline::Op::by_tag(tag))
            .take(3)
            .map(|op| {
                let (st, st_hov, img_id, vw) = (store.clone(), store.clone(), img.id.clone(), view.clone());
                let title = op.title();
                let op_hov = op.clone();
                let op_click = op.clone();
                h_flex()
                    .id(SharedString::from(format!("next-chip-{}", op.tag())))
                    .px(px(10.))
                    .h(px(24.))
                    .gap(px(4.))
                    .items_center()
                    .rounded(px(999.))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::panel())
                    .hover(|d| d.border_color(color::accent_mark()).bg(color::raised()))
                    .cursor_pointer()
                    .text_xs()
                    .text_color(color::text())
                    .child(Icon::new(IconName::Plus).xsmall().text_color(color::accent_mark()))
                    .child(title)
                    .on_hover(move |enter, _, cx| {
                        let op = (*enter).then(|| op_hov.clone());
                        st_hov.update(cx, |s, cx| s.set_hovered_op(op, cx));
                    })
                    .on_click(move |_, _, cx| {
                        let (op, image, view) = (op_click.clone(), img_id.clone(), vw.clone());
                        st.update(cx, |s, cx| {
                            s.set_hovered_op(None, cx);
                            if s.execute(Command::ApplyOp { op, view, image, channel: c }, cx).is_ok() {
                                s.set_status(format!("{title} added"), cx);
                            }
                        });
                    })
                    .into_any_element()
            })
            .collect();
        let next_row = (!next_chips.is_empty()).then(|| {
            v_flex()
                .p(px(theme::SM))
                .gap(px(6.))
                .rounded(px(12.))
                .bg(color::control())
                .child(
                    h_flex()
                        .gap(px(6.))
                        .items_center()
                        .child(Icon::new(IconName::Sparkles).xsmall().text_color(color::accent_mark()))
                        .child(div().text_xs().font_weight(FontWeight::MEDIUM).text_color(color::text()).child(prompt)),
                )
                .child(h_flex().flex_wrap().gap(px(6.)).children(next_chips))
        });
        let _ = (&step_rows, &selected_steps);
        let body = match page {
            ChannelPage::Colour => colour.into_any_element(),
            ChannelPage::Levels => levels.into_any_element(),
        };
        v_flex()
            .gap(px(theme::SM))
            .child(head)
            .child(tabs)
            .child(body)
            .child(proc_bar)
            .children(next_row)
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
                self.params.update(cx, |p, cx| p.set_op(id, op, commit, window, cx));
                let (st, id2, on) = (store, id.to_string(), *enabled);
                let docs_url = op.docs_url();
                let desc = op.description();
                v_flex()
                    .gap(px(theme::SM))
                    .child(
                        h_flex()
                            .gap(px(theme::SM))
                            .items_center()
                            .child(Icon::new(family_icon(op.family())).small())
                            .child(div().flex_1().min_w_0().truncate().text_sm().font_weight(FontWeight::SEMIBOLD).child(op.title()))
                            .child(
                                Button::new("step-docs")
                                    .ghost()
                                    .xsmall()
                                    .icon(Icon::new(IconName::ExternalLink))
                                    .label("Docs")
                                    .tooltip(format!("Open GPU kernel source ({})", op.source_rel_path()))
                                    .on_click(move |_, _, cx| cx.open_url(&docs_url)),
                            ),
                    )
                    .child(div().text_xs().text_color(color::text_muted()).child(desc))
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
    /// Goal-first contextual quick actions when an image is open and no layer is selected yet.
    fn starter_cards(&self, cx: &mut Context<Self>) -> AnyElement {
        if self.starters_dismissed {
            return hint("Select a channel, shape class, result, or step below to edit it here.");
        }
        let s = self.store.read(cx);
        let Some(img) = s.doc.active().cloned() else {
            return hint("Open an image to start.");
        };
        let is_volume = s.active_pyramid().is_some_and(|p| p.is_volume());
        let store = self.store.clone();
        let img_id = img.id.clone();
        let apply = |id: &'static str, icon: IconName, title: &'static str, sub: &'static str, op: lucem_pipeline::Op| {
            let (st, img_id) = (store.clone(), img_id.clone());
            Button::new(id)
                .ghost()
                .small()
                .w_full()
                .justify_start()
                .icon(Icon::new(icon))
                .label(title)
                .tooltip(sub)
                .on_click(move |_, _, cx| {
                    let (op, image) = (op.clone(), img_id.clone());
                    st.update(cx, |s, cx| {
                        let (_, view) = s.doc.graph.channel_chain(&image, 0);
                        let name = op.title();
                        if s.execute(Command::ApplyOp { op, view, image, channel: 0 }, cx).is_ok() {
                            s.select_layer("channel:0".into(), false, cx);
                            s.set_status(format!("{name} applied to channel 0"), cx);
                        }
                    });
                })
                .into_any_element()
        };
        let mut actions: Vec<AnyElement> = Vec::new();
        if is_volume {
            let st_vol = store.clone();
            actions.push(
                Button::new("starter-3d")
                    .ghost()
                    .small()
                    .w_full()
                    .justify_start()
                    .icon(Icon::new(IconName::Box))
                    .label("Open 3D Volume View")
                    .tooltip("Inspect this 3D stack with clipping planes and depth-cued MIP")
                    .on_click(move |_, _, cx| {
                        st_vol.update(cx, |s, cx| s.request_view(crate::data::store::ViewRequest::Add(crate::data::store::ViewKind::Volume), cx));
                    })
                    .into_any_element(),
            );
            actions.push(apply("starter-frangi", IconName::SlidersHorizontal, "Enhance vessels / neurites", "Frangi multi-scale ridge filter on channel 0", lucem_pipeline::Op::Frangi { sigma_min: 1.0, sigma_max: 5.0, black_ridges: false }));
        } else {
            actions.push(apply("starter-bg", IconName::Sparkles, "Subtract background", "Rolling ball illumination correction on channel 0", lucem_pipeline::Op::RollingBall { radius: 25.0 }));
            actions.push(apply("starter-li", IconName::Contrast, "Auto-threshold nuclei", "Li minimum cross-entropy segmentation on channel 0", lucem_pipeline::Op::LiThreshold { bins: 256 }));
        }
        actions.push(apply("starter-blur", IconName::SlidersHorizontal, "Denoise & smooth", "Gaussian blur (sigma 2) on channel 0", lucem_pipeline::Op::GaussianBlur { sigma: 2.0 }));
        v_flex()
            .gap(px(theme::XS))
            .child(
                h_flex()
                    .items_center()
                    .child(div().flex_1().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child("QUICK START"))
                    .child(
                        Button::new("starter-dismiss")
                            .ghost()
                            .xsmall()
                            .icon(Icon::new(IconName::Close))
                            .tooltip("Dismiss quick start")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.starters_dismissed = true;
                                cx.notify();
                            })),
                    ),
            )
            .children(actions)
            .child(div().pt(px(2.)).text_xs().text_color(color::text_muted()).child("Or pick a channel below, or press F3 to search 52 GPU ops."))
            .into_any_element()
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
            Target::Nothing => self.starter_cards(cx),
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

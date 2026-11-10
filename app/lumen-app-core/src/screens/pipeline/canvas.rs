//! The Pipeline canvas (dsp-kitchen's, with typed multi-input ports): the project's graph on an
//! infinite plane, navigated as in Figma — the wheel (or a two-finger swipe) pans, Ctrl + wheel
//! zooms at the pointer, Space + drag or the middle button drags the plane; a drag on empty space
//! selects with a rectangle.
//!
//! Nodes are cards with one handle per input on the left (coloured by type: image, mask,
//! labels, table) and their output on the right. Dragging from an output onto an input connects
//! them when the types fit (onto a node body: its first free input that fits); dragging an input
//! off its edge disconnects it. Delete removes a node and reconnects around it where the types
//! allow (Shift+Delete removes it alone). Selecting one node highlights its path. Each card shows
//! its result on the whole image (Blender's node previews), from the whole-image analysis.

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::{h_flex, v_flex, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    canvas, div, fill, img, point, px, size, App, ObjectFit, StyledImage as _, Bounds, Context, Entity, EventEmitter, FocusHandle, FontWeight, Hsla, InteractiveElement as _, IntoElement, KeyDownEvent, KeyUpEvent,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, PathBuilder, Pixels, Render, ScrollEvent, SharedString, Styled as _, Subscription, Window,
};

use crate::data::store::{AppEvent, Store};
use crate::domain::command::{Command, NodeMove};
use crate::domain::graph::{GraphDoc, NodeKind, COLUMN};
use crate::navigation::route::Workspace;
use crate::ui::theme::color;
use crate::ui::widgets::icon_button;

use super::catalog::{describe, kind_color, port_color, Template};

/// An edge as drawn: from, to, colour, whether it is selected, whether it is live.
type EdgeLine = ((f32, f32), (f32, f32), Hsla, bool, bool);

/// A node's size in canvas units.
pub const NODE_W: f32 = 216.;
pub const NODE_H: f32 = 184.;
/// The title area of a card (the preview fills the rest); handles sit in it.
const HEADER_H: f32 = 56.;
const HANDLE_R: f32 = 5.;
const HANDLE_GRAB: f32 = 11.;
const GRID: f32 = 24.;
const SNAP: f32 = 8.;
const MIN_ZOOM: f32 = 0.2;
const MAX_ZOOM: f32 = 2.5;
const DRAG_SLOP: f32 = 3.;
const MAP_W: f32 = 200.;
const MAP_H: f32 = 128.;

/// An item dragged from the library.
#[derive(Debug, Clone)]
pub struct CatalogDrag {
    pub label: SharedString,
    pub template: Template,
}

#[derive(Debug, Clone)]
enum Drag {
    Pan { mouse: (f32, f32), pan: (f32, f32) },
    Nodes { mouse: (f32, f32), origins: Vec<(String, f32, f32)>, moved: bool },
    /// A new edge from `from`'s output, its loose end at the pointer (canvas-local px).
    Connect { from: String, to: (f32, f32) },
    Marquee { a: (f32, f32), b: (f32, f32), additive: bool },
}

pub enum CanvasEvent {
    SelectionChanged,
}

/// Where a handle is: `None` the output, `Some(port)` an input.
type Handle = (String, Option<usize>);

pub struct GraphCanvas {
    store: Entity<Store>,
    pub focus: FocusHandle,
    pan: (f32, f32),
    zoom: f32,
    pub selected: BTreeSet<String>,
    drag: Option<Drag>,
    live: HashMap<String, (f32, f32)>,
    space: bool,
    pub hand: bool,
    pub show_grid: bool,
    pub show_minimap: bool,
    bounds: Rc<Cell<Bounds<Pixels>>>,
    fit_pending: bool,
    _subs: Vec<Subscription>,
}

impl EventEmitter<CanvasEvent> for GraphCanvas {}

impl GraphCanvas {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |this, s, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::Results) {
                let g = &s.read(cx).doc.graph;
                let before = this.selected.len();
                this.selected.retain(|id| g.contains(id));
                if this.selected.len() != before {
                    cx.emit(CanvasEvent::SelectionChanged);
                }
                cx.notify();
            }
        })];
        Self {
            store,
            focus: cx.focus_handle(),
            pan: (40.0, 40.0),
            zoom: 1.0,
            selected: BTreeSet::new(),
            drag: None,
            live: HashMap::new(),
            space: false,
            hand: false,
            show_grid: true,
            // Off until asked for: it covers nodes in a small pane
            show_minimap: false,
            bounds: Rc::new(Cell::new(Bounds::default())),
            fit_pending: true,
            _subs: subs,
        }
    }

    fn graph<'a>(&self, cx: &'a App) -> &'a GraphDoc {
        &self.store.read(cx).doc.graph
    }

    fn pos(&self, id: &str, x: f32, y: f32) -> (f32, f32) {
        self.live.get(id).copied().unwrap_or((x, y))
    }

    fn local(&self, p: gpui_kit::Point<Pixels>) -> (f32, f32) {
        let o = self.bounds.get().origin;
        ((p.x - o.x).as_f32(), (p.y - o.y).as_f32())
    }

    fn world(&self, (x, y): (f32, f32)) -> (f32, f32) {
        ((x - self.pan.0) / self.zoom, (y - self.pan.1) / self.zoom)
    }

    fn screen(&self, (x, y): (f32, f32)) -> (f32, f32) {
        (x * self.zoom + self.pan.0, y * self.zoom + self.pan.1)
    }

    /// The y (canvas units, from the node's top) of input `port` of `n` ports: in the title area.
    fn port_y(port: usize, n: usize) -> f32 {
        if n <= 1 { HEADER_H / 2.0 } else { 14.0 + port as f32 * ((HEADER_H - 20.0) / (n - 1) as f32).min(22.0) }
    }

    /// The y of the output handle.
    const OUT_Y: f32 = HEADER_H / 2.0;

    fn node_at(&self, p: (f32, f32), cx: &App) -> Option<String> {
        self.graph(cx).nodes.iter().rev().find_map(|n| {
            let (x, y) = self.pos(&n.id, n.x, n.y);
            (p.0 >= x && p.0 <= x + NODE_W && p.1 >= y && p.1 <= y + NODE_H).then(|| n.id.clone())
        })
    }

    fn handle_at(&self, p: (f32, f32), cx: &App) -> Option<Handle> {
        let grab = HANDLE_GRAB / self.zoom;
        self.graph(cx).nodes.iter().rev().find_map(|n| {
            let (x, y) = self.pos(&n.id, n.x, n.y);
            let near = |hx: f32, hy: f32| (p.0 - hx).hypot(p.1 - hy) <= grab;
            if n.kind.has_output() && near(x + NODE_W, y + Self::OUT_Y) {
                return Some((n.id.clone(), None));
            }
            let ins = n.kind.inputs().len();
            (0..ins).find(|&i| near(x, y + Self::port_y(i, ins))).map(|i| (n.id.clone(), Some(i)))
        })
    }

    fn exec(&self, cmd: Command, cx: &mut Context<Self>) -> bool {
        self.store.update(cx, |s, cx| s.execute(cmd, cx)).is_ok()
    }

    pub fn select(&mut self, ids: impl IntoIterator<Item = String>, cx: &mut Context<Self>) {
        self.selected = ids.into_iter().collect();
        let single = self.single().cloned();
        self.store.update(cx, |s, _| s.selected_node = single);
        cx.emit(CanvasEvent::SelectionChanged);
        cx.notify();
    }

    pub fn single(&self) -> Option<&String> {
        (self.selected.len() == 1).then(|| self.selected.iter().next()).flatten()
    }

    // ------------------------------------------------------------------------
    // View
    // ------------------------------------------------------------------------

    pub fn zoom_by(&mut self, factor: f32, anchor: Option<(f32, f32)>, cx: &mut Context<Self>) {
        let b = self.bounds.get().size;
        let anchor = anchor.unwrap_or((b.width.as_f32() / 2.0, b.height.as_f32() / 2.0));
        let w = self.world(anchor);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.pan = (anchor.0 - w.0 * self.zoom, anchor.1 - w.1 * self.zoom);
        cx.notify();
    }

    pub fn zoom_to(&mut self, zoom: f32, cx: &mut Context<Self>) {
        self.zoom_by(zoom / self.zoom, None, cx);
    }

    pub fn fit(&mut self, cx: &mut Context<Self>) {
        let b = self.bounds.get().size;
        let (w, h) = (b.width.as_f32(), b.height.as_f32());
        if w < 10.0 || h < 10.0 {
            self.fit_pending = true;
            return;
        }
        self.fit_pending = false;
        let Some((x0, y0, x1, y1)) = extent(self.graph(cx)) else {
            self.zoom = 1.0;
            self.pan = (40.0, 40.0);
            cx.notify();
            return;
        };
        let pad = 120.0;
        self.zoom = ((w - pad) / (x1 - x0).max(1.0)).min((h - pad) / (y1 - y0).max(1.0)).clamp(MIN_ZOOM, 1.2);
        self.pan = (w / 2.0 - (x0 + x1) / 2.0 * self.zoom, h / 2.0 - (y0 + y1) / 2.0 * self.zoom);
        cx.notify();
    }

    fn centre_on(&mut self, p: (f32, f32), cx: &mut Context<Self>) {
        let b = self.bounds.get().size;
        self.pan = (b.width.as_f32() / 2.0 - p.0 * self.zoom, b.height.as_f32() / 2.0 - p.1 * self.zoom);
        cx.notify();
    }

    fn new_place(&self, cx: &App) -> (f32, f32) {
        if let Some(n) = self.single().and_then(|id| self.graph(cx).node(id)) {
            return (n.x + COLUMN, n.y);
        }
        let b = self.bounds.get().size;
        let (x, y) = self.world((b.width.as_f32() / 2.0, b.height.as_f32() / 2.0));
        (snap(x - NODE_W / 2.0), snap(y - NODE_H / 2.0))
    }

    // ------------------------------------------------------------------------
    // Editing
    // ------------------------------------------------------------------------

    /// Adds `template` at `at` (default: after the selected node, fed by it when it fits).
    pub fn add(&mut self, template: &Template, at: Option<(f32, f32)>, cx: &mut Context<Self>) {
        let kind = template.kind();
        let after = if at.is_none() {
            self.single().filter(|id| {
                let g = self.graph(cx);
                let out = g.node(id).and_then(|n| n.kind.output());
                out.is_some() && kind.inputs().iter().any(|(t, _)| t.is_none_or(|t| Some(t) == out))
            }).cloned()
        } else {
            None
        };
        let (x, y) = at.unwrap_or_else(|| self.new_place(cx));
        let before: BTreeSet<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).collect();
        if self.exec(Command::AddNode { kind, x, y, id: None, after }, cx) {
            let new: Vec<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).filter(|id| !before.contains(id)).collect();
            self.select(new, cx);
        }
    }

    /// Deletes the selection; `bridge`: reconnects around each removed node where types allow.
    pub fn delete_selected(&mut self, bridge: bool, cx: &mut Context<Self>) {
        if self.selected.is_empty() {
            return;
        }
        if bridge {
            for id in self.selected.clone() {
                self.exec(Command::RemoveBridging { id }, cx);
            }
        } else {
            let nodes: Vec<String> = self.selected.iter().cloned().collect();
            self.exec(Command::RemoveNodes { nodes }, cx);
        }
        self.select([], cx);
    }

    pub fn duplicate_selected(&mut self, cx: &mut Context<Self>) {
        let g = self.graph(cx).clone();
        let mut map: HashMap<String, String> = HashMap::new();
        for id in &self.selected {
            let Some(n) = g.node(id) else { continue };
            let before: BTreeSet<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).collect();
            let mut kind = n.kind.clone();
            if let NodeKind::View { name, .. } = &mut kind {
                *name = format!("{name} (copy)");
            }
            if self.exec(Command::AddNode { kind, x: n.x + 32.0, y: n.y + 32.0, id: None, after: None }, cx)
                && let Some(new) = self.graph(cx).nodes.iter().map(|n| n.id.clone()).find(|i| !before.contains(i)) {
                    map.insert(id.clone(), new);
                }
        }
        for e in &g.edges {
            if let (Some(a), Some(b)) = (map.get(&e.from), map.get(&e.to)) {
                self.exec(Command::Connect { from: a.clone(), to: b.clone(), port: e.port }, cx);
            }
        }
        self.select(map.into_values(), cx);
    }

    pub fn reveal(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(n) = self.graph(cx).node(id) {
            let p = (n.x + NODE_W / 2.0, n.y + NODE_H / 2.0);
            self.centre_on(p, cx);
        }
        self.select([id.to_string()], cx);
    }

    pub fn select_all(&mut self, cx: &mut Context<Self>) {
        let all: Vec<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).collect();
        self.select(all, cx);
    }

    /// A double-click: a view shows the image in Explore.
    pub fn open(&mut self, id: &str, cx: &mut Context<Self>) {
        let g = self.graph(cx);
        let image = g.images_of(id).into_iter().next();
        if matches!(g.node(id).map(|n| &n.kind), Some(NodeKind::View { .. }) | Some(NodeKind::Source { .. }) | Some(NodeKind::Annotations { .. })) {
            self.store.update(cx, |s, cx| {
                if let Some(img) = image {
                    let _ = s.execute(Command::SelectImage { id: img }, cx);
                }
                s.set_workspace(Workspace::Explore, cx);
            });
        }
    }

    // ------------------------------------------------------------------------
    // Input
    // ------------------------------------------------------------------------

    fn on_down(&mut self, e: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.focus, cx);
        let local = self.local(e.position);
        if e.button == MouseButton::Middle || self.space || self.hand || e.modifiers.alt {
            self.drag = Some(Drag::Pan { mouse: local, pan: self.pan });
            return;
        }
        let p = self.world(local);
        let additive = e.modifiers.shift || e.modifiers.control || e.modifiers.platform;
        if let Some((id, port)) = self.handle_at(p, cx) {
            match port {
                None => {
                    self.drag = Some(Drag::Connect { from: id, to: local });
                    return;
                }
                Some(port) => {
                    // Pull an input off its edge: the edge follows the pointer from its source
                    if let Some(from) = self.graph(cx).input(&id, port).map(str::to_string) {
                        self.exec(Command::Disconnect { to: id, port }, cx);
                        self.drag = Some(Drag::Connect { from, to: local });
                        return;
                    }
                }
            }
        }
        if let Some(id) = self.node_at(p, cx) {
            if e.click_count == 2 {
                self.open(&id, cx);
                return;
            }
            if additive {
                let mut s = self.selected.clone();
                if !s.remove(&id) {
                    s.insert(id.clone());
                }
                self.select(s, cx);
            } else if !self.selected.contains(&id) {
                self.select([id.clone()], cx);
            }
            let g = self.graph(cx);
            let origins = self.selected.iter().filter_map(|i| g.node(i).map(|n| (i.clone(), n.x, n.y))).collect();
            self.drag = Some(Drag::Nodes { mouse: local, origins, moved: false });
            return;
        }
        if !additive && !self.selected.is_empty() {
            self.select([], cx);
        }
        self.drag = Some(Drag::Marquee { a: local, b: local, additive });
    }

    fn on_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if e.pressed_button.is_none() {
            if self.drag.take().is_some() {
                self.live.clear();
                cx.notify();
            }
            return;
        }
        let local = self.local(e.position);
        let zoom = self.zoom;
        match &mut self.drag {
            Some(Drag::Pan { mouse, pan }) => self.pan = (pan.0 + local.0 - mouse.0, pan.1 + local.1 - mouse.1),
            Some(Drag::Nodes { mouse, origins, moved }) => {
                let (dx, dy) = (local.0 - mouse.0, local.1 - mouse.1);
                if !*moved && dx.hypot(dy) < DRAG_SLOP {
                    return;
                }
                *moved = true;
                self.live = origins.iter().map(|(id, x, y)| (id.clone(), (snap(x + dx / zoom), snap(y + dy / zoom)))).collect();
            }
            Some(Drag::Connect { to, .. }) => *to = local,
            Some(Drag::Marquee { b, .. }) => *b = local,
            None => return,
        }
        cx.notify();
    }

    fn on_up(&mut self, e: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        let Some(drag) = self.drag.take() else { return };
        match drag {
            Drag::Pan { .. } => {}
            Drag::Nodes { moved, .. } => {
                if moved {
                    let moves = self.live.iter().map(|(id, &(x, y))| NodeMove { id: id.clone(), x, y }).collect();
                    self.exec(Command::MoveNodes { moves }, cx);
                }
            }
            Drag::Connect { from, .. } => {
                let p = self.world(self.local(e.position));
                let target = match self.handle_at(p, cx) {
                    Some((id, Some(port))) => Some((id, Some(port))),
                    _ => self.node_at(p, cx).map(|id| (id, None)),
                };
                if let Some((to, port)) = target.filter(|(t, _)| *t != from) {
                    let port = port.or_else(|| self.graph(cx).port_for(&from, &to));
                    match port {
                        Some(port) => {
                            if self.exec(Command::Connect { from, to: to.clone(), port }, cx) {
                                self.select([to], cx);
                            }
                        }
                        None => self.store.update(cx, |s, cx| s.set_status(format!("{from}'s output fits no input of {to}"), cx)),
                    }
                }
            }
            Drag::Marquee { a, b, additive } => {
                let (wa, wb) = (self.world(a), self.world(b));
                let (x0, x1, y0, y1) = (wa.0.min(wb.0), wa.0.max(wb.0), wa.1.min(wb.1), wa.1.max(wb.1));
                if (x1 - x0) * self.zoom > 2.0 || (y1 - y0) * self.zoom > 2.0 {
                    let hit: Vec<String> = self.graph(cx).nodes.iter().filter(|n| n.x < x1 && n.x + NODE_W > x0 && n.y < y1 && n.y + NODE_H > y0).map(|n| n.id.clone()).collect();
                    let mut s = if additive { self.selected.clone() } else { BTreeSet::new() };
                    s.extend(hit);
                    self.select(s, cx);
                }
            }
        }
        self.live.clear();
        cx.notify();
    }

    fn on_wheel(&mut self, e: &ScrollEvent, _: &mut Window, cx: &mut Context<Self>) {
        let d = e.delta.pixel_delta(px(20.));
        let (dx, dy) = (d.x.as_f32(), d.y.as_f32());
        let m = e.modifiers;
        if m.control || m.platform {
            self.zoom_by((dy * 0.003).exp(), Some(self.local(e.position)), cx);
        } else if m.shift && dx == 0.0 {
            self.pan.0 += dy;
        } else {
            self.pan = (self.pan.0 + dx, self.pan.1 + dy);
        }
        cx.stop_propagation();
        cx.notify();
    }

    fn on_key(&mut self, e: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        let k = &e.keystroke;
        let cmd = k.modifiers.control || k.modifiers.platform;
        match k.key.as_str() {
            "space" => self.space = true,
            "delete" | "backspace" => self.delete_selected(!k.modifiers.shift, cx),
            "escape" => self.select([], cx),
            "a" if cmd => self.select_all(cx),
            "d" if cmd => self.duplicate_selected(cx),
            "0" if cmd => self.zoom_to(1.0, cx),
            "=" | "+" if cmd => self.zoom_by(1.25, None, cx),
            "-" if cmd => self.zoom_by(0.8, None, cx),
            "f" if !cmd => self.fit(cx),
            "v" if !cmd => {
                self.hand = false;
                cx.notify();
            }
            "h" if !cmd => {
                self.hand = true;
                cx.notify();
            }
            "enter" => {
                if let Some(id) = self.single().cloned() {
                    self.open(&id, cx);
                }
            }
            _ => return,
        }
        cx.stop_propagation();
    }

    fn on_key_up(&mut self, e: &KeyUpEvent, _: &mut Window, _: &mut Context<Self>) {
        if e.keystroke.key == "space" {
            self.space = false;
        }
    }

    // ------------------------------------------------------------------------
    // Drawing
    // ------------------------------------------------------------------------

    fn highlight(&self, g: &GraphDoc) -> Option<BTreeSet<String>> {
        let id = self.single()?;
        let mut set = g.ancestors(id);
        set.extend(g.descendants(id));
        set.insert(id.clone());
        Some(set)
    }

    #[allow(clippy::too_many_arguments)]
    fn node_card(&self, id: &str, kind: &NodeKind, (sx, sy): (f32, f32), selected: bool, faded: bool, status: Option<(String, bool)>, cx: &App) -> impl IntoElement + use<> {
        let z = self.zoom;
        let doc = &self.store.read(cx).doc;
        let (title, sub, icon) = describe(kind, doc);
        let enabled = !matches!(kind, NodeKind::Op { enabled: false, .. });
        let accent = kind_color(kind);
        let r = (HANDLE_R * z.max(0.6)).max(3.0);
        let handle = |top: f32, left: bool, c: Hsla, connected: bool| {
            div()
                .absolute()
                .top(px(top * z - r))
                .when(left, |d| d.left(px(-r)))
                .when(!left, |d| d.right(px(-r)))
                .size(px(r * 2.0))
                .rounded_full()
                .border_2()
                .border_color(c)
                .bg(if connected { c } else { color::panel() })
        };
        let inputs = kind.inputs();
        let n_in = inputs.len();
        let g = &doc.graph;
        let in_handles: Vec<_> = inputs.iter().enumerate().map(|(i, (t, _))| handle(Self::port_y(i, n_in), true, port_color(*t), g.input(id, i).is_some())).collect();
        let out = kind.output().map(|t| handle(Self::OUT_Y, false, port_color(Some(t)), !g.outputs(id).is_empty()));
        let preview = self.store.read(cx).analysis.as_ref().and_then(|a| a.previews.get(id).cloned());
        let table_rows = self.store.read(cx).analysis.as_ref().and_then(|a| a.tables.get(id).map(|t| t.rows.len()));
        div()
            .id(SharedString::from(format!("node-{id}")))
            .absolute()
            .left(px(sx))
            .top(px(sy))
            .w(px(NODE_W * z))
            .h(px(NODE_H * z))
            .rounded(px(10.0 * z))
            .when(selected, |d| d.border_2())
            .when(!selected, |d| d.border_1())
            .border_color(if selected { color::accent_hover() } else { color::border() })
            .bg(color::raised())
            .shadow_sm()
            .when(faded, |d| d.opacity(0.35))
            .when(!enabled && !faded, |d| d.opacity(0.55))
            .child(
                v_flex()
                    .size_full()
                    .overflow_hidden()
                    .rounded(px(10.0 * z))
                    .child(div().w_full().h(px(3.0 * z.max(0.5))).bg(accent))
                    .child(
                        v_flex()
                            .flex_none()
                            .h(px((HEADER_H - 3.0) * z))
                            .w_full()
                            .min_w_0()
                            .px(px(12.0 * z))
                            .gap(px(3.0 * z))
                            .justify_center()
                            .child(
                                h_flex()
                                    .gap(px(6.0 * z))
                                    .min_w_0()
                                    .child(Icon::new(icon).size(px(14.0 * z)).text_color(accent))
                                    .child(div().flex_1().min_w_0().truncate().text_size(px(13.0 * z)).font_weight(FontWeight::SEMIBOLD).child(title))
                                    .when(!enabled, |d| d.child(div().flex_none().text_size(px(10.0 * z)).text_color(color::text_muted()).child("off")))
                                    .children(status.map(|(t, bad)| div().flex_none().text_size(px(10.0 * z)).text_color(if bad { color::danger() } else { color::text_muted() }).child(t))),
                            )
                            .when(z >= 0.45, |d| d.child(div().truncate().text_size(px(11.0 * z)).font_family("monospace").text_color(color::text_muted()).child(sub))),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_h_0()
                            .mx(px(8.0 * z))
                            .mb(px(8.0 * z))
                            .rounded(px(6.0 * z))
                            .overflow_hidden()
                            .bg(color::canvas())
                            .flex()
                            .items_center()
                            .justify_center()
                            .map(|d| match (&preview, table_rows) {
                                (_, Some(n)) => d.child(div().text_size(px(12.0 * z)).text_color(color::text_muted()).child(format!("{n} objects"))),
                                (Some(p), _) => d.child(img(p.clone()).size_full().object_fit(ObjectFit::Contain)),
                                (None, _) => d.child(div().text_size(px(11.0 * z)).text_color(color::text_disabled()).child("no preview")),
                            }),
                    ),
            )
            .children(in_handles)
            .children(out)
    }

    /// Buttons over the selected node.
    fn toolbar(&self, id: &str, kind: &NodeKind, (sx, sy): (f32, f32), cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let me = cx.entity().downgrade();
        let mut bar = h_flex().gap(px(2.));
        let button = |name: &'static str, icon: IconName, tip: &'static str, f: fn(&mut GraphCanvas, &str, &mut Context<GraphCanvas>)| {
            let (me, id) = (me.clone(), id.to_string());
            icon_button(SharedString::from(format!("node-tool-{name}")), Icon::new(icon), tip).xsmall().on_click(move |_, _, cx| {
                let _ = me.update(cx, |c, cx| f(c, &id, cx));
            })
        };
        if let NodeKind::Op { enabled, .. } = kind {
            let (icon, tip) = if *enabled { (IconName::EyeOff, "Turn off (kept, passes its input through)") } else { (IconName::Eye, "Turn on") };
            let on = !*enabled;
            let (me, id) = (me.clone(), id.to_string());
            bar = bar.child(icon_button("node-tool-enable", Icon::new(icon), tip).xsmall().on_click(move |_, _, cx| {
                let _ = me.update(cx, |c, cx| c.exec(Command::EnableNode { id: id.clone(), enabled: on }, cx));
            }));
        }
        if matches!(kind, NodeKind::View { .. }) {
            bar = bar.child(button("show", IconName::ExternalLink, "Show in Explore (double-click)", |c, id, cx| c.open(id, cx)));
        }
        bar = bar
            .child(button("duplicate", IconName::Copy, "Duplicate (Ctrl+D)", |c, _, cx| c.duplicate_selected(cx)))
            .child(button("delete", IconName::Trash, "Delete, reconnecting around it (Del)", |c, _, cx| c.delete_selected(true, cx)));
        div()
            .absolute()
            .left(px(sx))
            .top(px(sy - 34.0))
            .p(px(2.))
            .rounded(px(6.))
            .border_1()
            .border_color(color::border())
            .bg(color::raised())
            .shadow_md()
            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(bar)
    }

    fn controls(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let me = cx.entity().downgrade();
        let b = |name: &'static str, icon: IconName, tip: &'static str, f: fn(&mut GraphCanvas, &mut Context<GraphCanvas>)| {
            let me = me.clone();
            icon_button(SharedString::from(format!("canvas-{name}")), Icon::new(icon), tip).on_click(move |_, _, cx| {
                let _ = me.update(cx, f);
            })
        };
        h_flex()
            .absolute()
            .left_3()
            .bottom_3()
            .gap(px(2.))
            .p(px(2.))
            .rounded(px(6.))
            .border_1()
            .border_color(color::border())
            .bg(color::raised())
            .shadow_md()
            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(b("zoom-out", IconName::ZoomOut, "Zoom out (Ctrl −)", |c, cx| c.zoom_by(0.8, None, cx)))
            .child(div().w(px(48.)).text_xs().child({
                let me = me.clone();
                crate::ui::widgets::scrub_label("canvas-zoom", format!("{:.0} %", self.zoom * 100.0), move |d, _, cx| {
                    let _ = me.update(cx, |c, cx| c.zoom_by(if d > 0.0 { 1.05 } else { 1.0 / 1.05 }, None, cx));
                })
            }))
            .child(b("zoom-in", IconName::ZoomIn, "Zoom in (Ctrl +)", |c, cx| c.zoom_by(1.25, None, cx)))
            .child(b("fit", IconName::Maximize, "Fit the graph (F)", |c, cx| c.fit(cx)))
            .child(b("grid", IconName::LayoutGrid, "Dot grid", |c, cx| {
                c.show_grid = !c.show_grid;
                cx.notify();
            }))
            .child(b("minimap", IconName::Map, "Minimap", |c, cx| {
                c.show_minimap = !c.show_minimap;
                cx.notify();
            }))
    }

    fn minimap(&self, g: &GraphDoc, cx: &mut Context<Self>) -> Option<impl IntoElement + use<>> {
        let (x0, y0, x1, y1) = extent(g)?;
        let size_now = self.bounds.get().size;
        let (vw0, vh0) = self.world((0.0, 0.0));
        let (vw1, vh1) = self.world((size_now.width.as_f32(), size_now.height.as_f32()));
        let (ex0, ey0, ex1, ey1) = (x0.min(vw0), y0.min(vh0), x1.max(vw1), y1.max(vh1));
        let scale = ((MAP_W - 12.0) / (ex1 - ex0).max(1.0)).min((MAP_H - 12.0) / (ey1 - ey0).max(1.0));
        let rects: Vec<((f32, f32, f32, f32), Hsla)> = g
            .nodes
            .iter()
            .map(|n| {
                let (x, y) = self.pos(&n.id, n.x, n.y);
                (((x - ex0) * scale + 6.0, (y - ey0) * scale + 6.0, NODE_W * scale, NODE_H * scale), kind_color(&n.kind))
            })
            .collect();
        let view = ((vw0 - ex0) * scale + 6.0, (vh0 - ey0) * scale + 6.0, (vw1 - vw0) * scale, (vh1 - vh0) * scale);
        let frame = color::accent_hover();
        let me = cx.entity().downgrade();
        let to_world = move |local: (f32, f32)| ((local.0 - 6.0) / scale + ex0, (local.1 - 6.0) / scale + ey0);
        let map_bounds = Rc::new(Cell::new(Bounds::<Pixels>::default()));
        let mb = map_bounds.clone();
        Some(
            div()
                .id("canvas-minimap")
                .absolute()
                .right_3()
                .bottom_3()
                .w(px(MAP_W))
                .h(px(MAP_H))
                .rounded(px(6.))
                .border_1()
                .border_color(color::border())
                .bg(color::raised())
                .shadow_md()
                .overflow_hidden()
                .on_mouse_down(MouseButton::Left, move |e, _, cx| {
                    cx.stop_propagation();
                    let o = map_bounds.get().origin;
                    let p = to_world(((e.position.x - o.x).as_f32(), (e.position.y - o.y).as_f32()));
                    let _ = me.update(cx, |c, cx| c.centre_on(p, cx));
                })
                .child(
                    canvas(
                        move |bounds, _, _| mb.set(bounds),
                        move |bounds, _, window, _| {
                            let o = bounds.origin;
                            for ((x, y, w, h), c) in &rects {
                                window.paint_quad(fill(Bounds { origin: point(o.x + px(*x), o.y + px(*y)), size: size(px(w.max(2.0)), px(h.max(2.0))) }, c.opacity(0.85)));
                            }
                            let (x, y, w, h) = view;
                            let r = Bounds { origin: point(o.x + px(x), o.y + px(y)), size: size(px(w), px(h)) };
                            window.paint_quad(gpui_kit::outline(r, frame, gpui_kit::BorderStyle::Solid));
                        },
                    )
                    .size_full(),
                ),
        )
    }
}

fn snap(v: f32) -> f32 {
    (v / SNAP).round() * SNAP
}

fn extent(g: &GraphDoc) -> Option<(f32, f32, f32, f32)> {
    let mut it = g.nodes.iter();
    let first = it.next()?;
    let init = (first.x, first.y, first.x + NODE_W, first.y + NODE_H);
    Some(it.fold(init, |(a, b, c, d), n| (a.min(n.x), b.min(n.y), c.max(n.x + NODE_W), d.max(n.y + NODE_H))))
}

fn edge_path(a: (f32, f32), b: (f32, f32), width: f32) -> Option<gpui_kit::Path<Pixels>> {
    let mut p = PathBuilder::stroke(px(width));
    let dx = ((b.0 - a.0).abs() * 0.5).max(40.0);
    p.move_to(point(px(a.0), px(a.1)));
    p.cubic_bezier_to(point(px(b.0), px(b.1)), point(px(a.0 + dx), px(a.1)), point(px(b.0 - dx), px(b.1)));
    p.build().ok()
}

impl Render for GraphCanvas {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let s = self.store.read(cx);
        let g = s.doc.graph.clone();
        let results = s.results.clone();
        let analysis = s.analysis.clone();
        let z = self.zoom;
        let highlight = self.highlight(&g);
        let faded = |id: &str| highlight.as_ref().is_some_and(|h| !h.contains(id));

        // Edges coloured by what they carry
        let mut edges: Vec<EdgeLine> = Vec::new();
        for e in &g.edges {
            let (Some(a), Some(b)) = (g.node(&e.from), g.node(&e.to)) else { continue };
            let pa = self.pos(&a.id, a.x, a.y);
            let pb = self.pos(&b.id, b.x, b.y);
            let n_in = b.kind.inputs().len().max(1);
            let from = self.screen((pa.0 + NODE_W, pa.1 + Self::OUT_Y));
            let to = self.screen((pb.0, pb.1 + Self::port_y(e.port, n_in)));
            let lit = highlight.as_ref().is_some_and(|h| h.contains(&e.from) && h.contains(&e.to));
            let dim = highlight.is_some() && !lit;
            edges.push((from, to, port_color(a.kind.output()), lit, dim));
        }
        let pending = match &self.drag {
            Some(Drag::Connect { from, to }) => g.node(from).map(|n| {
                let p = self.pos(&n.id, n.x, n.y);
                (self.screen((p.0 + NODE_W, p.1 + Self::OUT_Y)), *to, port_color(n.kind.output()))
            }),
            _ => None,
        };
        let marquee = match &self.drag {
            Some(Drag::Marquee { a, b, .. }) => Some((*a, *b)),
            _ => None,
        };
        let (pan, show_grid) = (self.pan, self.show_grid);
        let (dot, sel_line) = (color::text_muted().opacity(0.25), color::accent_hover());
        let bounds_cell = self.bounds.clone();
        let plane = canvas(
            move |bounds, _, _| bounds_cell.set(bounds),
            move |bounds, _, window, _| {
                let o = bounds.origin;
                let (w, h) = (bounds.size.width.as_f32(), bounds.size.height.as_f32());
                if show_grid {
                    let mut step = GRID * z;
                    while step < 10.0 {
                        step *= 2.0;
                    }
                    let r = (1.2 * z.clamp(0.6, 1.4)).max(1.0);
                    let mut x = pan.0.rem_euclid(step);
                    while x < w {
                        let mut y = pan.1.rem_euclid(step);
                        while y < h {
                            window.paint_quad(fill(Bounds { origin: point(o.x + px(x - r / 2.0), o.y + px(y - r / 2.0)), size: size(px(r), px(r)) }, dot));
                            y += step;
                        }
                        x += step;
                    }
                }
                let at = |p: (f32, f32)| (p.0 + o.x.as_f32(), p.1 + o.y.as_f32());
                let mut order: Vec<&EdgeLine> = edges.iter().collect();
                order.sort_by_key(|e| e.3);
                for (a, b, c, lit, dim) in order {
                    let (col, width) = if *lit { (*c, 3.0) } else if *dim { (c.opacity(0.2), 1.5) } else { (c.opacity(0.75), 2.0) };
                    if let Some(p) = edge_path(at(*a), at(*b), width * z.clamp(0.7, 1.3)) {
                        window.paint_path(p, col);
                    }
                }
                if let Some((a, b, c)) = pending
                    && let Some(p) = edge_path(at(a), at(b), 2.0) {
                        window.paint_path(p, c);
                    }
                if let Some((a, b)) = marquee {
                    let (a, b) = (at(a), at(b));
                    let r = Bounds { origin: point(px(a.0.min(b.0)), px(a.1.min(b.1))), size: size(px((a.0 - b.0).abs()), px((a.1 - b.1).abs())) };
                    window.paint_quad(fill(r, sel_line.opacity(0.1)));
                    window.paint_quad(gpui_kit::outline(r, sel_line, gpui_kit::BorderStyle::Solid));
                }
            },
        )
        .absolute()
        .size_full();

        let cards: Vec<_> = g
            .nodes
            .iter()
            .map(|n| {
                let sp = self.screen(self.pos(&n.id, n.x, n.y));
                let status = match (analysis.as_ref().and_then(|a| a.errors.get(&n.id)), results.get(&n.id)) {
                    (Some(_), _) => Some(("error".to_string(), true)),
                    (None, Some(r)) if r.error.is_some() => Some(("error".to_string(), true)),
                    (None, Some(r)) => Some((format!("{:.0} ms", r.ms), false)),
                    _ => None,
                };
                self.node_card(&n.id, &n.kind, sp, self.selected.contains(&n.id), faded(&n.id), status, cx).into_any_element()
            })
            .collect();
        let toolbar = self.single().cloned().and_then(|id| {
            let n = g.node(&id)?;
            let sp = self.screen(self.pos(&n.id, n.x, n.y));
            Some(self.toolbar(&id, &n.kind, sp, cx).into_any_element())
        });
        let empty = g.nodes.is_empty().then(|| {
            v_flex()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .items_center()
                .justify_center()
                .gap_1()
                .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("An empty pipeline"))
                .child(div().text_sm().text_color(color::text_muted()).child("Drag a source and operations from the library on the left, or apply an operation in Explore."))
        });
        // Never over a pane too small to work around it (the Pipeline docked under the views)
        let roomy = { let b = self.bounds.get().size; b.height.as_f32() > 3.0 * MAP_H && b.width.as_f32() > 3.0 * MAP_W };
        let minimap = if self.show_minimap && roomy { self.minimap(&g, cx).map(IntoElement::into_any_element) } else { None };
        if self.fit_pending && self.bounds.get().size.width.as_f32() > 10.0 {
            self.fit_pending = false;
            let me = cx.entity().downgrade();
            cx.defer(move |cx| {
                let _ = me.update(cx, |c, cx| c.fit(cx));
            });
        }
        let cursor = match self.drag {
            Some(Drag::Pan { .. }) => gpui_kit::CursorStyle::ClosedHand,
            _ if self.space || self.hand => gpui_kit::CursorStyle::OpenHand,
            _ => gpui_kit::CursorStyle::Arrow,
        };
        div()
            .id("graph-canvas")
            .track_focus(&self.focus)
            .key_context("GraphCanvas")
            .relative()
            .size_full()
            .overflow_hidden()
            .bg(color::canvas())
            .cursor(cursor)
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_down))
            .on_mouse_down(MouseButton::Middle, cx.listener(Self::on_down))
            .on_mouse_move(cx.listener(Self::on_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_up))
            .on_mouse_up(MouseButton::Middle, cx.listener(Self::on_up))
            .on_scroll(cx.listener(Self::on_wheel))
            .on_key_down(cx.listener(Self::on_key))
            .on_key_up(cx.listener(Self::on_key_up))
            .on_drop(cx.listener(|this, d: &CatalogDrag, window, cx| {
                let local = this.local(window.mouse_position());
                let (x, y) = this.world(local);
                this.add(&d.template, Some((snap(x - NODE_W / 2.0), snap(y - NODE_H / 2.0))), cx);
            }))
            .child(plane)
            .children(cards)
            .children(toolbar)
            .children(empty)
            .child(self.controls(cx))
            .children(minimap)
    }
}

/// What a dragged library item looks like under the pointer.
pub struct DragGhost(pub SharedString);

impl Render for DragGhost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().px_2().py_1().rounded(px(5.)).border_1().border_color(color::accent()).bg(color::raised()).shadow_md().text_sm().child(self.0.clone())
    }
}

//! The Pipeline workspace (dsp-kitchen's layout): a menu bar on top (Photoshop style: Sources,
//! then one menu per family of operators, Output; an item is added after the selected node, fed
//! by it when it fits), the canvas with a floating tool bar, the inspector of the selection on the
//! right. Every edit is a command, so Explore, the canvas and an agent see the same graph.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _,
    StyledImage as _, Styled as _, Subscription, WeakEntity, Window,
};

use crate::data::results::COLORMAPS;
use crate::data::store::{AppEvent, Store};
use crate::domain::command::Command;
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::ui::param_editor::{Commit, ParamEditor};
use crate::ui::theme::{self, color};
use crate::ui::widgets::{divider_v, row, stepper, MenuSelect, Section};

use super::canvas::{CanvasEvent, GraphCanvas};
use super::catalog::{describe, operations, port_color, sources, Template};

const INSPECTOR_W: f32 = 312.;

pub struct PipelineView {
    store: Entity<Store>,
    pub canvas: Entity<GraphCanvas>,
    inspector: Entity<Inspector>,
    _subs: Vec<Subscription>,
}

impl PipelineView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let canvas = cx.new(|cx| GraphCanvas::new(store.clone(), cx));
        let inspector = cx.new(|cx| Inspector::new(store.clone(), canvas.clone(), window, cx));
        let subs = vec![
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::Document(_) | AppEvent::Results | AppEvent::ImageReady(_)) {
                    cx.notify();
                }
            }),
            cx.subscribe(&canvas, |_, _, _: &CanvasEvent, cx| cx.notify()),
        ];
        Self { store, canvas, inspector, _subs: subs }
    }

    fn canvas_do(canvas: &WeakEntity<GraphCanvas>, cx: &mut App, f: impl FnOnce(&mut GraphCanvas, &mut Context<GraphCanvas>)) {
        let _ = canvas.update(cx, f);
    }

    /// The menu bar: Sources, Edit, one menu per family of operators, Output, View.
    fn menu_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let canvas = self.canvas.downgrade();
        let selected = !self.canvas.read(cx).selected.is_empty();
        let menu = |id: &'static str, title: &'static str, icon: IconName, items: Vec<(String, String, Template)>| {
            let canvas = canvas.clone();
            Button::new(id).ghost().small().icon(Icon::new(icon)).label(title).dropdown_menu(move |m: PopupMenu, _, _| {
                let mut m = m.scrollable(true).max_h(px(520.));
                if items.is_empty() {
                    m = m.label("Open an image first");
                }
                for (label, hint, template) in &items {
                    let (canvas, template) = (canvas.clone(), template.clone());
                    m = m.item(PopupMenuItem::new(format!("{label}   {hint}")).on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, |c, cx| c.add(&template, None, cx))));
                }
                m
            })
        };
        let src = sources(&self.store.read(cx).doc);
        let mut bar = h_flex()
            .id("pipeline-menu-bar")
            .flex_none()
            .h(px(theme::OPTIONS_BAR_H))
            .px(px(theme::SM))
            .gap(px(2.))
            .items_center()
            .bg(color::panel())
            .border_b_1()
            .border_color(color::border())
            .child(menu("pipeline-menu-sources", "Sources", IconName::Image, src.items.into_iter().map(|i| (i.label, i.hint, i.template)).collect()));
        let edit = {
            let canvas = canvas.clone();
            Button::new("pipeline-menu-edit").ghost().small().label("Edit").dropdown_menu(move |m: PopupMenu, _, _| {
                let item = |label: &'static str, enabled: bool, f: fn(&mut GraphCanvas, &mut Context<GraphCanvas>)| {
                    let canvas = canvas.clone();
                    PopupMenuItem::new(label).disabled(!enabled).on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, f))
                };
                m.item(item("Duplicate  Ctrl+D", selected, |c, cx| c.duplicate_selected(cx)))
                    .item(item("Delete, reconnecting  Del", selected, |c, cx| c.delete_selected(true, cx)))
                    .item(item("Delete alone  Shift+Del", selected, |c, cx| c.delete_selected(false, cx)))
                    .separator()
                    .item(item("Select all  Ctrl+A", true, |c, cx| c.select_all(cx)))
            })
        };
        bar = bar.child(edit).child(divider_v());
        for c in operations() {
            let id: &'static str = match c.name {
                "Filter" => "pipeline-menu-filter",
                "Intensity" => "pipeline-menu-intensity",
                "Restore" => "pipeline-menu-restore",
                "Threshold" => "pipeline-menu-threshold",
                "Mask" => "pipeline-menu-mask",
                "Segment" => "pipeline-menu-segment",
                _ => "pipeline-menu-measure",
            };
            bar = bar.child(menu(id, c.name, c.icon, c.items.into_iter().map(|i| (i.label, i.hint, i.template)).collect()));
        }
        bar = bar.child(divider_v()).child(menu("pipeline-menu-output", "Output", IconName::Eye, vec![("Result view".into(), "shown in Explore / Analyze".into(), Template::View)]));
        let view = {
            let canvas = canvas.clone();
            Button::new("pipeline-menu-view").ghost().small().label("View").dropdown_menu(move |m: PopupMenu, _, _| {
                let item = |label: &'static str, f: fn(&mut GraphCanvas, &mut Context<GraphCanvas>)| {
                    let canvas = canvas.clone();
                    PopupMenuItem::new(label).on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, f))
                };
                m.item(item("Fit the graph  F", |c, cx| c.fit(cx)))
                    .item(item("Actual size  Ctrl+0", |c, cx| c.zoom_to(1.0, cx)))
                    .separator()
                    .item(item("Dot grid", |c, cx| {
                        c.show_grid = !c.show_grid;
                        cx.notify();
                    }))
                    .item(item("Minimap", |c, cx| {
                        c.show_minimap = !c.show_minimap;
                        cx.notify();
                    }))
            })
        };
        bar.child(view)
            .child(div().flex_1())
            .children([("image", lumen_pipeline::PortType::Image), ("mask", lumen_pipeline::PortType::Mask), ("labels", lumen_pipeline::PortType::Labels), ("table", lumen_pipeline::PortType::Table)].into_iter().map(|(n, t)| {
                h_flex().gap(px(theme::XS)).px(px(theme::XS)).child(div().size(px(8.)).rounded_full().bg(port_color(Some(t)))).child(div().text_xs().text_color(color::text_muted()).child(n))
            }))
    }

    /// The tool bar floating at the bottom of the canvas.
    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let c = self.canvas.read(cx);
        let (hand, selected) = (c.hand, !c.selected.is_empty());
        let canvas = self.canvas.downgrade();
        let tool = |id: &'static str, icon: IconName, tip: &'static str, on: bool, f: fn(&mut GraphCanvas, &mut Context<GraphCanvas>)| {
            let canvas = canvas.clone();
            Button::new(id).ghost().small().icon(Icon::new(icon)).tooltip(tip).selected(on).on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, f))
        };
        let store = self.store.clone();
        h_flex()
            .id("pipeline-toolbar")
            .occlude()
            .absolute()
            .bottom_4()
            .left_0()
            .right_0()
            .justify_center()
            .child(
                h_flex()
                    .gap(px(2.))
                    .p(px(theme::XS))
                    .rounded(px(8.))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::raised())
                    .shadow_lg()
                    .child(tool("tool-select", IconName::MousePointer2, "Select (V)", !hand, |c, cx| {
                        c.hand = false;
                        cx.notify();
                    }))
                    .child(tool("tool-hand", IconName::Hand, "Move the canvas (H, or hold Space)", hand, |c, cx| {
                        c.hand = true;
                        cx.notify();
                    }))
                    .child(divider_v())
                    .child(tool("tool-duplicate", IconName::Copy, "Duplicate (Ctrl+D)", false, |c, cx| c.duplicate_selected(cx)))
                    .child({
                        let canvas = canvas.clone();
                        Button::new("tool-delete").ghost().small().icon(Icon::new(IconName::Trash)).tooltip("Delete, reconnecting around it (Del; Shift+Del alone)").disabled(!selected).on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, |c, cx| c.delete_selected(true, cx)))
                    })
                    .child(divider_v())
                    .child(tool("tool-fit", IconName::Maximize, "Fit the graph (F)", false, |c, cx| c.fit(cx)))
                    .child(Button::new("tool-run").ghost().small().icon(Icon::new(IconName::Play)).label("Run on view").tooltip("Compute the results again over what Explore shows").on_click(move |_, _, cx| store.update(cx, |s, cx| s.recompute(cx)))),
            )
    }
}

impl Render for PipelineView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex().id("pipeline-workspace").size_full().child(self.menu_bar(cx)).child(
            h_flex()
            .flex_1()
            .min_h_0()
            .w_full()
            .child(div().relative().flex_1().min_w_0().h_full().child(self.canvas.clone()).child(self.toolbar(cx)))
            .child(
                v_flex()
                    .flex_none()
                    .w(px(INSPECTOR_W))
                    .h_full()
                    .bg(color::panel())
                    .border_l_1()
                    .border_color(color::border())
                    .child(self.inspector.clone()),
            ))
    }
}

/// The selected node: what it is, its values, its result.
pub struct Inspector {
    store: Entity<Store>,
    canvas: Entity<GraphCanvas>,
    params: Entity<ParamEditor>,
    name: Entity<InputState>,
    name_for: String,
    _subs: Vec<Subscription>,
}

impl Inspector {
    fn new(store: Entity<Store>, canvas: Entity<GraphCanvas>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let name = cx.new(|cx| InputState::new(window, cx));
        let subs = vec![
            cx.subscribe(&canvas, |_, _, _: &CanvasEvent, cx| cx.notify()),
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::Document(_) | AppEvent::Results) {
                    cx.notify();
                }
            }),
            cx.subscribe(&name, |this, input, e: &InputEvent, cx| {
                if matches!(e, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                    let text = input.read(cx).value().trim().to_string();
                    let id = this.name_for.clone();
                    if text.is_empty() || id.is_empty() {
                        return;
                    }
                    let style = match this.store.read(cx).doc.graph.node(&id).map(|n| &n.kind) {
                        Some(NodeKind::View { style, name }) if *name != text => style.clone(),
                        _ => return,
                    };
                    this.store.update(cx, |s, cx| {
                        let _ = s.execute(Command::SetView { id, name: text, style }, cx);
                    });
                }
            }),
        ];
        Self { store, canvas, params: cx.new(|_| ParamEditor::new()), name, name_for: String::new(), _subs: subs }
    }

    fn exec(store: &Entity<Store>, cmd: Command, cx: &mut App) {
        store.update(cx, |s, cx| {
            let _ = s.execute(cmd, cx);
        });
    }

    fn body(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let selected = self.canvas.read(cx).selected.clone();
        let s = self.store.read(cx);
        let doc = s.doc.clone();
        let g = &doc.graph;
        if selected.len() > 1 {
            let canvas = self.canvas.downgrade();
            return Section::new(format!("{} nodes selected", selected.len()))
                .child(div().text_xs().text_color(color::text_muted()).child("Drag one to move them together; Ctrl+D duplicates them with their edges."))
                .child(Button::new("inspector-delete").outline().small().label("Delete them").on_click(move |_, _, cx| {
                    let _ = canvas.update(cx, |c, cx| c.delete_selected(true, cx));
                }))
                .into_any_element();
        }
        let Some(node) = selected.iter().next().and_then(|id| g.node(id)).cloned() else {
            return Section::new("Nothing selected")
                .child(div().text_xs().text_color(color::text_muted()).child("Click a node to see it here; its path through the graph lights up."))
                .child(div().text_xs().text_color(color::text_muted()).child("Handles are coloured by what they carry: image, mask, labels, table. Drag an output onto an input of the same colour."))
                .child(div().text_xs().text_color(color::text_muted()).child("A result view draws what feeds it over the image in Explore (tables go to Analyze)."))
                .into_any_element();
        };
        let id = node.id.clone();
        let (title, sub, icon) = describe(&node.kind, &doc);
        let analysis = s.analysis.clone();
        let layer = s.results.get(&id).cloned();
        let preview = analysis.as_ref().and_then(|a| a.previews.get(&id).cloned());
        let header = v_flex()
            .gap(px(2.))
            .child(h_flex().gap(px(theme::SM)).child(Icon::new(icon).small()).child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(title)))
            .child(div().text_xs().font_family("monospace").text_color(color::text_muted()).truncate().child(sub))
            .child(div().text_xs().text_color(color::text_muted()).child(id.clone()));
        // Its inputs and what feeds them
        let inputs = node.kind.inputs();
        let ports = (!inputs.is_empty()).then(|| {
            let mut sec = Section::new("Inputs");
            for (i, (t, name)) in inputs.iter().enumerate() {
                let from = g.input(&id, i).map(str::to_string);
                let (store, id2) = (self.store.clone(), id.clone());
                sec = sec.child(
                    h_flex()
                        .gap(px(theme::SM))
                        .child(div().size(px(8.)).rounded_full().bg(port_color(*t)))
                        .child(div().w(px(80.)).text_sm().child(name.to_string()))
                        .child(div().flex_1().min_w_0().truncate().text_sm().text_color(if from.is_some() { color::text() } else { color::warning() }).child(from.clone().unwrap_or_else(|| "not connected".into())))
                        .when(from.is_some(), |d| {
                            d.child(Button::new(SharedString::from(format!("unlink-{i}"))).ghost().xsmall().icon(Icon::new(IconName::Unlink)).tooltip("Disconnect").on_click(move |_, _, cx| Self::exec(&store, Command::Disconnect { to: id2.clone(), port: i }, cx)))
                        }),
                );
            }
            sec
        });
        let body: AnyElement = match &node.kind {
            NodeKind::Op { enabled, op } => {
                let commit: Commit = {
                    let (store, id) = (self.store.clone(), id.clone());
                    Rc::new(move |op, cx| store.update(cx, |s, cx| s.execute(Command::SetOp { id: id.clone(), op }, cx)).map(|_| ()).map_err(|e| e.to_string()))
                };
                self.params.update(cx, |p, cx| p.set_target(&id, op, commit, window, cx));
                let (store, id2, on) = (self.store.clone(), id.clone(), *enabled);
                v_flex()
                    .gap(px(theme::MD))
                    .child(Checkbox::new("inspector-enabled").checked(on).label("On (off: passes its input through)").on_click(move |v: &bool, _, cx| Self::exec(&store, Command::EnableNode { id: id2.clone(), enabled: *v }, cx)))
                    .child(Section::new("Parameters").child(self.params.clone()))
                    .child(div().text_xs().text_color(color::text_muted()).child(format!("{} · margin {} px", super::catalog::signature(op), op.halo())))
                    .into_any_element()
            }
            NodeKind::Source { image, channel } => {
                let img = doc.image(image);
                let names: Vec<SharedString> = img.map(|i| i.channels.iter().map(|c| SharedString::from(c.label.clone())).collect()).unwrap_or_default();
                let store = self.store.clone();
                let id2 = id.clone();
                v_flex()
                    .gap(px(theme::SM))
                    .child(row("Image", div().text_sm().truncate().child(img.map_or(image.clone(), |i| i.name.clone()))))
                    .child(row("Channel", MenuSelect::new("inspector-channel", names, Some(*channel), move |i, _, cx| Self::exec(&store, Command::SetSourceChannel { id: id2.clone(), channel: i }, cx))))
                    .child(div().text_xs().text_color(color::text_muted()).child("Values scaled to 0–1 by the type's range (16-bit: ÷ 65535)."))
                    .into_any_element()
            }
            NodeKind::Annotations { image, class } => {
                let mut names: Vec<SharedString> = vec!["All classes".into()];
                names.extend(doc.classes.iter().map(|c| SharedString::from(c.name.clone())));
                let raw: Vec<String> = doc.classes.iter().map(|c| c.name.clone()).collect();
                let current = class.as_ref().and_then(|c| raw.iter().position(|r| r == c)).map_or(0, |i| i + 1);
                let (store, id2, img, x, y) = (self.store.clone(), id.clone(), image.clone(), node.x, node.y);
                v_flex()
                    .gap(px(theme::SM))
                    .child(row(
                        "Class",
                        MenuSelect::new("inspector-class", names, Some(current), move |i, _, cx| {
                            // Swap the node for one of the chosen class, keeping its edges
                            let class = if i == 0 { None } else { Some(raw[i - 1].clone()) };
                            let kind = NodeKind::Annotations { image: img.clone(), class };
                            let id = id2.clone();
                            store.update(cx, |s, cx| {
                                let outs: Vec<(String, usize)> = s.doc.graph.outputs(&id).into_iter().map(|(o, p)| (o.to_string(), p)).collect();
                                let fresh = s.doc.graph.fresh_id("annotations");
                                if s.execute(Command::AddNode { kind, x, y, id: Some(fresh.clone()), after: None }, cx).is_ok() {
                                    for (o, p) in outs {
                                        let _ = s.execute(Command::Connect { from: fresh.clone(), to: o, port: p }, cx);
                                    }
                                    let _ = s.execute(Command::RemoveNodes { nodes: vec![id] }, cx);
                                }
                            });
                        }),
                    ))
                    .child(div().text_xs().text_color(color::text_muted()).child("Regions and brush strokes fill the mask; eraser strokes clear it."))
                    .into_any_element()
            }
            NodeKind::View { name, style } => {
                if self.name_for != id {
                    self.name_for = id.clone();
                    let name = name.clone();
                    self.name.update(cx, |i, cx| i.set_value(name, window, cx));
                }
                let set = {
                    let (store, id, name, style) = (self.store.clone(), id.clone(), name.clone(), style.clone());
                    Rc::new(move |f: &dyn Fn(&mut ViewStyle), cx: &mut App| {
                        let mut st = style.clone();
                        f(&mut st);
                        Self::exec(&store, Command::SetView { id: id.clone(), name: name.clone(), style: st }, cx);
                    })
                };
                let (s1, s2, s3) = (set.clone(), set.clone(), set);
                let lut_idx = COLORMAPS.iter().position(|m| *m == style.colormap);
                v_flex()
                    .gap(px(theme::SM))
                    .child(row("Name", div().w_full().child(Input::new(&self.name).small())))
                    .child(Checkbox::new("view-visible").checked(style.visible).label("Drawn over the image").on_click(move |v: &bool, _, cx| {
                        let v = *v;
                        s1(&move |st: &mut ViewStyle| st.visible = v, cx)
                    }))
                    .child(row("Opacity", stepper("view-opacity", format!("{:.0} %", style.opacity * 100.0), move |d, _, cx| s2(&move |st: &mut ViewStyle| st.opacity = (st.opacity + d * 0.1).clamp(0.0, 1.0), cx))))
                    .child(row(
                        "Colours",
                        MenuSelect::new("view-lut", COLORMAPS.iter().map(|m| SharedString::from(*m)).collect(), lut_idx, move |i, _, cx| s3(&move |st: &mut ViewStyle| st.colormap = COLORMAPS[i].to_string(), cx)),
                    ))
                    .into_any_element()
            }
        };
        let status = {
            let mut sec = Section::new("Result");
            if let Some(a) = &analysis {
                if let Some(e) = a.errors.get(&id) {
                    sec = sec.child(div().text_xs().text_color(color::danger()).child(e.clone()));
                }
                if let Some(t) = a.thresholds.get(&id) {
                    sec = sec.child(div().text_xs().text_color(color::text_muted()).child(format!("Otsu on the whole image: {t:.4} (of 0–1)")));
                }
                if let Some(t) = a.tables.get(&id) {
                    sec = sec.child(div().text_xs().text_color(color::text_muted()).child(format!("{} objects on the whole image (level {}) · see Analyze", t.rows.len(), a.level)));
                }
            }
            if let Some(p) = preview {
                sec = sec.child(div().w_full().h(px(200.)).rounded(px(6.)).overflow_hidden().bg(color::canvas()).child(gpui_kit::img(p).size_full().object_fit(gpui_kit::ObjectFit::Contain)));
            }
            if let Some(r) = layer {
                let line = match &r.error {
                    Some(e) => e.clone(),
                    None => format!("In view: level {} · {:.0} ms", r.level, r.ms),
                };
                sec = sec.child(div().text_xs().text_color(if r.error.is_some() { color::danger() } else { color::text_muted() }).child(line));
            }
            Some(sec)
        };
        v_flex().gap(px(theme::XL)).child(header).child(body).children(ports).children(status).into_any_element()
    }
}

impl Render for Inspector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body = self.body(window, cx);
        v_flex()
            .id("pipeline-inspector")
            .size_full()
            .child(h_flex().flex_none().h(px(32.)).px(px(theme::MD)).border_b_1().border_color(color::border()).child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child("INSPECTOR")))
            .child(div().id("inspector-body").flex_1().min_h_0().overflow_y_scroll().p(px(theme::MD)).child(body))
    }
}

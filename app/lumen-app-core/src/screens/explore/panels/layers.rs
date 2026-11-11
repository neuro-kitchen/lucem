//! Layers (Figma's layer tree): everything on the image as one tree, top to bottom as it is
//! drawn: the pipeline's results, then the shapes by class (each shape a row; drawn regions,
//! points and detected objects alike), then the image's channels, each unfolding to the
//! processing steps applied to it. The editor above the tree edits the row selected. Classes live here: a click picks the class the tools draw with, its menu
//! changes its colour; "New class" adds one. Rows fold; the eye shows or hides a layer; a click
//! on an annotation selects it and brings it into view, on its plane.
//!
//! **Combine**: results that are masks and annotation classes can be selected (Shift or Ctrl
//! adds) and combined (union, intersect, subtract, invert, grow, shrink); each makes a new result
//! layer through the Pipeline and leaves the originals as they were. The bar shows when masks are
//! selected; the same commands are in the right-click menu of a mask row.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::menu::{ContextMenuExt as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, img, px, AnyElement, App, ClickEvent, Context, Entity, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _, Window};
use lumen_pipeline::{Op, PortType};

use crate::data::store::{MaskOperation, Store};
use crate::domain::annotation::{Annotation, ClassDoc, Shape};
use crate::domain::command::{Command, MEASUREMENT};
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::screens::explore::canvas::ImageCanvas;
use crate::ui::theme::{self, color};
use crate::ui::widgets::group_digits;

/// Thumbnail of a result at the top level of a group, and the icon / dot of other rows.
const THUMB: f32 = 28.;
const MARK: f32 = 16.;
/// Indent per level of the tree (px).
const INDENT: f32 = 14.;
/// Annotation rows listed per class before "… more".
const MAX_ROWS: usize = 200;
use super::display::{chain_of, CLASS_COLORS};

type OnEye = Box<dyn Fn(&mut App)>;
type OnRowClick = Box<dyn Fn(&ClickEvent, &mut App)>;
type MenuFn = Rc<dyn Fn(PopupMenu, &mut Window, &mut Context<PopupMenu>) -> PopupMenu>;

fn select(store: &Entity<Store>, key: &str, e: &ClickEvent, cx: &mut App) {
    let additive = e.modifiers().shift || e.modifiers().control || e.modifiers().platform;
    let key = key.to_string();
    store.update(cx, |s, cx| s.select_layer(key, additive, cx));
}

/// What a row of the tree shows and does.
struct Row {
    key: String,
    depth: usize,
    /// `Some(open)`: the row folds.
    fold: Option<bool>,
    mark: AnyElement,
    name: String,
    /// Shown after the name, muted (under it on a top-level row).
    kind: String,
    selected: bool,
    /// `Some(shown)`: the row has an eye.
    visible: Option<bool>,
    on_eye: Option<OnEye>,
    on_click: OnRowClick,
    menu: Option<MenuFn>,
}

fn row(store: &Entity<Store>, r: Row) -> AnyElement {
    let fold_key = r.key.clone();
    let st = store.clone();
    let dim = r.visible == Some(false);
    let top = r.depth == 0;
    let label = if top {
        h_flex()
            .flex_1()
            .min_w_0()
            .gap(px(theme::SM))
            .child(div().flex_1().min_w_0().truncate().text_sm().font_weight(gpui_kit::FontWeight::SEMIBOLD).child(r.name))
            .child(div().flex_none().text_xs().text_color(color::text_muted()).child(r.kind))
            .into_any_element()
    } else {
        h_flex()
            .flex_1()
            .min_w_0()
            .gap(px(theme::SM))
            .child(div().flex_shrink(1.0).min_w_0().truncate().text_sm().when(dim, |d| d.text_color(color::text_muted())).child(r.name))
            .child(div().flex_none().text_xs().text_color(color::text_muted()).child(r.kind))
            .into_any_element()
    };
    let eye_key = r.key.clone();
    let base = h_flex()
        .id(SharedString::from(format!("layer-{}", r.key)))
        .h(px(if top { 30. } else { 26. }))
        .when(top, |d| d.mt(px(theme::XS)))
        .pl(px(theme::XS + r.depth as f32 * INDENT))
        .pr(px(theme::XS))
        .gap(px(6.))
        .rounded(px(4.))
        .cursor_pointer()
        .when(r.selected, |d| d.bg(color::accent().opacity(0.3)))
        .when(!r.selected, |d| d.hover(|d| d.bg(color::control())))
        .child(match r.fold {
            Some(open) => Button::new(SharedString::from(format!("layer-fold-{}", r.key)))
                .ghost()
                .xsmall()
                .icon(Icon::new(if open { IconName::ChevronDown } else { IconName::ChevronRight }))
                .on_click(move |_, _, cx| {
                    cx.stop_propagation();
                    st.update(cx, |s, cx| s.toggle_layer_open(&fold_key, cx));
                })
                .into_any_element(),
            None => div().flex_none().w(px(20.)).into_any_element(),
        })
        .child(r.mark)
        .child(label)
        .children(r.visible.zip(r.on_eye).map(|(shown, on_eye)| {
            Button::new(SharedString::from(format!("layer-eye-{eye_key}"))).ghost().xsmall().icon(Icon::new(if shown { IconName::Eye } else { IconName::EyeOff })).on_click(move |_, _, cx| {
                cx.stop_propagation();
                on_eye(cx);
            })
        }))
        .on_click(move |e, _, cx| (r.on_click)(e, cx));
    match r.menu {
        Some(m) => base.context_menu(move |p, w, cx| m(p, w, cx)).into_any_element(),
        None => base.into_any_element(),
    }
}

/// A colour dot.
fn dot(c: [u8; 3]) -> AnyElement {
    div().flex_none().size(px(MARK)).flex().items_center().justify_center().child(div().size(px(10.)).rounded_full().bg(theme::rgb8(c))).into_any_element()
}

fn icon_mark(icon: IconName, tint: Option<[u8; 3]>) -> AnyElement {
    let i = Icon::new(icon).small();
    div().flex_none().size(px(MARK)).flex().items_center().justify_center().child(match tint {
        Some(c) => i.text_color(theme::rgb8(c)),
        None => i.text_color(color::text_muted()),
    }).into_any_element()
}

/// A shape's name and icon in the tree.
fn shape_label(shape: &Shape) -> (&'static str, IconName) {
    match shape {
        Shape::Rect { .. } => ("Rectangle", IconName::Square),
        Shape::Ellipse { .. } => ("Ellipse", IconName::Circle),
        Shape::Polygon { .. } => ("Polygon", IconName::Pentagon),
        Shape::Freehand { .. } => ("Lasso region", IconName::Lasso),
        Shape::Point { .. } => ("Point", IconName::MapPin),
        Shape::Line { .. } => ("Measurement", IconName::Ruler),
        Shape::Stroke { erase: true, .. } => ("Eraser stroke", IconName::Eraser),
        Shape::Stroke { .. } => ("Brush stroke", IconName::Brush),
    }
}

/// Selects annotation `a` (Shift adds) and brings it into view, on its plane.
fn reveal(store: &Entity<Store>, canvas: &Entity<ImageCanvas>, a: &Annotation, additive: bool, cx: &mut App) {
    let (id, image, z) = (a.id.clone(), a.image.clone(), a.z);
    let (x0, y0, x1, y1) = a.shape.bounds();
    store.update(cx, |s, cx| {
        let mut sel = if additive { s.selection.clone() } else { Default::default() };
        if !sel.remove(&id) || !additive {
            sel.insert(id.clone());
        }
        s.select(sel, cx);
        if let (Some(z), Some(mut at)) = (z, s.cursor_of(&image)) {
            at[0] = z;
            s.set_cursor_of(&image, at, cx);
        }
    });
    canvas.update(cx, |c, cx| c.reveal([(x0 + x1) / 2.0, (y0 + y1) / 2.0], cx));
}

/// An annotation's right-click menu: it acts on the whole selection when the annotation is in it,
/// else on the annotation alone.
fn annotation_menu(store: &Entity<Store>, a: &Annotation, s: &Store) -> MenuFn {
    let ids: Vec<String> = if s.selection.contains(&a.id) { s.selection.iter().cloned().collect() } else { vec![a.id.clone()] };
    let classes: Vec<String> = s.doc.classes.iter().filter(|c| c.name != MEASUREMENT).map(|c| c.name.clone()).collect();
    let (store, current) = (store.clone(), a.class.clone());
    Rc::new(move |m: PopupMenu, w: &mut Window, cx: &mut Context<PopupMenu>| {
        let n = ids.len();
        let mut m = m.label(if n == 1 { "Annotation".to_string() } else { format!("{n} selected annotations") });
        {
            let (st, ids, classes, current) = (store.clone(), ids.clone(), classes.clone(), current.clone());
            m = m.submenu("Move to class", w, cx, move |mut sub, _, _| {
                for c in &classes {
                    let (st, ids, c2) = (st.clone(), ids.clone(), c.clone());
                    sub = sub.item(PopupMenuItem::new(c.clone()).checked(n == 1 && *c == current).on_click(move |_, _, cx| st.update(cx, |s, cx| s.move_to_class(ids.clone(), c2.clone(), cx))));
                }
                sub
            });
        }
        let (st, ids) = (store.clone(), ids.clone());
        m.separator().item(PopupMenuItem::new(if n == 1 { "Delete  Del".to_string() } else { format!("Delete {n}  Del") }).icon(Icon::new(IconName::Trash)).on_click(move |_, _, cx| {
            st.update(cx, |s, cx| {
                if s.execute(Command::RemoveAnnotations { ids: ids.clone() }, cx).is_ok() {
                    s.set_status(format!("Deleted {} annotation(s) (Ctrl+Z brings them back)", ids.len()), cx);
                }
            })
        }))
    })
}

/// The Combine commands for the selection with `key` in it, for a row's right-click menu.
fn combine_menu(store: &Entity<Store>, key: String, radius: usize, extra: Option<MenuFn>) -> MenuFn {
    let store = store.clone();
    Rc::new(move |menu: PopupMenu, w: &mut Window, cx: &mut Context<PopupMenu>| {
        let item = |label: String, op: MaskOperation| {
            let (st, key) = (store.clone(), key.clone());
            PopupMenuItem::new(label).on_click(move |_, _, cx| {
                st.update(cx, |s, cx| {
                    if !s.layers_selected.contains(&key) {
                        s.layers_selected.push(key.clone());
                    }
                    s.apply_mask_operation(op.clone(), cx)
                })
            })
        };
        let mut m = menu
            .label("Combine with the selected masks")
            .item(item("Union".into(), MaskOperation::Union))
            .item(item("Intersect".into(), MaskOperation::Intersect))
            .item(item("Subtract the others from it".into(), MaskOperation::Subtract))
            .separator()
            .item(item("Invert".into(), MaskOperation::Invert))
            .item(item(format!("Grow {radius} px"), MaskOperation::Op(Op::Dilate { radius })))
            .item(item(format!("Shrink {radius} px"), MaskOperation::Op(Op::Erode { radius })));
        if let Some(extra) = &extra {
            m = extra(m.separator(), w, cx);
        }
        m
    })
}

pub fn panel(store: &Entity<Store>, canvas: &Entity<ImageCanvas>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let Some(active) = s.doc.active().cloned() else { return div().p(px(theme::MD)).text_sm().text_color(color::text_muted()).child("No image").into_any_element() };
    let image = active.id.clone();
    let open = |key: &str| !s.layers_collapsed.contains(key);
    let mut list = v_flex().id("layers-list").w_full().gap(px(1.));
    let noop = |_: &ClickEvent, _: &mut App| {};
    let radius = s.mask_radius;

    // Channels first: the image itself, what everything else is drawn over
    let key = format!("channels:{image}");
    list = list.child(row(store, Row { key: key.clone(), depth: 0, fold: Some(open(&key)), mark: icon_mark(IconName::Layers, None), name: "Channels".into(), kind: active.channels.len().to_string(), selected: false, visible: None, on_eye: None, on_click: Box::new(noop), menu: None }));
    if open(&key) {
        for (i, c) in active.channels.iter().enumerate().rev() {
            let (st, doc, img_id) = (store.clone(), c.clone(), image.clone());
            let ch_key = format!("channel:{i}");
            let (steps, _) = chain_of(s, &image, i);
            let (st_sel, k_sel) = (store.clone(), ch_key.clone());
            list = list.child(row(
                store,
                Row {
                    key: ch_key.clone(),
                    depth: 1,
                    fold: (!steps.is_empty()).then(|| open(&ch_key)),
                    mark: dot(c.color),
                    name: c.label.clone(),
                    kind: if steps.is_empty() { String::new() } else { format!("{} step(s)", steps.len()) },
                    selected: s.layers_selected.contains(&ch_key),
                    visible: Some(c.visible),
                    on_eye: Some(Box::new(move |cx| {
                        let mut display = doc.clone();
                        display.visible = !display.visible;
                        let image = img_id.clone();
                        st.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetChannel { image, channel: i, display }, cx);
                        });
                    })),
                    on_click: Box::new(move |e, cx| select(&st_sel, &k_sel, e, cx)),
                    menu: None,
                },
            ));
            // Its processing steps, in order (the editor above edits the one selected)
            if !steps.is_empty() && open(&ch_key) {
                for sid in &steps {
                    let Some(n) = s.doc.graph.node(sid) else { continue };
                    let (name, icon, on) = match &n.kind {
                        NodeKind::Op { op, enabled } => (op.title().to_string(), crate::screens::pipeline::catalog::family_icon(op.family()), *enabled),
                        NodeKind::Model { model, enabled, .. } => (model.clone(), IconName::BrainCircuit, *enabled),
                        NodeKind::Script { path, .. } => (std::path::Path::new(path).file_name().map_or(path.clone(), |f| f.to_string_lossy().into_owned()), IconName::SquareTerminal, true),
                        _ => continue,
                    };
                    let key = format!("node:{sid}");
                    let (st, k) = (store.clone(), key.clone());
                    list = list.child(row(
                        store,
                        Row { key: key.clone(), depth: 2, fold: None, mark: icon_mark(icon, None), name, kind: if on { String::new() } else { "off".into() }, selected: s.layers_selected.contains(&key), visible: None, on_eye: None, on_click: Box::new(move |e, cx| select(&st, &k, e, cx)), menu: None },
                    ));
                }
            }
        }
    }
    // Results: on top of the image, as they are drawn
    let views: Vec<_> = s.doc.graph.views().filter(|v| s.doc.graph.images_of(&v.id).contains(&image)).cloned().collect();
    if !views.is_empty() {
        let key = format!("results:{image}");
        list = list.child(row(store, Row { key: key.clone(), depth: 0, fold: Some(open(&key)), mark: icon_mark(IconName::ChartColumn, None), name: "Results".into(), kind: views.len().to_string(), selected: false, visible: None, on_eye: None, on_click: Box::new(noop), menu: None }));
        if open(&key) {
            for v in views.iter().rev() {
                let NodeKind::View { name, style } = &v.kind else { continue };
                let port = s.doc.graph.input(&v.id, 0).and_then(|i| s.doc.graph.node(i)).and_then(|n| n.kind.output());
                let layer = format!("view:{}", v.id);
                let preview = s.analysis.as_ref().and_then(|a| a.previews.get(&v.id).cloned());
                let mark = match preview {
                    Some(p) => div().flex_none().size(px(THUMB)).rounded(px(3.)).overflow_hidden().border_1().border_color(color::border()).child(img(p).size_full()).into_any_element(),
                    None => dot(style.color),
                };
                let (st, id, nm, sty) = (store.clone(), v.id.clone(), name.clone(), style.clone());
                let (st2, key2) = (store.clone(), layer.clone());
                list = list.child(row(
                    store,
                    Row {
                        key: layer.clone(),
                        depth: 1,
                        fold: None,
                        mark,
                        name: name.clone(),
                        kind: port.map_or("—", PortType::name).to_string(),
                        selected: s.layers_selected.contains(&layer),
                        visible: Some(style.visible),
                        on_eye: Some(Box::new(move |cx| {
                            let style = ViewStyle { visible: !sty.visible, ..sty.clone() };
                            let (id, name) = (id.clone(), nm.clone());
                            st.update(cx, |s, cx| {
                                let _ = s.execute(Command::SetView { id, name, style }, cx);
                            });
                        })),
                        on_click: Box::new(move |e, cx| select(&st2, &key2, e, cx)),
                        menu: Some({
                            let remove: MenuFn = {
                                let (st, id) = (store.clone(), v.id.clone());
                                Rc::new(move |m: PopupMenu, _: &mut Window, _: &mut Context<PopupMenu>| {
                                    let (st, id) = (st.clone(), id.clone());
                                    m.item(PopupMenuItem::new("Remove result").icon(Icon::new(IconName::Trash)).on_click(move |_, _, cx| st.update(cx, |s, cx| s.remove_result(id.clone(), cx))))
                                })
                            };
                            if port == Some(PortType::Mask) { combine_menu(store, layer.clone(), radius, Some(remove)) } else { remove }
                        }),
                    },
                ));
            }
        }
    }

    // Annotations by class (every class: they are managed here), each annotation a row
    let all: Vec<&Annotation> = s.doc.annotations_of(&image).collect();
    // The hierarchy: how many cells each region holds (with the regions inside it)
    let cells = s.hist.cells.get(&image).cloned().unwrap_or_default();
    let hier = s.hierarchy(&image);
    let cells_in = |id: &str| hier.in_subtree.get(id).copied().unwrap_or(0);
    let key = format!("annotations:{image}");
    list = list.child(row(store, Row { key: key.clone(), depth: 0, fold: Some(open(&key)), mark: icon_mark(IconName::Shapes, None), name: "Shapes".into(), kind: group_digits(all.len()), selected: false, visible: None, on_eye: None, on_click: Box::new(noop), menu: None }));
    if open(&key) {
        // Classes with annotations first; the empty ones folded under one row (closed by default),
        // Measurement (the rulers' class) only when it has rulers
        let count = |name: &str| all.iter().filter(|a| a.class == name).count();
        let mut ordered: Vec<&ClassDoc> = s.doc.classes.iter().filter(|c| count(&c.name) > 0).collect();
        let empties: Vec<&ClassDoc> = s.doc.classes.iter().filter(|c| count(&c.name) == 0 && c.name != "Measurement").collect();
        let show_empty_key = format!("show-empty:{image}");
        let show_empty = s.layers_collapsed.contains(&show_empty_key);
        if !empties.is_empty() {
            let names: Vec<&str> = empties.iter().map(|c| c.name.as_str()).collect();
            let _ = names;
            ordered.extend(empties.iter().copied());
        }
        let mut fold_shown = false;
        for c in ordered {
            let of: Vec<&&Annotation> = all.iter().filter(|a| a.class == c.name).collect();
            if of.is_empty() && !fold_shown {
                fold_shown = true;
                list = list.child(row(
                    store,
                    Row {
                        key: show_empty_key.clone(),
                        depth: 1,
                        fold: Some(show_empty),
                        mark: icon_mark(IconName::Shapes, None),
                        name: format!("{} empty class(es)", empties.len()),
                        kind: if show_empty { String::new() } else { empties.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ") },
                        selected: false,
                        visible: None,
                        on_eye: None,
                        on_click: Box::new(noop),
                        menu: Some({
                            let st = store.clone();
                            Rc::new(move |m: PopupMenu, _: &mut Window, _: &mut Context<PopupMenu>| {
                                let st = st.clone();
                                m.item(PopupMenuItem::new("Delete the empty classes").icon(Icon::new(IconName::Trash)).on_click(move |_, _, cx| st.update(cx, |s, cx| s.delete_empty_classes(cx))))
                            })
                        }),
                    },
                ));
            }
            if of.is_empty() && !show_empty {
                break;
            }
            let layer = format!("class:{}", c.name);
            let drawing = s.tool.class == c.name;
            let (st, cls) = (store.clone(), c.clone());
            let (st2, key2, name2) = (store.clone(), layer.clone(), c.name.clone());
            // The class's own menu items: colour, select its annotations
            let others: Vec<String> = s.doc.classes.iter().filter(|o| o.name != c.name && o.name != MEASUREMENT).map(|o| o.name.clone()).collect();
            let deletable = c.name != MEASUREMENT && !others.is_empty();
            let extra: MenuFn = {
                let (st, cls, ids) = (store.clone(), c.clone(), of.iter().map(|a| a.id.clone()).collect::<Vec<_>>());
                let others = others.clone();
                Rc::new(move |m: PopupMenu, w: &mut Window, cx: &mut Context<PopupMenu>| {
                    let mut m = m.label(format!("Class {}", cls.name));
                    {
                        let (st, ids) = (st.clone(), ids.clone());
                        m = m.item(PopupMenuItem::new(format!("Select its {} annotation(s)", ids.len())).disabled(ids.is_empty()).on_click(move |_, _, cx| st.update(cx, |s, cx| s.select(ids.clone(), cx))));
                    }
                    for (cname, rgb) in CLASS_COLORS {
                        let (st, cls) = (st.clone(), cls.clone());
                        m = m.item(PopupMenuItem::new(format!("Colour: {cname}")).checked(cls.color == rgb).on_click(move |_, _, cx| {
                            let class = ClassDoc { color: rgb, ..cls.clone() };
                            st.update(cx, |s, cx| {
                                let _ = s.execute(Command::SetClass { class }, cx);
                            });
                        }));
                    }
                    {
                        let (st, cls) = (st.clone(), cls.clone());
                        m = m.separator().submenu(format!("Opacity: {:.0} %", cls.opacity * 100.0), w, cx, move |mut sub, _, _| {
                            for pct in [100u32, 75, 50, 25] {
                                let (st, cls) = (st.clone(), cls.clone());
                                let v = pct as f32 / 100.0;
                                sub = sub.item(PopupMenuItem::new(format!("{pct} %")).checked((cls.opacity - v).abs() < 0.01).on_click(move |_, _, cx| {
                                    let class = ClassDoc { opacity: v, ..cls.clone() };
                                    st.update(cx, |s, cx| {
                                        let _ = s.execute(Command::SetClass { class }, cx);
                                    });
                                }));
                            }
                            sub
                        });
                    }
                    // Deleting: with its annotations, or keeping them in another class
                    m = m.separator();
                    if !ids.is_empty() && deletable {
                        let (st, name, others) = (st.clone(), cls.name.clone(), others.clone());
                        m = m.submenu(format!("Delete class, keep its {} annotation(s) as", ids.len()), w, cx, move |mut sub, _, _| {
                            for to in &others {
                                let (st, name, to) = (st.clone(), name.clone(), to.clone());
                                sub = sub.item(PopupMenuItem::new(to.clone()).on_click(move |_, _, cx| st.update(cx, |s, cx| s.delete_class(name.clone(), Some(to.clone()), cx))));
                            }
                            sub
                        });
                    }
                    let label = if ids.is_empty() { "Delete class".to_string() } else { format!("Delete class and its {} annotation(s)", ids.len()) };
                    let (st2, name) = (st.clone(), cls.name.clone());
                    m.item(PopupMenuItem::new(label).icon(Icon::new(IconName::Trash)).disabled(!deletable).on_click(move |_, _, cx| st2.update(cx, |s, cx| s.delete_class(name.clone(), None, cx))))
                })
            };
            list = list.child(row(
                store,
                Row {
                    key: layer.clone(),
                    depth: 1,
                    fold: (!of.is_empty()).then(|| open(&layer)),
                    mark: dot(c.color),
                    name: c.name.clone(),
                    kind: if drawing { format!("{} · drawing", group_digits(of.len())) } else { group_digits(of.len()) },
                    selected: s.layers_selected.contains(&layer),
                    visible: Some(c.visible),
                    on_eye: Some(Box::new(move |cx| {
                        let class = ClassDoc { visible: !cls.visible, ..cls.clone() };
                        st.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetClass { class }, cx);
                        });
                    })),
                    // A click picks the class the tools draw with, and selects it (for Combine)
                    on_click: Box::new(move |e, cx| {
                        let name = name2.clone();
                        st2.update(cx, |s, cx| s.update_tool(|t| t.class = name, cx));
                        select(&st2, &key2, e, cx)
                    }),
                    menu: Some(combine_menu(store, layer.clone(), radius, Some(extra))),
                },
            ));
            if of.is_empty() || !open(&layer) {
                continue;
            }
            for a in of.iter().rev().take(MAX_ROWS) {
                let (label, icon) = shape_label(&a.shape);
                let n_cells = if a.shape.is_region() && !cells.is_empty() { cells_in(&a.id) } else { 0 };
                let plane = [a.z.map(|z| format!("z {z}")), (n_cells > 0).then(|| format!("{} cells", group_digits(n_cells)))].into_iter().flatten().collect::<Vec<_>>().join(" · ");
                let (st, cv, ann) = (store.clone(), canvas.clone(), (**a).clone());
                list = list.child(row(
                    store,
                    Row {
                        key: format!("ann:{}", a.id),
                        depth: 2,
                        fold: None,
                        mark: icon_mark(icon, Some(c.color)),
                        name: if a.note.is_empty() { format!("{label} {}", a.id) } else { a.note.clone() },
                        kind: plane,
                        selected: s.selection.contains(&a.id),
                        visible: None,
                        on_eye: None,
                        on_click: Box::new(move |e, cx| {
                            let additive = e.modifiers().shift || e.modifiers().control || e.modifiers().platform;
                            reveal(&st, &cv, &ann, additive, cx)
                        }),
                        menu: Some(annotation_menu(store, a, s)),
                    },
                ));
            }
            if of.len() > MAX_ROWS {
                list = list.child(div().pl(px(theme::XS + 2.0 * INDENT + 26.0)).py(px(2.)).text_xs().text_color(color::text_muted()).child(format!("… {} more (newest shown; all in the Analyze workspace)", group_digits(of.len() - MAX_ROWS))));
            }
        }
        // New class
        let st = store.clone();
        let n = s.doc.classes.len();
        list = list.child(
            div()
                .id("layers-new-class")
                .h(px(26.))
                .pl(px(theme::XS + INDENT + 26.0))
                .flex()
                .items_center()
                .gap(px(theme::XS))
                .rounded(px(4.))
                .cursor_pointer()
                .hover(|d| d.bg(color::control()))
                .text_sm()
                .text_color(color::text_muted())
                .child(Icon::new(IconName::Plus).small())
                .child("New class")
                .on_click(move |_, _, cx| {
                    st.update(cx, |s, cx| {
                        let name = (1..).map(|i| format!("Class {i}")).find(|c| s.doc.class(c).is_none()).unwrap_or_default();
                        let class = ClassDoc { name: name.clone(), color: CLASS_COLORS[n % CLASS_COLORS.len()].1, visible: true, opacity: 1.0 };
                        if s.execute(Command::SetClass { class }, cx).is_ok() {
                            s.update_tool(|t| t.class = name, cx);
                        }
                    })
                }),
        );
    }

    // Detections: the cells and tiles found, by class (thousands: drawn as a raster, listed here
    // by class; each one is in the Analyze workspace, under the region it belongs to)
    let tiles = s.hist.tiles.get(&image).cloned().unwrap_or_default();
    if !cells.is_empty() || !tiles.is_empty() {
        let key = format!("detections:{image}");
        let shown = s.hist.show;
        let st = store.clone();
        list = list.child(row(
            store,
            Row {
                key: key.clone(),
                depth: 0,
                fold: Some(open(&key)),
                mark: icon_mark(IconName::Sparkles, None),
                name: "Detections".into(),
                kind: group_digits(cells.len() + tiles.len()),
                selected: false,
                visible: Some(shown),
                on_eye: Some(Box::new(move |cx| {
                    st.update(cx, |s, cx| {
                        s.hist.show = !s.hist.show;
                        s.hist.rev += 1;
                        cx.notify();
                    })
                })),
                on_click: Box::new(noop),
                menu: None,
            },
        ));
        if open(&key) {
            for (what, objects, icon) in [("Cells", &cells, IconName::Sparkles), ("Tiles", &tiles, IconName::Grid3x3)] {
                if objects.is_empty() {
                    continue;
                }
                let mut by_class: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
                for o in objects.iter() {
                    *by_class.entry(o.class.clone().unwrap_or_else(|| "Unclassified".into())).or_default() += 1;
                }
                let gkey = format!("detections:{image}:{what}");
                list = list.child(row(store, Row { key: gkey.clone(), depth: 1, fold: Some(open(&gkey)), mark: icon_mark(icon, None), name: what.into(), kind: group_digits(objects.len()), selected: false, visible: None, on_eye: None, on_click: Box::new(noop), menu: None }));
                if open(&gkey) {
                    for (class, n) in by_class {
                        let color = s.doc.class(&class).map(|c| c.color).unwrap_or([90, 200, 255]);
                        list = list.child(row(store, Row { key: format!("{gkey}:{class}"), depth: 2, fold: None, mark: dot(color), name: class, kind: group_digits(n), selected: false, visible: None, on_eye: None, on_click: Box::new(noop), menu: None }));
                    }
                }
            }
        }
    }
    finish(store, s, list)
}

/// The tree (the Combine commands are in Shapes ▸ Combine and the options bar, not in a panel).
fn finish(_store: &Entity<Store>, _s: &Store, list: gpui_kit::Stateful<gpui_kit::Div>) -> AnyElement {
    v_flex().size_full().child(list.flex_1().min_h_0().overflow_y_scroll().p(px(theme::SM))).into_any_element()
}

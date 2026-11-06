//! The Explore workspace (`.tasks/2026-11-06_ui-ux-redesign.md` §4): the tool options bar on
//! top (Photoshop), one icon tool bar on the left, the image canvas, and one sidebar on the right
//! of stacked sections — the image, its channels, the results of the pipeline, the annotations
//! and notes. Tab hides the sidebar.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _,
    Styled as _, Subscription, Window,
};

use crate::data::results::COLORMAPS;
use crate::data::store::{AppEvent, Store};
use crate::domain::annotation::Shape;
use crate::domain::command::Command;
use crate::domain::graph::NodeKind;
use crate::domain::project::{ChannelDoc, NoteDoc};
use crate::domain::tool::Tool;
use crate::navigation::route::Workspace;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{divider_v, format_area, format_length, list_row, row, stepper, swatch, EmptyState, MenuSelect, Section};

use super::canvas::ImageCanvas;

/// Colours a channel cycles through when its swatch is clicked.
const CHANNEL_COLORS: [[u8; 3]; 8] = [[64, 128, 255], [0, 255, 255], [64, 255, 96], [255, 255, 0], [255, 128, 0], [255, 64, 64], [255, 0, 255], [255, 255, 255]];

pub struct ExploreView {
    store: Entity<Store>,
    pub canvas: Entity<ImageCanvas>,
    note: Entity<InputState>,
    _subs: Vec<Subscription>,
}

impl ExploreView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let canvas = cx.new(|cx| ImageCanvas::new(store.clone(), cx));
        let note = cx.new(|cx| InputState::new(window, cx).placeholder("Add a note…"));
        let subs = vec![
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if !matches!(e, AppEvent::Status) {
                    cx.notify();
                }
            }),
            cx.subscribe_in(&note, window, |this, input, e: &InputEvent, window, cx| {
                if matches!(e, InputEvent::PressEnter { .. }) {
                    this.add_note(input, window, cx);
                }
            }),
        ];
        Self { store, canvas, note, _subs: subs }
    }

    fn add_note(&mut self, input: &Entity<InputState>, window: &mut Window, cx: &mut Context<Self>) {
        let text = input.read(cx).value().trim().to_string();
        if text.is_empty() {
            return;
        }
        let image = self.store.read(cx).doc.active_image.clone();
        let at = image.as_ref().map(|_| {
            let c = self.canvas.read(cx).center;
            [c.0, c.1]
        });
        let note = NoteDoc { id: String::new(), text, image, at };
        let ok = self.store.update(cx, |s, cx| s.execute(Command::SetNote { note }, cx)).is_ok();
        if ok {
            input.update(cx, |i, cx| i.set_value("", window, cx));
        }
    }

    // ------------------------------------------------------------------------
    // Tool bar and options bar
    // ------------------------------------------------------------------------

    fn tool_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let current = self.store.read(cx).tool.tool;
        let mut bar = v_flex().id("tool-bar").flex_none().w(px(theme::TOOLBAR_W)).h_full().py(px(theme::SM)).gap(px(2.)).items_center().bg(color::chrome()).border_r_1().border_color(color::border());
        for slot in Tool::BAR {
            match slot {
                Some(t) => {
                    let store = self.store.clone();
                    let canvas = self.canvas.clone();
                    bar = bar.child(
                        Button::new(SharedString::from(format!("tool-{}", t.key())))
                            .ghost()
                            .icon(Icon::new(t.icon()))
                            .selected(t == current)
                            .tooltip(format!("{} ({})  —  {}", t.name(), t.key().to_uppercase(), t.hint()))
                            .on_click(move |_, window, cx| {
                                store.update(cx, |s, cx| s.set_tool(t, cx));
                                let f = canvas.read(cx).focus.clone();
                                window.focus(&f, cx);
                            }),
                    )
                }
                None => bar = bar.child(div().w(px(24.)).h(px(1.)).my(px(theme::XS)).bg(color::border())),
            }
        }
        bar
    }

    fn options_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let s = self.store.read(cx);
        let t = s.tool.clone();
        let classes: Vec<SharedString> = s.doc.classes.iter().filter(|c| c.name != "Measurement").map(|c| SharedString::from(c.name.clone())).collect();
        let class_names: Vec<String> = classes.iter().map(|c| c.to_string()).collect();
        let class_color = s.doc.class(&t.class).map_or([255, 255, 255], |c| c.color);
        let selected = s.selection.len();
        let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
        let zoom = self.canvas.read(cx).zoom;
        let store = self.store.clone();
        let mut bar = h_flex().id("options-bar").flex_none().h(px(theme::OPTIONS_BAR_H)).px(px(theme::MD)).gap(px(theme::SM)).items_center().bg(color::panel()).border_b_1().border_color(color::border());
        bar = bar.child(Icon::new(t.tool.icon()).small().text_color(color::text_muted())).child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(t.tool.name())).child(divider_v());

        let class_picker = |id: &'static str| {
            let store = store.clone();
            let names = class_names.clone();
            h_flex().gap(px(theme::XS)).child(swatch(class_color, 12.)).child(div().text_sm().text_color(color::text_muted()).child("Class")).child(div().w(px(130.)).child(MenuSelect::new(id, classes.clone(), class_names.iter().position(|c| *c == t.class), move |i, _, cx| {
                let name = names[i].clone();
                store.update(cx, |s, cx| s.update_tool(|t| t.class = name, cx));
            })))
        };
        match t.tool {
            Tool::Brush | Tool::Eraser => {
                let (st1, st2) = (store.clone(), store.clone());
                let size = if t.brush_in_image {
                    format!("{} ", format_length(t.brush_image_px * 2.0, um))
                } else {
                    format!("{:.0} px", t.brush_px * 2.0)
                };
                bar = bar
                    .child(class_picker("options-class"))
                    .child(divider_v())
                    .child(div().text_sm().text_color(color::text_muted()).child("Size"))
                    .child(stepper("brush-size", size, move |d, _, cx| {
                        st1.update(cx, |s, cx| {
                            s.update_tool(
                                |t| {
                                    if t.brush_in_image {
                                        t.brush_image_px = (t.brush_image_px * if d > 0.0 { 1.25 } else { 0.8 }).clamp(1.0, 5000.0);
                                    } else {
                                        t.brush_px = crate::domain::tool::step_brush(t.brush_px, d);
                                    }
                                },
                                cx,
                            )
                        })
                    }))
                    .child(Checkbox::new("brush-in-image").checked(t.brush_in_image).label("Fixed on the image").on_click(move |on: &bool, _, cx| {
                        let on = *on;
                        st2.update(cx, |s, cx| {
                            s.update_tool(
                                |t| {
                                    if on {
                                        t.brush_image_px = t.brush_px as f64 / zoom;
                                    }
                                    t.brush_in_image = on;
                                },
                                cx,
                            )
                        });
                    }))
                    .child(div().text_xs().text_color(color::text_muted()).child("[ ] resize"));
            }
            Tool::Rect | Tool::Ellipse | Tool::Lasso | Tool::Point => bar = bar.child(class_picker("options-class")),
            Tool::Polygon => bar = bar.child(class_picker("options-class")).child(div().text_xs().text_color(color::text_muted()).child("Click corners · Enter or double-click closes · Esc cancels")),
            Tool::Ruler => bar = bar.child(div().text_sm().text_color(color::text_muted()).child(if um.is_some() { "Drag to measure (µm from the image's pixel size)" } else { "Drag to measure (px: the image has no pixel size)" })),
            Tool::Select => {
                let st = store.clone();
                bar = bar.child(div().text_sm().text_color(color::text_muted()).child(if selected == 0 { "Click an annotation".to_string() } else { format!("{selected} selected") })).when(selected > 0, |b| {
                    b.child(Button::new("options-delete").ghost().small().icon(IconName::Trash).label("Delete").on_click(move |_, _, cx| st.update(cx, |s, cx| s.delete_selection(cx))))
                });
            }
            Tool::Zoom => bar = bar.child(div().text_sm().text_color(color::text_muted()).child("Click to zoom in · Alt+click to zoom out")),
            Tool::Hand | Tool::Probe => bar = bar.child(div().text_sm().text_color(color::text_muted()).child(t.tool.hint())),
        }
        let (canvas1, canvas2) = (self.canvas.clone(), self.canvas.clone());
        let (st_r, st_v) = (store.clone(), store.clone());
        bar.child(div().flex_1())
            .child(Button::new("options-results").ghost().small().icon(IconName::Layers).label("Results").selected(t.show_results).tooltip("Draw the pipeline's results over the image").on_click(move |_, _, cx| {
                st_v.update(cx, |s, cx| s.update_tool(|t| t.show_results = !t.show_results, cx))
            }))
            .child(Button::new("options-rulers").ghost().small().icon(IconName::Ruler).selected(t.rulers).tooltip("Rulers (Ctrl+R)").on_click(move |_, _, cx| st_r.update(cx, |s, cx| s.update_tool(|t| t.rulers = !t.rulers, cx))))
            .child(divider_v())
            .child(Button::new("options-fit").ghost().small().icon(IconName::Maximize).tooltip("Fit (F · Ctrl+0)").on_click(move |_, _, cx| canvas1.update(cx, |c, cx| c.fit(cx))))
            .child(Button::new("options-100").ghost().small().label("1:1").tooltip("Actual pixels (Ctrl+1)").on_click(move |_, _, cx| canvas2.update(cx, |c, cx| c.actual_size(cx))))
    }

    // ------------------------------------------------------------------------
    // Sidebar
    // ------------------------------------------------------------------------

    fn image_section(&self, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let Some(img) = s.doc.active() else { return div().into_any_element() };
        let py = s.pyramids.get(&img.id).cloned();
        let names: Vec<SharedString> = s.doc.images.iter().map(|i| SharedString::from(i.name.clone())).collect();
        let ids: Vec<String> = s.doc.images.iter().map(|i| i.id.clone()).collect();
        let current = ids.iter().position(|i| *i == img.id);
        let store = self.store.clone();
        let mut sec = Section::new("Image");
        if ids.len() > 1 {
            sec = sec.child(MenuSelect::new("image-pick", names, current, move |i, _, cx| {
                let id = ids[i].clone();
                store.update(cx, |s, cx| {
                    let _ = s.execute(Command::SelectImage { id }, cx);
                });
            }).full_width());
        } else {
            sec = sec.child(div().text_sm().font_weight(FontWeight::SEMIBOLD).truncate().child(img.name.clone()));
        }
        if let Some(py) = py {
            sec = sec
                .child(row("Size", div().text_sm().child(format!("{} × {} px", py.width(), py.height()))))
                .child(row("Pixel", div().text_sm().child(py.pixel_size_um.map_or("unknown".to_string(), |u| format!("{u:.4} µm")))))
                .child(row("Levels", div().text_sm().child(format!("{} · {}", py.levels.len(), py.dtype))));
        } else {
            sec = sec.child(div().text_xs().text_color(color::text_muted()).child("Opening…"));
        }
        sec.into_any_element()
    }

    fn channels_section(&self, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let Some(img) = s.doc.active() else { return div().into_any_element() };
        let id = img.id.clone();
        let py = s.pyramids.get(&id).cloned();
        let mut sec = Section::new("Channels");
        for (c, ch) in img.channels.iter().enumerate() {
            let range = py.as_ref().and_then(|p| p.channels.get(c)).map_or((0.0, 1.0), |i| i.range);
            let step = ((range.1 - range.0) * 0.01).max(1e-6);
            let set = {
                let (store, id, ch) = (self.store.clone(), id.clone(), ch.clone());
                Rc::new(move |f: &dyn Fn(&mut ChannelDoc), cx: &mut App| {
                    let mut d = ch.clone();
                    f(&mut d);
                    let id = id.clone();
                    store.update(cx, |s, cx| {
                        let _ = s.execute(Command::SetChannel { image: id, channel: c, display: d }, cx);
                    });
                })
            };
            let (s1, s2, s3, s4, s5) = (set.clone(), set.clone(), set.clone(), set.clone(), set);
            let auto = {
                let (store, id, ch, py) = (self.store.clone(), id.clone(), ch.clone(), py.clone());
                move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| {
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
            let color_now = ch.color;
            sec = sec.child(
                v_flex()
                    .w_full()
                    .gap(px(theme::XS))
                    .pb(px(theme::SM))
                    .border_b_1()
                    .border_color(color::border())
                    .child(
                        h_flex()
                            .gap(px(theme::SM))
                            .child(Checkbox::new(SharedString::from(format!("ch-visible-{c}"))).checked(ch.visible).on_click(move |on: &bool, _, cx| {
                                let on = *on;
                                s1(&move |d: &mut ChannelDoc| d.visible = on, cx)
                            }))
                            .child(
                                div()
                                    .id(SharedString::from(format!("ch-color-{c}")))
                                    .cursor_pointer()
                                    .child(swatch(ch.color, 14.))
                                    .on_click(move |_, _, cx| {
                                        let next = CHANNEL_COLORS.iter().position(|k| *k == color_now).map_or(0, |i| (i + 1) % CHANNEL_COLORS.len());
                                        s2(&move |d: &mut ChannelDoc| d.color = CHANNEL_COLORS[next], cx)
                                    }),
                            )
                            .child(div().flex_1().min_w_0().truncate().text_sm().child(ch.label.clone()))
                            .child(Button::new(SharedString::from(format!("ch-auto-{c}"))).ghost().xsmall().label("Auto").tooltip("Window from the 0.1–99.8 % percentiles").on_click(auto)),
                    )
                    .child(row("Black", stepper(&format!("ch-min-{c}"), trim(ch.min), move |d, _, cx| s3(&move |x: &mut ChannelDoc| x.min = (x.min + d * step).min(x.max - step), cx))))
                    .child(row("White", stepper(&format!("ch-max-{c}"), trim(ch.max), move |d, _, cx| s4(&move |x: &mut ChannelDoc| x.max = (x.max + d * step * 2.0).max(x.min + step), cx))))
                    .child(row("Gamma", stepper(&format!("ch-gamma-{c}"), format!("{:.2}", ch.gamma), move |d, _, cx| s5(&move |x: &mut ChannelDoc| x.gamma = (x.gamma * if d > 0.0 { 1.1 } else { 1.0 / 1.1 }).clamp(0.1, 5.0), cx))))
                    ,
            );
        }
        sec.into_any_element()
    }

    fn results_section(&self, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let Some(img) = s.doc.active_image.clone() else { return div().into_any_element() };
        let views: Vec<_> = s.doc.graph.views().filter(|v| s.doc.graph.images_of(&v.id).contains(&img)).cloned().collect();
        let store = self.store.clone();
        let go = Button::new("results-pipeline").ghost().xsmall().icon(IconName::Workflow).tooltip("Open the Pipeline").on_click(move |_, _, cx| store.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx)));
        let mut sec = Section::new("Results").extra(go);
        if views.is_empty() {
            sec = sec.child(div().text_xs().text_color(color::text_muted()).child("Apply an operation (Operations menu, or Ctrl+K) and its result shows here and over the image."));
        }
        for v in views {
            let NodeKind::View { name, style } = &v.kind else { continue };
            let r = s.results.get(&v.id);
            let status = match r {
                Some(r) if r.error.is_some() => "error".to_string(),
                Some(r) if r.table.is_some() => format!("{} objects", r.table.as_ref().map_or(0, |t| t.len())),
                Some(r) => format!("{:.0} ms", r.ms),
                None if s.computing => "…".into(),
                None => String::new(),
            };
            let err = r.and_then(|r| r.error.clone());
            let (st1, st2, st3) = (self.store.clone(), self.store.clone(), self.store.clone());
            let (id1, id2, id3) = (v.id.clone(), v.id.clone(), v.id.clone());
            let (n1, n2, n3) = (name.clone(), name.clone(), name.clone());
            let (sty1, sty2, sty3) = (style.clone(), style.clone(), style.clone());
            let lut_idx = COLORMAPS.iter().position(|m| *m == style.colormap);
            sec = sec.child(
                v_flex()
                    .w_full()
                    .gap(px(theme::XS))
                    .child(
                        h_flex()
                            .gap(px(theme::SM))
                            .child(Button::new(SharedString::from(format!("view-eye-{}", v.id))).ghost().xsmall().icon(if style.visible { IconName::Eye } else { IconName::EyeOff }).on_click(move |_, _, cx| {
                                let style = crate::domain::graph::ViewStyle { visible: !sty1.visible, ..sty1.clone() };
                                st1.update(cx, |s, cx| {
                                    let _ = s.execute(Command::SetView { id: id1.clone(), name: n1.clone(), style }, cx);
                                });
                            }))
                            .child(div().flex_1().min_w_0().truncate().text_sm().child(name.clone()))
                            .child(div().text_xs().text_color(if err.is_some() { color::danger() } else { color::text_muted() }).child(status)),
                    )
                    .child(row("Opacity", stepper(&format!("view-opacity-{}", v.id), format!("{:.0} %", style.opacity * 100.0), move |d, _, cx| {
                        let style = crate::domain::graph::ViewStyle { opacity: (sty2.opacity + d * 0.1).clamp(0.0, 1.0), ..sty2.clone() };
                        st2.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetView { id: id2.clone(), name: n2.clone(), style }, cx);
                        });
                    })))
                    .child(row(
                        "Colours",
                        MenuSelect::new(SharedString::from(format!("view-lut-{}", v.id)), COLORMAPS.iter().map(|m| SharedString::from(*m)).collect(), lut_idx, move |i, _, cx| {
                            let style = crate::domain::graph::ViewStyle { colormap: COLORMAPS[i].to_string(), ..sty3.clone() };
                            st3.update(cx, |s, cx| {
                                let _ = s.execute(Command::SetView { id: id3.clone(), name: n3.clone(), style }, cx);
                            });
                        }),
                    ))
                    .children(err.map(|e| div().text_xs().text_color(color::danger()).child(e))),
            );
        }
        sec.into_any_element()
    }

    fn annotations_section(&self, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        let Some(img) = s.doc.active_image.clone() else { return div().into_any_element() };
        let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
        let mut sec = Section::new("Annotations");
        // Classes: colour, count, visibility; a click makes it the drawing class
        let mut classes = v_flex().w_full().gap(px(2.));
        for c in &s.doc.classes {
            let n = s.doc.annotations_of(&img).filter(|a| a.class == c.name).count();
            let active = s.tool.class == c.name;
            let (st1, st2, cls1, cls2) = (self.store.clone(), self.store.clone(), c.clone(), c.name.clone());
            classes = classes.child(
                h_flex()
                    .id(SharedString::from(format!("class-{}", c.name)))
                    .h(px(26.))
                    .px(px(theme::XS))
                    .gap(px(theme::SM))
                    .rounded(px(4.))
                    .cursor_pointer()
                    .when(active, |d| d.bg(color::accent().opacity(0.25)))
                    .hover(|d| d.bg(color::control()))
                    .child(Button::new(SharedString::from(format!("class-eye-{}", c.name))).ghost().xsmall().icon(if c.visible { IconName::Eye } else { IconName::EyeOff }).on_click(move |_, _, cx| {
                        cx.stop_propagation();
                        let class = crate::domain::annotation::ClassDoc { visible: !cls1.visible, ..cls1.clone() };
                        st1.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetClass { class }, cx);
                        });
                    }))
                    .child(swatch(c.color, 12.))
                    .child(div().flex_1().text_sm().child(c.name.clone()))
                    .child(div().text_xs().text_color(color::text_muted()).child(n.to_string()))
                    .on_click(move |_, _, cx| {
                        let name = cls2.clone();
                        st2.update(cx, |s, cx| s.update_tool(|t| t.class = name, cx));
                    }),
            );
        }
        sec = sec.child(classes);
        // The annotations themselves
        let list: Vec<_> = s.doc.annotations_of(&img).collect();
        if list.is_empty() {
            sec = sec.child(div().text_xs().text_color(color::text_muted()).child("Draw with the tools on the left (M, O, P, L, N, B, R)."));
        }
        let mut rows = v_flex().w_full().gap(px(1.));
        for a in list.iter().rev().take(150) {
            let measure = match (&a.shape, a.shape.area()) {
                (_, Some(area)) => format_area(area, um),
                (Shape::Line { .. }, _) | (Shape::Stroke { .. }, _) => a.shape.length().map_or(String::new(), |l| format_length(l, um)),
                _ => String::new(),
            };
            let selected = s.selection.contains(&a.id);
            let (store, canvas, id) = (self.store.clone(), self.canvas.clone(), a.id.clone());
            let (bx0, by0, bx1, by1) = a.shape.bounds();
            let centre = [(bx0 + bx1) / 2.0, (by0 + by1) / 2.0];
            rows = rows.child(list_row(SharedString::from(format!("ann-{}", a.id)), None, format!("{} · {}", a.shape.kind(), a.class), measure, selected, cx).on_click(move |_, _, cx| {
                let id = id.clone();
                store.update(cx, |s, cx| s.select([id], cx));
                canvas.update(cx, |c, cx| c.reveal(centre, cx));
            }));
        }
        sec = sec.child(rows);
        // The selection: its class
        if s.selection.len() == 1 {
            if let Some(a) = s.doc.annotations.iter().find(|a| s.selection.contains(&a.id)).cloned() {
                let names: Vec<SharedString> = s.doc.classes.iter().map(|c| SharedString::from(c.name.clone())).collect();
                let raw: Vec<String> = s.doc.classes.iter().map(|c| c.name.clone()).collect();
                let current = raw.iter().position(|c| *c == a.class);
                let store = self.store.clone();
                sec = sec.child(row(
                    "Class",
                    MenuSelect::new("ann-class", names, current, move |i, _, cx| {
                        let annotation = crate::domain::annotation::Annotation { class: raw[i].clone(), ..a.clone() };
                        store.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetAnnotation { annotation }, cx);
                        });
                    }),
                ));
            }
        }
        sec.into_any_element()
    }

    fn notes_section(&self, cx: &mut Context<Self>) -> AnyElement {
        let notes = self.store.read(cx).doc.notes.clone();
        let mut sec = Section::new("Notes").child(Input::new(&self.note).small());
        for n in notes.iter().rev() {
            let (store, canvas, id, at) = (self.store.clone(), self.canvas.clone(), n.id.clone(), n.at);
            let store2 = self.store.clone();
            let id2 = n.id.clone();
            sec = sec.child(
                h_flex()
                    .id(SharedString::from(format!("note-{}", n.id)))
                    .w_full()
                    .gap(px(theme::SM))
                    .p(px(theme::XS))
                    .rounded(px(4.))
                    .hover(|d| d.bg(color::control()))
                    .cursor_pointer()
                    .child(Icon::new(IconName::StickyNote).small().text_color(color::text_muted()))
                    .child(div().flex_1().min_w_0().text_sm().child(n.text.clone()))
                    .child(Button::new(SharedString::from(format!("note-del-{}", n.id))).ghost().xsmall().icon(IconName::X).on_click(move |_, _, cx| {
                        cx.stop_propagation();
                        store2.update(cx, |s, cx| {
                            let _ = s.execute(Command::RemoveNote { id: id2.clone() }, cx);
                        });
                    }))
                    .on_click(move |_, _, cx| {
                        if let Some(p) = at {
                            canvas.update(cx, |c, cx| c.reveal(p, cx));
                        }
                        let _ = (&store, &id);
                    }),
            );
        }
        sec.into_any_element()
    }

    fn sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let sections = [self.image_section(cx), self.channels_section(cx), self.results_section(cx), self.annotations_section(cx), self.notes_section(cx)];
        v_flex()
            .id("explore-sidebar")
            .flex_none()
            .w(px(theme::SIDEBAR_W))
            .h_full()
            .bg(color::panel())
            .border_l_1()
            .border_color(color::border())
            .overflow_y_scroll()
            .p(px(theme::MD))
            .gap(px(theme::XL))
            .children(sections)
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

impl Render for ExploreView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let s = self.store.read(cx);
        if s.doc.images.is_empty() && s.opening.is_empty() {
            return v_flex().size_full().child(EmptyState::new(IconName::Microscope, "No image yet", "Open an image or an OME-Zarr container from the File menu (Ctrl+O), or drop one on the window.")).into_any_element();
        }
        let sidebar_open = s.prefs.sidebar_open;
        v_flex()
            .size_full()
            .child(self.options_bar(cx))
            .child(h_flex().flex_1().min_h_0().w_full().child(self.tool_bar(cx)).child(div().flex_1().min_w_0().h_full().child(self.canvas.clone())).when(sidebar_open, |d| d.child(self.sidebar(cx))))
            .into_any_element()
    }
}

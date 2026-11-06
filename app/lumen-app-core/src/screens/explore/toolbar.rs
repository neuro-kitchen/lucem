//! The image tools: a floating tool bar over the canvas (Figma / Affinity style: no extra strip of
//! chrome) and the options bar on top, whose controls follow the active tool (Photoshop).

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, App, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, Styled as _};

use crate::data::store::Store;
use crate::domain::tool::{step_brush, Tool};
use crate::screens::explore::canvas::ImageCanvas;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{divider_v, format_area, format_length, stepper, swatch, MenuSelect};

/// The tool bar floating at the canvas's left edge.
pub fn tool_bar(store: &Entity<Store>, canvas: &Entity<ImageCanvas>, cx: &App) -> impl IntoElement + use<> {
    let current = store.read(cx).tool.tool;
    let mut bar = v_flex()
        .id("tool-bar")
        .occlude()
        .absolute()
        .left(px(theme::MD))
        .top(px(theme::MD + theme::RULER))
        .p(px(theme::XS))
        .gap(px(2.))
        .items_center()
        .rounded(px(8.))
        .border_1()
        .border_color(color::border())
        .bg(color::raised().opacity(0.96))
        .shadow_lg();
    for slot in Tool::BAR {
        match slot {
            Some(t) => {
                let (store, canvas) = (store.clone(), canvas.clone());
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
            None => bar = bar.child(div().w(px(20.)).h(px(1.)).my(px(2.)).bg(color::border())),
        }
    }
    bar
}

fn label(text: &'static str) -> impl IntoElement {
    div().text_sm().text_color(color::text_muted()).child(text)
}

fn hint(text: impl Into<SharedString>) -> impl IntoElement {
    div().text_xs().text_color(color::text_muted()).child(text.into())
}

/// The options bar of the active tool.
pub fn options_bar(store: &Entity<Store>, canvas: &Entity<ImageCanvas>, cx: &App) -> impl IntoElement + use<> {
    let s = store.read(cx);
    let t = s.tool.clone();
    let classes: Vec<String> = s.doc.classes.iter().filter(|c| c.name != "Measurement").map(|c| c.name.clone()).collect();
    let class_color = s.doc.class(&t.class).map_or([255, 255, 255], |c| c.color);
    let selected = s.selection.len();
    let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
    let zoom = canvas.read(cx).zoom;
    let img = s.doc.active().cloned();
    let hist = img.as_ref().and_then(|i| s.histograms.get(&i.id)).and_then(|h| h.get(t.threshold_channel).cloned());
    let mut bar = h_flex().id("options-bar").flex_none().h(px(theme::OPTIONS_BAR_H)).px(px(theme::MD)).gap(px(theme::SM)).items_center().bg(color::panel()).border_b_1().border_color(color::border());
    bar = bar.child(Icon::new(t.tool.icon()).small().text_color(color::text_muted())).child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(t.tool.name())).child(divider_v());

    let class_picker = || {
        let st = store.clone();
        let names = classes.clone();
        h_flex().gap(px(theme::XS)).child(swatch(class_color, 12.)).child(label("Class")).child(div().w(px(124.)).child(MenuSelect::new("options-class", classes.iter().map(|c| SharedString::from(c.clone())).collect(), classes.iter().position(|c| *c == t.class), move |i, _, cx| {
            let name = names[i].clone();
            st.update(cx, |s, cx| s.update_tool(|t| t.class = name, cx));
        })))
    };
    let brush_size = || {
        let size = if t.brush_in_image { format_length(t.brush_image_px * 2.0, um) } else { format!("{:.0} px", t.brush_px * 2.0) };
        let st = store.clone();
        h_flex().gap(px(theme::XS)).child(label("Size")).child(stepper("brush-size", size, move |d, _, cx| {
            st.update(cx, |s, cx| {
                s.update_tool(
                    |t| {
                        if t.brush_in_image {
                            t.brush_image_px = (t.brush_image_px * if d > 0.0 { 1.25 } else { 0.8 }).clamp(1.0, 5000.0);
                        } else {
                            t.brush_px = step_brush(t.brush_px, d);
                        }
                    },
                    cx,
                )
            })
        }))
    };
    match t.tool {
        Tool::Brush | Tool::Eraser | Tool::Wand => {
            let st = store.clone();
            bar = bar.child(class_picker()).child(divider_v()).child(brush_size()).child(Checkbox::new("brush-in-image").checked(t.brush_in_image).label("Fixed on the image").on_click(move |on: &bool, _, cx| {
                let on = *on;
                st.update(cx, |s, cx| {
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
            }));
            if t.tool == Tool::Wand {
                let st = store.clone();
                bar = bar.child(divider_v()).child(label("Tolerance")).child(stepper("wand-tol", format!("{:.0} %", t.wand_tolerance * 100.0), move |d, _, cx| {
                    st.update(cx, |s, cx| s.update_tool(|t| t.wand_tolerance = (t.wand_tolerance + d * 0.05).clamp(0.01, 1.0), cx))
                }));
                bar = bar.child(hint("Fills pixels like the one under the centre (visible channels)"));
            } else {
                bar = bar.child(hint("[ ] resize"));
            }
        }
        Tool::Threshold => {
            let names: Vec<SharedString> = img.as_ref().map(|i| i.channels.iter().map(|c| SharedString::from(c.label.clone())).collect()).unwrap_or_default();
            let span = hist.as_ref().map_or(1.0, |h| (h.hi - h.lo).max(1e-6));
            let st_c = store.clone();
            let st_v = store.clone();
            let otsu = hist.as_ref().map(|h| h.otsu());
            let st_o = store.clone();
            let st_a = store.clone();
            let st_s = store.clone();
            let st_m = store.clone();
            let (st_make, st_pipe) = (store.clone(), store.clone());
            bar = bar
                .child(label("Channel"))
                .child(div().w(px(120.)).child(MenuSelect::new("thr-channel", names, Some(t.threshold_channel), move |i, _, cx| {
                    st_c.update(cx, |s, cx| {
                        let o = s.doc.active_image.clone().and_then(|id| s.histograms.get(&id)).and_then(|h| h.get(i)).map(|h| h.otsu());
                        s.update_tool(
                            |t| {
                                t.threshold_channel = i;
                                if let Some(o) = o {
                                    t.threshold_value = o;
                                }
                            },
                            cx,
                        )
                    })
                })))
                .child(label("Value"))
                .child(stepper("thr-value", trim(t.threshold_value), move |d, _, cx| st_v.update(cx, |s, cx| s.update_tool(|t| t.threshold_value += d * span * 0.01, cx))))
                .children(otsu.map(|o| {
                    Button::new("thr-otsu").ghost().xsmall().label("Otsu").tooltip("The Otsu threshold of the whole image (coarsest level)").on_click(move |_, _, cx| st_o.update(cx, |s, cx| s.update_tool(|t| t.threshold_value = o, cx)))
                }))
                .child(Button::new("thr-above").ghost().xsmall().label(if t.threshold_above { "Above" } else { "Below" }).tooltip("Keep pixels above (bright objects) or below the value").on_click(move |_, _, cx| st_a.update(cx, |s, cx| s.update_tool(|t| t.threshold_above = !t.threshold_above, cx))))
                .child(divider_v())
                .child(label("Smooth"))
                .child(stepper("thr-sigma", format_length(t.threshold_sigma, um), move |d, _, cx| st_s.update(cx, |s, cx| s.update_tool(|t| t.threshold_sigma = (t.threshold_sigma * if d > 0.0 { 1.5 } else { 1.0 / 1.5 }).clamp(0.0, 200.0).max(if d > 0.0 { 0.5 } else { 0.0 }), cx))))
                .child(label("Min"))
                .child(stepper("thr-min", format_area(t.threshold_min_area, um), move |d, _, cx| st_m.update(cx, |s, cx| s.update_tool(|t| t.threshold_min_area = (t.threshold_min_area * if d > 0.0 { 1.5 } else { 1.0 / 1.5 }).clamp(1.0, 1.0e9), cx))))
                .child(divider_v())
                .child(class_picker())
                .child(Button::new("thr-make").primary().small().icon(Icon::new(IconName::Shapes)).label("Create annotations").tooltip("Segment the whole image with these settings").on_click(move |_, _, cx| st_make.update(cx, |s, cx| s.threshold_to_annotations(cx))))
                .child(Button::new("thr-pipe").ghost().small().icon(Icon::new(IconName::Workflow)).label("Add to pipeline").on_click(move |_, _, cx| st_pipe.update(cx, |s, cx| s.threshold_to_pipeline(cx))));
        }
        Tool::Rect | Tool::Ellipse | Tool::Lasso | Tool::Point => bar = bar.child(class_picker()),
        Tool::Polygon => bar = bar.child(class_picker()).child(hint("Click corners · Enter or double-click closes · Esc cancels")),
        Tool::Ruler => bar = bar.child(hint(if um.is_some() { "Drag to measure (µm from the image's pixel size)" } else { "Drag to measure (px: the image has no pixel size)" })),
        Tool::Select => {
            let st = store.clone();
            bar = bar.child(hint(if selected == 0 { "Click an annotation; Shift adds".to_string() } else { format!("{selected} selected") })).when(selected > 0, |b| {
                b.child(Button::new("options-delete").ghost().small().icon(Icon::new(IconName::Trash)).label("Delete").on_click(move |_, _, cx| st.update(cx, |s, cx| s.delete_selection(cx))))
            });
        }
        Tool::Zoom => bar = bar.child(hint("Click to zoom in · Alt+click to zoom out")),
        Tool::Hand | Tool::Probe => bar = bar.child(hint(t.tool.hint())),
    }
    let (c1, c2) = (canvas.clone(), canvas.clone());
    let (st_r, st_v) = (store.clone(), store.clone());
    bar.child(div().flex_1())
        .child(Button::new("options-results").ghost().small().icon(Icon::new(IconName::Layers)).label("Results").selected(t.show_results).tooltip("Draw the pipeline's results over the image").on_click(move |_, _, cx| {
            st_v.update(cx, |s, cx| s.update_tool(|t| t.show_results = !t.show_results, cx))
        }))
        .child(Button::new("options-rulers").ghost().small().icon(Icon::new(IconName::Ruler)).selected(t.rulers).tooltip("Rulers (Ctrl+R)").on_click(move |_, _, cx| st_r.update(cx, |s, cx| s.update_tool(|t| t.rulers = !t.rulers, cx))))
        .child(divider_v())
        .child(Button::new("options-fit").ghost().small().icon(Icon::new(IconName::Maximize)).tooltip("Fit (F · Ctrl+0)").on_click(move |_, _, cx| c1.update(cx, |c, cx| c.fit(cx))))
        .child(Button::new("options-100").ghost().small().label("1:1").tooltip("Actual pixels (Ctrl+1)").on_click(move |_, _, cx| c2.update(cx, |c, cx| c.actual_size(cx))))
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

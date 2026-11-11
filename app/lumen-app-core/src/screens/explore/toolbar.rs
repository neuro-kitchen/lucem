//! The image tools: a tool column docked at the left edge (Affinity), so it never covers the
//! image, and the options bar on top, whose controls follow the active tool (Photoshop).

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, App, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, Styled as _};

use crate::data::store::Store;
use crate::domain::tool::{step_brush, Tool};
use crate::screens::explore::canvas::ImageCanvas;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{divider_v, format_area, format_length, stepper, swatch, MenuSelect};

/// The tool column at the left edge of Explore.
pub fn tool_bar(store: &Entity<Store>, canvas: &Entity<ImageCanvas>, cx: &App) -> impl IntoElement + use<> {
    let current = store.read(cx).tool.tool;
    let mut bar = v_flex()
        .id("tool-bar")
        .flex_none()
        .w(px(theme::TOOLBAR_W))
        .h_full()
        .py(px(theme::SM))
        .gap(px(2.))
        .items_center()
        .bg(color::chrome())
        .border_r_1()
        .border_color(color::border());
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
pub fn options_bar(store: &Entity<Store>, canvas: &Entity<ImageCanvas>, trailing: gpui_kit::AnyElement, cx: &App) -> impl IntoElement + use<> {
    let s = store.read(cx);
    let t = s.tool.clone();
    let classes: Vec<String> = s.doc.classes.iter().filter(|c| c.name != "Measurement").map(|c| c.name.clone()).collect();
    let class_color = s.doc.class(&t.class).map_or([255, 255, 255], |c| c.color);
    let selected = s.selection.len();
    let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
    let volume = s.active_pyramid().is_some_and(|p| p.is_volume());
    let zoom = canvas.read(cx).zoom;
    let img = s.doc.active().cloned();
    let hist = img.as_ref().and_then(|i| s.histograms.get(&i.id)).and_then(|h| h.get(t.threshold_channel).cloned());
    let mut bar = h_flex().id("options-bar").flex_none().h(px(theme::OPTIONS_BAR_H)).px(px(theme::MD)).gap(px(theme::SM)).items_center().bg(color::panel()).border_b_1().border_color(color::border());
    bar = bar.child(Icon::new(t.tool.icon()).small().text_color(color::text_muted())).child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(t.tool.name())).child(divider_v());
    // Combine (Affinity's context toolbar): only while masks or classes are selected in Layers
    let masks = s.selected_masks().len();
    if masks > 0 {
        use crate::data::store::MaskOperation;
        let r = s.mask_radius;
        let op = |id: &'static str, icon: IconName, tip: String, need: usize, o: MaskOperation| {
            let st = store.clone();
            Button::new(id).ghost().xsmall().icon(Icon::new(icon)).tooltip(tip).disabled(masks < need).on_click(move |_, _, cx| st.update(cx, |s, cx| s.apply_mask_operation(o.clone(), cx)))
        };
        bar = bar
            .child(label("Combine"))
            .child(op("ctx-union", IconName::SquaresUnite, "Union: on where any is on".into(), 2, MaskOperation::Union))
            .child(op("ctx-intersect", IconName::SquaresIntersect, "Intersect: on where all are on".into(), 2, MaskOperation::Intersect))
            .child(op("ctx-subtract", IconName::SquaresSubtract, "Subtract the others from the first".into(), 2, MaskOperation::Subtract))
            .child(op("ctx-invert", IconName::Contrast, "Invert the first".into(), 1, MaskOperation::Invert))
            .child(op("ctx-grow", IconName::Expand, format!("Grow the first by {r} px"), 1, MaskOperation::Op(lumen_pipeline::Op::Dilate { radius: r })))
            .child(op("ctx-shrink", IconName::Shrink, format!("Shrink the first by {r} px"), 1, MaskOperation::Op(lumen_pipeline::Op::Erode { radius: r })))
            .child(divider_v());
    }
    // Shapes (Affinity's Add, Subtract, Intersect, Xor): while two or more regions are selected
    let regions = s.doc.active_image.as_ref().map_or(0, |img| s.doc.annotations_of(img).filter(|a| s.selection.contains(&a.id) && a.shape.is_region()).count());
    if regions >= 2 {
        use crate::domain::boolean::ShapeOp;
        bar = bar.child(label("Shapes"));
        for op in ShapeOp::ALL {
            let st = store.clone();
            let icon = match op {
                ShapeOp::Add => IconName::SquaresUnite,
                ShapeOp::Subtract => IconName::SquaresSubtract,
                ShapeOp::Intersect => IconName::SquaresIntersect,
                ShapeOp::Xor => IconName::SquaresExclude,
            };
            bar = bar.child(Button::new(SharedString::from(format!("ctx-shape-{}", op.title().to_lowercase()))).ghost().xsmall().icon(Icon::new(icon)).tooltip(op.hint()).on_click(move |_, _, cx| st.update(cx, |s, cx| s.combine_shapes(op, cx))));
        }
        bar = bar.child(divider_v());
    }

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
                if t.tool == Tool::Brush {
                    let st = store.clone();
                    bar = bar.child(Checkbox::new("brush-fill").checked(t.brush_fill).label("Fill when closed").on_click(move |on: &bool, _, cx| {
                        let on = *on;
                        st.update(cx, |s, cx| s.update_tool(|t| t.brush_fill = on, cx));
                    }));
                }
                bar = bar.child(hint("[ ] resize · Shift: straight line"));
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
        Tool::Segment => {
            // The model that will run, and whether it can
            let model = s.ml.segmenter(s.prefs.ml_model.as_deref());
            let state = match (&s.ml.busy, &model) {
                (Some((what, f)), _) => format!("{what}{}", f.map_or(String::new(), |f| format!(" {:.0} %", f * 100.0))),
                (None, Some((_, id, true))) => format!("Model: {id}"),
                (None, Some((_, id, false))) => format!("{id} is not installed"),
                (None, None) if !s.ml.scanned || s.ml.scanning => "Looking for model plugins…".to_string(),
                (None, None) => "No model plugin found".to_string(),
            };
            let ready = matches!(model, Some((_, _, true)));
            bar = bar
                .child(class_picker())
                .child(divider_v())
                .child(div().text_sm().text_color(if ready || s.ml.busy.is_some() { color::text() } else { color::warning() }).child(state))
                .child(Button::new("segment-models").ghost().small().icon(Icon::new(IconName::Settings)).label("Models…").on_click(|_, window, cx| window.dispatch_action(Box::new(crate::navigation::actions::OpenSettings), cx)));
            if volume {
                let st = store.clone();
                bar = bar.child(divider_v()).child(Checkbox::new("segment-through-z").checked(t.sam_through_z).label("Through z").on_click(move |on: &bool, _, cx| {
                    let on = *on;
                    st.update(cx, |s, cx| s.update_tool(|t| t.sam_through_z = on, cx));
                }));
            }
            bar = bar.child(hint(if volume && t.sam_through_z { "Click an object: it is followed plane by plane while the model is sure of it" } else { "Click an object · Alt+click: not this · drag a box" }));
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
        Tool::Marquee => {
            let objects = s.selected_objects.len();
            let st = store.clone();
            bar = bar.child(hint(if selected + objects == 0 { "Drag a box around shapes or cells; Shift adds".to_string() } else { format!("{selected} shape(s), {objects} object(s) selected") })).when(selected > 0, |b| {
                b.child(Button::new("options-delete").ghost().small().icon(Icon::new(IconName::Trash)).label("Delete").on_click(move |_, _, cx| st.update(cx, |s, cx| s.delete_selection(cx))))
            });
        }
        Tool::Zoom => bar = bar.child(hint("Click to zoom in · Alt+click to zoom out")),
        Tool::Hand | Tool::Probe => bar = bar.child(hint(t.tool.hint())),
    }
    let (c1, c2) = (canvas.clone(), canvas.clone());
    let (st_r, st_v) = (store.clone(), store.clone());
    // Each tool shows its own hint above; then what the caller puts on the right (Overlays)
    let _ = (st_r, st_v, c1, c2);
    bar.child(div().flex_1().min_w_0()).child(trailing)
}

/// The Overlays popover (the reference viewer's toggles, in one place): annotations, their fill
/// and names, the pipeline's results, the crosshair, the rulers, and the annotations' opacity.
/// The overlays as buttons on the options bar (what is drawn over the image, and how opaque),
/// always in reach: the first one switches between the overlays shared by every view and the
/// focused view's own.
pub fn overlay_toggles(store: &Entity<Store>, canvas: &Entity<ImageCanvas>, cx: &App) -> impl IntoElement + use<> {
    let own = canvas.read(cx).overlays;
    let shared = store.read(cx).tool.overlays();
    let o = own.unwrap_or(shared);
    let toggle = |id: &'static str, icon: IconName, tip: &'static str, on: bool, f: fn(&mut crate::domain::tool::Overlays)| {
        let cv = canvas.clone();
        Button::new(id).ghost().small().icon(Icon::new(icon)).selected(on).tooltip(tip).on_click(move |_, _, cx| cv.update(cx, |c, cx| c.toggle_overlay(f, cx)))
    };
    let scope = {
        let cv = canvas.clone();
        Button::new("ov-scope")
            .ghost()
            .small()
            .icon(Icon::new(if own.is_some() { IconName::Unlink } else { IconName::Link }))
            .selected(own.is_some())
            .tooltip(if own.is_some() { "This view has its own overlays (click: follow the shared ones)" } else { "Overlays shared by every view (click: this view only)" })
            .on_click(move |_, _, cx| {
                cv.update(cx, |c, cx| {
                    c.overlays = if c.overlays.is_some() { None } else { Some(shared) };
                    cx.notify();
                })
            })
    };
    let cv = canvas.clone();
    let opacity = crate::ui::widgets::slider("ov-opacity", o.opacity, (0.0, 1.0), |v| format!("{:.0} %", v * 100.0), move |v, _, _, cx| {
        cv.update(cx, |c, cx| c.toggle_overlay(move |o| o.opacity = v, cx))
    });
    h_flex()
        .flex_none()
        .gap(px(1.))
        .items_center()
        .child(scope)
        .child(divider_v())
        .child(toggle("ov-annotations", IconName::Shapes, "Shapes (A)", o.annotations, |o| o.annotations = !o.annotations))
        .child(toggle("ov-fill", IconName::PaintBucket, "Fill regions (Shift+F)", o.fill, |o| o.fill = !o.fill))
        .child(toggle("ov-names", IconName::Type, "Names (Shift+N)", o.names, |o| o.names = !o.names))
        .child(toggle("ov-results", IconName::ChartColumn, "Pipeline results", o.results, |o| o.results = !o.results))
        .child(toggle("ov-crosshair", IconName::Crosshair, "Crosshair (3D cursor)", o.crosshair, |o| o.crosshair = !o.crosshair))
        .child(toggle("ov-rulers", IconName::Ruler, "Rulers (Ctrl+R)", o.rulers, |o| o.rulers = !o.rulers))
        .child(divider_v())
        .child(div().flex_none().text_xs().text_color(color::text_muted()).pl(px(theme::XS)).pr(px(theme::SM)).child("Opacity"))
        // One line, fixed width: never wraps
        .child(div().flex_none().w(px(150.)).child(opacity))
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

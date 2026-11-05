//! Right-Side Vector & Paint Tool Properties + Right-Edge Tool Strip (`render_right_inspector_dock`).
//!
//! Matches the Blender 4.x layout:
//! - Left column (`246px`): Active Tool / Stroke & Brush Settings, Vector Path & ROI Geometry, and
//!   Viewport Shading & LUT controls.
//! - Right-edge strip (`42px`): Vertical icon toolbar (`Pan`, `Zoom`, `Brush Paint`, `Eraser`,
//!   `Vector Pen`, `Polygon ROI`, `Iso-Contour`, `Measure Caliper`, `Pixel Eyedropper`).

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::Icon;
use gpui_kit::{div, prelude::*, px, App, IntoElement, SharedString, Window};
use lumen_view::ColormapLut;

use crate::domain::ActiveTool;
use crate::screens::studio::studio_event::StudioEvent;
use crate::screens::studio::studio_state::StudioUiState;
use crate::ui::theme::palette;

pub fn render_right_inspector_dock(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let tool_title: SharedString = format!("Tool · {}", state.active_tool.label()).into();
    let tool_cat: SharedString = state.active_tool.category().into();

    div()
        .id("studio-right-tool-dock")
        .w(px(288.0))
        .h_full()
        .bg(palette::panel_bg())
        .border_l_1()
        .border_color(palette::border_subtle())
        .flex()
        .flex_row()
        .overflow_hidden()
        // 1. Active Tool Properties & Viewport Shading Panel (246px)
        .child(
            div()
                .flex_1()
                .h_full()
                .flex()
                .flex_col()
                .overflow_hidden()
                .child(
                    div()
                        .w_full()
                        .px(px(10.0))
                        .py(px(6.0))
                        .bg(palette::header_bg())
                        .border_b_1()
                        .border_color(palette::border_subtle())
                        .flex()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .text_size(px(11.5))
                                .text_color(palette::text_primary())
                                .child(tool_title),
                        )
                        .child(
                            div()
                                .text_size(px(10.5))
                                .text_color(palette::text_dim())
                                .child(tool_cat),
                        ),
                )
                .child(
                    div()
                        .id("right-tool-props-scroll")
                        .flex_1()
                        .overflow_y_scroll()
                        .p(px(8.0))
                        .flex()
                        .flex_col()
                        .gap(px(8.0))
                        .child(render_brush_and_stroke_subpanel(state, on_event.clone()))
                        .child(render_vector_and_roi_subpanel(state, on_event.clone()))
                        .child(render_shading_and_lut_subpanel(state, on_event.clone())),
                ),
        )
        // 2. Right-Edge Vertical Tool Strip (42px)
        .child(render_vertical_tool_strip(state.active_tool, on_event))
}

fn render_brush_and_stroke_subpanel(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    let cb_r_dec = on_event.clone();
    let cb_r_inc = on_event.clone();
    let cb_s_dec = on_event.clone();
    let cb_s_inc = on_event;

    let radius_str: SharedString = format!("{:.0} px", state.brush_radius).into();
    let strength_str: SharedString = format!("{:.2}", state.brush_strength).into();

    div()
        .w_full()
        .p(px(8.0))
        .rounded(px(4.0))
        .bg(palette::surface_card())
        .flex()
        .flex_col()
        .gap(px(6.0))
        .child(
            div()
                .text_size(px(11.0))
                .text_color(palette::text_primary())
                .child("Stroke & Brush Settings"),
        )
        // Brush Radius
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(10.5))
                        .text_color(palette::text_secondary())
                        .child("Radius"),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(3.0))
                        .child(
                            div()
                                .id("brush-radius-dec")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_r_dec(StudioEvent::AdjustBrushRadius(-2.0), window, cx);
                                })
                                .child("-"),
                        )
                        .child(
                            div()
                                .px(px(6.0))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .child(radius_str),
                        )
                        .child(
                            div()
                                .id("brush-radius-inc")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_r_inc(StudioEvent::AdjustBrushRadius(2.0), window, cx);
                                })
                                .child("+"),
                        ),
                ),
        )
        // Brush Strength / Opacity
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(10.5))
                        .text_color(palette::text_secondary())
                        .child("Strength / Opacity"),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(3.0))
                        .child(
                            div()
                                .id("brush-strength-dec")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_s_dec(StudioEvent::AdjustBrushStrength(-0.10), window, cx);
                                })
                                .child("-"),
                        )
                        .child(
                            div()
                                .px(px(6.0))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .child(strength_str),
                        )
                        .child(
                            div()
                                .id("brush-strength-inc")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_s_inc(StudioEvent::AdjustBrushStrength(0.10), window, cx);
                                })
                                .child("+"),
                        ),
                ),
        )
        // Falloff & Blend Mode info row
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(10.5))
                        .text_color(palette::text_secondary())
                        .child("Falloff / Blend"),
                )
                .child(
                    div()
                        .px(px(6.0))
                        .py(px(1.5))
                        .rounded(px(3.0))
                        .bg(palette::widget_bg())
                        .text_size(px(10.5))
                        .text_color(palette::text_secondary())
                        .child("Smooth · Screen"),
                ),
        )
}

fn render_vector_and_roi_subpanel(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    let cb_w_dec = on_event.clone();
    let cb_w_inc = on_event.clone();
    let cb_contours = on_event.clone();
    let cb_grid = on_event;

    let sw_str: SharedString = format!("{:.1} px", state.stroke_width).into();

    div()
        .w_full()
        .p(px(8.0))
        .rounded(px(4.0))
        .bg(palette::surface_card())
        .flex()
        .flex_col()
        .gap(px(6.0))
        .child(
            div()
                .text_size(px(11.0))
                .text_color(palette::text_primary())
                .child("Vector Path & ROI Geometry"),
        )
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(10.5))
                        .text_color(palette::text_secondary())
                        .child("Stroke Width"),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(3.0))
                        .child(
                            div()
                                .id("vec-stroke-dec")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_w_dec(StudioEvent::AdjustStrokeWidth(-0.5), window, cx);
                                })
                                .child("-"),
                        )
                        .child(
                            div()
                                .px(px(6.0))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .child(sw_str),
                        )
                        .child(
                            div()
                                .id("vec-stroke-inc")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_w_inc(StudioEvent::AdjustStrokeWidth(0.5), window, cx);
                                })
                                .child("+"),
                        ),
                ),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap(px(4.0))
                .child(
                    div()
                        .id("vec-toggle-contours")
                        .cursor_pointer()
                        .flex_1()
                        .py(px(3.5))
                        .rounded(px(3.0))
                        .bg(if state.camera.show_contours {
                            palette::accent_blue()
                        } else {
                            palette::widget_bg()
                        })
                        .hover(|s| s.bg(palette::surface_hover()))
                        .flex()
                        .justify_center()
                        .text_size(px(10.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            cb_contours(StudioEvent::ToggleContours, window, cx);
                        })
                        .child("Iso-Contours"),
                )
                .child(
                    div()
                        .id("vec-toggle-grid")
                        .cursor_pointer()
                        .flex_1()
                        .py(px(3.5))
                        .rounded(px(3.0))
                        .bg(if state.camera.show_grid {
                            palette::accent_blue()
                        } else {
                            palette::widget_bg()
                        })
                        .hover(|s| s.bg(palette::surface_hover()))
                        .flex()
                        .justify_center()
                        .text_size(px(10.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            cb_grid(StudioEvent::ToggleGrid, window, cx);
                        })
                        .child("Canvas Grid"),
                ),
        )
}

fn render_shading_and_lut_subpanel(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    let cb_exp_dec = on_event.clone();
    let cb_exp_inc = on_event.clone();
    let cb_gam_dec = on_event.clone();
    let cb_gam_inc = on_event.clone();
    let cb_inv = on_event.clone();
    let cb_rst = on_event.clone();
    let cb_fit = on_event.clone();
    let cb_one = on_event.clone();
    let cb_zout = on_event.clone();
    let cb_zin = on_event.clone();

    let exp_val: SharedString = format!("{:+.2}", state.tonemap.exposure).into();
    let gam_val: SharedString = format!("{:.2}", state.tonemap.gamma).into();

    div()
        .w_full()
        .p(px(8.0))
        .rounded(px(4.0))
        .bg(palette::surface_card())
        .flex()
        .flex_col()
        .gap(px(6.0))
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(11.0))
                        .text_color(palette::text_primary())
                        .child("Viewport Shading & LUT"),
                )
                .child(
                    div()
                        .id("shading-reset-btn")
                        .cursor_pointer()
                        .px(px(5.0))
                        .py(px(1.0))
                        .rounded(px(3.0))
                        .bg(palette::widget_bg())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .text_size(px(10.0))
                        .text_color(palette::text_secondary())
                        .on_click(move |_, window, cx| {
                            cb_rst(StudioEvent::ResetTonemap, window, cx);
                        })
                        .child("Reset"),
                ),
        )
        // Colormap LUT grid
        .child(
            div()
                .grid()
                .grid_cols(2)
                .gap(px(3.0))
                .children(ColormapLut::ALL.iter().enumerate().map(|(idx, &lut)| {
                    let is_active = lut == state.tonemap.colormap;
                    let cb = on_event.clone();
                    div()
                        .id(("right-lut-chip", idx))
                        .cursor_pointer()
                        .px(px(6.0))
                        .py(px(3.0))
                        .rounded(px(3.0))
                        .bg(if is_active {
                            palette::accent_blue()
                        } else {
                            palette::widget_bg()
                        })
                        .hover(|s| s.bg(palette::surface_hover()))
                        .text_size(px(10.5))
                        .text_color(if is_active {
                            palette::text_primary()
                        } else {
                            palette::text_secondary()
                        })
                        .on_click(move |_, window, cx| {
                            cb(StudioEvent::SelectColormap(lut), window, cx);
                        })
                        .child(lut.label())
                })),
        )
        // Exposure row
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(10.5))
                        .text_color(palette::text_secondary())
                        .child("Exposure"),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(3.0))
                        .child(
                            div()
                                .id("shading-exp-dec")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_exp_dec(StudioEvent::AdjustExposure(-0.15), window, cx);
                                })
                                .child("-"),
                        )
                        .child(
                            div()
                                .px(px(6.0))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .child(exp_val),
                        )
                        .child(
                            div()
                                .id("shading-exp-inc")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_exp_inc(StudioEvent::AdjustExposure(0.15), window, cx);
                                })
                                .child("+"),
                        ),
                ),
        )
        // Gamma + Invert row
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(10.5))
                        .text_color(palette::text_secondary())
                        .child("Gamma"),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(3.0))
                        .child(
                            div()
                                .id("shading-gam-dec")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_gam_dec(StudioEvent::AdjustGamma(-0.10), window, cx);
                                })
                                .child("-"),
                        )
                        .child(
                            div()
                                .px(px(6.0))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .child(gam_val),
                        )
                        .child(
                            div()
                                .id("shading-gam-inc")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(palette::widget_bg())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_gam_inc(StudioEvent::AdjustGamma(0.10), window, cx);
                                })
                                .child("+"),
                        )
                        .child(
                            div()
                                .id("shading-inv-btn")
                                .cursor_pointer()
                                .px(px(6.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .bg(if state.tonemap.invert {
                                    palette::accent_blue()
                                } else {
                                    palette::widget_bg()
                                })
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(10.5))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    cb_inv(StudioEvent::ToggleInvert, window, cx);
                                })
                                .child("Inv"),
                        ),
                ),
        )
        // Camera Zoom quick row
        .child(
            div()
                .pt(px(2.0))
                .flex()
                .items_center()
                .gap(px(3.0))
                .child(
                    div()
                        .id("right-zoom-fit")
                        .cursor_pointer()
                        .flex_1()
                        .py(px(3.0))
                        .rounded(px(3.0))
                        .bg(palette::widget_bg())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .flex()
                        .justify_center()
                        .text_size(px(10.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            cb_fit(StudioEvent::FitToViewport, window, cx);
                        })
                        .child("Fit"),
                )
                .child(
                    div()
                        .id("right-zoom-1to1")
                        .cursor_pointer()
                        .flex_1()
                        .py(px(3.0))
                        .rounded(px(3.0))
                        .bg(palette::widget_bg())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .flex()
                        .justify_center()
                        .text_size(px(10.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            cb_one(StudioEvent::ZoomOneToOne, window, cx);
                        })
                        .child("1:1"),
                )
                .child(
                    div()
                        .id("right-zoom-out")
                        .cursor_pointer()
                        .px(px(8.0))
                        .py(px(3.0))
                        .rounded(px(3.0))
                        .bg(palette::widget_bg())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .text_size(px(10.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            cb_zout(StudioEvent::ZoomOut, window, cx);
                        })
                        .child("-"),
                )
                .child(
                    div()
                        .id("right-zoom-in")
                        .cursor_pointer()
                        .px(px(8.0))
                        .py(px(3.0))
                        .rounded(px(3.0))
                        .bg(palette::widget_bg())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .text_size(px(10.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            cb_zin(StudioEvent::ZoomIn, window, cx);
                        })
                        .child("+"),
                ),
        )
}

fn render_vertical_tool_strip(
    active_tool: ActiveTool,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    let tools: [(ActiveTool, IconName); 9] = [
        (ActiveTool::ExplorePan, IconName::Maximize2),
        (ActiveTool::ZoomLens, IconName::Search),
        (ActiveTool::BrushPaint, IconName::Palette),
        (ActiveTool::Eraser, IconName::Delete),
        (ActiveTool::VectorPen, IconName::Sparkles),
        (ActiveTool::PolygonRoi, IconName::Frame),
        (ActiveTool::VectorContour, IconName::SlidersHorizontal),
        (ActiveTool::MeasureCaliper, IconName::PanelBottom),
        (ActiveTool::PixelProbe, IconName::Eye),
    ];

    div()
        .w(px(42.0))
        .h_full()
        .py(px(6.0))
        .bg(palette::header_bg())
        .border_l_1()
        .border_color(palette::border_subtle())
        .flex()
        .flex_col()
        .items_center()
        .gap(px(4.0))
        .children(tools.into_iter().enumerate().map(|(idx, (tool, icon))| {
            let is_active = tool == active_tool;
            let cb = on_event.clone();
            div()
                .id(("right-strip-tool-btn", idx))
                .cursor_pointer()
                .w(px(34.0))
                .py(px(5.0))
                .rounded(px(4.0))
                .bg(if is_active {
                    palette::accent_blue()
                } else {
                    palette::widget_bg()
                })
                .hover(|s| s.bg(palette::surface_hover()))
                .flex()
                .flex_col()
                .items_center()
                .gap(px(1.5))
                .text_color(if is_active {
                    palette::text_primary()
                } else {
                    palette::text_secondary()
                })
                .on_click(move |_, window, cx| {
                    cb(StudioEvent::SelectTool(tool), window, cx);
                })
                .child(Icon::new(icon))
                .child(
                    div()
                        .text_size(px(8.5))
                        .child(tool.short_label()),
                )
        }))
}

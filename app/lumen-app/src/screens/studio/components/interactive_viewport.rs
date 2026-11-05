//! Interactive GPU Explore Viewport (`render_interactive_viewport`):
//! Supports cursor-anchored scroll-wheel / trackpad zoom (`on_scroll`, `on_pinch`) with automatic
//! OME-Zarr pyramid level streaming (`s3 ↔ s2 ↔ s1 ↔ s0`), click-and-drag panning or brush painting,
//! OS file drag-and-drop (`ExternalPaths`), and the top `Operations ▾` dropdown menu overlay.

use std::rc::Rc;

use gpui_kit::{
    div, img, prelude::*, px, App, ExternalPaths, ImageSource, IntoElement, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, PinchEvent, ScrollEvent, SharedString, Window,
};
use lumen_pipeline::ModifierKind;

use crate::screens::studio::studio_event::StudioEvent;
use crate::screens::studio::studio_state::StudioUiState;
use crate::ui::theme::palette;

/// Left Outliner width (`268px`), Top Titlebar height (`34px`), and Right Tool Dock width (`288px`).
const VIEWPORT_LEFT_OFFSET: f32 = 268.0;
const VIEWPORT_TOP_OFFSET: f32 = 34.0;
const RIGHT_DOCK_WIDTH: f32 = 288.0;
const STATUS_FOOTER_HEIGHT: f32 = 28.0;
const ANALYTICS_DRAWER_HEIGHT: f32 = 184.0;

fn compute_viewport_bounds(window: &Window, drawer_open: bool) -> (f32, f32) {
    let win_size = window.viewport_size();
    let win_w = f32::from(win_size.width);
    let win_h = f32::from(win_size.height);
    let bottom_h = STATUS_FOOTER_HEIGHT + if drawer_open { ANALYTICS_DRAWER_HEIGHT } else { 0.0 };
    let view_w = (win_w - VIEWPORT_LEFT_OFFSET - RIGHT_DOCK_WIDTH).max(240.0);
    let view_h = (win_h - VIEWPORT_TOP_OFFSET - bottom_h).max(180.0);
    (view_w, view_h)
}

pub fn render_interactive_viewport(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let img_w = state.source_image.width;
    let img_h = state.source_image.height;
    let (view_w, view_h) = state.viewport_size;
    let (left, top, sw, sh) = state.camera.image_rect(img_w, img_h, view_w, view_h);
    let render_img = state.render_image.clone();
    let show_grid = state.camera.show_grid;
    let show_ops_menu = state.show_operations_menu;
    let drawer_open = state.show_analytics_drawer;

    let on_scroll_cb = on_event.clone();
    let on_pinch_cb = on_event.clone();
    let on_down_left = on_event.clone();
    let on_down_mid = on_event.clone();
    let on_move_cb = on_event.clone();
    let on_up_left = on_event.clone();
    let on_up_mid = on_event.clone();
    let on_up_out_left = on_event.clone();
    let on_drop_cb = on_event.clone();

    div()
        .id("studio-interactive-viewport")
        .flex_1()
        .h_full()
        .relative()
        .overflow_hidden()
        .bg(if show_grid {
            palette::canvas_checker_a()
        } else {
            palette::canvas_void()
        })
        .cursor_grab()
        // Scroll-wheel / trackpad 2-finger zoom anchored at cursor (automatically pulls s3..s0 pyramid)
        .on_scroll(move |event: &ScrollEvent, window, cx| {
            let (vw, vh) = compute_viewport_bounds(window, drawer_open);
            let local_x = f32::from(event.position.x) - VIEWPORT_LEFT_OFFSET;
            let local_y = f32::from(event.position.y) - VIEWPORT_TOP_OFFSET;
            let rel_x = local_x - 0.5 * vw;
            let rel_y = local_y - 0.5 * vh;

            let delta = event.delta.pixel_delta(px(24.0));
            let dy = f32::from(delta.y);
            if dy.abs() > 0.01 {
                let factor = if dy > 0.0 {
                    (1.0 + (dy / 160.0).min(0.35)).max(1.04)
                } else {
                    (1.0 / (1.0 + ((-dy) / 160.0).min(0.35))).min(0.96)
                };
                on_scroll_cb(
                    StudioEvent::ZoomAtCursor {
                        factor,
                        rel_x,
                        rel_y,
                    },
                    window,
                    cx,
                );
            }
        })
        // Trackpad pinch-to-zoom gesture
        .on_pinch(move |event: &PinchEvent, window, cx| {
            let (vw, vh) = compute_viewport_bounds(window, drawer_open);
            let local_x = f32::from(event.position.x) - VIEWPORT_LEFT_OFFSET;
            let local_y = f32::from(event.position.y) - VIEWPORT_TOP_OFFSET;
            let rel_x = local_x - 0.5 * vw;
            let rel_y = local_y - 0.5 * vh;
            let factor = (1.0 + event.delta).clamp(0.7, 1.45);
            on_pinch_cb(
                StudioEvent::ZoomAtCursor {
                    factor,
                    rel_x,
                    rel_y,
                },
                window,
                cx,
            );
        })
        // Mouse down (Left or Middle) starts panning / brush painting / tool click
        .on_mouse_down(MouseButton::Left, move |event: &MouseDownEvent, window, cx| {
            let (vw, vh) = compute_viewport_bounds(window, drawer_open);
            let local_x = f32::from(event.position.x) - VIEWPORT_LEFT_OFFSET;
            let local_y = f32::from(event.position.y) - VIEWPORT_TOP_OFFSET;
            on_down_left(
                StudioEvent::BeginPan {
                    x: local_x,
                    y: local_y,
                },
                window,
                cx,
            );
            on_down_left(
                StudioEvent::CanvasClick {
                    x: local_x,
                    y: local_y,
                    view_w: vw,
                    view_h: vh,
                },
                window,
                cx,
            );
        })
        .on_mouse_down(MouseButton::Middle, move |event: &MouseDownEvent, window, cx| {
            let local_x = f32::from(event.position.x) - VIEWPORT_LEFT_OFFSET;
            let local_y = f32::from(event.position.y) - VIEWPORT_TOP_OFFSET;
            on_down_mid(
                StudioEvent::BeginPan {
                    x: local_x,
                    y: local_y,
                },
                window,
                cx,
            );
        })
        // Mouse move updates pan drag / brush stroke and live pixel coordinate/intensity probe
        .on_mouse_move(move |event: &MouseMoveEvent, window, cx| {
            let (vw, vh) = compute_viewport_bounds(window, drawer_open);
            let local_x = f32::from(event.position.x) - VIEWPORT_LEFT_OFFSET;
            let local_y = f32::from(event.position.y) - VIEWPORT_TOP_OFFSET;
            on_move_cb(
                StudioEvent::PointerMove {
                    x: local_x,
                    y: local_y,
                    view_w: vw,
                    view_h: vh,
                },
                window,
                cx,
            );
        })
        .on_mouse_up(MouseButton::Left, move |_: &MouseUpEvent, window, cx| {
            on_up_left(StudioEvent::EndPan, window, cx);
        })
        .on_mouse_up(MouseButton::Middle, move |_: &MouseUpEvent, window, cx| {
            on_up_mid(StudioEvent::EndPan, window, cx);
        })
        .on_mouse_up_out(MouseButton::Left, move |_: &MouseUpEvent, window, cx| {
            on_up_out_left(StudioEvent::EndPan, window, cx);
        })
        // Drag and drop image files or `.ome.zarr` folders from OS file manager directly onto the viewport
        .on_drop(move |paths: &ExternalPaths, window, cx| {
            if let Some(first_path) = paths.paths().first() {
                on_drop_cb(StudioEvent::LoadImageFile(first_path.clone()), window, cx);
            }
        })
        // Positioned & zoomed GPU RenderImage canvas (pure neutral 1px dark border)
        .child(
            div()
                .absolute()
                .left(px(left))
                .top(px(top))
                .w(px(sw.max(4.0)))
                .h(px(sh.max(4.0)))
                .border_1()
                .border_color(palette::border_subtle())
                .shadow_lg()
                .overflow_hidden()
                .child(
                    img(ImageSource::Render(render_img))
                        .w(px(sw.max(4.0)))
                        .h(px(sh.max(4.0))),
                ),
        )
        // Top-bar `Operations ▾` Categorized Dropdown Menu Overlay
        .when(show_ops_menu, |el| {
            el.child(render_operations_dropdown_menu(on_event))
        })
}

fn render_operations_dropdown_menu(
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    let filter_ops = [
        ModifierKind::GaussianBlur,
        ModifierKind::TvChambolleDenoise,
        ModifierKind::RollingBallBackground,
    ];
    let contrast_ops = [ModifierKind::ClaheContrast, ModifierKind::GammaCurve];
    let edge_ops = [
        ModifierKind::SobelEdges,
        ModifierKind::CannyEdges,
        ModifierKind::MorphGradient,
        ModifierKind::OtsuThreshold,
    ];

    div()
        .w(px(280.0))
        .absolute()
        .top(px(4.0))
        .left(px(8.0))
        .py(px(6.0))
        .rounded(px(4.0))
        .bg(palette::panel_bg())
        .border_1()
        .border_color(palette::border_subtle())
        .shadow_lg()
        .flex()
        .flex_col()
        .gap(px(2.0))
        .child(
            div()
                .px(px(12.0))
                .py(px(3.0))
                .text_size(px(10.0))
                .text_color(palette::text_dim())
                .child("FILTER & RESTORE"),
        )
        .children(filter_ops.into_iter().enumerate().map(|(idx, kind)| {
            render_op_menu_item(("ops-menu-filter", idx), kind, on_event.clone())
        }))
        .child(
            div()
                .my(px(3.0))
                .w_full()
                .border_t_1()
                .border_color(palette::border_subtle()),
        )
        .child(
            div()
                .px(px(12.0))
                .py(px(3.0))
                .text_size(px(10.0))
                .text_color(palette::text_dim())
                .child("CONTRAST & INTENSITY"),
        )
        .children(contrast_ops.into_iter().enumerate().map(|(idx, kind)| {
            render_op_menu_item(("ops-menu-contrast", idx), kind, on_event.clone())
        }))
        .child(
            div()
                .my(px(3.0))
                .w_full()
                .border_t_1()
                .border_color(palette::border_subtle()),
        )
        .child(
            div()
                .px(px(12.0))
                .py(px(3.0))
                .text_size(px(10.0))
                .text_color(palette::text_dim())
                .child("EDGES, MORPHOLOGY & SEGMENT"),
        )
        .children(edge_ops.into_iter().enumerate().map(|(idx, kind)| {
            render_op_menu_item(("ops-menu-edge", idx), kind, on_event.clone())
        }))
}

fn render_op_menu_item(
    id: (&'static str, usize),
    kind: ModifierKind,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let title: SharedString = kind.title().into();
    let badge: SharedString = kind.category().into();
    div()
        .id(id)
        .cursor_pointer()
        .px(px(12.0))
        .py(px(5.0))
        .hover(|s| s.bg(palette::accent_blue()))
        .flex()
        .items_center()
        .justify_between()
        .on_click(move |_, window, cx| {
            on_event(StudioEvent::AddModifier(kind), window, cx);
        })
        .child(
            div()
                .text_size(px(11.5))
                .text_color(palette::text_primary())
                .child(title),
        )
        .child(
            div()
                .text_size(px(10.0))
                .text_color(palette::text_secondary())
                .child(badge),
        )
}

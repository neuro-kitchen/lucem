//! Interactive GPU Explore Viewport (`render_interactive_viewport`):
//! Supports cursor-anchored scroll-wheel / trackpad zoom (`on_scroll`, `on_pinch`), click-and-drag
//! panning (`on_mouse_down`, `on_mouse_move`, `on_mouse_up`, `on_mouse_up_out`), OS file drag-and-drop
//! (`ExternalPaths`), and live `(y, x)` pixel coordinate & intensity probing without obstructing the
//! canvas with floating bars.

use std::rc::Rc;

use gpui_kit::{
    div, img, prelude::*, px, App, ExternalPaths, ImageSource, IntoElement, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, PinchEvent, ScrollEvent, Window,
};

use crate::screens::studio::studio_event::StudioEvent;
use crate::screens::studio::studio_state::StudioUiState;
use crate::ui::theme::palette;

/// Estimated left chrome offset (`left_tool_shelf` width = 68px) and top chrome offset (`top_mode_bar` height = 44px)
/// used to convert window mouse coordinates into viewport-local coordinates.
const VIEWPORT_LEFT_OFFSET: f32 = 68.0;
const VIEWPORT_TOP_OFFSET: f32 = 44.0;
const RIGHT_DOCK_WIDTH: f32 = 340.0;
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
    let drawer_open = state.show_analytics_drawer;

    let on_scroll_cb = on_event.clone();
    let on_pinch_cb = on_event.clone();
    let on_down_left = on_event.clone();
    let on_down_mid = on_event.clone();
    let on_move_cb = on_event.clone();
    let on_up_left = on_event.clone();
    let on_up_mid = on_event.clone();
    let on_up_out_left = on_event.clone();
    let on_drop_cb = on_event;

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
        // Scroll-wheel / trackpad 2-finger zoom anchored at cursor
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
        // Mouse down (Left or Middle) starts panning / tool click
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
        // Mouse move updates pan drag and live pixel coordinate/intensity probe
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
        // Subtle viewport grid lines when `show_grid` is active
        .when(show_grid, |el| {
            el.child(
                div()
                    .absolute()
                    .top(px(0.0))
                    .left(px(0.0))
                    .right(px(0.0))
                    .bottom(px(0.0))
                    .border_1()
                    .border_color(palette::border_subtle()),
            )
        })
        // Positioned & zoomed GPU RenderImage canvas
        .child(
            div()
                .absolute()
                .left(px(left))
                .top(px(top))
                .w(px(sw.max(4.0)))
                .h(px(sh.max(4.0)))
                .rounded(px(2.0))
                .border_1()
                .border_color(palette::border_strong())
                .shadow_lg()
                .overflow_hidden()
                .child(
                    img(ImageSource::Render(render_img))
                        .w(px(sw.max(4.0)))
                        .h(px(sh.max(4.0))),
                ),
        )
}

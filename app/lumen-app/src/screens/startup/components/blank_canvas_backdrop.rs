//! Blank infinite canvas backdrop rendered underneath the Startup Project Hub.

use gpui_kit::{div, prelude::*, px, IntoElement, SharedString};

use crate::ui::theme::palette;

/// Renders the dark infinite canvas with subtle grid rulers and a blank artboard preview frame in
/// the background while the Startup Project Hub is open.
pub fn render_blank_canvas_backdrop(
    canvas_width: usize,
    canvas_height: usize,
    compute_backend: &str,
) -> impl IntoElement {
    let dim_tag: SharedString = format!("BLANK ARTBOARD · {canvas_width} × {canvas_height} px").into();
    let backend_tag: SharedString = format!("DEVICE · {compute_backend}").into();

    div()
        .size_full()
        .bg(palette::canvas_bg())
        .flex()
        .flex_col()
        .justify_between()
        // Top coordinate ruler strip
        .child(
            div()
                .w_full()
                .h(px(22.0))
                .bg(palette::panel_bg())
                .border_b_1()
                .border_color(palette::border_subtle())
                .px(px(16.0))
                .flex()
                .items_center()
                .justify_between()
                .text_size(px(10.0))
                .text_color(palette::text_dim())
                .child("0 px        256 px        512 px        768 px        1024 px        1280 px")
                .child(dim_tag),
        )
        // Center faint blank artboard outline
        .child(
            div()
                .flex_1()
                .w_full()
                .flex()
                .items_center()
                .justify_center()
                .child(
                    div()
                        .w(px(760.0))
                        .h(px(520.0))
                        .rounded(px(8.0))
                        .border_1()
                        .border_color(palette::canvas_grid())
                        .bg(palette::panel_bg())
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(
                            div()
                                .text_size(px(11.0))
                                .text_color(palette::text_dim())
                                .child("+  CANVAS ORIGIN (0, 0)"),
                        ),
                ),
        )
        // Bottom subtle status bar on the blank canvas
        .child(
            div()
                .w_full()
                .h(px(26.0))
                .bg(palette::panel_bg())
                .border_t_1()
                .border_color(palette::border_subtle())
                .px(px(16.0))
                .flex()
                .items_center()
                .justify_between()
                .text_size(px(11.0))
                .text_color(palette::text_dim())
                .child("READY · Drop an image file anywhere or select a project in the center hub")
                .child(backend_tag),
        )
}

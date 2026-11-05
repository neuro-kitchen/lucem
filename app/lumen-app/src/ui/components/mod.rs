//! Shared cross-screen UI primitives (`pill_badge`, `section_header`, `scrubber_row`).

use gpui_kit::{div, prelude::*, px, Hsla, IntoElement, SharedString};

use crate::ui::theme::palette;

/// Renders a compact uppercase pill badge (for dtypes, GPU backend, categories, or FPS).
pub fn pill_badge(label: impl Into<SharedString>, bg_color: Hsla, text_color: Hsla) -> impl IntoElement {
    let text: SharedString = label.into();
    div()
        .px(px(7.0))
        .py(px(2.0))
        .rounded(px(4.0))
        .bg(bg_color)
        .text_size(px(10.5))
        .text_color(text_color)
        .child(text)
}

/// Renders a Blender-style uppercase dock section title.
pub fn section_header(title: impl Into<SharedString>, right_tag: Option<SharedString>) -> impl IntoElement {
    let title_str: SharedString = title.into();
    div()
        .w_full()
        .flex()
        .items_center()
        .justify_between()
        .px(px(12.0))
        .py(px(8.0))
        .border_b_1()
        .border_color(palette::border_subtle())
        .child(
            div()
                .text_size(px(11.0))
                .text_color(palette::text_secondary())
                .child(title_str),
        )
        .when_some(right_tag, |el, tag| {
            el.child(
                div()
                    .text_size(px(10.5))
                    .text_color(palette::text_dim())
                    .child(tag),
            )
        })
}

//! Left-hand ArtCraft × Blender tool shelf (`render_left_tool_shelf`) for switching interactive
//! viewport tools (`Pan & Explore`, `Zoom Lens`, `Pixel Probe`, `Contour Trace`) and toggling
//! canvas overlays (Grid, Contours, Invert).

use std::rc::Rc;

use gpui_kit::{div, prelude::*, px, App, IntoElement, Window};

use crate::domain::ActiveTool;
use crate::screens::studio::studio_event::StudioEvent;
use crate::screens::studio::studio_state::StudioUiState;
use crate::ui::theme::palette;

pub fn render_left_tool_shelf(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let active_tool = state.active_tool;
    let show_grid = state.camera.show_grid;
    let show_contours = state.camera.show_contours;
    let invert = state.tonemap.invert;

    let on_grid = on_event.clone();
    let on_contours = on_event.clone();
    let on_invert = on_event.clone();
    let on_reset_tonemap = on_event.clone();

    div()
        .w(px(68.0))
        .h_full()
        .bg(palette::panel_bg())
        .border_r_1()
        .border_color(palette::border_subtle())
        .py(px(12.0))
        .px(px(6.0))
        .flex()
        .flex_col()
        .justify_between()
        .items_center()
        .child(
            div()
                .w_full()
                .flex()
                .flex_col()
                .gap(px(6.0))
                .child(
                    div()
                        .text_size(px(9.5))
                        .text_color(palette::text_dim())
                        .px(px(4.0))
                        .child("TOOLS"),
                )
                .children(ActiveTool::ALL.iter().copied().enumerate().map(|(idx, tool)| {
                    let selected = tool == active_tool;
                    let cb = on_event.clone();
                    div()
                        .id(("tool-shelf-item", idx))
                        .w_full()
                        .py(px(8.0))
                        .px(px(4.0))
                        .rounded(px(6.0))
                        .cursor_pointer()
                        .bg(if selected {
                            palette::accent_cyan_muted()
                        } else {
                            palette::card_bg()
                        })
                        .border_1()
                        .border_color(if selected {
                            palette::accent_cyan()
                        } else {
                            palette::border_subtle()
                        })
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap(px(2.0))
                        .on_click(move |_, window, cx| {
                            cb(StudioEvent::SelectTool(tool), window, cx);
                        })
                        .child(
                            div()
                                .text_size(px(11.0))
                                .text_color(if selected {
                                    palette::accent_cyan()
                                } else {
                                    palette::text_primary()
                                })
                                .child(tool.short_label()),
                        )
                })),
        )
        .child(
            // Bottom overlay toggles
            div()
                .w_full()
                .flex()
                .flex_col()
                .gap(px(6.0))
                .child(
                    div()
                        .text_size(px(9.5))
                        .text_color(palette::text_dim())
                        .px(px(4.0))
                        .child("VIEW"),
                )
                .child(
                    div()
                        .id("shelf-toggle-grid")
                        .w_full()
                        .py(px(6.0))
                        .rounded(px(6.0))
                        .cursor_pointer()
                        .bg(if show_grid {
                            palette::card_hover_bg()
                        } else {
                            palette::card_bg()
                        })
                        .border_1()
                        .border_color(palette::border_subtle())
                        .flex()
                        .justify_center()
                        .on_click(move |_, window, cx| {
                            on_grid(StudioEvent::ToggleGrid, window, cx);
                        })
                        .child(
                            div()
                                .text_size(px(10.5))
                                .text_color(if show_grid {
                                    palette::text_primary()
                                } else {
                                    palette::text_dim()
                                })
                                .child("Grid"),
                        ),
                )
                .child(
                    div()
                        .id("shelf-toggle-contours")
                        .w_full()
                        .py(px(6.0))
                        .rounded(px(6.0))
                        .cursor_pointer()
                        .bg(if show_contours {
                            palette::accent_cyan_muted()
                        } else {
                            palette::card_bg()
                        })
                        .border_1()
                        .border_color(if show_contours {
                            palette::accent_cyan()
                        } else {
                            palette::border_subtle()
                        })
                        .flex()
                        .justify_center()
                        .on_click(move |_, window, cx| {
                            on_contours(StudioEvent::ToggleContours, window, cx);
                        })
                        .child(
                            div()
                                .text_size(px(10.5))
                                .text_color(if show_contours {
                                    palette::accent_cyan()
                                } else {
                                    palette::text_secondary()
                                })
                                .child("Contours"),
                        ),
                )
                .child(
                    div()
                        .id("shelf-toggle-invert")
                        .w_full()
                        .py(px(6.0))
                        .rounded(px(6.0))
                        .cursor_pointer()
                        .bg(if invert {
                            palette::accent_amber()
                        } else {
                            palette::card_bg()
                        })
                        .border_1()
                        .border_color(palette::border_subtle())
                        .flex()
                        .justify_center()
                        .on_click(move |_, window, cx| {
                            on_invert(StudioEvent::ToggleInvert, window, cx);
                        })
                        .child(
                            div()
                                .text_size(px(10.5))
                                .text_color(if invert {
                                    palette::canvas_void()
                                } else {
                                    palette::text_secondary()
                                })
                                .child("Invert"),
                        ),
                )
                .child(
                    div()
                        .id("shelf-reset-tonemap")
                        .w_full()
                        .py(px(6.0))
                        .rounded(px(6.0))
                        .cursor_pointer()
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .flex()
                        .justify_center()
                        .on_click(move |_, window, cx| {
                            on_reset_tonemap(StudioEvent::ResetTonemap, window, cx);
                        })
                        .child(
                            div()
                                .text_size(px(10.5))
                                .text_color(palette::text_secondary())
                                .child("Reset LUT"),
                        ),
                ),
        )
}

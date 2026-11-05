//! Blender-style top header mode bar (`render_top_mode_bar`) with Project Hub switcher, Open Image
//! action, workspace mode tabs (`Craft & Explore`, `GPU Pipeline`, `Polars Analytics`), image
//! badge, and quick zoom controls.

use std::rc::Rc;

use gpui_kit::{div, prelude::*, px, App, IntoElement, SharedString, Window};

use crate::domain::WorkspaceMode;
use crate::screens::studio::studio_event::StudioEvent;
use crate::screens::studio::studio_state::StudioUiState;
use crate::ui::components::pill_badge;
use crate::ui::theme::palette;

pub fn render_top_mode_bar(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let active_mode = state.active_mode;
    let img_info = format!(
        "{} — {}×{} ({})",
        state.source_image.name,
        state.source_image.width,
        state.source_image.height,
        state.source_image.dtype_label
    );
    let zoom_pct = state.camera.zoom_percent_label();
    let drawer_open = state.show_analytics_drawer;

    let on_hub = on_event.clone();
    let on_open = on_event.clone();
    let on_zoom_out = on_event.clone();
    let on_zoom_in = on_event.clone();
    let on_zoom_1to1 = on_event.clone();
    let on_fit = on_event.clone();
    let on_drawer = on_event.clone();

    div()
        .w_full()
        .h(px(44.0))
        .bg(palette::header_bg())
        .border_b_1()
        .border_color(palette::border_subtle())
        .px(px(14.0))
        .flex()
        .items_center()
        .justify_between()
        .child(
            // Left group: Brand + Project Hub button + Open File button
            div()
                .flex()
                .items_center()
                .gap(px(8.0))
                .child(
                    div()
                        .id("top-btn-project-hub")
                        .px(px(10.0))
                        .py(px(5.0))
                        .rounded(px(6.0))
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_strong())
                        .cursor_pointer()
                        .hover(|s| s.bg(palette::card_hover_bg()))
                        .on_click(move |_, window, cx| {
                            on_hub(StudioEvent::OpenProjectHub, window, cx);
                        })
                        .child(
                            div()
                                .text_size(px(11.5))
                                .text_color(palette::accent_cyan())
                                .child("LUMEN HUB"),
                        ),
                )
                .child(
                    div()
                        .id("top-btn-open-image")
                        .px(px(10.0))
                        .py(px(5.0))
                        .rounded(px(6.0))
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .cursor_pointer()
                        .hover(|s| s.bg(palette::card_hover_bg()))
                        .on_click(move |_, window, cx| {
                            on_open(StudioEvent::PromptOpenImage, window, cx);
                        })
                        .child(
                            div()
                                .text_size(px(11.5))
                                .text_color(palette::text_primary())
                                .child("Open Image..."),
                        ),
                )
                .child(pill_badge(
                    SharedString::from(img_info),
                    palette::card_bg(),
                    palette::text_secondary(),
                )),
        )
        .child(
            // Center group: Blender-style Workspace Mode Tabs
            div()
                .flex()
                .items_center()
                .gap(px(4.0))
                .p(px(3.0))
                .rounded(px(7.0))
                .bg(palette::canvas_void())
                .border_1()
                .border_color(palette::border_subtle())
                .children(WorkspaceMode::ALL.iter().copied().enumerate().map(|(idx, mode)| {
                    let selected = mode == active_mode;
                    let cb = on_event.clone();
                    div()
                        .id(("workspace-mode-tab", idx))
                        .px(px(12.0))
                        .py(px(4.0))
                        .rounded(px(5.0))
                        .cursor_pointer()
                        .bg(if selected {
                            palette::accent_cyan_muted()
                        } else {
                            palette::canvas_void()
                        })
                        .text_size(px(11.5))
                        .text_color(if selected {
                            palette::accent_cyan()
                        } else {
                            palette::text_secondary()
                        })
                        .on_click(move |_, window, cx| {
                            cb(StudioEvent::SwitchWorkspaceMode(mode), window, cx);
                        })
                        .child(mode.label())
                })),
        )
        .child(
            // Right group: Explore Zoom Bar + Analytics Drawer toggle
            div()
                .flex()
                .items_center()
                .gap(px(6.0))
                .child(
                    div()
                        .id("top-zoom-out")
                        .px(px(8.0))
                        .py(px(4.0))
                        .rounded(px(5.0))
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .cursor_pointer()
                        .hover(|s| s.bg(palette::card_hover_bg()))
                        .on_click(move |_, window, cx| {
                            on_zoom_out(StudioEvent::ZoomOut, window, cx);
                        })
                        .child(div().text_size(px(12.0)).text_color(palette::text_primary()).child("-")),
                )
                .child(
                    div()
                        .px(px(8.0))
                        .py(px(4.0))
                        .rounded(px(5.0))
                        .bg(palette::canvas_void())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .text_size(px(11.5))
                        .text_color(palette::accent_cyan())
                        .child(zoom_pct),
                )
                .child(
                    div()
                        .id("top-zoom-in")
                        .px(px(8.0))
                        .py(px(4.0))
                        .rounded(px(5.0))
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .cursor_pointer()
                        .hover(|s| s.bg(palette::card_hover_bg()))
                        .on_click(move |_, window, cx| {
                            on_zoom_in(StudioEvent::ZoomIn, window, cx);
                        })
                        .child(div().text_size(px(12.0)).text_color(palette::text_primary()).child("+")),
                )
                .child(
                    div()
                        .id("top-zoom-1to1")
                        .px(px(8.0))
                        .py(px(4.0))
                        .rounded(px(5.0))
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .cursor_pointer()
                        .hover(|s| s.bg(palette::card_hover_bg()))
                        .on_click(move |_, window, cx| {
                            on_zoom_1to1(StudioEvent::ZoomOneToOne, window, cx);
                        })
                        .child(div().text_size(px(11.0)).text_color(palette::text_secondary()).child("1:1")),
                )
                .child(
                    div()
                        .id("top-zoom-fit")
                        .px(px(8.0))
                        .py(px(4.0))
                        .rounded(px(5.0))
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .cursor_pointer()
                        .hover(|s| s.bg(palette::card_hover_bg()))
                        .on_click(move |_, window, cx| {
                            on_fit(StudioEvent::FitToViewport, window, cx);
                        })
                        .child(div().text_size(px(11.0)).text_color(palette::text_secondary()).child("Fit")),
                )
                .child(
                    div()
                        .id("top-toggle-analytics")
                        .px(px(9.0))
                        .py(px(4.0))
                        .rounded(px(5.0))
                        .bg(if drawer_open {
                            palette::accent_cyan_muted()
                        } else {
                            palette::card_bg()
                        })
                        .border_1()
                        .border_color(if drawer_open {
                            palette::accent_cyan()
                        } else {
                            palette::border_subtle()
                        })
                        .cursor_pointer()
                        .on_click(move |_, window, cx| {
                            on_drawer(StudioEvent::ToggleAnalyticsDrawer, window, cx);
                        })
                        .child(
                            div()
                                .text_size(px(11.0))
                                .text_color(if drawer_open {
                                    palette::accent_cyan()
                                } else {
                                    palette::text_secondary()
                                })
                                .child("Polars Table"),
                        ),
                ),
        )
}

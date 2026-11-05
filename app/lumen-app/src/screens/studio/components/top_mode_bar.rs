//! Blender 4.x-style Top Titlebar + App Menu Bar (`render_top_mode_bar`).
//!
//! Includes:
//! - Left: `[LUMEN]` project hub button, `File`, `Zarr`, `Operations ▾` dropdown menu trigger, `View`, `Analyze`
//! - Center: Workspace tabs (`Compositing`, `Vector & Paint`, `RegionProps`) + OME-Zarr Auto-LOD status
//! - Right: Window controls (`—` Minimize, `□` Maximize/Restore, `✕` Close) and drag area.

use std::rc::Rc;

use gpui_kit::{
    div, prelude::*, px, App, IntoElement, SharedString, Window, WindowControlArea,
};

use crate::domain::WorkspaceMode;
use crate::screens::studio::studio_event::StudioEvent;
use crate::screens::studio::studio_state::StudioUiState;
use crate::ui::theme::palette;

pub fn render_top_mode_bar(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let on_hub = on_event.clone();
    let on_open_file = on_event.clone();
    let on_open_zarr = on_event.clone();
    let on_ops_menu = on_event.clone();
    let on_fit_view = on_event.clone();
    let on_analyze = on_event.clone();
    let on_auto_lod = on_event.clone();

    let ops_open = state.show_operations_menu;
    let ops_count = state.modifiers.len();
    let ops_label: SharedString = if ops_count > 0 {
        format!("Operations ({ops_count}) ▾").into()
    } else {
        "Operations ▾".into()
    };

    let lod_badge: Option<(SharedString, bool)> = state.source_image.ome_zarr.as_ref().map(|z| {
        let mode_str = if z.auto_pyramid { "Auto-LOD" } else { "Pinned" };
        let txt = format!(
            "OME-Zarr {mode_str}: {} ({}×{})",
            z.active_level, state.source_image.width, state.source_image.height
        );
        (SharedString::from(txt), z.auto_pyramid)
    });

    div()
        .id("studio-top-titlebar")
        .w_full()
        .h(px(34.0))
        .px(px(8.0))
        .bg(palette::header_bg())
        .border_b_1()
        .border_color(palette::border_subtle())
        .window_control_area(WindowControlArea::Drag)
        .flex()
        .items_center()
        .justify_between()
        // Left: Brand + Application Menu Bar (`File`, `Zarr`, `Operations ▾`, `View`, `Analyze`)
        .child(
            div()
                .flex()
                .items_center()
                .gap(px(3.0))
                .child(
                    div()
                        .id("top-menu-hub-btn")
                        .cursor_pointer()
                        .px(px(8.0))
                        .py(px(3.0))
                        .rounded(px(4.0))
                        .bg(palette::widget_bg())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .text_size(px(11.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_hub(StudioEvent::OpenProjectHub, window, cx);
                        })
                        .child("LUMEN"),
                )
                .child(
                    div()
                        .id("top-menu-file-btn")
                        .cursor_pointer()
                        .px(px(8.0))
                        .py(px(3.0))
                        .rounded(px(4.0))
                        .hover(|s| s.bg(palette::widget_bg()))
                        .text_size(px(11.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_open_file(StudioEvent::PromptOpenImage, window, cx);
                        })
                        .child("File"),
                )
                .child(
                    div()
                        .id("top-menu-zarr-btn")
                        .cursor_pointer()
                        .px(px(8.0))
                        .py(px(3.0))
                        .rounded(px(4.0))
                        .hover(|s| s.bg(palette::widget_bg()))
                        .text_size(px(11.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_open_zarr(StudioEvent::PromptOpenZarrFolder, window, cx);
                        })
                        .child("Open Zarr"),
                )
                .child(
                    div()
                        .id("top-menu-operations-btn")
                        .cursor_pointer()
                        .px(px(9.0))
                        .py(px(3.0))
                        .rounded(px(4.0))
                        .bg(if ops_open {
                            palette::accent_blue()
                        } else {
                            palette::widget_bg()
                        })
                        .hover(|s| s.bg(palette::accent_blue()))
                        .text_size(px(11.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_ops_menu(StudioEvent::ToggleOperationsMenu, window, cx);
                        })
                        .child(ops_label),
                )
                .child(
                    div()
                        .id("top-menu-view-btn")
                        .cursor_pointer()
                        .px(px(8.0))
                        .py(px(3.0))
                        .rounded(px(4.0))
                        .hover(|s| s.bg(palette::widget_bg()))
                        .text_size(px(11.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_fit_view(StudioEvent::FitToViewport, window, cx);
                        })
                        .child("View"),
                )
                .child(
                    div()
                        .id("top-menu-analyze-btn")
                        .cursor_pointer()
                        .px(px(8.0))
                        .py(px(3.0))
                        .rounded(px(4.0))
                        .bg(if state.show_analytics_drawer {
                            palette::surface_card()
                        } else {
                            palette::header_bg()
                        })
                        .hover(|s| s.bg(palette::widget_bg()))
                        .text_size(px(11.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_analyze(StudioEvent::ToggleAnalyticsDrawer, window, cx);
                        })
                        .child("Analyze"),
                ),
        )
        // Center: Blender Workspace Mode Tabs + OME-Zarr Auto-LOD Indicator
        .child(
            div()
                .flex()
                .items_center()
                .gap(px(4.0))
                .children(WorkspaceMode::ALL.iter().enumerate().map(|(idx, &mode)| {
                    let is_active = mode == state.active_mode;
                    let cb = on_event.clone();
                    div()
                        .id(("top-workspace-tab", idx))
                        .cursor_pointer()
                        .px(px(10.0))
                        .py(px(3.0))
                        .rounded(px(4.0))
                        .bg(if is_active {
                            palette::accent_blue()
                        } else {
                            palette::surface_card()
                        })
                        .hover(|s| s.bg(palette::surface_hover()))
                        .text_size(px(11.5))
                        .text_color(if is_active {
                            palette::text_primary()
                        } else {
                            palette::text_secondary()
                        })
                        .on_click(move |_, window, cx| {
                            cb(StudioEvent::SwitchWorkspaceMode(mode), window, cx);
                        })
                        .child(mode.label())
                }))
                .when_some(lod_badge, |el, (badge_txt, is_auto)| {
                    el.child(
                        div()
                            .id("top-ome-zarr-lod-badge")
                            .cursor_pointer()
                            .ml(px(8.0))
                            .px(px(8.0))
                            .py(px(2.5))
                            .rounded(px(4.0))
                            .bg(if is_auto {
                                palette::surface_card()
                            } else {
                                palette::widget_bg()
                            })
                            .hover(|s| s.bg(palette::surface_hover()))
                            .text_size(px(11.0))
                            .text_color(palette::text_secondary())
                            .on_click(move |_, window, cx| {
                                on_auto_lod(StudioEvent::ToggleOmeZarrAutoPyramid, window, cx);
                            })
                            .child(badge_txt),
                    )
                }),
        )
        // Right: Zoom percentage readout + Window Controls (Minimize, Maximize/Restore, Close)
        .child(
            div()
                .flex()
                .items_center()
                .gap(px(4.0))
                .child(
                    div()
                        .px(px(6.0))
                        .text_size(px(11.0))
                        .text_color(palette::text_dim())
                        .child(SharedString::from(format!(
                            "{:.0}%",
                            state.camera.zoom * 100.0
                        ))),
                )
                .child(
                    div()
                        .id("win-ctrl-minimize")
                        .cursor_pointer()
                        .window_control_area(WindowControlArea::Min)
                        .w(px(24.0))
                        .h(px(22.0))
                        .rounded(px(3.0))
                        .hover(|s| s.bg(palette::widget_bg()))
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_size(px(12.0))
                        .text_color(palette::text_secondary())
                        .on_click(|_, window, _| {
                            window.minimize_window();
                        })
                        .child("—"),
                )
                .child(
                    div()
                        .id("win-ctrl-maximize")
                        .cursor_pointer()
                        .window_control_area(WindowControlArea::Max)
                        .w(px(24.0))
                        .h(px(22.0))
                        .rounded(px(3.0))
                        .hover(|s| s.bg(palette::widget_bg()))
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_size(px(11.5))
                        .text_color(palette::text_secondary())
                        .on_click(|_, window, _| {
                            window.zoom_window();
                        })
                        .child("□"),
                )
                .child(
                    div()
                        .id("win-ctrl-close")
                        .cursor_pointer()
                        .window_control_area(WindowControlArea::Close)
                        .w(px(24.0))
                        .h(px(22.0))
                        .rounded(px(3.0))
                        .hover(|s| s.bg(palette::accent_coral()))
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_size(px(11.5))
                        .text_color(palette::text_secondary())
                        .on_click(|_, window, _| {
                            window.remove_window();
                        })
                        .child("✕"),
                ),
        )
}

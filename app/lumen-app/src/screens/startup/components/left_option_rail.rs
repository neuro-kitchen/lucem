//! Persistent Left Option Menu / Rail (`left_option_rail.rs`) shown on the left edge of the app.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::Icon;
use gpui_kit::{div, prelude::*, px, App, IntoElement, SharedString, Window};

use crate::navigation::LeftRailTab;
use crate::screens::startup::StartupEvent;
use crate::ui::theme::palette;

/// Renders the left option menu with quick project, file, canvas, phantom, and analytics actions.
pub fn render_left_option_rail(
    active_tab: LeftRailTab,
    compute_backend: &str,
    on_event: Rc<dyn Fn(StartupEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let items: [(LeftRailTab, IconName, &'static str); 5] = [
        (LeftRailTab::ProjectHub, IconName::LayoutDashboard, "Project Hub"),
        (LeftRailTab::NewCanvas, IconName::Frame, "Blank Canvas"),
        (LeftRailTab::SamplePhantoms, IconName::Sparkles, "Sample Lab"),
        (LeftRailTab::StudioCanvas, IconName::SlidersHorizontal, "Studio View"),
        (LeftRailTab::AnalyticsDrawer, IconName::ChartPie, "Polars Table"),
    ];

    let on_open = on_event.clone();
    let on_open_zarr = on_event.clone();
    let on_blank = on_event.clone();
    let backend_str: SharedString = compute_backend.to_string().into();

    div()
        .w(px(210.0))
        .h_full()
        .bg(palette::panel_bg())
        .border_r_1()
        .border_color(palette::border_subtle())
        .flex()
        .flex_col()
        .justify_between()
        .p(px(12.0))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(14.0))
                // Brand header: LUMEN (ArtCraft × VectorCraft × Blender)
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(10.0))
                        .px(px(6.0))
                        .py(px(4.0))
                        .child(
                            div()
                                .w(px(28.0))
                                .h(px(28.0))
                                .rounded(px(6.0))
                                .bg(palette::accent_blue())
                                .flex()
                                .items_center()
                                .justify_center()
                                .text_size(px(14.0))
                                .text_color(palette::text_primary())
                                .child("L"),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .child(
                                    div()
                                        .text_size(px(14.0))
                                        .text_color(palette::text_primary())
                                        .child("LUMEN STUDIO"),
                                )
                                .child(
                                    div()
                                        .text_size(px(10.0))
                                        .text_color(palette::text_dim())
                                        .child("Art × Vector × Blender"),
                                ),
                        ),
                )
                // Primary action button: Open Image from disk
                .child(
                    div()
                        .id("rail-open-image-btn")
                        .cursor_pointer()
                        .w_full()
                        .px(px(12.0))
                        .py(px(9.0))
                        .rounded(px(6.0))
                        .bg(palette::accent_blue())
                        .hover(|s| s.opacity(0.92))
                        .flex()
                        .items_center()
                        .justify_center()
                        .gap(px(8.0))
                        .text_size(px(12.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_open(StartupEvent::PromptOpenImageFile, window, cx);
                        })
                        .child(Icon::new(IconName::FolderOpen))
                        .child("Open Image File..."),
                )
                // Open .ome.zarr Container Folder button
                .child(
                    div()
                        .id("rail-open-zarr-btn")
                        .cursor_pointer()
                        .w_full()
                        .px(px(12.0))
                        .py(px(8.0))
                        .rounded(px(6.0))
                        .bg(palette::accent_cyan_muted())
                        .border_1()
                        .border_color(palette::accent_cyan())
                        .hover(|s| s.opacity(0.92))
                        .flex()
                        .items_center()
                        .justify_center()
                        .gap(px(8.0))
                        .text_size(px(12.0))
                        .text_color(palette::accent_cyan())
                        .on_click(move |_, window, cx| {
                            on_open_zarr(StartupEvent::PromptOpenZarrFolder, window, cx);
                        })
                        .child(Icon::new(IconName::Folder))
                        .child("Open .ome.zarr..."),
                )
                // Secondary action button: New Blank Canvas
                .child(
                    div()
                        .id("rail-enter-blank-btn")
                        .cursor_pointer()
                        .w_full()
                        .px(px(12.0))
                        .py(px(8.0))
                        .rounded(px(6.0))
                        .bg(palette::surface_card())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .flex()
                        .items_center()
                        .justify_center()
                        .gap(px(8.0))
                        .text_size(px(12.0))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_blank(StartupEvent::EnterBlankCanvasNow, window, cx);
                        })
                        .child(Icon::new(IconName::Plus))
                        .child("Start Blank Canvas"),
                )
                // Navigation section
                .child(
                    div()
                        .pt(px(6.0))
                        .flex()
                        .flex_col()
                        .gap(px(4.0))
                        .child(
                            div()
                                .px(px(8.0))
                                .pb(px(4.0))
                                .text_size(px(10.0))
                                .text_color(palette::text_dim())
                                .child("WORKSPACE MENU"),
                        )
                        .children(items.into_iter().enumerate().map(|(idx, (tab, icon, label))| {
                            let is_active = tab == active_tab;
                            let cb = on_event.clone();
                            div()
                                .id(("left-rail-item", idx))
                                .cursor_pointer()
                                .w_full()
                                .px(px(10.0))
                                .py(px(8.0))
                                .rounded(px(6.0))
                                .bg(if is_active {
                                    palette::accent_blue_soft()
                                } else {
                                    palette::panel_bg()
                                })
                                .border_1()
                                .border_color(if is_active {
                                    palette::border_accent()
                                } else {
                                    palette::panel_bg()
                                })
                                .hover(|s| s.bg(palette::surface_hover()))
                                .flex()
                                .items_center()
                                .gap(px(9.0))
                                .text_size(px(12.5))
                                .text_color(if is_active {
                                    palette::text_primary()
                                } else {
                                    palette::text_secondary()
                                })
                                .on_click(move |_, window, cx| {
                                    cb(StartupEvent::SelectRailTab(tab), window, cx);
                                })
                                .child(Icon::new(icon))
                                .child(label)
                        })),
                ),
        )
        // Bottom CubeCL device card
        .child(
            div()
                .w_full()
                .p(px(10.0))
                .rounded(px(6.0))
                .bg(palette::surface_card())
                .border_1()
                .border_color(palette::border_subtle())
                .flex()
                .flex_col()
                .gap(px(4.0))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(6.0))
                        .text_size(px(10.5))
                        .text_color(palette::accent_emerald())
                        .child(Icon::new(IconName::Cpu))
                        .child("CUBECL GPU RUNTIME"),
                )
                .child(
                    div()
                        .text_size(px(11.0))
                        .text_color(palette::text_secondary())
                        .child(backend_str),
                ),
        )
}

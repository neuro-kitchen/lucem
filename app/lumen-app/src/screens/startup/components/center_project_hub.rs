//! VectorCraft-style central floating Project Hub modal (`center_project_hub.rs`) over the blank canvas.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::Icon;
use gpui_kit::{div, prelude::*, px, App, IntoElement, SharedString, Window};
use lumen_io::PhantomKind;

use crate::navigation::LeftRailTab;
use crate::screens::startup::{StartupEvent, StartupUiState};
use crate::ui::components::pill_badge;
use crate::ui::theme::palette;

/// Renders the central VectorCraft-style project selector card floating in the middle of the blank canvas.
pub fn render_center_project_hub(
    state: &StartupUiState,
    on_event: Rc<dyn Fn(StartupEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let on_open_file = on_event.clone();
    let on_dismiss = on_event.clone();

    div()
        .w(px(780.0))
        .rounded(px(12.0))
        .bg(palette::panel_bg())
        .border_1()
        .border_color(palette::border_subtle())
        .flex()
        .flex_col()
        .overflow_hidden()
        // Modal top header bar
        .child(
            div()
                .w_full()
                .px(px(22.0))
                .py(px(16.0))
                .bg(palette::surface_card())
                .border_b_1()
                .border_color(palette::border_subtle())
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(3.0))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(10.0))
                                .child(
                                    div()
                                        .text_size(px(17.0))
                                        .text_color(palette::text_primary())
                                        .child("Select or Create a Lumen Project"),
                                )
                                .child(pill_badge(
                                    "PHASE 1 · GPU EXPLORE",
                                    palette::accent_blue_soft(),
                                    palette::accent_blue(),
                                )),
                        )
                        .child(
                            div()
                                .text_size(px(12.0))
                                .text_color(palette::text_secondary())
                                .child("Open an image from disk, start on a blank canvas, or launch a live GPU pipeline preset."),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.0))
                        .child(
                            div()
                                .id("hub-open-image-btn")
                                .cursor_pointer()
                                .px(px(13.0))
                                .py(px(7.0))
                                .rounded(px(6.0))
                                .bg(palette::accent_blue())
                                .hover(|s| s.opacity(0.92))
                                .flex()
                                .items_center()
                                .gap(px(6.0))
                                .text_size(px(12.0))
                                .text_color(palette::text_primary())
                                .on_click(move |_, window, cx| {
                                    on_open_file(StartupEvent::PromptOpenImageFile, window, cx);
                                })
                                .child(Icon::new(IconName::FolderOpen))
                                .child("Load Image..."),
                        )
                        .child(
                            div()
                                .id("hub-close-to-canvas-btn")
                                .cursor_pointer()
                                .px(px(11.0))
                                .py(px(7.0))
                                .rounded(px(6.0))
                                .bg(palette::panel_bg())
                                .border_1()
                                .border_color(palette::border_subtle())
                                .hover(|s| s.bg(palette::surface_hover()))
                                .text_size(px(12.0))
                                .text_color(palette::text_secondary())
                                .on_click(move |_, window, cx| {
                                    on_dismiss(StartupEvent::EnterBlankCanvasNow, window, cx);
                                })
                                .child("Blank Canvas →"),
                        ),
                ),
        )
        // Optional error/status banner
        .when_some(state.status_banner.clone(), |el, msg| {
            let banner_str: SharedString = msg.into();
            el.child(
                div()
                    .w_full()
                    .px(px(22.0))
                    .py(px(8.0))
                    .bg(palette::accent_blue_soft())
                    .border_b_1()
                    .border_color(palette::border_accent())
                    .text_size(px(12.0))
                    .text_color(palette::text_primary())
                    .child(banner_str),
            )
        })
        // Body switches between New Blank Canvas configurator, Sample Phantoms, or Curated Presets
        .child(match state.active_rail_tab {
            LeftRailTab::NewCanvas => render_new_canvas_tab(state, on_event),
            LeftRailTab::SamplePhantoms => render_phantoms_tab(on_event),
            _ => render_presets_and_dropzone_tab(state, on_event),
        })
}

fn render_presets_and_dropzone_tab(
    state: &StartupUiState,
    on_event: Rc<dyn Fn(StartupEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    let on_drop_click = on_event.clone();
    div()
        .p(px(20.0))
        .flex()
        .flex_col()
        .gap(px(16.0))
        // Drag-and-drop / click to load image banner
        .child(
            div()
                .id("hub-dropzone-card")
                .cursor_pointer()
                .w_full()
                .p(px(16.0))
                .rounded(px(8.0))
                .bg(palette::surface_card())
                .border_1()
                .border_color(palette::border_accent())
                .hover(|s| s.bg(palette::surface_hover()))
                .flex()
                .items_center()
                .justify_between()
                .on_click(move |_, window, cx| {
                    on_drop_click(StartupEvent::PromptOpenImageFile, window, cx);
                })
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(14.0))
                        .child(
                            div()
                                .w(px(40.0))
                                .h(px(40.0))
                                .rounded(px(8.0))
                                .bg(palette::accent_blue_soft())
                                .flex()
                                .items_center()
                                .justify_center()
                                .text_color(palette::accent_blue())
                                .child(Icon::new(IconName::Image)),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(2.0))
                                .child(
                                    div()
                                        .text_size(px(13.5))
                                        .text_color(palette::text_primary())
                                        .child("Open Image from Disk or Drag & Drop File onto Window"),
                                )
                                .child(
                                    div()
                                        .text_size(px(11.5))
                                        .text_color(palette::text_secondary())
                                        .child("Supports PNG, TIFF (8/16/32-bit), JPEG, WebP, BMP · Automatic CubeCL GPU upload"),
                                ),
                        ),
                )
                .child(pill_badge(
                    "BROWSE FILES",
                    palette::accent_blue(),
                    palette::text_primary(),
                )),
        )
        // Discovered `.data/` & recent datasets section
        .when(!state.recent_files.is_empty(), |el| {
            el.child(
                div()
                    .w_full()
                    .flex()
                    .flex_col()
                    .gap(px(8.0))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_size(px(11.5))
                                    .text_color(palette::accent_cyan())
                                    .child("WORKSPACE .data/ & RECENT DATASETS (ONE-CLICK GPU BENCHMARK)"),
                            )
                            .child(
                                div()
                                    .text_size(px(10.5))
                                    .text_color(palette::text_dim())
                                    .child("OME-Zarr v0.4 (Blosc-LZ4) & Image Files"),
                            ),
                    )
                    .child(
                        div()
                            .w_full()
                            .flex()
                            .flex_wrap()
                            .gap(px(8.0))
                            .children(state.recent_files.iter().enumerate().map(|(idx, path)| {
                                let p_clone = path.clone();
                                let cb = on_event.clone();
                                let display_str = path
                                    .strip_prefix(
                                        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
                                    )
                                    .unwrap_or(path.as_path())
                                    .display()
                                    .to_string();
                                div()
                                    .id(("hub-data-path-chip", idx))
                                    .cursor_pointer()
                                    .px(px(11.0))
                                    .py(px(7.0))
                                    .rounded(px(6.0))
                                    .bg(palette::surface_card())
                                    .border_1()
                                    .border_color(palette::accent_cyan())
                                    .hover(|s| s.bg(palette::surface_hover()))
                                    .flex()
                                    .items_center()
                                    .gap(px(6.0))
                                    .on_click(move |_, window, cx| {
                                        cb(StartupEvent::OpenImagePath(p_clone.clone()), window, cx);
                                    })
                                    .child(pill_badge(
                                        ".DATA",
                                        palette::accent_cyan_muted(),
                                        palette::accent_cyan(),
                                    ))
                                    .child(
                                        div()
                                            .text_size(px(11.5))
                                            .text_color(palette::text_primary())
                                            .child(SharedString::from(display_str)),
                                    )
                            })),
                    ),
            )
        })
        // Curated Project Presets header
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(11.5))
                        .text_color(palette::text_dim())
                        .child("WORKFLOW TEMPLATES (ARTCRAFT × VECTORCRAFT × BLENDER)"),
                )
                .child(
                    div()
                        .text_size(px(11.0))
                        .text_color(palette::text_dim())
                        .child("Click any card to open in Studio"),
                ),
        )
        // 2x2 grid of curated presets
        .child(
            div()
                .grid()
                .grid_cols(2)
                .gap(px(12.0))
                .children(state.presets.iter().enumerate().map(|(idx, preset)| {
                    let cb = on_event.clone();
                    let title: SharedString = preset.title.into();
                    let cat: SharedString = preset.category.into();
                    let desc: SharedString = preset.description.into();
                    let lut_label: SharedString = format!("LUT: {}", preset.colormap.label()).into();

                    div()
                        .id(("hub-preset-card", idx))
                        .cursor_pointer()
                        .p(px(14.0))
                        .rounded(px(8.0))
                        .bg(palette::surface_card())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .hover(|s| s.bg(palette::surface_hover()).border_color(palette::border_accent()))
                        .flex()
                        .flex_col()
                        .justify_between()
                        .gap(px(8.0))
                        .on_click(move |_, window, cx| {
                            cb(StartupEvent::LaunchPreset(idx), window, cx);
                        })
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(5.0))
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .child(pill_badge(
                                            cat,
                                            palette::accent_blue_soft(),
                                            palette::accent_blue(),
                                        ))
                                        .child(
                                            div()
                                                .text_size(px(10.5))
                                                .text_color(palette::text_dim())
                                                .child(lut_label),
                                        ),
                                )
                                .child(
                                    div()
                                        .text_size(px(13.5))
                                        .text_color(palette::text_primary())
                                        .child(title),
                                )
                                .child(
                                    div()
                                        .text_size(px(11.5))
                                        .text_color(palette::text_secondary())
                                        .child(desc),
                                ),
                        )
                })),
        )
}

fn render_new_canvas_tab(
    state: &StartupUiState,
    on_event: Rc<dyn Fn(StartupEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    let sizes: [(usize, usize, &'static str); 4] = [
        (512, 512, "512 × 512 (Square Tile)"),
        (1024, 768, "1024 × 768 (Standard Studio)"),
        (1920, 1080, "1920 × 1080 (Full HD Canvas)"),
        (2048, 2048, "2048 × 2048 (High-Res 4MP)"),
    ];
    let backgrounds: [(f32, &'static str); 4] = [
        (0.0, " Pitch Black (0.00)"),
        (0.08, "Obsidian Studio (0.08)"),
        (0.5, "Neutral Gray (0.50)"),
        (1.0, "Paper White (1.00)"),
    ];

    let on_create = on_event.clone();

    div()
        .p(px(22.0))
        .flex()
        .flex_col()
        .gap(px(16.0))
        .child(
            div()
                .text_size(px(13.0))
                .text_color(palette::text_primary())
                .child("Configure Blank Canvas Dimensions & Background"),
        )
        .child(
            div()
                .grid()
                .grid_cols(2)
                .gap(px(10.0))
                .children(sizes.into_iter().enumerate().map(|(i, (w, h, label))| {
                    let active = state.blank_spec.width == w && state.blank_spec.height == h;
                    let cb = on_event.clone();
                    div()
                        .id(("blank-size-opt", i))
                        .cursor_pointer()
                        .p(px(12.0))
                        .rounded(px(6.0))
                        .bg(if active {
                            palette::accent_blue_soft()
                        } else {
                            palette::surface_card()
                        })
                        .border_1()
                        .border_color(if active {
                            palette::border_accent()
                        } else {
                            palette::border_subtle()
                        })
                        .text_size(px(12.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            cb(StartupEvent::SelectBlankSize(w, h), window, cx);
                        })
                        .child(label)
                })),
        )
        .child(
            div()
                .text_size(px(12.0))
                .text_color(palette::text_secondary())
                .child("Canvas Fill Intensity"),
        )
        .child(
            div()
                .grid()
                .grid_cols(4)
                .gap(px(10.0))
                .children(backgrounds.into_iter().enumerate().map(|(i, (val, label))| {
                    let active = (state.blank_spec.background_intensity - val).abs() < 1e-3;
                    let cb = on_event.clone();
                    div()
                        .id(("blank-bg-opt", i))
                        .cursor_pointer()
                        .p(px(10.0))
                        .rounded(px(6.0))
                        .bg(if active {
                            palette::accent_blue_soft()
                        } else {
                            palette::surface_card()
                        })
                        .border_1()
                        .border_color(if active {
                            palette::border_accent()
                        } else {
                            palette::border_subtle()
                        })
                        .text_size(px(11.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            cb(StartupEvent::SelectBlankBackground(val), window, cx);
                        })
                        .child(label)
                })),
        )
        .child(
            div()
                .pt(px(6.0))
                .flex()
                .justify_end()
                .child(
                    div()
                        .id("create-blank-canvas-confirm")
                        .cursor_pointer()
                        .px(px(18.0))
                        .py(px(9.0))
                        .rounded(px(6.0))
                        .bg(palette::accent_blue())
                        .hover(|s| s.opacity(0.92))
                        .text_size(px(13.0))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_create(StartupEvent::CreateBlankCanvas, window, cx);
                        })
                        .child("Create Blank Canvas & Open Studio"),
                ),
        )
}

fn render_phantoms_tab(
    on_event: Rc<dyn Fn(StartupEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    div()
        .p(px(20.0))
        .flex()
        .flex_col()
        .gap(px(12.0))
        .child(
            div()
                .text_size(px(12.5))
                .text_color(palette::text_secondary())
                .child("Synthetic GPU Calibration & Bio-Imaging Phantoms (Generated in RAM & Uploaded to CubeCL)"),
        )
        .child(
            div()
                .grid()
                .grid_cols(2)
                .gap(px(12.0))
                .children(PhantomKind::ALL.iter().copied().enumerate().map(|(idx, kind)| {
                    let cb = on_event.clone();
                    div()
                        .id(("phantom-card", idx))
                        .cursor_pointer()
                        .p(px(14.0))
                        .rounded(px(8.0))
                        .bg(palette::surface_card())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .hover(|s| s.bg(palette::surface_hover()).border_color(palette::border_accent()))
                        .flex()
                        .flex_col()
                        .gap(px(6.0))
                        .on_click(move |_, window, cx| {
                            cb(StartupEvent::LoadPhantom(kind), window, cx);
                        })
                        .child(pill_badge(
                            kind.badge(),
                            palette::accent_blue_soft(),
                            palette::accent_emerald(),
                        ))
                        .child(
                            div()
                                .text_size(px(13.5))
                                .text_color(palette::text_primary())
                                .child(kind.title()),
                        )
                        .child(
                            div()
                                .text_size(px(11.5))
                                .text_color(palette::text_secondary())
                                .child(kind.subtitle()),
                        )
                })),
        )
}

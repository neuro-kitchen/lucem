//! Left Blender-style Outliner · Operations Tree Pane (`render_left_tool_shelf`).
//!
//! Displays the non-destructive processing pipeline as a clean hierarchical tree:
//! - `▾ [00] Source Input` (with OME-Zarr Auto-LOD / Pyramid Scale `s0..s3`, Channels, Label Masks, Layers)
//! - `├─ [01..N]` Applied GPU Operations (with enable/bypass, parameter `- / +`, and `✕` remove)
//! - `└─ [Out] Display & LUT` summary

use std::rc::Rc;

use gpui_kit::{div, prelude::*, px, App, IntoElement, SharedString, Window};

use crate::screens::studio::studio_event::StudioEvent;
use crate::screens::studio::studio_state::StudioUiState;
use crate::ui::theme::palette;

pub fn render_left_tool_shelf(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let on_clear = on_event.clone();
    let on_open_ops = on_event.clone();
    let has_mods = !state.modifiers.is_empty();

    div()
        .id("studio-left-outliner-tree")
        .w(px(268.0))
        .h_full()
        .bg(palette::panel_bg())
        .border_r_1()
        .border_color(palette::border_subtle())
        .flex()
        .flex_col()
        .overflow_hidden()
        // Outliner Header
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
                        .child("Outliner · Operations Tree"),
                )
                .child(if has_mods {
                    div()
                        .id("outliner-clear-ops-btn")
                        .cursor_pointer()
                        .px(px(6.0))
                        .py(px(1.5))
                        .rounded(px(3.0))
                        .bg(palette::widget_bg())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .text_size(px(10.5))
                        .text_color(palette::text_secondary())
                        .on_click(move |_, window, cx| {
                            on_clear(StudioEvent::ClearModifiers, window, cx);
                        })
                        .child("Clear")
                } else {
                    div()
                        .id("outliner-add-op-shortcut")
                        .cursor_pointer()
                        .px(px(6.0))
                        .py(px(1.5))
                        .rounded(px(3.0))
                        .bg(palette::widget_bg())
                        .hover(|s| s.bg(palette::accent_blue()))
                        .text_size(px(10.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_open_ops(StudioEvent::ToggleOperationsMenu, window, cx);
                        })
                        .child("+ Add Op")
                }),
        )
        // Scrollable Tree Body
        .child(
            div()
                .id("outliner-tree-scroll")
                .flex_1()
                .overflow_y_scroll()
                .p(px(8.0))
                .flex()
                .flex_col()
                .gap(px(6.0))
                .child(render_root_source_node(state, on_event.clone()))
                .child(render_operation_steps(state, on_event))
                .child(render_output_summary_node(state)),
        )
}

fn render_root_source_node(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    let src_name: SharedString = state.source_image.name.clone().into();
    let res_str: SharedString = format!(
        "{}×{} · {}",
        state.source_image.width, state.source_image.height, state.source_image.dtype_label
    )
    .into();

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
                        .text_size(px(11.5))
                        .text_color(palette::text_primary())
                        .child("▾ [00] Source Input"),
                )
                .child(
                    div()
                        .text_size(px(10.5))
                        .text_color(palette::text_dim())
                        .child(res_str),
                ),
        )
        .child(
            div()
                .pl(px(10.0))
                .text_size(px(11.0))
                .text_color(palette::text_secondary())
                .child(src_name),
        )
        // OME-Zarr Pyramid LOD + Channel + Label Mask controls
        .when_some(state.source_image.ome_zarr.clone(), |el, info| {
            let active_lvl = info.active_level.clone();
            let is_auto = info.auto_pyramid;
            let active_ch = info.active_channel;
            let active_lbl = info.active_label_group.clone();
            let cb_auto = on_event.clone();
            let cb_comp = on_event.clone();

            el.child(
                div()
                    .pl(px(10.0))
                    .pt(px(2.0))
                    .flex()
                    .flex_col()
                    .gap(px(5.0))
                    // Pyramid LOD Row
                    .child(
                        div()
                            .flex()
                            .items_center()
                            . justify_between()
                            .child(
                                div()
                                    .text_size(px(10.5))
                                    .text_color(palette::text_dim())
                                    .child("Pyramid LOD"),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(3.0))
                                    .child(
                                        div()
                                            .id("tree-zarr-auto-lod")
                                            .cursor_pointer()
                                            .px(px(6.0))
                                            .py(px(1.5))
                                            .rounded(px(3.0))
                                            .bg(if is_auto {
                                                palette::accent_blue()
                                            } else {
                                                palette::widget_bg()
                                            })
                                            .hover(|s| s.bg(palette::surface_hover()))
                                            .text_size(px(10.5))
                                            .text_color(palette::text_primary())
                                            .on_click(move |_, window, cx| {
                                                cb_auto(
                                                    StudioEvent::ToggleOmeZarrAutoPyramid,
                                                    window,
                                                    cx,
                                                );
                                            })
                                            .child("Auto"),
                                    )
                                    .children(info.pyramid_levels.into_iter().enumerate().map(
                                        |(idx, lvl)| {
                                            let is_curr = lvl == active_lvl;
                                            let lvl_arg = lvl.clone();
                                            let cb = on_event.clone();
                                            div()
                                                .id(("tree-zarr-lvl", idx))
                                                .cursor_pointer()
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(3.0))
                                                .bg(if is_curr && !is_auto {
                                                    palette::accent_blue()
                                                } else if is_curr {
                                                    palette::surface_hover()
                                                } else {
                                                    palette::widget_bg()
                                                })
                                                .hover(|s| s.bg(palette::surface_hover()))
                                                .text_size(px(10.5))
                                                .text_color(if is_curr {
                                                    palette::text_primary()
                                                } else {
                                                    palette::text_secondary()
                                                })
                                                .on_click(move |_, window, cx| {
                                                    cb(
                                                        StudioEvent::SelectOmeZarrLevel(
                                                            lvl_arg.clone(),
                                                        ),
                                                        window,
                                                        cx,
                                                    );
                                                })
                                                .child(SharedString::from(lvl))
                                        },
                                    )),
                            ),
                    )
                    // Channel & Mask Row
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .gap(px(3.0))
                            .child({
                                let is_comp = active_ch.is_none() && active_lbl.is_none();
                                div()
                                    .id("tree-zarr-ch-comp")
                                    .cursor_pointer()
                                    .px(px(6.0))
                                    .py(px(2.0))
                                    .rounded(px(3.0))
                                    .bg(if is_comp {
                                        palette::accent_blue()
                                    } else {
                                        palette::widget_bg()
                                    })
                                    .hover(|s| s.bg(palette::surface_hover()))
                                    .text_size(px(10.5))
                                    .text_color(if is_comp {
                                        palette::text_primary()
                                    } else {
                                        palette::text_secondary()
                                    })
                                    .on_click(move |_, window, cx| {
                                        cb_comp(StudioEvent::SelectOmeZarrChannel(None), window, cx);
                                    })
                                    .child("Composite")
                            })
                            .children(info.channel_labels.into_iter().enumerate().map(
                                |(ch_i, label)| {
                                    let is_active =
                                        active_ch == Some(ch_i) && active_lbl.is_none();
                                    let cb = on_event.clone();
                                    div()
                                        .id(("tree-zarr-ch-item", ch_i))
                                        .cursor_pointer()
                                        .px(px(6.0))
                                        .py(px(2.0))
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
                                            cb(
                                                StudioEvent::SelectOmeZarrChannel(Some(ch_i)),
                                                window,
                                                cx,
                                            );
                                        })
                                        .child(SharedString::from(label))
                                },
                            ))
                            .children(info.label_groups.into_iter().enumerate().map(
                                |(lg_i, group)| {
                                    let is_active = active_lbl.as_deref() == Some(group.as_str());
                                    let grp_arg = group.clone();
                                    let cb = on_event.clone();
                                    div()
                                        .id(("tree-zarr-lbl-item", lg_i))
                                        .cursor_pointer()
                                        .px(px(6.0))
                                        .py(px(2.0))
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
                                            cb(
                                                StudioEvent::SelectOmeZarrLabelGroup(Some(
                                                    grp_arg.clone(),
                                                )),
                                                window,
                                                cx,
                                            );
                                        })
                                        .child(SharedString::from(format!("mask:{group}")))
                                },
                            )),
                    ),
            )
        })
        // Layers sub-row
        .child(
            div()
                .pl(px(10.0))
                .pt(px(2.0))
                .flex()
                .flex_col()
                .gap(px(2.0))
                .children(state.layers.iter().enumerate().map(|(idx, layer)| {
                    let lid = layer.id;
                    let visible = layer.visible;
                    let cb = on_event.clone();
                    let label: SharedString = format!(
                        "{} {}",
                        if visible { "●" } else { "○" },
                        layer.kind.badge()
                    )
                    .into();
                    div()
                        .id(("tree-layer-item", idx))
                        .cursor_pointer()
                        .px(px(6.0))
                        .py(px(2.5))
                        .rounded(px(3.0))
                        .bg(palette::widget_bg())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .flex()
                        .items_center()
                        .justify_between()
                        .on_click(move |_, window, cx| {
                            cb(StudioEvent::ToggleLayerVisibility(lid), window, cx);
                        })
                        .child(
                            div()
                                .text_size(px(10.5))
                                .text_color(if visible {
                                    palette::text_primary()
                                } else {
                                    palette::text_dim()
                                })
                                .child(label),
                        )
                        .child(
                            div()
                                .text_size(px(10.0))
                                .text_color(palette::text_dim())
                                .child(if visible { "Visible" } else { "Hidden" }),
                        )
                })),
        )
}

fn render_operation_steps(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> gpui_kit::Div {
    let cb_open_menu = on_event.clone();
    div()
        .pl(px(6.0))
        .flex()
        .flex_col()
        .gap(px(5.0))
        .when(state.modifiers.is_empty(), |el| {
            el.child(
                div()
                    .id("tree-empty-ops-prompt")
                    .cursor_pointer()
                    .p(px(8.0))
                    .rounded(px(4.0))
                    .bg(palette::surface_card())
                    .hover(|s| s.bg(palette::surface_hover()))
                    .on_click(move |_, window, cx| {
                        cb_open_menu(StudioEvent::ToggleOperationsMenu, window, cx);
                    })
                    .child(
                        div()
                            .text_size(px(11.0))
                            .text_color(palette::text_secondary())
                            .child("├─ [No GPU operations applied]"),
                    )
                    .child(
                        div()
                            .pl(px(12.0))
                            .text_size(px(10.5))
                            .text_color(palette::text_dim())
                            .child("Click 'Operations ▾' in top menu to add steps"),
                    ),
            )
        })
        .children(state.modifiers.iter().enumerate().map(|(idx, node)| {
            let id = node.id;
            let enabled = node.enabled;
            let step = node.kind.step();
            let cb_toggle = on_event.clone();
            let cb_dec = on_event.clone();
            let cb_inc = on_event.clone();
            let cb_rm = on_event.clone();

            let title: SharedString =
                format!("├─ [{:02}] {}", idx + 1, node.kind.title()).into();
            let param_txt: SharedString =
                format!("{}: {:.2}", node.kind.primary_label(), node.primary_param).into();

            div()
                .w_full()
                .p(px(7.0))
                .rounded(px(4.0))
                .bg(palette::surface_card())
                .flex()
                .flex_col()
                .gap(px(4.0))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .id(("tree-op-toggle", idx))
                                .cursor_pointer()
                                .text_size(px(11.5))
                                .text_color(if enabled {
                                    palette::text_primary()
                                } else {
                                    palette::text_dim()
                                })
                                .on_click(move |_, window, cx| {
                                    cb_toggle(StudioEvent::ToggleModifier(id), window, cx);
                                })
                                .child(title),
                        )
                        .child(
                            div()
                                .id(("tree-op-rm", idx))
                                .cursor_pointer()
                                .px(px(5.0))
                                .py(px(1.0))
                                .rounded(px(3.0))
                                .hover(|s| s.bg(palette::accent_coral()))
                                .text_size(px(10.5))
                                .text_color(palette::text_dim())
                                .on_click(move |_, window, cx| {
                                    cb_rm(StudioEvent::RemoveModifier(id), window, cx);
                                })
                                .child("✕"),
                        ),
                )
                .child(
                    div()
                        .pl(px(12.0))
                        .flex()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .text_size(px(10.5))
                                .text_color(palette::text_secondary())
                                .child(param_txt),
                        )
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(3.0))
                                .child(
                                    div()
                                        .id(("tree-op-dec", idx))
                                        .cursor_pointer()
                                        .px(px(6.0))
                                        .py(px(1.0))
                                        .rounded(px(3.0))
                                        .bg(palette::widget_bg())
                                        .hover(|s| s.bg(palette::surface_hover()))
                                        .text_size(px(11.0))
                                        .text_color(palette::text_primary())
                                        .on_click(move |_, window, cx| {
                                            cb_dec(
                                                StudioEvent::AdjustModifierPrimary {
                                                    id,
                                                    delta: -step,
                                                },
                                                window,
                                                cx,
                                            );
                                        })
                                        .child("-"),
                                )
                                .child(
                                    div()
                                        .id(("tree-op-inc", idx))
                                        .cursor_pointer()
                                        .px(px(6.0))
                                        .py(px(1.0))
                                        .rounded(px(3.0))
                                        .bg(palette::widget_bg())
                                        .hover(|s| s.bg(palette::surface_hover()))
                                        .text_size(px(11.0))
                                        .text_color(palette::text_primary())
                                        .on_click(move |_, window, cx| {
                                            cb_inc(
                                                StudioEvent::AdjustModifierPrimary {
                                                    id,
                                                    delta: step,
                                                },
                                                window,
                                                cx,
                                            );
                                        })
                                        .child("+"),
                                ),
                        ),
                )
        }))
}

fn render_output_summary_node(state: &StudioUiState) -> gpui_kit::Div {
    let summary: SharedString = format!(
        "{} · Exp {:+.2} · γ {:.2}",
        state.tonemap.colormap.label(),
        state.tonemap.exposure,
        state.tonemap.gamma
    )
    .into();

    div()
        .w_full()
        .p(px(8.0))
        .rounded(px(4.0))
        .bg(palette::surface_card())
        .flex()
        .flex_col()
        .gap(px(2.0))
        .child(
            div()
                .text_size(px(11.5))
                .text_color(palette::text_primary())
                .child("└─ [Out] Display & LUT"),
        )
        .child(
            div()
                .pl(px(12.0))
                .text_size(px(10.5))
                .text_color(palette::text_secondary())
                .child(summary),
        )
}

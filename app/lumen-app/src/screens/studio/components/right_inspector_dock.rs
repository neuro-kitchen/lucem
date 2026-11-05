//! `OPERATIONS` Tree Pane (`render_right_inspector_dock`):
//! Renders a hierarchical tree structure showing every transform and operation applied to the image
//! at each step in execution order:
//! - `▾ [00] Source Input` (Dimensions, dtype, and OME-Zarr Pyramid / Channel / Label branches)
//! - `├─ [01..N] Pipeline Steps` (Ordered GPU operations with per-step parameter leaves and bypass/remove controls)
//! - `├─ [+] Append Operation` (Catalog of CubeCL GPU operators to append as the next step)
//! - `└─ [Out] Display & Viewport Transform` (Zoom, Fit, Colormap LUT, Exposure, Gamma, and Contour/Layer overlays)

use std::rc::Rc;

use gpui_kit::{div, prelude::*, px, App, IntoElement, SharedString, Window};
use lumen_pipeline::ModifierKind;
use lumen_view::ColormapLut;

use crate::screens::studio::studio_event::StudioEvent;
use crate::screens::studio::studio_state::StudioUiState;
use crate::ui::components::{pill_badge, section_header};
use crate::ui::theme::palette;

pub fn render_right_inspector_dock(
    state: &StudioUiState,
    on_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let exposure = state.tonemap.exposure;
    let gamma = state.tonemap.gamma;
    let active_lut = state.tonemap.colormap;
    let zoom_label = state.camera.zoom_percent_label();
    let show_contours = state.camera.show_contours;

    let total_steps = state.modifiers.len();
    let active_steps = state.modifiers.iter().filter(|m| m.enabled).count();
    let gpu_ms = state
        .pipeline_result
        .as_ref()
        .map(|r| format!("{:.1} ms", r.elapsed_ms))
        .unwrap_or_else(|| "0.0 ms".to_string());

    let src_summary = format!(
        "{}×{} · {}",
        state.source_image.width, state.source_image.height, state.source_image.dtype_label
    );

    let exp_dec = on_event.clone();
    let exp_inc = on_event.clone();
    let gam_dec = on_event.clone();
    let gam_inc = on_event.clone();
    let clear_mods = on_event.clone();
    let zoom_out_cb = on_event.clone();
    let zoom_in_cb = on_event.clone();
    let zoom_1to1_cb = on_event.clone();
    let zoom_fit_cb = on_event.clone();
    let contours_cb = on_event.clone();

    div()
        .w(px(340.0))
        .h_full()
        .bg(palette::panel_bg())
        .border_l_1()
        .border_color(palette::border_subtle())
        .flex()
        .flex_col()
        .overflow_hidden()
        // Top header for the Operations Tree Pane
        .child(section_header(
            "OPERATIONS",
            Some(SharedString::from(format!(
                "{active_steps}/{total_steps} steps · {gpu_ms}"
            ))),
        ))
        .child(
            div()
                .flex_1()
                .w_full()
                .p(px(10.0))
                .flex()
                .flex_col()
                .gap(px(8.0))
                .overflow_hidden()
                // =========================================================================
                // STEP 00: SOURCE INPUT NODE
                // =========================================================================
                .child(
                    div()
                        .w_full()
                        .p(px(8.0))
                        .rounded(px(6.0))
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_strong())
                        .flex()
                        .flex_col()
                        .gap(px(5.0))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(6.0))
                                        .child(pill_badge(
                                            "00",
                                            palette::accent_cyan_muted(),
                                            palette::accent_cyan(),
                                        ))
                                        .child(
                                            div()
                                                .text_size(px(11.5))
                                                .text_color(palette::text_primary())
                                                .child("▾ Source Input"),
                                        ),
                                )
                                .child(
                                    div()
                                        .text_size(px(10.0))
                                        .text_color(palette::text_dim())
                                        .child(SharedString::from(src_summary)),
                                ),
                        )
                        .child(
                            div()
                                .pl(px(10.0))
                                .text_size(px(10.5))
                                .text_color(palette::text_secondary())
                                .child(SharedString::from(format!(
                                    "└─ {}",
                                    state.source_image.name
                                ))),
                        )
                        // OME-Zarr container child branches (`├─ Pyramid`, `├─ Channel`, `└─ Labels`)
                        .when_some(state.source_image.ome_zarr.clone(), |el, zarr| {
                            el.child(
                                div()
                                    .pl(px(10.0))
                                    .pt(px(2.0))
                                    .flex()
                                    .flex_col()
                                    .gap(px(5.0))
                                    // Pyramid branch
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .justify_between()
                                            .child(
                                                div()
                                                    .text_size(px(10.5))
                                                    .text_color(palette::text_dim())
                                                    .child("├─ Pyramid"),
                                            )
                                            .child(
                                                div()
                                                    .flex()
                                                    .items_center()
                                                    .gap(px(3.0))
                                                    .children(
                                                        zarr.pyramid_levels.iter().enumerate().map(
                                                            |(idx, lvl)| {
                                                                let is_active =
                                                                    *lvl == zarr.active_level;
                                                                let lvl_clone = lvl.clone();
                                                                let lvl_text: SharedString =
                                                                    lvl.clone().into();
                                                                let cb = on_event.clone();
                                                                div()
                                                                    .id(("zarr-lvl-btn", idx))
                                                                    .px(px(6.0))
                                                                    .py(px(1.5))
                                                                    .rounded(px(4.0))
                                                                    .bg(if is_active {
                                                                        palette::accent_cyan_muted()
                                                                    } else {
                                                                        palette::canvas_void()
                                                                    })
                                                                    .border_1()
                                                                    .border_color(if is_active {
                                                                        palette::accent_cyan()
                                                                    } else {
                                                                        palette::border_subtle()
                                                                    })
                                                                    .cursor_pointer()
                                                                    .on_click(
                                                                        move |_, window, cx| {
                                                                            cb(
                                                                                StudioEvent::SelectOmeZarrLevel(
                                                                                    lvl_clone.clone(),
                                                                                ),
                                                                                window,
                                                                                cx,
                                                                            );
                                                                        },
                                                                    )
                                                                    .child(
                                                                        div()
                                                                            .text_size(px(10.0))
                                                                            .text_color(
                                                                                if is_active {
                                                                                    palette::accent_cyan()
                                                                                } else {
                                                                                    palette::text_secondary()
                                                                                },
                                                                            )
                                                                            .child(lvl_text),
                                                                    )
                                                            },
                                                        ),
                                                    ),
                                            ),
                                    )
                                    // Channel branch
                                    .when(zarr.channel_labels.len() > 1, |el_ch| {
                                        let is_comp = zarr.active_channel.is_none()
                                            && zarr.active_label_group.is_none();
                                        let cb_comp = on_event.clone();
                                        el_ch.child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .justify_between()
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_dim())
                                                        .child("├─ Channel"),
                                                )
                                                .child(
                                                    div()
                                                        .flex()
                                                        .flex_wrap()
                                                        .items_center()
                                                        .gap(px(3.0))
                                                        .child(
                                                            div()
                                                                .id("zarr-ch-comp")
                                                                .px(px(5.0))
                                                                .py(px(1.5))
                                                                .rounded(px(4.0))
                                                                .bg(if is_comp {
                                                                    palette::accent_cyan_muted()
                                                                } else {
                                                                    palette::canvas_void()
                                                                })
                                                                .border_1()
                                                                .border_color(if is_comp {
                                                                    palette::accent_cyan()
                                                                } else {
                                                                    palette::border_subtle()
                                                                })
                                                                .cursor_pointer()
                                                                .on_click(move |_, window, cx| {
                                                                    cb_comp(
                                                                        StudioEvent::SelectOmeZarrChannel(
                                                                            None,
                                                                        ),
                                                                        window,
                                                                        cx,
                                                                    );
                                                                })
                                                                .child(
                                                                    div()
                                                                        .text_size(px(9.5))
                                                                        .text_color(if is_comp {
                                                                            palette::accent_cyan()
                                                                        } else {
                                                                            palette::text_secondary()
                                                                        })
                                                                        .child("Composite"),
                                                                ),
                                                        )
                                                        .children(
                                                            zarr.channel_labels
                                                                .iter()
                                                                .enumerate()
                                                                .map(|(c_idx, ch_lbl)| {
                                                                    let is_sel = zarr
                                                                        .active_label_group
                                                                        .is_none()
                                                                        && zarr.active_channel
                                                                            == Some(c_idx);
                                                                    let lbl_str: SharedString =
                                                                        ch_lbl.clone().into();
                                                                    let cb = on_event.clone();
                                                                    div()
                                                                        .id(("zarr-ch-btn", c_idx))
                                                                        .px(px(5.0))
                                                                        .py(px(1.5))
                                                                        .rounded(px(4.0))
                                                                        .bg(if is_sel {
                                                                            palette::accent_cyan_muted()
                                                                        } else {
                                                                            palette::canvas_void()
                                                                        })
                                                                        .border_1()
                                                                        .border_color(if is_sel {
                                                                            palette::accent_cyan()
                                                                        } else {
                                                                            palette::border_subtle()
                                                                        })
                                                                        .cursor_pointer()
                                                                        .on_click(
                                                                            move |_, window, cx| {
                                                                                cb(
                                                                                    StudioEvent::SelectOmeZarrChannel(
                                                                                        Some(c_idx),
                                                                                    ),
                                                                                    window,
                                                                                    cx,
                                                                                );
                                                                            },
                                                                        )
                                                                        .child(
                                                                            div()
                                                                                .text_size(px(9.5))
                                                                                .text_color(
                                                                                    if is_sel {
                                                                                        palette::accent_cyan()
                                                                                    } else {
                                                                                        palette::text_secondary()
                                                                                    },
                                                                                )
                                                                                .child(lbl_str),
                                                                        )
                                                                }),
                                                        ),
                                                ),
                                        )
                                    })
                                    // Labels branch
                                    .when(!zarr.label_groups.is_empty(), |el_lbl| {
                                        let is_img = zarr.active_label_group.is_none();
                                        let cb_img = on_event.clone();
                                        el_lbl.child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .justify_between()
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_dim())
                                                        .child("└─ Plane"),
                                                )
                                                .child(
                                                    div()
                                                        .flex()
                                                        .items_center()
                                                        .gap(px(3.0))
                                                        .child(
                                                            div()
                                                                .id("zarr-lbl-none")
                                                                .px(px(5.0))
                                                                .py(px(1.5))
                                                                .rounded(px(4.0))
                                                                .bg(if is_img {
                                                                    palette::accent_cyan_muted()
                                                                } else {
                                                                    palette::canvas_void()
                                                                })
                                                                .border_1()
                                                                .border_color(if is_img {
                                                                    palette::accent_cyan()
                                                                } else {
                                                                    palette::border_subtle()
                                                                })
                                                                .cursor_pointer()
                                                                .on_click(move |_, window, cx| {
                                                                    cb_img(
                                                                        StudioEvent::SelectOmeZarrLabelGroup(
                                                                            None,
                                                                        ),
                                                                        window,
                                                                        cx,
                                                                    );
                                                                })
                                                                .child(
                                                                    div()
                                                                        .text_size(px(9.5))
                                                                        .text_color(if is_img {
                                                                            palette::accent_cyan()
                                                                        } else {
                                                                            palette::text_secondary()
                                                                        })
                                                                        .child("Image"),
                                                                ),
                                                        )
                                                        .children(
                                                            zarr.label_groups
                                                                .iter()
                                                                .enumerate()
                                                                .map(|(l_idx, grp)| {
                                                                    let is_sel = zarr
                                                                        .active_label_group
                                                                        .as_deref()
                                                                        == Some(grp.as_str());
                                                                    let grp_clone = grp.clone();
                                                                    let grp_str: SharedString =
                                                                        grp.clone().into();
                                                                    let cb = on_event.clone();
                                                                    div()
                                                                        .id(("zarr-lbl-btn", l_idx))
                                                                        .px(px(5.0))
                                                                        .py(px(1.5))
                                                                        .rounded(px(4.0))
                                                                        .bg(if is_sel {
                                                                            palette::accent_cyan_muted()
                                                                        } else {
                                                                            palette::canvas_void()
                                                                        })
                                                                        .border_1()
                                                                        .border_color(if is_sel {
                                                                            palette::accent_cyan()
                                                                        } else {
                                                                            palette::border_subtle()
                                                                        })
                                                                        .cursor_pointer()
                                                                        .on_click(
                                                                            move |_, window, cx| {
                                                                                cb(
                                                                                    StudioEvent::SelectOmeZarrLabelGroup(
                                                                                        Some(grp_clone.clone()),
                                                                                    ),
                                                                                    window,
                                                                                    cx,
                                                                                );
                                                                            },
                                                                        )
                                                                        .child(
                                                                            div()
                                                                                .text_size(px(9.5))
                                                                                .text_color(
                                                                                    if is_sel {
                                                                                        palette::accent_cyan()
                                                                                    } else {
                                                                                        palette::text_secondary()
                                                                                    },
                                                                                )
                                                                                .child(grp_str),
                                                                        )
                                                                }),
                                                        ),
                                                ),
                                        )
                                    }),
                            )
                        }),
                )
                // =========================================================================
                // STEPS 01..N: ORDERED GPU PIPELINE TREE NODES
                // =========================================================================
                .child(
                    div()
                        .w_full()
                        .pl(px(8.0))
                        .border_l_1()
                        .border_color(palette::border_strong())
                        .flex()
                        .flex_col()
                        .gap(px(6.0))
                        .when(state.modifiers.is_empty(), |el| {
                            el.child(
                                div()
                                    .px(px(8.0))
                                    .py(px(5.0))
                                    .rounded(px(5.0))
                                    .bg(palette::canvas_void())
                                    .border_1()
                                    .border_color(palette::border_subtle())
                                    .text_size(px(10.5))
                                    .text_color(palette::text_dim())
                                    .child("├─ [01] No GPU ops queued — append a step below"),
                            )
                        })
                        .children(state.modifiers.iter().enumerate().map(|(idx, node)| {
                            let step_num = format!("{:02}", idx + 1);
                            let mod_id = node.id;
                            let kind = node.kind;
                            let enabled = node.enabled;
                            let param_val = node.primary_param;
                            let step = kind.step();

                            let cb_toggle = on_event.clone();
                            let cb_dec = on_event.clone();
                            let cb_inc = on_event.clone();
                            let cb_rm = on_event.clone();

                            div()
                                .id(("ops-tree-step", idx))
                                .w_full()
                                .p(px(7.0))
                                .rounded(px(6.0))
                                .bg(palette::card_bg())
                                .border_1()
                                .border_color(if enabled {
                                    palette::border_strong()
                                } else {
                                    palette::border_subtle()
                                })
                                .flex()
                                .flex_col()
                                .gap(px(4.0))
                                // Tree node header: `├─ [01] ● Operator Name` ... `×`
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .child(
                                            div()
                                                .id(("toggle-mod-btn", idx))
                                                .flex()
                                                .items_center()
                                                .gap(px(5.0))
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    cb_toggle(
                                                        StudioEvent::ToggleModifier(mod_id),
                                                        window,
                                                        cx,
                                                    );
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_dim())
                                                        .child("├─"),
                                                )
                                                .child(pill_badge(
                                                    step_num,
                                                    if enabled {
                                                        palette::accent_cyan_muted()
                                                    } else {
                                                        palette::canvas_void()
                                                    },
                                                    if enabled {
                                                        palette::accent_cyan()
                                                    } else {
                                                        palette::text_dim()
                                                    },
                                                ))
                                                .child(
                                                    div()
                                                        .text_size(px(11.0))
                                                        .text_color(if enabled {
                                                            palette::text_primary()
                                                        } else {
                                                            palette::text_dim()
                                                        })
                                                        .child(kind.title()),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .id(("rm-mod-btn", idx))
                                                .px(px(5.0))
                                                .py(px(1.0))
                                                .rounded(px(4.0))
                                                .cursor_pointer()
                                                .hover(|s| s.bg(palette::card_hover_bg()))
                                                .on_click(move |_, window, cx| {
                                                    cb_rm(
                                                        StudioEvent::RemoveModifier(mod_id),
                                                        window,
                                                        cx,
                                                    );
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::accent_coral())
                                                        .child("×"),
                                                ),
                                        ),
                                )
                                // Indented child parameter leaf: `└─ param_label: [−] val [+]`
                                .child(
                                    div()
                                        .pl(px(14.0))
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .child(
                                            div()
                                                .text_size(px(10.0))
                                                .text_color(palette::text_secondary())
                                                .child(format!("└─ {}", kind.primary_label())),
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap(px(4.0))
                                                .child(
                                                    div()
                                                        .id(("mod-dec-btn", idx))
                                                        .px(px(6.0))
                                                        .py(px(1.0))
                                                        .rounded(px(4.0))
                                                        .bg(palette::canvas_void())
                                                        .cursor_pointer()
                                                        .on_click(move |_, window, cx| {
                                                            cb_dec(
                                                                StudioEvent::AdjustModifierPrimary {
                                                                    id: mod_id,
                                                                    delta: -step,
                                                                },
                                                                window,
                                                                cx,
                                                            );
                                                        })
                                                        .child(
                                                            div()
                                                                .text_size(px(10.5))
                                                                .text_color(palette::text_primary())
                                                                .child("−"),
                                                        ),
                                                )
                                                .child(
                                                    div()
                                                        .px(px(6.0))
                                                        .py(px(1.0))
                                                        .rounded(px(4.0))
                                                        .bg(palette::canvas_void())
                                                        .text_size(px(10.0))
                                                        .text_color(palette::accent_cyan())
                                                        .child(format!("{param_val:.3}")),
                                                )
                                                .child(
                                                    div()
                                                        .id(("mod-inc-btn", idx))
                                                        .px(px(6.0))
                                                        .py(px(1.0))
                                                        .rounded(px(4.0))
                                                        .bg(palette::canvas_void())
                                                        .cursor_pointer()
                                                        .on_click(move |_, window, cx| {
                                                            cb_inc(
                                                                StudioEvent::AdjustModifierPrimary {
                                                                    id: mod_id,
                                                                    delta: step,
                                                                },
                                                                window,
                                                                cx,
                                                            );
                                                        })
                                                        .child(
                                                            div()
                                                                .text_size(px(10.5))
                                                                .text_color(palette::text_primary())
                                                                .child("+"),
                                                        ),
                                                ),
                                        ),
                                )
                        })),
                )
                // =========================================================================
                // APPEND OPERATION STEP BRANCH (`├─ [+] Append Operation`)
                // =========================================================================
                .child(
                    div()
                        .w_full()
                        .p(px(8.0))
                        .rounded(px(6.0))
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_subtle())
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
                                        .text_size(px(10.5))
                                        .text_color(palette::text_secondary())
                                        .child("├─ [+] Append Operation Step"),
                                )
                                .child(
                                    div()
                                        .id("btn-clear-modifiers")
                                        .px(px(6.0))
                                        .py(px(1.5))
                                        .rounded(px(4.0))
                                        .bg(palette::canvas_void())
                                        .cursor_pointer()
                                        .hover(|s| s.bg(palette::card_hover_bg()))
                                        .on_click(move |_, window, cx| {
                                            clear_mods(StudioEvent::ClearModifiers, window, cx);
                                        })
                                        .child(
                                            div()
                                                .text_size(px(9.5))
                                                .text_color(palette::text_dim())
                                                .child("Reset Tree"),
                                        ),
                                ),
                        )
                        .child(
                            div()
                                .w_full()
                                .flex()
                                .flex_wrap()
                                .gap(px(4.0))
                                .children(ModifierKind::ALL.iter().copied().enumerate().map(
                                    |(idx, kind)| {
                                        let cb = on_event.clone();
                                        div()
                                            .id(("add-mod-btn", idx))
                                            .px(px(6.0))
                                            .py(px(3.0))
                                            .rounded(px(4.0))
                                            .bg(palette::canvas_void())
                                            .border_1()
                                            .border_color(palette::border_subtle())
                                            .cursor_pointer()
                                            .hover(|s| {
                                                s.bg(palette::card_hover_bg())
                                                    .border_color(palette::accent_cyan())
                                            })
                                            .on_click(move |_, window, cx| {
                                                cb(StudioEvent::AddModifier(kind), window, cx);
                                            })
                                            .child(
                                                div()
                                                    .text_size(px(10.0))
                                                    .text_color(palette::text_primary())
                                                    .child(format!("+ {}", kind.title())),
                                            )
                                    },
                                )),
                        ),
                )
                // =========================================================================
                // TERMINAL TREE NODE: `└─ [Out] Display & Viewport Transform`
                // =========================================================================
                .child(
                    div()
                        .w_full()
                        .p(px(8.0))
                        .rounded(px(6.0))
                        .bg(palette::card_bg())
                        .border_1()
                        .border_color(palette::border_strong())
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
                                        .flex()
                                        .items_center()
                                        .gap(px(6.0))
                                        .child(pill_badge(
                                            "OUT",
                                            palette::accent_blue_soft(),
                                            palette::accent_blue(),
                                        ))
                                        .child(
                                            div()
                                                .text_size(px(11.5))
                                                .text_color(palette::text_primary())
                                                .child("▾ Display & Viewport Transform"),
                                        ),
                                )
                                .child(
                                    div()
                                        .text_size(px(10.0))
                                        .text_color(palette::accent_cyan())
                                        .child(SharedString::from(active_lut.label())),
                                ),
                        )
                        // `├─ Zoom & Fit` leaf
                        .child(
                            div()
                                .pl(px(10.0))
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_size(px(10.5))
                                        .text_color(palette::text_secondary())
                                        .child("├─ Camera Zoom"),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(3.0))
                                        .child(
                                            div()
                                                .id("tree-zoom-out")
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    zoom_out_cb(StudioEvent::ZoomOut, window, cx);
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_primary())
                                                        .child("−"),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .text_size(px(10.0))
                                                .text_color(palette::accent_cyan())
                                                .child(SharedString::from(zoom_label)),
                                        )
                                        .child(
                                            div()
                                                .id("tree-zoom-in")
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    zoom_in_cb(StudioEvent::ZoomIn, window, cx);
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_primary())
                                                        .child("+"),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .id("tree-zoom-1to1")
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    zoom_1to1_cb(
                                                        StudioEvent::ZoomOneToOne,
                                                        window,
                                                        cx,
                                                    );
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.0))
                                                        .text_color(palette::text_secondary())
                                                        .child("1:1"),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .id("tree-zoom-fit")
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    zoom_fit_cb(
                                                        StudioEvent::FitToViewport,
                                                        window,
                                                        cx,
                                                    );
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.0))
                                                        .text_color(palette::text_secondary())
                                                        .child("Fit"),
                                                ),
                                        ),
                                ),
                        )
                        // `├─ Exposure` leaf
                        .child(
                            div()
                                .pl(px(10.0))
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_size(px(10.5))
                                        .text_color(palette::text_secondary())
                                        .child("├─ Exposure Gain"),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(4.0))
                                        .child(
                                            div()
                                                .id("exp-dec")
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    exp_dec(
                                                        StudioEvent::AdjustExposure(-0.15),
                                                        window,
                                                        cx,
                                                    );
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_primary())
                                                        .child("−"),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .px(px(7.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .text_size(px(10.0))
                                                .text_color(palette::accent_cyan())
                                                .child(format!("{exposure:.2}x")),
                                        )
                                        .child(
                                            div()
                                                .id("exp-inc")
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    exp_inc(
                                                        StudioEvent::AdjustExposure(0.15),
                                                        window,
                                                        cx,
                                                    );
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_primary())
                                                        .child("+"),
                                                ),
                                        ),
                                ),
                        )
                        // `├─ Gamma` leaf
                        .child(
                            div()
                                .pl(px(10.0))
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_size(px(10.5))
                                        .text_color(palette::text_secondary())
                                        .child("├─ Display Gamma"),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(4.0))
                                        .child(
                                            div()
                                                .id("gam-dec")
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    gam_dec(
                                                        StudioEvent::AdjustGamma(-0.1),
                                                        window,
                                                        cx,
                                                    );
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_primary())
                                                        .child("−"),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .px(px(7.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .text_size(px(10.0))
                                                .text_color(palette::accent_cyan())
                                                .child(format!("{gamma:.2}")),
                                        )
                                        .child(
                                            div()
                                                .id("gam-inc")
                                                .px(px(6.0))
                                                .py(px(1.5))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    gam_inc(
                                                        StudioEvent::AdjustGamma(0.1),
                                                        window,
                                                        cx,
                                                    );
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_primary())
                                                        .child("+"),
                                                ),
                                        ),
                                ),
                        )
                        // `├─ Colormap LUT` leaf
                        .child(
                            div()
                                .pl(px(10.0))
                                .flex()
                                .flex_col()
                                .gap(px(4.0))
                                .child(
                                    div()
                                        .text_size(px(10.5))
                                        .text_color(palette::text_secondary())
                                        .child("├─ Colormap LUT"),
                                )
                                .child(
                                    div()
                                        .pl(px(12.0))
                                        .flex()
                                        .flex_wrap()
                                        .gap(px(4.0))
                                        .children(ColormapLut::ALL.iter().copied().enumerate().map(
                                            |(idx, lut)| {
                                                let is_active = lut == active_lut;
                                                let cb = on_event.clone();
                                                div()
                                                    .id(("tree-lut-btn", idx))
                                                    .px(px(6.0))
                                                    .py(px(2.0))
                                                    .rounded(px(4.0))
                                                    .bg(if is_active {
                                                        palette::accent_cyan_muted()
                                                    } else {
                                                        palette::canvas_void()
                                                    })
                                                    .border_1()
                                                    .border_color(if is_active {
                                                        palette::accent_cyan()
                                                    } else {
                                                        palette::border_subtle()
                                                    })
                                                    .cursor_pointer()
                                                    .on_click(move |_, window, cx| {
                                                        cb(
                                                            StudioEvent::SelectColormap(lut),
                                                            window,
                                                            cx,
                                                        );
                                                    })
                                                    .child(
                                                        div()
                                                            .text_size(px(9.5))
                                                            .text_color(if is_active {
                                                                palette::accent_cyan()
                                                            } else {
                                                                palette::text_secondary()
                                                            })
                                                            .child(lut.label()),
                                                    )
                                            },
                                        )),
                                ),
                        )
                        // `└─ Overlays & Layers` leaf
                        .child(
                            div()
                                .pl(px(10.0))
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_size(px(10.5))
                                        .text_color(palette::text_secondary())
                                        .child("└─ Vector Overlay"),
                                )
                                .child(
                                    div()
                                        .id("tree-toggle-contours")
                                        .px(px(7.0))
                                        .py(px(2.0))
                                        .rounded(px(4.0))
                                        .bg(if show_contours {
                                            palette::accent_cyan_muted()
                                        } else {
                                            palette::canvas_void()
                                        })
                                        .border_1()
                                        .border_color(if show_contours {
                                            palette::accent_cyan()
                                        } else {
                                            palette::border_subtle()
                                        })
                                        .cursor_pointer()
                                        .on_click(move |_, window, cx| {
                                            contours_cb(StudioEvent::ToggleContours, window, cx);
                                        })
                                        .child(
                                            div()
                                                .text_size(px(10.0))
                                                .text_color(if show_contours {
                                                    palette::accent_cyan()
                                                } else {
                                                    palette::text_secondary()
                                                })
                                                .child("Iso-Contours (Marching Squares)"),
                                        ),
                                ),
                        ),
                ),
        )
}

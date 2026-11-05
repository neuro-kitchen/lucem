//! Right-hand ArtCraft × Blender Inspector Dock (`render_right_inspector_dock`):
//! 1. Display & Tonemap Inspector (Exposure, Gamma, Colormap LUT).
//! 2. ArtCraft Layer Stack (Base Raster, VectorCraft Contours, Connected Regions).
//! 3. Blender-style Non-Destructive GPU Modifier Stack (Add Modifier buttons + live parameter scrubbers).

use std::rc::Rc;

use gpui_kit::{div, prelude::*, px, App, IntoElement, SharedString, Window};
use lumen_pipeline::ModifierKind;

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

    let exp_dec = on_event.clone();
    let exp_inc = on_event.clone();
    let gam_dec = on_event.clone();
    let gam_inc = on_event.clone();
    let clear_mods = on_event.clone();

    div()
        .w(px(328.0))
        .h_full()
        .bg(palette::panel_bg())
        .border_l_1()
        .border_color(palette::border_subtle())
        .flex()
        .flex_col()
        .justify_between()
        .overflow_hidden()
        .child(
            div()
                .w_full()
                .flex()
                .flex_col()
                // 1. Display & Tonemap section
                .child(section_header(
                    "DISPLAY & TONEMAP",
                    Some(SharedString::from(state.tonemap.colormap.label())),
                ))
                .child(
                    div()
                        .p(px(10.0))
                        .flex()
                        .flex_col()
                        .gap(px(6.0))
                        .border_b_1()
                        .border_color(palette::border_subtle())
                        .child(
                            // Exposure scrubber row
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_size(px(11.5))
                                        .text_color(palette::text_secondary())
                                        .child("Exposure Gain"),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(4.0))
                                        .child(
                                            div()
                                                .id("exp-dec")
                                                .px(px(7.0))
                                                .py(px(2.0))
                                                .rounded(px(4.0))
                                                .bg(palette::card_bg())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    exp_dec(StudioEvent::AdjustExposure(-0.15), window, cx);
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(11.0))
                                                        .text_color(palette::text_primary())
                                                        .child("−"),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .px(px(8.0))
                                                .py(px(2.0))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .text_size(px(11.0))
                                                .text_color(palette::accent_cyan())
                                                .child(format!("{exposure:.2}x")),
                                        )
                                        .child(
                                            div()
                                                .id("exp-inc")
                                                .px(px(7.0))
                                                .py(px(2.0))
                                                .rounded(px(4.0))
                                                .bg(palette::card_bg())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    exp_inc(StudioEvent::AdjustExposure(0.15), window, cx);
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(11.0))
                                                        .text_color(palette::text_primary())
                                                        .child("+"),
                                                ),
                                        ),
                                ),
                        )
                        .child(
                            // Gamma scrubber row
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_size(px(11.5))
                                        .text_color(palette::text_secondary())
                                        .child("Display Gamma"),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(4.0))
                                        .child(
                                            div()
                                                .id("gam-dec")
                                                .px(px(7.0))
                                                .py(px(2.0))
                                                .rounded(px(4.0))
                                                .bg(palette::card_bg())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    gam_dec(StudioEvent::AdjustGamma(-0.1), window, cx);
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(11.0))
                                                        .text_color(palette::text_primary())
                                                        .child("−"),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .px(px(8.0))
                                                .py(px(2.0))
                                                .rounded(px(4.0))
                                                .bg(palette::canvas_void())
                                                .text_size(px(11.0))
                                                .text_color(palette::accent_cyan())
                                                .child(format!("{gamma:.2}")),
                                        )
                                        .child(
                                            div()
                                                .id("gam-inc")
                                                .px(px(7.0))
                                                .py(px(2.0))
                                                .rounded(px(4.0))
                                                .bg(palette::card_bg())
                                                .cursor_pointer()
                                                .on_click(move |_, window, cx| {
                                                    gam_inc(StudioEvent::AdjustGamma(0.1), window, cx);
                                                })
                                                .child(
                                                    div()
                                                        .text_size(px(11.0))
                                                        .text_color(palette::text_primary())
                                                        .child("+"),
                                                ),
                                        ),
                                ),
                        ),
                )
                // 2. ArtCraft Layer Stack section
                .child(section_header(
                    "ARTCRAFT LAYER STACK",
                    Some(SharedString::from(format!("{} layers", state.layers.len()))),
                ))
                .child(
                    div()
                        .p(px(10.0))
                        .flex()
                        .flex_col()
                        .gap(px(5.0))
                        .border_b_1()
                        .border_color(palette::border_subtle())
                        .children(state.layers.iter().enumerate().map(|(idx, layer)| {
                            let layer_id = layer.id;
                            let vis = layer.visible;
                            let name = layer.name.clone();
                            let badge = layer.kind.badge();
                            let cb = on_event.clone();
                            div()
                                .id(("layer-row", idx))
                                .w_full()
                                .px(px(8.0))
                                .py(px(5.0))
                                .rounded(px(6.0))
                                .bg(palette::card_bg())
                                .border_1()
                                .border_color(if vis {
                                    palette::border_strong()
                                } else {
                                    palette::border_subtle()
                                })
                                .cursor_pointer()
                                .flex()
                                .items_center()
                                .justify_between()
                                .on_click(move |_, window, cx| {
                                    cb(StudioEvent::ToggleLayerVisibility(layer_id), window, cx);
                                })
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(6.0))
                                        .child(
                                            div()
                                                .w(px(8.0))
                                                .h(px(8.0))
                                                .rounded(px(4.0))
                                                .bg(if vis {
                                                    palette::accent_emerald()
                                                } else {
                                                    palette::text_dim()
                                                }),
                                        )
                                        .child(
                                            div()
                                                .text_size(px(11.0))
                                                .text_color(if vis {
                                                    palette::text_primary()
                                                } else {
                                                    palette::text_dim()
                                                })
                                                .child(name),
                                        ),
                                )
                                .child(pill_badge(badge, palette::canvas_void(), palette::text_dim()))
                        })),
                )
                // 3. Blender Non-Destructive GPU Modifier Stack
                .child(section_header(
                    "BLENDER GPU MODIFIER STACK",
                    Some(SharedString::from(format!(
                        "{} active",
                        state.modifiers.iter().filter(|m| m.enabled).count()
                    ))),
                ))
                .child(
                    // Add Modifier palette grid
                    div()
                        .p(px(10.0))
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
                                        .text_size(px(10.5))
                                        .text_color(palette::text_dim())
                                        .child("ADD GPU OPERATOR"),
                                )
                                .child(
                                    div()
                                        .id("btn-clear-modifiers")
                                        .px(px(7.0))
                                        .py(px(2.0))
                                        .rounded(px(4.0))
                                        .bg(palette::card_bg())
                                        .cursor_pointer()
                                        .hover(|s| s.bg(palette::card_hover_bg()))
                                        .on_click(move |_, window, cx| {
                                            clear_mods(StudioEvent::ClearModifiers, window, cx);
                                        })
                                        .child(
                                            div()
                                                .text_size(px(10.0))
                                                .text_color(palette::text_secondary())
                                                .child("Clear Stack"),
                                        ),
                                ),
                        )
                        .child(
                            div()
                                .w_full()
                                .flex()
                                .flex_wrap()
                                .gap(px(5.0))
                                .children(ModifierKind::ALL.iter().copied().enumerate().map(|(idx, kind)| {
                                    let cb = on_event.clone();
                                    div()
                                        .id(("add-mod-btn", idx))
                                        .px(px(7.0))
                                        .py(px(4.0))
                                        .rounded(px(5.0))
                                        .bg(palette::card_bg())
                                        .border_1()
                                        .border_color(palette::border_subtle())
                                        .cursor_pointer()
                                        .hover(|s| s.bg(palette::card_hover_bg()).border_color(palette::accent_cyan()))
                                        .on_click(move |_, window, cx| {
                                            cb(StudioEvent::AddModifier(kind), window, cx);
                                        })
                                        .child(
                                            div()
                                                .text_size(px(10.5))
                                                .text_color(palette::text_primary())
                                                .child(format!("+ {}", kind.title())),
                                        )
                                })),
                        )
                        // Active modifier cards
                        .child(
                            div()
                                .w_full()
                                .flex()
                                .flex_col()
                                .gap(px(6.0))
                                .children(state.modifiers.iter().enumerate().map(|(idx, node)| {
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
                                        .id(("active-mod-card", idx))
                                        .w_full()
                                        .p(px(8.0))
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
                                        .gap(px(6.0))
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
                                                        .gap(px(6.0))
                                                        .cursor_pointer()
                                                        .on_click(move |_, window, cx| {
                                                            cb_toggle(StudioEvent::ToggleModifier(mod_id), window, cx);
                                                        })
                                                        .child(
                                                            div()
                                                                .w(px(8.0))
                                                                .h(px(8.0))
                                                                .rounded(px(4.0))
                                                                .bg(if enabled {
                                                                    palette::accent_cyan()
                                                                } else {
                                                                    palette::text_dim()
                                                                }),
                                                        )
                                                        .child(
                                                            div()
                                                                .text_size(px(11.5))
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
                                                        .px(px(6.0))
                                                        .py(px(1.0))
                                                        .rounded(px(4.0))
                                                        .cursor_pointer()
                                                        .hover(|s| s.bg(palette::card_hover_bg()))
                                                        .on_click(move |_, window, cx| {
                                                            cb_rm(StudioEvent::RemoveModifier(mod_id), window, cx);
                                                        })
                                                        .child(
                                                            div()
                                                                .text_size(px(10.5))
                                                                .text_color(palette::accent_coral())
                                                                .child("×"),
                                                        ),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .justify_between()
                                                .child(
                                                    div()
                                                        .text_size(px(10.5))
                                                        .text_color(palette::text_secondary())
                                                        .child(kind.primary_label()),
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
                                                                        .text_size(px(11.0))
                                                                        .text_color(palette::text_primary())
                                                                        .child("−"),
                                                                ),
                                                        )
                                                        .child(
                                                            div()
                                                                .px(px(7.0))
                                                                .py(px(1.0))
                                                                .rounded(px(4.0))
                                                                .bg(palette::canvas_void())
                                                                .text_size(px(10.5))
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
                                                                        .text_size(px(11.0))
                                                                        .text_color(palette::text_primary())
                                                                        .child("+"),
                                                                ),
                                                        ),
                                                ),
                                        )
                                })),
                        ),
                ),
        )
}

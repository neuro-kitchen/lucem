//! Sortable Polars `DataFrame` / Apache Arrow `RecordBatch` RegionProps table (`region_props_table.rs`).

use std::rc::Rc;

use gpui_kit::{div, prelude::*, px, App, IntoElement, SharedString, Window};

use crate::screens::analytics::analytics_state::RegionSortColumn;
use crate::screens::analytics::{AnalyticsEvent, AnalyticsUiState};
use crate::ui::components::pill_badge;
use crate::ui::theme::palette;

pub fn render_region_props_table(
    state: &AnalyticsUiState,
    on_event: Rc<dyn Fn(AnalyticsEvent, &mut Window, &mut App)>,
) -> impl IntoElement {
    let summary_str: SharedString = state.polars_summary_line.clone().into();
    let filter_str: SharedString = format!("Min Area ≥ {} px", state.min_area_filter).into();
    let on_filter = on_event.clone();

    let sortable_cols: [(RegionSortColumn, f32); 4] = [
        (RegionSortColumn::Label, 68.0),
        (RegionSortColumn::Area, 82.0),
        (RegionSortColumn::MeanIntensity, 92.0),
        (RegionSortColumn::Eccentricity, 92.0),
    ];

    div()
        .flex_1()
        .h_full()
        .flex()
        .flex_col()
        .overflow_hidden()
        // Table toolbar header
        .child(
            div()
                .w_full()
                .px(px(12.0))
                .py(px(7.0))
                .bg(palette::surface_card())
                .border_b_1()
                .border_color(palette::border_subtle())
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.0))
                        .child(pill_badge(
                            "POLARS + ARROW",
                            palette::accent_blue_soft(),
                            palette::accent_emerald(),
                        ))
                        .child(
                            div()
                                .text_size(px(11.0))
                                .text_color(palette::text_secondary())
                                .child(summary_str),
                        ),
                )
                .child(
                    div()
                        .id("analytics-cycle-min-area-btn")
                        .cursor_pointer()
                        .px(px(9.0))
                        .py(px(3.0))
                        .rounded(px(4.0))
                        .bg(palette::panel_bg())
                        .border_1()
                        .border_color(palette::border_subtle())
                        .hover(|s| s.bg(palette::surface_hover()))
                        .text_size(px(10.5))
                        .text_color(palette::text_primary())
                        .on_click(move |_, window, cx| {
                            on_filter(AnalyticsEvent::CycleMinAreaFilter, window, cx);
                        })
                        .child(filter_str),
                ),
        )
        // Column headers
        .child(
            div()
                .w_full()
                .px(px(12.0))
                .py(px(5.0))
                .bg(palette::panel_bg())
                .border_b_1()
                .border_color(palette::border_subtle())
                .flex()
                .items_center()
                .gap(px(8.0))
                .text_size(px(10.5))
                .text_color(palette::text_dim())
                .children(sortable_cols.into_iter().enumerate().map(|(i, (col, width_px))| {
                    let is_sorted = state.sort_column == col;
                    let arrow_mark = if is_sorted {
                        if state.sort_descending { " ↓" } else { " ↑" }
                    } else {
                        ""
                    };
                    let lbl: SharedString = format!("{}{}", col.label(), arrow_mark).into();
                    let cb = on_event.clone();
                    div()
                        .id(("analytics-sort-col", i))
                        .cursor_pointer()
                        .w(px(width_px))
                        .text_color(if is_sorted {
                            palette::accent_blue()
                        } else {
                            palette::text_secondary()
                        })
                        .on_click(move |_, window, cx| {
                            cb(AnalyticsEvent::SortBy(col), window, cx);
                        })
                        .child(lbl)
                }))
                .child(div().w(px(110.0)).child("Centroid (y, x)"))
                .child(div().w(px(82.0)).child("Equiv. Ø"))
                .child(div().w(px(82.0)).child("Perimeter"))
                .child(div().w(px(82.0)).child("Max Int.")),
        )
        // Rows body
        .child(
            div()
                .flex_1()
                .w_full()
                .overflow_hidden()
                .flex()
                .flex_col()
                .children(state.rows.iter().take(6).enumerate().map(|(row_idx, row)| {
                    let c_lbl: SharedString = format!("#{}", row.label).into();
                    let c_area: SharedString = format!("{}", row.area).into();
                    let c_mean: SharedString = format!("{:.3}", row.mean_intensity).into();
                    let c_ecc: SharedString = format!("{:.3}", row.eccentricity).into();
                    let c_cent: SharedString =
                        format!("({:.1}, {:.1})", row.centroid_y, row.centroid_x).into();
                    let c_diam: SharedString = format!("{:.1} px", row.equivalent_diameter).into();
                    let c_perim: SharedString = format!("{:.1} px", row.perimeter).into();
                    let c_max: SharedString = format!("{:.3}", row.max_intensity).into();

                    div()
                        .w_full()
                        .px(px(12.0))
                        .py(px(4.0))
                        .bg(if row_idx % 2 == 0 {
                            palette::canvas_bg()
                        } else {
                            palette::panel_bg()
                        })
                        .flex()
                        .items_center()
                        .gap(px(8.0))
                        .text_size(px(11.0))
                        .text_color(palette::text_primary())
                        .child(div().w(px(68.0)).text_color(palette::accent_emerald()).child(c_lbl))
                        .child(div().w(px(82.0)).child(c_area))
                        .child(div().w(px(92.0)).child(c_mean))
                        .child(div().w(px(92.0)).child(c_ecc))
                        .child(div().w(px(110.0)).text_color(palette::text_secondary()).child(c_cent))
                        .child(div().w(px(82.0)).text_color(palette::text_secondary()).child(c_diam))
                        .child(div().w(px(82.0)).text_color(palette::text_secondary()).child(c_perim))
                        .child(div().w(px(82.0)).text_color(palette::text_secondary()).child(c_max))
                })),
        )
}

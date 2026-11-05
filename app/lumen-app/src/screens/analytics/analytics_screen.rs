//! Analytics Drawer Screen (`AnalyticsScreen`): Combines the live 64-bin GPU Histogram chart and
//! the Polars/Arrow RegionProps table.

use std::rc::Rc;

use gpui_kit::{div, prelude::*, px, App, IntoElement, Window};

use super::analytics_event::AnalyticsEvent;
use super::analytics_state::AnalyticsUiState;
use super::components::histogram_chart::render_histogram_chart;
use super::components::region_props_table::render_region_props_table;
use crate::ui::theme::palette;

pub struct AnalyticsScreen;

impl AnalyticsScreen {
    pub fn render(
        state: &AnalyticsUiState,
        on_event: Rc<dyn Fn(AnalyticsEvent, &mut Window, &mut App)>,
    ) -> impl IntoElement {
        div()
            .w_full()
            .h(px(184.0))
            .bg(palette::panel_bg())
            .border_t_1()
            .border_color(palette::border_subtle())
            .flex()
            .flex_row()
            .child(render_histogram_chart(
                &state.hist_counts,
                state.otsu_level,
                state.entropy_bits,
                state.sharpness_score,
            ))
            .child(render_region_props_table(state, on_event))
    }
}

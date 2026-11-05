//! Studio & Explore Workspace Screen (`StudioScreen`):
//! Composes the Blender-style top mode bar, ArtCraft left tool shelf, interactive pan/zoom GPU
//! viewport, right-hand layer & modifier dock, bottom Polars/Arrow analytics drawer, and status bar.

use std::rc::Rc;

use gpui_kit::{div, prelude::*, App, IntoElement, Window};

use super::components::interactive_viewport::render_interactive_viewport;
use super::components::left_tool_shelf::render_left_tool_shelf;
use super::components::right_inspector_dock::render_right_inspector_dock;
use super::components::status_footer::render_status_footer;
use super::components::top_mode_bar::render_top_mode_bar;
use super::studio_event::StudioEvent;
use super::studio_state::StudioUiState;
use crate::screens::analytics::analytics_event::AnalyticsEvent;
use crate::screens::analytics::analytics_screen::AnalyticsScreen;
use crate::screens::analytics::analytics_state::AnalyticsUiState;
use crate::ui::theme::palette;

pub struct StudioScreen;

impl StudioScreen {
    pub fn render(
        state: &StudioUiState,
        analytics_state: &AnalyticsUiState,
        on_studio_event: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)>,
        on_analytics_event: Rc<dyn Fn(AnalyticsEvent, &mut Window, &mut App)>,
    ) -> impl IntoElement {
        let show_drawer = state.show_analytics_drawer;

        div()
            .size_full()
            .bg(palette::canvas_void())
            .flex()
            .flex_col()
            .overflow_hidden()
            // Top mode bar
            .child(render_top_mode_bar(state, on_studio_event.clone()))
            // Middle workspace row: Left Tool Shelf + Center Viewport (+ optional Polars Drawer) + Right Dock
            .child(
                div()
                    .flex_1()
                    .w_full()
                    .flex()
                    .flex_row()
                    .overflow_hidden()
                    .child(render_left_tool_shelf(state, on_studio_event.clone()))
                    .child(
                        div()
                            .flex_1()
                            .h_full()
                            .flex()
                            .flex_col()
                            .overflow_hidden()
                            .child(render_interactive_viewport(state, on_studio_event.clone()))
                            .when(show_drawer, |el| {
                                el.child(AnalyticsScreen::render(analytics_state, on_analytics_event))
                            }),
                    )
                    .child(render_right_inspector_dock(state, on_studio_event)),
            )
            // Bottom status bar
            .child(render_status_footer(state))
    }
}

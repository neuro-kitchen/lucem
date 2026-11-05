//! Startup Screen (`StartupScreen`): Composes the blank canvas backdrop, Left Option Rail, and
//! Center VectorCraft Project Hub.

use std::rc::Rc;

use gpui_kit::{div, prelude::*, App, IntoElement, Window};

use super::components::blank_canvas_backdrop::render_blank_canvas_backdrop;
use super::components::center_project_hub::render_center_project_hub;
use super::components::left_option_rail::render_left_option_rail;
use super::startup_event::StartupEvent;
use super::startup_state::StartupUiState;
use crate::ui::theme::palette;

pub struct StartupScreen;

impl StartupScreen {
    pub fn render(
        state: &StartupUiState,
        on_event: Rc<dyn Fn(StartupEvent, &mut Window, &mut App)>,
    ) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_row()
            .bg(palette::canvas_bg())
            // Left Option Menu (persistent)
            .child(render_left_option_rail(
                state.active_rail_tab,
                &state.compute_backend,
                on_event.clone(),
            ))
            // Right area: Blank Canvas Backdrop + Centered VectorCraft Project Hub overlay
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .relative()
                    .child(render_blank_canvas_backdrop(
                        state.blank_spec.width,
                        state.blank_spec.height,
                        &state.compute_backend,
                    ))
                    .child(
                        div()
                            .occlude()
                            .absolute()
                            .top_0()
                            .left_0()
                            .size_full()
                            .bg(palette::glass_overlay())
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(render_center_project_hub(state, on_event)),
                    ),
            )
    }
}

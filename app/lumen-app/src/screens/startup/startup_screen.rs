//! Startup Screen (`StartupScreen`): Composes the top Blender window titlebar, blank canvas
//! backdrop, Left Option Rail, and Center Project Hub.

use std::rc::Rc;

use gpui_kit::{div, prelude::*, px, App, IntoElement, Window, WindowControlArea};

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
        let on_enter_studio = on_event.clone();
        let on_open_file = on_event.clone();
        let on_open_zarr = on_event.clone();

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(palette::canvas_bg())
            // Top Blender Window Bar with Minimize / Maximize / Close controls
            .child(
                div()
                    .id("startup-top-titlebar")
                    .w_full()
                    .h(px(32.0))
                    .px(px(10.0))
                    .bg(palette::header_bg())
                    .border_b_1()
                    .border_color(palette::border_subtle())
                    .window_control_area(WindowControlArea::Drag)
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.0))
                            .child(
                                div()
                                    .px(px(8.0))
                                    .py(px(2.5))
                                    .rounded(px(3.0))
                                    .bg(palette::widget_bg())
                                    .text_size(px(11.5))
                                    .text_color(palette::text_primary())
                                    .child("LUMEN"),
                            )
                            .child(
                                div()
                                    .id("startup-bar-open-file")
                                    .cursor_pointer()
                                    .px(px(8.0))
                                    .py(px(2.5))
                                    .rounded(px(3.0))
                                    .hover(|s| s.bg(palette::widget_bg()))
                                    .text_size(px(11.5))
                                    .text_color(palette::text_primary())
                                    .on_click(move |_, window, cx| {
                                        on_open_file(StartupEvent::PromptOpenImageFile, window, cx);
                                    })
                                    .child("File"),
                            )
                            .child(
                                div()
                                    .id("startup-bar-open-zarr")
                                    .cursor_pointer()
                                    .px(px(8.0))
                                    .py(px(2.5))
                                    .rounded(px(3.0))
                                    .hover(|s| s.bg(palette::widget_bg()))
                                    .text_size(px(11.5))
                                    .text_color(palette::text_primary())
                                    .on_click(move |_, window, cx| {
                                        on_open_zarr(StartupEvent::PromptOpenZarrFolder, window, cx);
                                    })
                                    .child("Open Zarr"),
                            )
                            .child(
                                div()
                                    .id("startup-bar-enter-studio")
                                    .cursor_pointer()
                                    .px(px(8.0))
                                    .py(px(2.5))
                                    .rounded(px(3.0))
                                    .hover(|s| s.bg(palette::widget_bg()))
                                    .text_size(px(11.5))
                                    .text_color(palette::text_primary())
                                    .on_click(move |_, window, cx| {
                                        on_enter_studio(
                                            StartupEvent::EnterBlankCanvasNow,
                                            window,
                                            cx,
                                        );
                                    })
                                    .child("Studio Workspace"),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(4.0))
                            .child(
                                div()
                                    .id("startup-win-min")
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
                                    .id("startup-win-max")
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
                                    .id("startup-win-close")
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
                    ),
            )
            // Body row: Left Option Rail + Right Blank Canvas & Center Project Hub
            .child(
                div()
                    .flex_1()
                    .w_full()
                    .flex()
                    .flex_row()
                    .overflow_hidden()
                    .child(render_left_option_rail(
                        state.active_rail_tab,
                        &state.compute_backend,
                        on_event.clone(),
                    ))
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
                    ),
            )
    }
}

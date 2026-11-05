//! Bottom Studio status footer (`render_status_footer`) displaying GPU execution time, image
//! statistics (`min`, `max`, `mean`, `std`, `otsu`), and active CubeCL backend.

use gpui_kit::{div, prelude::*, px, IntoElement, SharedString};

use crate::screens::studio::studio_state::StudioUiState;
use crate::ui::components::pill_badge;
use crate::ui::theme::palette;

pub fn render_status_footer(state: &StudioUiState) -> impl IntoElement {
    let stats_summary = if let Some(ref res) = state.pipeline_result {
        format!(
            "Min {:.3}  •  Max {:.3}  •  Mean {:.3} ± {:.3}  •  Otsu {:.3}  •  Contours {}  •  Regions {}",
            res.min_val,
            res.max_val,
            res.mean_val,
            res.std_val,
            res.otsu_level,
            res.contours.len(),
            res.regions.len()
        )
    } else {
        "No pipeline metrics".to_string()
    };

    let backend_badge = format!("CubeCL: {}", state.compute_target_name);

    div()
        .w_full()
        .h(px(28.0))
        .bg(palette::header_bg())
        .border_t_1()
        .border_color(palette::border_subtle())
        .px(px(14.0))
        .flex()
        .items_center()
        .justify_between()
        .child(
            div()
                .text_size(px(11.0))
                .text_color(palette::text_secondary())
                .child(state.status_banner.clone()),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap(px(10.0))
                .child(
                    div()
                        .text_size(px(11.0))
                        .text_color(palette::text_dim())
                        .child(stats_summary),
                )
                .child(pill_badge(
                    SharedString::from(backend_badge),
                    palette::accent_cyan_muted(),
                    palette::accent_cyan(),
                )),
        )
}

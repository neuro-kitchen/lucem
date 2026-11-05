//! Live 64-bin GPU Histogram bar chart with Otsu threshold indicator (`histogram_chart.rs`).

use gpui_kit::{div, prelude::*, px, IntoElement, SharedString};

use crate::ui::theme::palette;

pub fn render_histogram_chart(
    hist_counts: &[u32],
    otsu_level: f64,
    entropy_bits: f64,
    sharpness_score: f64,
) -> impl IntoElement {
    let max_count = hist_counts.iter().copied().max().unwrap_or(1).max(1) as f32;
    let otsu_bin = ((otsu_level.clamp(0.0, 1.0) * 63.0).round() as usize).min(63);

    let otsu_label: SharedString = format!("Otsu: {:.3}", otsu_level).into();
    let ent_label: SharedString = format!("Entropy: {:.2} bits", entropy_bits).into();
    let blur_label: SharedString = format!("Sharpness: {:.4}", sharpness_score).into();

    div()
        .w(px(300.0))
        .h_full()
        .p(px(12.0))
        .border_r_1()
        .border_color(palette::border_subtle())
        .flex()
        .flex_col()
        .justify_between()
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(11.0))
                        .text_color(palette::text_primary())
                        .child("GPU INTENSITY HISTOGRAM (64 BINS)"),
                )
                .child(
                    div()
                        .text_size(px(10.5))
                        .text_color(palette::accent_amber())
                        .child(otsu_label),
                ),
        )
        // Bar chart area
        .child(
            div()
                .w_full()
                .h(px(96.0))
                .p(px(6.0))
                .rounded(px(6.0))
                .bg(palette::canvas_bg())
                .border_1()
                .border_color(palette::border_subtle())
                .flex()
                .items_end()
                .gap(px(1.5))
                .children(hist_counts.iter().copied().enumerate().map(|(bin_idx, count)| {
                    let frac = (count as f32 / max_count).sqrt().clamp(0.04, 1.0);
                    let bar_h = (frac * 80.0).max(2.0);
                    let is_otsu = bin_idx == otsu_bin;
                    div()
                        .flex_1()
                        .h(px(bar_h))
                        .rounded(px(1.0))
                        .bg(if is_otsu {
                            palette::accent_amber()
                        } else {
                            palette::accent_blue()
                        })
                })),
        )
        // Summary metrics footer
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .text_size(px(10.5))
                .text_color(palette::text_secondary())
                .child(ent_label)
                .child(blur_label),
        )
}

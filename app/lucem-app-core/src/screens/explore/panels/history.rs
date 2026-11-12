//! History (Affinity's): every change of the project, oldest first; a click goes back (or
//! forward) to just after that change. Changes undone stay listed, dimmed, until a new change.
//!
//! Then the **session**: the analysis steps recorded (tissue, cells, classes, scores, exports)
//! with their settings — `session.json` next to the project, which `lucem session run` runs
//! again on other images without the app.

use gpui_kit::component::v_flex;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, App, Entity, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _};

use crate::data::store::Store;
use crate::ui::theme::{self, color};

pub fn panel(store: &Entity<Store>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let (done, undone) = s.history();
    let n = done.len();
    // Row k: the state after k changes (row 0: before any)
    let row = |k: usize, text: String, current: bool, ahead: bool| {
        let st = store.clone();
        div()
            .id(SharedString::from(format!("history-{k}")))
            .h(px(26.))
            .px(px(theme::SM))
            .flex()
            .items_center()
            .rounded(px(4.))
            .text_sm()
            .cursor_pointer()
            .when(current, |d| d.bg(color::accent().opacity(0.35)))
            .when(!current, |d| d.hover(|d| d.bg(color::control())))
            .when(ahead, |d| d.text_color(color::text_disabled()))
            .child(div().truncate().child(text))
            .on_click(move |_, _, cx| st.update(cx, |s, cx| s.undo_to(k, cx)))
    };
    let mut list = v_flex().gap(px(1.)).child(row(0, "Opened".to_string(), n == 0, false));
    for (i, label) in done.iter().enumerate() {
        list = list.child(row(i + 1, label.clone(), i + 1 == n, false));
    }
    for (j, label) in undone.iter().enumerate() {
        list = list.child(row(n + j + 1, (*label).clone(), false, true));
    }
    // The session: what the analysis did, replayable headless
    let steps = &s.hist.session.steps;
    let mut session = v_flex()
        .gap(px(2.))
        .pt(px(theme::LG))
        .child(div().text_base().font_weight(gpui_kit::FontWeight::SEMIBOLD).child("Session"))
        .child(div().text_xs().text_color(color::text_muted()).child(if steps.is_empty() { "The analysis steps you run are recorded here.".to_string() } else { format!("{} step(s) · session.json · lucem session run session.json <images…>", steps.len()) }));
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    for st in steps.iter().rev().take(200) {
        let ago = now.saturating_sub(st.at);
        let when = if ago < 60 { "now".to_string() } else if ago < 3600 { format!("{} min ago", ago / 60) } else if ago < 86_400 { format!("{} h ago", ago / 3600) } else { format!("{} d ago", ago / 86_400) };
        session = session.child(
            gpui_kit::component::h_flex()
                .h(px(24.))
                .px(px(theme::SM))
                .gap(px(theme::SM))
                .child(div().flex_1().min_w_0().truncate().text_sm().child(st.action.title()))
                .child(div().flex_none().max_w(px(120.)).truncate().text_xs().text_color(color::text_muted()).child(st.image.clone()))
                .child(div().flex_none().text_xs().text_color(color::text_muted()).child(when)),
        );
    }
    v_flex().id("history-panel").size_full().overflow_y_scroll().p(px(theme::SM)).child(list).child(session).into_any_element()
}

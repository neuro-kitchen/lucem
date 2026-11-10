//! History (Affinity's): every change of the project, oldest first; a click goes back (or
//! forward) to just after that change. Changes undone stay listed, dimmed, until a new change.

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
    v_flex().id("history-panel").size_full().overflow_y_scroll().p(px(theme::SM)).child(list).into_any_element()
}

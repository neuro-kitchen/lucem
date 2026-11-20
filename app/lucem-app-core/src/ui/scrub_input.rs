//! Dual-mode numeric input (Affinity / Blender / `kitchen-ui` pattern):
//! - **Drag horizontally:** scrubs continuously (window-level tracking, pointer may leave the control).
//! - **Click without dragging:** enters inline text editing (`Enter` commits parsed number,
//!   `Esc` or blur cancels).

use std::rc::Rc;

use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::Sizable as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, Focusable as _, InteractiveElement as _, IntoElement, ParentElement as _,
    SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};
use kitchen_ui::widgets::scrub_field;

/// Shared editing state for all [`scrub_number`] fields in one toolbar / panel (only one field
/// can be in click-to-type text mode at a time).
pub struct ScrubState {
    active_id: Option<String>,
    input: Entity<InputState>,
    commit: Option<Rc<dyn Fn(f64, &mut Window, &mut App)>>,
    _sub: Subscription,
}

impl ScrubState {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| InputState::new(window, cx));
        let _sub = cx.subscribe_in(&input, window, |this, input, e: &InputEvent, window, cx| match e {
            InputEvent::PressEnter { .. } => {
                let raw = input.read(cx).value().trim().to_string();
                let parsed = parse_number(&raw);
                let commit = this.commit.take();
                this.active_id = None;
                if let (Some(v), Some(cb)) = (parsed, commit) {
                    cb(v, window, cx);
                }
                cx.notify();
            }
            InputEvent::Blur => {
                this.active_id = None;
                this.commit = None;
                cx.notify();
            }
            _ => {}
        });
        Self { active_id: None, input, commit: None, _sub }
    }

    pub fn begin_edit(&mut self, id: String, initial: String, on_commit: Rc<dyn Fn(f64, &mut Window, &mut App)>, window: &mut Window, cx: &mut Context<Self>) {
        self.active_id = Some(id);
        self.commit = Some(on_commit);
        self.input.update(cx, |i, cx| {
            i.set_value(initial, window, cx);
            i.focus_handle(cx).focus(window, cx);
        });
        cx.notify();
    }
}

/// Extracts a leading/embedded finite `f64` from user text (tolerating trailing units like `48 px`
/// or `12.5 %`).
fn parse_number(text: &str) -> Option<f64> {
    let token = text.split_whitespace().next().unwrap_or(text).trim_end_matches('%');
    token.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Renders a clean `‹ value ›` numeric field backed by `kitchen_ui::widgets::scrub_field`:
/// **drag horizontally** to scrub (even after leaving the control) or **click** to type an exact value.
///
/// - `display`: formatted label shown when idle (e.g. `"48 px"`, `"16.41"`, `"12.4 µm²"`).
/// - `edit_value`: plain numeric string pre-filled when clicked (e.g. `"48"`, `"16.41"`).
/// - `on_step`: called during horizontal drag with signed step (`-1.0` / `+1.0`).
/// - `on_set`: called when the user types an exact number and presses `Enter`.
#[allow(clippy::too_many_arguments)]
pub fn scrub_number(
    state: &Entity<ScrubState>,
    id: impl Into<String>,
    display: impl Into<SharedString>,
    edit_value: String,
    tooltip: &'static str,
    on_step: impl Fn(f32, &mut Window, &mut App) + 'static,
    on_set: impl Fn(f64, &mut Window, &mut App) + 'static,
    cx: &App,
) -> AnyElement {
    let id = id.into();
    let s = state.read(cx);
    if s.active_id.as_deref() == Some(id.as_str()) {
        return div().w(px(76.)).child(Input::new(&s.input).xsmall()).into_any_element();
    }
    let st_edit = state.clone();
    let on_set: Rc<dyn Fn(f64, &mut Window, &mut App)> = Rc::new(on_set);
    let tip_id = SharedString::from(format!("scrub-tip-{id}"));
    let field_id = format!("scrub-{id}");
    div()
        .id(tip_id)
        .tooltip(move |window, cx| gpui_kit::component::tooltip::Tooltip::new(tooltip).build(window, cx))
        .child(scrub_field(
            field_id,
            display.into(),
            on_step,
            move |window, cx| {
                let initial = edit_value.clone();
                let cb = on_set.clone();
                let edit_id = id.clone();
                st_edit.update(cx, |s, cx| s.begin_edit(edit_id, initial, cb, window, cx));
            },
        ))
        .into_any_element()
}

//! The parameters of one op, editable (from dsp-kitchen): each number is an input (Enter or
//! leaving the field applies it). Whoever owns the op decides how a change is applied (`commit`:
//! a project command), so the Explore sidebar and the Pipeline inspector share it.

use std::rc::Rc;

use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{v_flex, Sizable as _};
use gpui_kit::{div, App, AppContext as _, Context, Entity, IntoElement, ParentElement as _, Render, Styled as _, Subscription, Window};
use lumen_pipeline::Op;
use serde_json::{json, Value};

use super::theme::color;
use super::widgets::row;

/// Sets the new op; an error is shown under the fields.
pub type Commit = Rc<dyn Fn(Op, &mut App) -> Result<(), String>>;

struct Field {
    key: String,
    label: String,
    input: Entity<InputState>,
}

#[derive(Default)]
pub struct ParamEditor {
    fields: Vec<Field>,
    /// What the fields were built for (key and op JSON).
    fields_for: String,
    field_subs: Vec<Subscription>,
    error: Option<String>,
    target: Option<(Op, Commit)>,
}

impl ParamEditor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Edits `op`, known as `key` (its node id); inputs are rebuilt when the key or the values
    /// change.
    pub fn set_target(&mut self, key: &str, op: &Op, commit: Commit, window: &mut Window, cx: &mut Context<Self>) {
        let value = serde_json::to_value(op).unwrap_or(Value::Null);
        self.target = Some((op.clone(), commit));
        let fresh = format!("{key}/{value}");
        if fresh == self.fields_for {
            return;
        }
        if !self.fields_for.starts_with(&format!("{key}/")) {
            self.error = None;
        }
        self.fields_for = fresh;
        self.fields.clear();
        self.field_subs.clear();
        let Value::Object(map) = &value else { return };
        for (k, v) in map {
            if let Value::Number(n) = v {
                let input = cx.new(|cx| InputState::new(window, cx).default_value(trim_number(n.as_f64().unwrap_or(0.0))));
                let sub = cx.subscribe(&input, |this, _, e: &InputEvent, cx| {
                    if matches!(e, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                        this.apply_fields(cx);
                    }
                });
                self.field_subs.push(sub);
                self.fields.push(Field { key: k.clone(), label: label_of(k), input });
            }
        }
    }

    /// The numbers typed in, on the op; a value that does not parse, or parameters the op rejects,
    /// leave it unchanged and say why.
    fn apply_fields(&mut self, cx: &mut Context<Self>) {
        let Some((op, commit)) = self.target.clone() else { return };
        let mut value = serde_json::to_value(&op).unwrap_or(Value::Null);
        for f in &self.fields {
            let text = f.input.read(cx).value().to_string();
            let Ok(n) = text.trim().parse::<f64>() else {
                self.error = Some(format!("{}: {text:?} is not a number", f.label));
                cx.notify();
                return;
            };
            if let Some(slot) = value.get_mut(&f.key) {
                // Integers stay integers (radii, counts)
                *slot = if slot.is_u64() || slot.is_i64() { json!(n.round().max(0.0) as u64) } else { json!(n) };
            }
        }
        match serde_json::from_value::<Op>(value) {
            Ok(new) if new == op => self.error = None,
            Ok(new) => self.error = commit(new, cx).err(),
            Err(e) => self.error = Some(e.to_string()),
        }
        cx.notify();
    }
}

impl Render for ParamEditor {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut out = v_flex().w_full().gap_2();
        for f in &self.fields {
            out = out.child(row(f.label.clone(), div().w(gpui_kit::px(120.)).child(Input::new(&f.input).small())));
        }
        if self.fields.is_empty() && self.target.is_some() {
            out = out.child(div().text_xs().text_color(color::text_muted()).child("No parameters."));
        }
        out.children(self.error.clone().map(|e| div().text_xs().text_color(color::danger()).child(e)))
    }
}

/// `clip_limit` → `Clip limit`
pub fn label_of(key: &str) -> String {
    let mut s = key.replace('_', " ");
    if let Some(c) = s.get_mut(0..1) {
        c.make_ascii_uppercase();
    }
    s
}

/// `300` rather than `300.0`, at most 6 decimals.
pub fn trim_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_and_numbers() {
        assert_eq!(label_of("clip_limit"), "Clip limit");
        assert_eq!(trim_number(300.0), "300");
        assert_eq!(trim_number(0.25), "0.25");
    }
}

//! The parameters of one op, editable: kitchen-ui's editor for lucem's pipeline ops. Each number,
//! short number list and switch is an input (Enter or leaving the field applies it). Whoever owns
//! the op decides how a change is applied (`commit`: a project command), so the Explore sidebar and
//! the Pipeline inspector share it.

use std::rc::Rc;

use gpui_kit::{App, Context, Window};
use lucem_pipeline::Op;
use serde::{Deserialize, Serialize};

pub use kitchen_ui::param_editor::{label_of, trim_number};

/// A pipeline op as kitchen-ui's editor sees it (the same JSON). A wrapper, so the pipeline engine
/// does not depend on the UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EditOp(pub Op);

impl kitchen_ui::Params for EditOp {}

/// The editor of a pipeline op.
pub type ParamEditor = kitchen_ui::ParamEditor<EditOp>;
/// Sets the new op; an error is shown under the fields.
pub type Commit = Rc<dyn Fn(Op, &mut App) -> Result<(), String>>;

/// Editing a pipeline op directly.
pub trait EditOps {
    /// Edits `op`, known as `key` (its node id); `commit` applies a change.
    fn set_op(&mut self, key: &str, op: &Op, commit: Commit, window: &mut Window, cx: &mut Context<Self>)
    where
        Self: Sized;
}

impl EditOps for ParamEditor {
    fn set_op(&mut self, key: &str, op: &Op, commit: Commit, window: &mut Window, cx: &mut Context<Self>) {
        self.set_target(key, &EditOp(op.clone()), Rc::new(move |e: EditOp, cx: &mut App| commit(e.0, cx)), window, cx);
    }
}

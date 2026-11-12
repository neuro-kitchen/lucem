//! The document side of the app: a change is a [`Command`] applied to the project document
//! ([`crate::domain::command::apply`]), recorded in the [`History`] to undo and redo, saved, and
//! followed by what depends on it (results recomputed, selections kept valid, events). Agents
//! and the menus change the document the same way: through `Store::execute`.

use std::path::{Path, PathBuf};

use gpui_kit::Context;

use crate::domain::command::{apply, Changed, Command, CommandError};
use crate::domain::graph::NodeKind;
use crate::domain::project::ProjectDoc;

use super::store::{AppEvent, Store, UNDO_DEPTH};

/// The changes made to the document: the states before each (oldest first) with what it did,
/// and the undone ones (the next redo last).
#[derive(Default)]
pub struct History {
    done: Vec<(ProjectDoc, String)>,
    undone: Vec<(ProjectDoc, String)>,
}

impl History {
    /// A change was made from `before`: it can be undone; what was undone is forgotten.
    pub fn record(&mut self, before: ProjectDoc, label: String) {
        self.done.push((before, label));
        if self.done.len() > UNDO_DEPTH {
            self.done.remove(0);
        }
        self.undone.clear();
    }

    /// The document before the last change (`current` kept to redo), if any.
    pub fn undo(&mut self, current: ProjectDoc) -> Option<ProjectDoc> {
        let (before, label) = self.done.pop()?;
        self.undone.push((current, label));
        Some(before)
    }

    /// The document after the last undone change (`current` kept to undo), if any.
    pub fn redo(&mut self, current: ProjectDoc) -> Option<ProjectDoc> {
        let (after, label) = self.undone.pop()?;
        self.done.push((current, label));
        Some(after)
    }

    pub fn can_undo(&self) -> bool {
        !self.done.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.undone.is_empty()
    }

    /// How many changes are applied.
    pub fn done(&self) -> usize {
        self.done.len()
    }

    /// What each change did: the applied ones (oldest first), then the undone ones (the next
    /// redo first).
    pub fn labels(&self) -> (Vec<String>, Vec<&String>) {
        (self.done.iter().map(|(_, l)| l.clone()).collect(), self.undone.iter().rev().map(|(_, l)| l).collect())
    }
}

impl Store {
    /// Runs a command on the document, saves it and says what changed.
    pub fn execute(&mut self, cmd: Command, cx: &mut Context<Self>) -> Result<Changed, CommandError> {
        let before = self.doc.clone();
        let label = cmd.label();
        let result = apply(&mut self.doc, cmd);
        match &result {
            Ok(change) => {
                if before != self.doc {
                    self.undo.record(before, label);
                    self.save();
                }
                self.after_change(change.clone(), cx);
            }
            Err(e) => self.set_status(e.to_string(), cx),
        }
        result
    }

    /// Back to the document before the last change (`redo`: forward again).
    pub fn undo(&mut self, redo: bool, cx: &mut Context<Self>) {
        let current = self.doc.clone();
        let Some(doc) = (if redo { self.undo.redo(current) } else { self.undo.undo(current) }) else {
            self.set_status(if redo { "Nothing to redo" } else { "Nothing to undo" }, cx);
            return;
        };
        self.doc = doc;
        self.save();
        self.sel.shapes.retain(|id| self.doc.annotations.iter().any(|a| &a.id == id));
        self.annotations_rev += 1;
        self.ensure_pyramids(cx);
        self.after_change(Changed::Graph, cx);
        cx.emit(AppEvent::Document(Changed::Images));
        cx.emit(AppEvent::Document(Changed::Annotations));
        self.set_status(if redo { "Redone" } else { "Undone" }, cx);
    }

    /// The History panel's rows: what each step did, done ones first (oldest first), then the
    /// undone ones (the next redo first).
    pub fn history(&self) -> (Vec<String>, Vec<&String>) {
        self.undo.labels()
    }

    /// Undoes or redoes until `done` steps of the history are applied (a click in History).
    pub fn undo_to(&mut self, done: usize, cx: &mut Context<Self>) {
        while self.undo.done() > done && self.undo.can_undo() {
            self.undo(false, cx);
        }
        while self.undo.done() < done && self.undo.can_redo() {
            self.undo(true, cx);
        }
    }

    pub fn can_undo(&self) -> bool {
        self.undo.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.undo.can_redo()
    }

    pub(crate) fn after_change(&mut self, change: Changed, cx: &mut Context<Self>) {
        match &change {
            Changed::Images => {
                // Selected layers and objects belong to the image shown before
                self.sel.layers.clear();
                self.sel.objects.clear();
                self.ensure_pyramids(cx);
                self.results.retain(|_, r| self.doc.image(&r.image).is_some());
            }
            Changed::Graph => {
                self.watch_scripts(cx);
                self.results.retain(|v, _| self.doc.graph.contains(v));
                self.recompute(cx);
            }
            Changed::Layout => {
                // A view's style changes how its layer is drawn
                self.recompute(cx);
            }
            Changed::Annotations => {
                self.annotations_rev += 1;
                self.sel.shapes.retain(|id| self.doc.annotations.iter().any(|a| &a.id == id));
                // A deleted or renamed class: draw with one that exists, unselect its layer
                if self.doc.class(&self.tool.class).is_none()
                    && let Some(c) = self.doc.classes.iter().find(|c| c.name != crate::domain::command::MEASUREMENT) {
                        self.tool.class = c.name.clone();
                    }
                let classes: Vec<String> = self.doc.classes.iter().map(|c| format!("class:{}", c.name)).collect();
                self.sel.layers.retain(|k| !k.starts_with("class:") || classes.contains(k));
                self.results.retain(|v, _| self.doc.graph.contains(v));
                let reads_annotations = self.doc.graph.nodes.iter().any(|n| matches!(n.kind, NodeKind::Annotations { .. }));
                if reads_annotations {
                    self.recompute(cx);
                }
            }
            // A channel with processing draws through its result (its colour, window and eye):
            // that layer is made again
            Changed::Display(image) => {
                let n = self.doc.image(image).map_or(0, |i| i.channels.len());
                if (0..n).any(|c| self.doc.graph.channel_chain(image, c).1.is_some()) {
                    self.results_for = None;
                    self.recompute(cx);
                }
            }
            _ => {}
        }
        cx.emit(AppEvent::Document(change));
        cx.notify();
    }

    /// Writes the project file when the project has a folder.
    pub(crate) fn save(&mut self) {
        if let Some(path) = &self.project_file
            && let Err(e) = self.doc.save(path) {
                self.status = format!("Project not saved: {e}");
            }
    }

    /// The project's folder.
    pub fn folder(&self) -> Option<PathBuf> {
        self.project_file.as_ref().and_then(|p| p.parent().map(Path::to_path_buf))
    }
}

#[cfg(test)]
mod tests {
    use super::History;
    use crate::domain::project::ProjectDoc;

    fn named(n: &str) -> ProjectDoc {
        ProjectDoc { name: n.into(), ..Default::default() }
    }

    #[test]
    fn undo_redo_and_a_new_change_forgets_what_was_undone() {
        let mut h = History::default();
        h.record(named("a"), "first".into());
        h.record(named("b"), "second".into());
        assert_eq!(h.undo(named("c")).unwrap().name, "b");
        assert_eq!(h.labels().1, vec!["second"]);
        assert_eq!(h.redo(named("b")).unwrap().name, "c");
        assert_eq!(h.done(), 2);
        h.undo(named("c"));
        h.record(named("b"), "other".into());
        assert!(!h.can_redo(), "a new change drops the undone one");
        assert_eq!(h.labels().0, vec!["first", "other"]);
    }
}

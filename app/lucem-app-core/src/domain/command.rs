//! Commands: every change to the project document. The UI, the command line and an agent (a
//! future MCP server) send the same commands, so there is one code path; each command is plain
//! data (JSON, `{"command": "add_annotation", …}`) and [`apply`] is pure (no I/O, no device), so
//! it is tested directly. A tool stroke, an op added from a menu, a node dragged on the Pipeline
//! canvas: each is a command, so each is undoable, saved, and open to an LLM.

use std::path::PathBuf;

use lucem_pipeline::Op;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::annotation::{Annotation, ClassDoc};
use super::graph::{GraphNode, NodeKind, ViewStyle, COLUMN};
use super::project::{ChannelDoc, ImageRef, NoteDoc, ProjectDoc};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    /// Adds an image (and shows it).
    AddImage {
        path: PathBuf,
        name: String,
        #[serde(default)]
        channels: Vec<ChannelDoc>,
    },
    /// Removes an image, its annotations and the nodes reading it.
    RemoveImage { id: String },
    /// Shows an image in Explore.
    SelectImage { id: String },
    /// How channel `channel` of `image` is shown.
    SetChannel { image: String, channel: usize, display: ChannelDoc },
    /// Adds a node at (`x`, `y`); `after`'s output feeds its first fitting input.
    AddNode {
        kind: NodeKind,
        #[serde(default)]
        x: f32,
        #[serde(default)]
        y: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        after: Option<String>,
    },
    /// Removes nodes and their edges.
    RemoveNodes { nodes: Vec<String> },
    /// Removes a node; its first input feeds what it fed (where the types allow).
    RemoveBridging { id: String },
    MoveNodes { moves: Vec<NodeMove> },
    /// Feeds input `port` of `to` from `from`.
    Connect {
        from: String,
        to: String,
        #[serde(default)]
        port: usize,
    },
    /// Empties input `port` of `to`.
    Disconnect {
        to: String,
        #[serde(default)]
        port: usize,
    },
    /// Replaces an op node's operator.
    SetOp { id: String, op: Op },
    EnableNode { id: String, enabled: bool },
    SetView { id: String, name: String, style: ViewStyle },
    /// Points a source node at another channel.
    SetSourceChannel { id: String, channel: usize },
    /// Replaces a script node's file, ports and numbers (edges past its inputs are dropped).
    SetScript { id: String, kind: NodeKind },
    /// Applies `op` to the result shown by `view` (inserted just before it); without a view, a
    /// new chain from channel `channel` of `image` into a new view.
    ApplyOp {
        op: Op,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        view: Option<String>,
        image: String,
        #[serde(default)]
        channel: usize,
    },
    /// Adds an annotation (an empty id gets one).
    AddAnnotation { annotation: Annotation },
    /// Adds many annotations in one step (ids given where empty).
    AddAnnotations { annotations: Vec<Annotation> },
    /// Replaces the annotation with its id.
    SetAnnotation { annotation: Annotation },
    RemoveAnnotations { ids: Vec<String> },
    MoveAnnotations { ids: Vec<String>, dx: f64, dy: f64 },
    /// Adds a class, or replaces the one with its name.
    SetClass { class: ClassDoc },
    /// Adds a note (an empty id gets one) or replaces the one with its id.
    SetNote { note: NoteDoc },
    RemoveNote { id: String },
    /// Deletes a class. Its annotations move to `move_to` when given, else they are deleted too;
    /// the pipeline's annotation sources of the class follow them (or go). The rulers' class
    /// (Measurement) and the last class stay.
    RemoveClass {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        move_to: Option<String>,
    },
    /// Renames a class, with its annotations and the pipeline's sources of it.
    RenameClass { from: String, to: String },
    /// Puts annotations in another class.
    SetAnnotationsClass { ids: Vec<String>, class: String },
    Rename { name: String },
    /// Several commands as one step: all of them apply, or none.
    Batch { commands: Vec<Command> },
}

impl Command {
    /// A short name for the History panel.
    pub fn label(&self) -> String {
        use Command as C;
        match self {
            C::AddImage { name, .. } => format!("Add image {name}"),
            C::RemoveImage { .. } => "Remove image".into(),
            C::SelectImage { .. } => "Show image".into(),
            C::SetChannel { display, .. } => format!("Display {}", display.label),
            C::AddNode { kind, .. } => match kind {
                NodeKind::Source { .. } => "Add source".into(),
                NodeKind::Annotations { .. } => "Add annotations source".into(),
                NodeKind::Op { op, .. } => format!("Add {}", op.title()),
                NodeKind::View { .. } => "Add view".into(),
                NodeKind::Script { path, .. } => format!("Add script {}", std::path::Path::new(path).file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())),
                NodeKind::Model { model, .. } => format!("Add model {model}"),
            },
            C::RemoveNodes { nodes } => format!("Remove {} node(s)", nodes.len()),
            C::RemoveBridging { .. } => "Remove node".into(),
            C::MoveNodes { .. } => "Move nodes".into(),
            C::Connect { .. } => "Connect".into(),
            C::Disconnect { .. } => "Disconnect".into(),
            C::SetOp { op, .. } => format!("Edit {}", op.title()),
            C::EnableNode { enabled, .. } => if *enabled { "Enable node".into() } else { "Bypass node".into() },
            C::SetView { name, .. } => format!("Edit view {name}"),
            C::SetSourceChannel { .. } => "Source channel".into(),
            C::SetScript { .. } => "Edit script".into(),
            C::ApplyOp { op, .. } => format!("Apply {}", op.title()),
            C::AddAnnotation { annotation } => format!("{} {}", shape_verb(&annotation.shape), annotation.class),
            C::AddAnnotations { annotations } => format!("Add {} annotation(s)", annotations.len()),
            C::SetAnnotation { .. } => "Edit annotation".into(),
            C::RemoveAnnotations { ids } => format!("Delete {} annotation(s)", ids.len()),
            C::MoveAnnotations { ids, .. } => format!("Move {} annotation(s)", ids.len()),
            C::SetClass { class } => format!("Class {}", class.name),
            C::SetNote { .. } => "Note".into(),
            C::RemoveNote { .. } => "Delete note".into(),
            C::RemoveClass { name, move_to: Some(to) } => format!("Delete class {name} (annotations to {to})"),
            C::RemoveClass { name, move_to: None } => format!("Delete class {name}"),
            C::RenameClass { from, to } => format!("Rename class {from} to {to}"),
            C::SetAnnotationsClass { ids, class } => format!("Move {} annotation(s) to {class}", ids.len()),
            C::Rename { name } => format!("Rename to {name}"),
            C::Batch { commands } => match commands.first() {
                Some(c) if commands.len() == 1 => c.label(),
                _ => format!("{} changes", commands.len()),
            },
        }
    }
}

/// What drawing an annotation of this shape is called.
fn shape_verb(shape: &super::annotation::Shape) -> &'static str {
    use super::annotation::Shape as S;
    match shape {
        S::Stroke { erase: true, .. } => "Erase",
        S::Stroke { .. } => "Paint",
        S::Point { .. } => "Point",
        S::Line { .. } => "Measure",
        _ => "Region",
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeMove {
    pub id: String,
    pub x: f32,
    pub y: f32,
}

/// What a command changed, so the app redraws (and recomputes) only that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Changed {
    /// Images added, removed, or another shown.
    Images,
    /// How an image's channels are shown.
    Display(String),
    /// The graph (results must be computed again).
    Graph,
    /// Only where nodes sit, or a view's style (nothing to compute).
    Layout,
    /// Annotations or classes.
    Annotations,
    Notes,
    Project,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommandError {
    #[error("no image {0:?}")]
    NoImage(String),
    #[error("no node {0:?}")]
    NoNode(String),
    #[error("no annotation {0:?}")]
    NoAnnotation(String),
    #[error("{0}")]
    Invalid(String),
}

/// Applies `cmd` to `doc`; on an error `doc` is unchanged.
pub fn apply(doc: &mut ProjectDoc, cmd: Command) -> Result<Changed, CommandError> {
    let mut work = doc.clone();
    let change = apply_inner(&mut work, cmd)?;
    *doc = work;
    Ok(change)
}

fn invalid(e: impl Into<String>) -> CommandError {
    CommandError::Invalid(e.into())
}

fn apply_inner(doc: &mut ProjectDoc, cmd: Command) -> Result<Changed, CommandError> {
    use Command as C;
    match cmd {
        C::AddImage { path, name, channels } => {
            let id = doc.new_image_id();
            doc.images.push(ImageRef { id: id.clone(), path, name, channels });
            doc.active_image = Some(id);
            Ok(Changed::Images)
        }
        C::RemoveImage { id } => {
            image(doc, &id)?;
            doc.images.retain(|i| i.id != id);
            doc.annotations.retain(|a| a.image != id);
            let reading: Vec<String> = doc.graph.nodes.iter().filter(|n| n.kind.image() == Some(id.as_str())).map(|n| n.id.clone()).collect();
            for n in reading {
                doc.graph.remove(&n);
            }
            if doc.active_image.as_deref() == Some(id.as_str()) {
                doc.active_image = doc.images.first().map(|i| i.id.clone());
            }
            Ok(Changed::Images)
        }
        C::SelectImage { id } => {
            image(doc, &id)?;
            doc.active_image = Some(id);
            Ok(Changed::Images)
        }
        C::SetChannel { image: id, channel, display } => {
            if !(display.min.is_finite() && display.max.is_finite() && display.max > display.min) {
                return Err(invalid("a channel's max must be above its min"));
            }
            if !(display.gamma.is_finite() && display.gamma > 0.0) {
                return Err(invalid("gamma must be positive"));
            }
            let img = doc.image_mut(&id).ok_or_else(|| CommandError::NoImage(id.clone()))?;
            let slot = img.channels.get_mut(channel).ok_or_else(|| invalid(format!("{id} has no channel {channel}")))?;
            *slot = display;
            Ok(Changed::Display(id))
        }
        C::AddNode { kind, x, y, id, after } => {
            check_kind(doc, &kind)?;
            let id = match id {
                Some(id) if doc.graph.contains(&id) => return Err(invalid(format!("node id {id:?} is taken"))),
                Some(id) if id.trim().is_empty() => return Err(invalid("a node id cannot be empty")),
                Some(id) => id,
                None => doc.graph.fresh_id(&base_id(&kind)),
            };
            doc.graph.nodes.push(GraphNode { id: id.clone(), x, y, kind });
            if let Some(from) = after {
                let port = doc.graph.port_for(&from, &id).ok_or_else(|| invalid(format!("{from}'s output fits no input of the new node")))?;
                doc.graph.connect(&from, &id, port).map_err(invalid)?;
            }
            Ok(Changed::Graph)
        }
        C::RemoveNodes { nodes } => {
            for id in &nodes {
                node(doc, id)?;
            }
            for id in &nodes {
                doc.graph.remove(id);
            }
            Ok(Changed::Graph)
        }
        C::RemoveBridging { id } => {
            node(doc, &id)?;
            doc.graph.remove_bridging(&id);
            Ok(Changed::Graph)
        }
        C::MoveNodes { moves } => {
            for m in moves {
                if !(m.x.is_finite() && m.y.is_finite()) {
                    return Err(invalid("node positions are numbers"));
                }
                let n = doc.graph.node_mut(&m.id).ok_or_else(|| CommandError::NoNode(m.id.clone()))?;
                (n.x, n.y) = (m.x, m.y);
            }
            Ok(Changed::Layout)
        }
        C::Connect { from, to, port } => {
            doc.graph.connect(&from, &to, port).map_err(invalid)?;
            Ok(Changed::Graph)
        }
        C::Disconnect { to, port } => {
            node(doc, &to)?;
            doc.graph.disconnect(&to, port);
            Ok(Changed::Graph)
        }
        C::SetOp { id, op } => {
            op.validate().map_err(invalid)?;
            let n = doc.graph.node_mut(&id).ok_or_else(|| CommandError::NoNode(id.clone()))?;
            match &mut n.kind {
                NodeKind::Op { op: o, .. } => {
                    // A new kind of op may take other inputs: keep the edges that still fit
                    let same_ports = o.inputs() == op.inputs() && o.output() == op.output();
                    *o = op;
                    if !same_ports {
                        let edges: Vec<_> = doc.graph.edges.iter().filter(|e| e.to == id || e.from == id).cloned().collect();
                        doc.graph.edges.retain(|e| e.to != id && e.from != id);
                        for e in edges {
                            let _ = doc.graph.connect(&e.from, &e.to, e.port);
                        }
                    }
                }
                _ => return Err(invalid(format!("{id} is not an op"))),
            }
            Ok(Changed::Graph)
        }
        C::EnableNode { id, enabled } => {
            match &mut doc.graph.node_mut(&id).ok_or_else(|| CommandError::NoNode(id.clone()))?.kind {
                NodeKind::Op { enabled: e, .. } => *e = enabled,
                _ => return Err(invalid(format!("{id} is not an op"))),
            }
            Ok(Changed::Graph)
        }
        C::SetView { id, name, style } => {
            if !(0.0..=1.0).contains(&style.opacity) {
                return Err(invalid("opacity is between 0 and 1"));
            }
            match &mut doc.graph.node_mut(&id).ok_or_else(|| CommandError::NoNode(id.clone()))?.kind {
                NodeKind::View { name: n, style: s } => {
                    *n = if name.trim().is_empty() { n.clone() } else { name };
                    *s = style;
                }
                _ => return Err(invalid(format!("{id} is not a view"))),
            }
            Ok(Changed::Layout)
        }
        C::SetScript { id, kind } => {
            let same = matches!((&kind, &node(doc, &id)?.kind), (NodeKind::Script { .. }, NodeKind::Script { .. }) | (NodeKind::Model { .. }, NodeKind::Model { .. }));
            if !same {
                return Err(invalid(format!("{id} is not a script or model node of that kind")));
            }
            let n = kind.inputs().len();
            if let Some(g) = doc.graph.node_mut(&id) {
                g.kind = kind;
            }
            doc.graph.edges.retain(|e| !(e.to == id && e.port >= n));
            Ok(Changed::Graph)
        }
        C::SetSourceChannel { id, channel } => {
            let img = match &node(doc, &id)?.kind {
                NodeKind::Source { image, .. } => image.clone(),
                _ => return Err(invalid(format!("{id} is not an image source"))),
            };
            let n = image(doc, &img)?.channels.len();
            if n > 0 && channel >= n {
                return Err(invalid(format!("{img} has {n} channel(s)")));
            }
            if let Some(GraphNode { kind: NodeKind::Source { channel: c, .. }, .. }) = doc.graph.node_mut(&id) {
                *c = channel;
            }
            Ok(Changed::Graph)
        }
        C::ApplyOp { op, view, image: img, channel } => {
            op.validate().map_err(invalid)?;
            image(doc, &img)?;
            match view {
                Some(v) => {
                    let n = node(doc, &v)?;
                    if !matches!(n.kind, NodeKind::View { .. }) {
                        return Err(invalid(format!("{v} is not a view")));
                    }
                    let (x, y) = (n.x, n.y);
                    let id = doc.graph.fresh_id(op.tag());
                    doc.graph.nodes.push(GraphNode { id: id.clone(), x, y, kind: NodeKind::Op { enabled: true, op } });
                    if let Some(n) = doc.graph.node_mut(&v) {
                        n.x += COLUMN;
                    }
                    if let Err(e) = doc.graph.splice(&v, 0, &id) {
                        return Err(invalid(e));
                    }
                }
                None => {
                    let y = doc.graph.next_row();
                    let s = doc.graph.fresh_id("source");
                    doc.graph.nodes.push(GraphNode { id: s.clone(), x: 0.0, y, kind: NodeKind::Source { image: img.clone(), channel } });
                    let id = doc.graph.fresh_id(op.tag());
                    let title = op.title().to_string();
                    doc.graph.nodes.push(GraphNode { id: id.clone(), x: COLUMN, y, kind: NodeKind::Op { enabled: true, op } });
                    let v = doc.graph.fresh_id("view");
                    doc.graph.nodes.push(GraphNode { id: v.clone(), x: 2.0 * COLUMN, y, kind: NodeKind::View { name: title, style: ViewStyle::default() } });
                    doc.graph.connect(&s, &id, 0).map_err(invalid)?;
                    doc.graph.connect(&id, &v, 0).map_err(invalid)?;
                }
            }
            Ok(Changed::Graph)
        }
        C::AddAnnotation { mut annotation } => {
            check_annotation(doc, &annotation)?;
            if annotation.id.is_empty() {
                annotation.id = doc.new_annotation_id();
            } else if doc.annotations.iter().any(|a| a.id == annotation.id) {
                return Err(invalid(format!("annotation id {:?} is taken", annotation.id)));
            }
            doc.annotations.push(annotation);
            Ok(Changed::Annotations)
        }
        C::AddAnnotations { annotations } => {
            for mut a in annotations {
                check_annotation(doc, &a)?;
                if a.id.is_empty() || doc.annotations.iter().any(|x| x.id == a.id) {
                    a.id = doc.new_annotation_id();
                }
                doc.annotations.push(a);
            }
            Ok(Changed::Annotations)
        }
        C::SetAnnotation { annotation } => {
            check_annotation(doc, &annotation)?;
            let slot = doc.annotations.iter_mut().find(|a| a.id == annotation.id).ok_or_else(|| CommandError::NoAnnotation(annotation.id.clone()))?;
            *slot = annotation;
            Ok(Changed::Annotations)
        }
        C::RemoveAnnotations { ids } => {
            for id in &ids {
                if !doc.annotations.iter().any(|a| &a.id == id) {
                    return Err(CommandError::NoAnnotation(id.clone()));
                }
            }
            doc.annotations.retain(|a| !ids.contains(&a.id));
            Ok(Changed::Annotations)
        }
        C::MoveAnnotations { ids, dx, dy } => {
            if !(dx.is_finite() && dy.is_finite()) {
                return Err(invalid("a move is a number of pixels"));
            }
            for id in &ids {
                let a = doc.annotations.iter_mut().find(|a| &a.id == id).ok_or_else(|| CommandError::NoAnnotation(id.clone()))?;
                a.shape = a.shape.translated(dx, dy);
            }
            Ok(Changed::Annotations)
        }
        C::SetClass { class } => {
            if class.name.trim().is_empty() {
                return Err(invalid("a class needs a name"));
            }
            match doc.classes.iter_mut().find(|c| c.name == class.name) {
                Some(c) => *c = class,
                None => doc.classes.push(class),
            }
            Ok(Changed::Annotations)
        }
        C::SetNote { mut note } => {
            if note.text.trim().is_empty() {
                return Err(invalid("a note needs text"));
            }
            if note.id.is_empty() {
                note.id = doc.new_note_id();
                doc.notes.push(note);
            } else {
                let slot = doc.notes.iter_mut().find(|n| n.id == note.id).ok_or_else(|| invalid(format!("no note {:?}", note.id)))?;
                *slot = note;
            }
            Ok(Changed::Notes)
        }
        C::RemoveNote { id } => {
            let n = doc.notes.len();
            doc.notes.retain(|x| x.id != id);
            if doc.notes.len() == n {
                return Err(invalid(format!("no note {id:?}")));
            }
            Ok(Changed::Notes)
        }
        C::RemoveClass { name, move_to } => {
            if doc.class(&name).is_none() {
                return Err(invalid(format!("no class {name:?}")));
            }
            if name == MEASUREMENT {
                return Err(invalid("the Measurement class holds the rulers; it stays"));
            }
            if doc.classes.iter().filter(|c| c.name != MEASUREMENT).count() <= 1 {
                return Err(invalid("a project keeps at least one class"));
            }
            match &move_to {
                Some(to) if to == &name => return Err(invalid("annotations cannot move to the class being deleted")),
                Some(to) if doc.class(to).is_none() => return Err(invalid(format!("no class {to:?}"))),
                _ => {}
            }
            match &move_to {
                Some(to) => doc.annotations.iter_mut().filter(|a| a.class == name).for_each(|a| a.class = to.clone()),
                None => doc.annotations.retain(|a| a.class != name),
            }
            let sources: Vec<String> = doc.graph.nodes.iter().filter(|n| matches!(&n.kind, NodeKind::Annotations { class: Some(c), .. } if c == &name)).map(|n| n.id.clone()).collect();
            for id in sources {
                match &move_to {
                    Some(to) => {
                        if let Some(NodeKind::Annotations { class, .. }) = doc.graph.nodes.iter_mut().find(|n| n.id == id).map(|n| &mut n.kind) {
                            *class = Some(to.clone());
                        }
                    }
                    None => doc.graph.remove(&id),
                }
            }
            doc.classes.retain(|c| c.name != name);
            Ok(Changed::Annotations)
        }
        C::RenameClass { from, to } => {
            let to = to.trim().to_string();
            if to.is_empty() {
                return Err(invalid("a class needs a name"));
            }
            if from == MEASUREMENT {
                return Err(invalid("the Measurement class holds the rulers; it keeps its name"));
            }
            if to != from && doc.class(&to).is_some() {
                return Err(invalid(format!("there is already a class {to:?}")));
            }
            let c = doc.classes.iter_mut().find(|c| c.name == from).ok_or_else(|| invalid(format!("no class {from:?}")))?;
            c.name = to.clone();
            doc.annotations.iter_mut().filter(|a| a.class == from).for_each(|a| a.class = to.clone());
            for n in doc.graph.nodes.iter_mut() {
                if let NodeKind::Annotations { class: Some(c), .. } = &mut n.kind
                    && *c == from {
                        *c = to.clone();
                    }
            }
            Ok(Changed::Annotations)
        }
        C::SetAnnotationsClass { ids, class } => {
            if doc.class(&class).is_none() {
                return Err(invalid(format!("no class {class:?}")));
            }
            for id in &ids {
                let a = doc.annotations.iter_mut().find(|a| &a.id == id).ok_or_else(|| CommandError::NoAnnotation(id.clone()))?;
                a.class = class.clone();
            }
            Ok(Changed::Annotations)
        }
        C::Batch { commands } => {
            let before = doc.clone();
            let mut changed = Changed::Graph;
            for c in commands {
                match apply(doc, c) {
                    Ok(ch) => changed = ch,
                    Err(e) => {
                        *doc = before;
                        return Err(e);
                    }
                }
            }
            Ok(changed)
        }
        C::Rename { name } => {
            if name.trim().is_empty() {
                return Err(invalid("a project needs a name"));
            }
            doc.name = name;
            Ok(Changed::Project)
        }
    }
}

fn image<'a>(doc: &'a ProjectDoc, id: &str) -> Result<&'a ImageRef, CommandError> {
    doc.image(id).ok_or_else(|| CommandError::NoImage(id.to_string()))
}

fn node<'a>(doc: &'a ProjectDoc, id: &str) -> Result<&'a GraphNode, CommandError> {
    doc.graph.node(id).ok_or_else(|| CommandError::NoNode(id.to_string()))
}

fn check_kind(doc: &ProjectDoc, kind: &NodeKind) -> Result<(), CommandError> {
    match kind {
        NodeKind::Source { image: i, channel } => {
            let n = image(doc, i)?.channels.len();
            if n > 0 && *channel >= n {
                return Err(invalid(format!("{i} has {n} channel(s)")));
            }
            Ok(())
        }
        NodeKind::Annotations { image: i, .. } => image(doc, i).map(|_| ()),
        NodeKind::Op { op, .. } => op.validate().map_err(invalid),
        NodeKind::View { .. } => Ok(()),
        NodeKind::Script { path, output, .. } => {
            if crate::data::script::Language::of(std::path::Path::new(path)).is_none() {
                return Err(invalid(format!("{path}: a script is a .py or .rs file")));
            }
            if *output == lucem_pipeline::PortType::Table {
                return Err(invalid("a script gives an image, a mask or labels"));
            }
            Ok(())
        }
        NodeKind::Model { image: i, channels, model, .. } => {
            if model.trim().is_empty() {
                return Err(invalid("a model node names its model"));
            }
            let n = image(doc, i)?.channels.len();
            if n > 0 && channels.iter().any(|&c| c >= n) {
                return Err(invalid(format!("{i} has {n} channel(s)")));
            }
            Ok(())
        }
    }
}

/// The rulers' class.
pub const MEASUREMENT: &str = "Measurement";

fn check_annotation(doc: &ProjectDoc, a: &Annotation) -> Result<(), CommandError> {
    image(doc, &a.image)?;
    a.shape.validate().map_err(invalid)?;
    if doc.class(&a.class).is_none() {
        return Err(invalid(format!("no class {:?}", a.class)));
    }
    Ok(())
}

/// A new node's id from what it is (`gaussian_blur`, `source`, `view`, …).
fn base_id(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Source { .. } => "source".into(),
        NodeKind::Annotations { .. } => "annotations".into(),
        NodeKind::Op { op, .. } => op.tag().into(),
        NodeKind::View { .. } => "view".into(),
        NodeKind::Script { .. } => "script".into(),
        NodeKind::Model { .. } => "model".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::annotation::Shape;

    fn opened() -> ProjectDoc {
        let mut d = ProjectDoc::default();
        let ch = |l: &str| ChannelDoc { label: l.into(), visible: true, color: [255, 255, 255], min: 0.0, max: 4095.0, gamma: 1.0 };
        apply(&mut d, Command::AddImage { path: "/data/s.ome.zarr".into(), name: "s".into(), channels: vec![ch("DAPI"), ch("RFP")] }).unwrap();
        d
    }

    #[test]
    fn apply_op_builds_a_chain_then_inserts_before_the_view() {
        let mut d = opened();
        apply(&mut d, Command::ApplyOp { op: Op::GaussianBlur { sigma: 2.0 }, view: None, image: "img1".into(), channel: 0 }).unwrap();
        assert_eq!(d.graph.nodes.len(), 3);
        apply(&mut d, Command::ApplyOp { op: Op::OtsuThreshold { bins: 256 }, view: Some("view".into()), image: "img1".into(), channel: 0 }).unwrap();
        assert_eq!(d.graph.input("view", 0), Some("otsu_threshold"));
        assert_eq!(d.graph.input("otsu_threshold", 0), Some("gaussian_blur"));
        // Label after a mask fits; a second Label after labels does not
        apply(&mut d, Command::ApplyOp { op: Op::Label {}, view: Some("view".into()), image: "img1".into(), channel: 0 }).unwrap();
        let before = d.clone();
        assert!(apply(&mut d, Command::ApplyOp { op: Op::Label {}, view: Some("view".into()), image: "img1".into(), channel: 0 }).is_err());
        assert_eq!(d, before, "a failed command leaves the document unchanged");
    }

    #[test]
    fn annotations_are_checked_and_numbered() {
        let mut d = opened();
        let a = Annotation { id: String::new(), image: "img1".into(), shape: Shape::Rect { x: 0.0, y: 0.0, w: 5.0, h: 5.0 }, class: "Tumor".into(), note: String::new(), z: None };
        apply(&mut d, Command::AddAnnotation { annotation: a.clone() }).unwrap();
        assert_eq!(d.annotations[0].id, "a1");
        assert!(apply(&mut d, Command::AddAnnotation { annotation: Annotation { class: "Nope".into(), ..a.clone() } }).is_err());
        apply(&mut d, Command::MoveAnnotations { ids: vec!["a1".into()], dx: 2.0, dy: 0.0 }).unwrap();
        assert_eq!(d.annotations[0].shape, Shape::Rect { x: 2.0, y: 0.0, w: 5.0, h: 5.0 });
        apply(&mut d, Command::RemoveImage { id: "img1".into() }).unwrap();
        assert!(d.annotations.is_empty() && d.active_image.is_none());
    }

    #[test]
    fn a_class_goes_with_or_without_its_annotations() {
        let mut d = opened();
        let ann = |class: &str| Annotation { id: String::new(), image: "img1".into(), class: class.into(), shape: super::super::annotation::Shape::Rect { x: 0.0, y: 0.0, w: 4.0, h: 4.0 }, note: String::new(), z: None };
        let first = d.classes[0].name.clone();
        let second = d.classes[1].name.clone();
        apply(&mut d, Command::AddAnnotations { annotations: vec![ann(&first), ann(&first), ann(&second)] }).unwrap();
        apply(&mut d, Command::RemoveClass { name: first.clone(), move_to: Some(second.clone()) }).unwrap();
        assert!(d.class(&first).is_none() && d.annotations.iter().all(|a| a.class == second) && d.annotations.len() == 3);
        apply(&mut d, Command::RenameClass { from: second.clone(), to: "Nuclei".into() }).unwrap();
        assert!(d.annotations.iter().all(|a| a.class == "Nuclei"));
        apply(&mut d, Command::RemoveClass { name: "Nuclei".into(), move_to: None }).unwrap();
        assert!(d.annotations.is_empty());
        assert!(apply(&mut d, Command::RemoveClass { name: MEASUREMENT.into(), move_to: None }).is_err());
        while d.classes.iter().filter(|c| c.name != MEASUREMENT).count() > 1 {
            let n = d.classes.iter().find(|c| c.name != MEASUREMENT).unwrap().name.clone();
            apply(&mut d, Command::RemoveClass { name: n, move_to: None }).unwrap();
        }
        let last = d.classes.iter().find(|c| c.name != MEASUREMENT).unwrap().name.clone();
        assert!(apply(&mut d, Command::RemoveClass { name: last, move_to: None }).is_err());
    }

    #[test]
    fn commands_are_json() {
        let mut d = opened();
        let cmd: Command = serde_json::from_str(r#"{"command": "apply_op", "op": {"op": "median", "radius": 2}, "image": "img1"}"#).unwrap();
        assert_eq!(apply(&mut d, cmd), Ok(Changed::Graph));
        let cmd: Command = serde_json::from_str(r#"{"command": "set_channel", "image": "img1", "channel": 1, "display": {"label": "RFP", "color": [255, 0, 0], "min": 10, "max": 900}}"#).unwrap();
        assert_eq!(apply(&mut d, cmd), Ok(Changed::Display("img1".into())));
        assert_eq!(d.images[0].channels[1].gamma, 1.0);
    }
}

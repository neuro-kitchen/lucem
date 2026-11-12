//! The project's processing graph: what the Pipeline canvas draws and edits (dsp-kitchen's graph
//! with typed, multi-input ports).
//!
//! Nodes are sources (a channel of an image; the annotations of a class as a mask), ops (a
//! `lucem_pipeline::Op`, whose inputs and output are typed) and views (a result shown over the
//! image in Explore, or as a table in Analyze). An edge carries `from`'s output into input port
//! `port` of `to`; a connection is checked (types, loops) and replaces what that port had.
//! Outputs fan out, so one channel can feed several branches. The graph is the truth: Explore's
//! result layers are computed from it.

use std::collections::{BTreeMap, BTreeSet};

use lucem_pipeline::{Op, Plan, PlanNode, PortType};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Spacing of nodes laid out by the app (canvas units).
pub const COLUMN: f32 = 260.;
pub const ROW: f32 = 230.;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GraphDoc {
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
    #[serde(default)]
    pub edges: Vec<EdgeDoc>,
}

/// A node and where it sits on the canvas (canvas units; y grows downwards).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GraphNode {
    pub id: String,
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
    pub kind: NodeKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NodeKind {
    /// One channel of an image (values scaled to 0–1 by the type's range).
    Source { image: String, channel: usize },
    /// The annotations of `class` on `image` (all regions and strokes when `None`), as a mask.
    Annotations {
        image: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        class: Option<String>,
    },
    /// An operator.
    Op {
        #[serde(default = "yes")]
        enabled: bool,
        op: Op,
    },
    /// A result: drawn over the image in Explore (rasters) or listed in Analyze (tables).
    View {
        name: String,
        #[serde(default)]
        style: ViewStyle,
    },
    /// A script file (Python `.py` or Rust `.rs`) run on its inputs (Blender's Script node): its
    /// input ports, output and numbers come from the file's `lucem:` header
    /// (`crate::data::script`), read again when the file changes.
    Script {
        path: String,
        #[serde(default)]
        inputs: Vec<ScriptPort>,
        #[serde(default = "image_port")]
        output: PortType,
        #[serde(default)]
        params: Vec<ScriptParam>,
        #[serde(default = "yes")]
        enabled: bool,
    },
    /// A model plugin run on an image (a source of its own: the model sees the image, not one
    /// channel): `channels` are the ones it reads as red, green, blue (empty: the image's three
    /// colour channels, else its first). A nucleus model gives every nucleus; Segment Anything
    /// gives one object per box or point annotation of class `prompts`. Gives labels.
    Model {
        image: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        channels: Vec<usize>,
        model: String,
        /// Where it runs: inside these region annotations only (`None`: wherever it is asked).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        roi: Option<Roi>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompts: Option<String>,
        #[serde(default = "yes")]
        enabled: bool,
    },
}

/// A model node's region of interest: the region annotations it runs inside.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum Roi {
    /// Every region annotation of a class.
    Class { class: String },
    /// One annotation.
    Annotation { id: String },
}

impl Roi {
    pub fn matches(&self, a: &crate::domain::annotation::Annotation) -> bool {
        a.shape.is_region()
            && match self {
                Roi::Class { class } => &a.class == class,
                Roi::Annotation { id } => &a.id == id,
            }
    }
}

/// An input of a script: its name (the argument) and what it takes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScriptPort {
    pub name: String,
    pub port: PortType,
}

/// A number a script takes (edited in the Inspector).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScriptParam {
    pub name: String,
    pub value: f64,
}

fn image_port() -> PortType {
    PortType::Image
}

fn yes() -> bool {
    true
}

/// How a view's raster is drawn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ViewStyle {
    #[serde(default = "yes")]
    pub visible: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    /// Colour map of an image result: `grayscale`, `viridis`, `magma`, `inferno`, `cyan`.
    #[serde(default = "default_lut")]
    pub colormap: String,
    /// Colour of a mask result.
    #[serde(default = "default_mask_color")]
    pub color: [u8; 3],
}

fn default_opacity() -> f32 {
    0.65
}
fn default_lut() -> String {
    "magma".into()
}
fn default_mask_color() -> [u8; 3] {
    [0, 220, 255]
}

impl Default for ViewStyle {
    fn default() -> Self {
        Self { visible: true, opacity: default_opacity(), colormap: default_lut(), color: default_mask_color() }
    }
}

/// `from`'s output into input `port` of `to`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeDoc {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub port: usize,
}

impl NodeKind {
    /// Input ports: type (`None`: any raster or table) and name.
    pub fn inputs(&self) -> Vec<(Option<PortType>, String)> {
        match self {
            NodeKind::Source { .. } | NodeKind::Annotations { .. } => Vec::new(),
            NodeKind::Op { op, .. } => op.inputs().into_iter().map(|(t, n)| (Some(t), n.to_string())).collect(),
            NodeKind::View { .. } => vec![(None, "result".to_string())],
            NodeKind::Script { inputs, .. } => inputs.iter().map(|p| (Some(p.port), p.name.clone())).collect(),
            NodeKind::Model { .. } => Vec::new(),
        }
    }

    pub fn output(&self) -> Option<PortType> {
        match self {
            NodeKind::Source { .. } => Some(PortType::Image),
            NodeKind::Annotations { .. } => Some(PortType::Mask),
            NodeKind::Op { op, .. } => Some(op.output()),
            NodeKind::View { .. } => None,
            NodeKind::Script { output, .. } => Some(*output),
            NodeKind::Model { .. } => Some(PortType::Labels),
        }
    }

    pub fn has_input(&self) -> bool {
        !self.inputs().is_empty()
    }

    pub fn has_output(&self) -> bool {
        self.output().is_some()
    }

    /// The image a source node reads.
    pub fn image(&self) -> Option<&str> {
        match self {
            NodeKind::Source { image, .. } | NodeKind::Annotations { image, .. } | NodeKind::Model { image, .. } => Some(image),
            _ => None,
        }
    }
}

impl GraphDoc {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: &str) -> Option<&mut GraphNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.node(id).is_some()
    }

    /// The node feeding port `port` of `id`.
    pub fn input(&self, id: &str, port: usize) -> Option<&str> {
        self.edges.iter().find(|e| e.to == id && e.port == port).map(|e| e.from.as_str())
    }

    /// What feeds each input port of `id`.
    pub fn inputs(&self, id: &str) -> Vec<Option<String>> {
        let n = self.node(id).map_or(0, |n| n.kind.inputs().len());
        (0..n).map(|p| self.input(id, p).map(str::to_string)).collect()
    }

    /// `(node, port)` pairs `id` feeds.
    pub fn outputs(&self, id: &str) -> Vec<(&str, usize)> {
        self.edges.iter().filter(|e| e.from == id).map(|e| (e.to.as_str(), e.port)).collect()
    }

    /// Every node upstream of `id` (through every port).
    pub fn ancestors(&self, id: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut stack = vec![id.to_string()];
        while let Some(n) = stack.pop() {
            for e in self.edges.iter().filter(|e| e.to == n) {
                if out.insert(e.from.clone()) {
                    stack.push(e.from.clone());
                }
            }
        }
        out
    }

    /// Every node downstream of `id`.
    pub fn descendants(&self, id: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut stack = vec![id.to_string()];
        while let Some(n) = stack.pop() {
            for (o, _) in self.outputs(&n) {
                if out.insert(o.to_string()) {
                    stack.push(o.to_string());
                }
            }
        }
        out
    }

    /// The images `id` reads (its source nodes upstream).
    pub fn images_of(&self, id: &str) -> BTreeSet<String> {
        let mut set = self.ancestors(id);
        set.insert(id.to_string());
        set.iter().filter_map(|n| self.node(n).and_then(|n| n.kind.image()).map(str::to_string)).collect()
    }

    /// An id not used yet: `base`, else `base1`, `base2`, …
    pub fn fresh_id(&self, base: &str) -> String {
        if !self.contains(base) {
            return base.to_string();
        }
        (1..).map(|i| format!("{base}{i}")).find(|id| !self.contains(id)).expect("ids are unbounded")
    }

    /// The y of a new row under every node.
    pub fn next_row(&self) -> f32 {
        self.nodes.iter().map(|n| n.y + ROW).fold(0.0, f32::max)
    }

    /// Feeds input `port` of `to` from `from` (replacing what that port had), if the types fit
    /// and it makes no loop.
    pub fn connect(&mut self, from: &str, to: &str, port: usize) -> Result<(), String> {
        let (Some(a), Some(b)) = (self.node(from), self.node(to)) else { return Err(format!("no node {from:?} or {to:?}")) };
        let Some(out) = a.kind.output() else { return Err(format!("{from} has no output (views end a chain)")) };
        let ports = b.kind.inputs();
        let Some((want, name)) = ports.get(port).cloned() else {
            return Err(if ports.is_empty() { format!("{to} takes no input") } else { format!("{to} has no input {}", port + 1) });
        };
        if let Some(want) = want
            && want != out {
                return Err(format!("{to}'s {name} input takes {}, {from} gives {}", want.name(), out.name()));
            }
        if from == to || self.descendants(to).contains(from) {
            return Err("that connection would make a loop".into());
        }
        self.edges.retain(|e| !(e.to == to && e.port == port));
        self.edges.push(EdgeDoc { from: from.to_string(), to: to.to_string(), port });
        Ok(())
    }

    /// The first input port of `to` that takes `from`'s output and is free (else the first that
    /// takes it).
    pub fn port_for(&self, from: &str, to: &str) -> Option<usize> {
        let out = self.node(from)?.kind.output()?;
        let ports = self.node(to)?.kind.inputs();
        let fits: Vec<usize> = ports.iter().enumerate().filter(|(_, (t, _))| t.is_none_or(|t| t == out)).map(|(i, _)| i).collect();
        fits.iter().copied().find(|&p| self.input(to, p).is_none()).or(fits.first().copied())
    }

    /// Empties input `port` of `to`.
    pub fn disconnect(&mut self, to: &str, port: usize) {
        self.edges.retain(|e| !(e.to == to && e.port == port));
    }

    /// Removes a node and its edges.
    pub fn remove(&mut self, id: &str) {
        self.nodes.retain(|n| n.id != id);
        self.edges.retain(|e| e.from != id && e.to != id);
    }

    /// Removes a node; what fed its first input feeds what it fed, where the types allow.
    pub fn remove_bridging(&mut self, id: &str) {
        let input = self.input(id, 0).map(str::to_string);
        let outputs: Vec<(String, usize)> = self.outputs(id).into_iter().map(|(o, p)| (o.to_string(), p)).collect();
        self.remove(id);
        if let Some(i) = input {
            for (o, p) in outputs {
                let _ = self.connect(&i, &o, p);
            }
        }
    }

    /// Puts `id` (first input, output) on the edge into port `port` of `b`.
    pub fn splice(&mut self, b: &str, port: usize, id: &str) -> Result<(), String> {
        let a = self.input(b, port).map(str::to_string);
        if let Some(a) = &a {
            self.connect(a, id, 0)?;
        }
        self.connect(id, b, port)
    }

    /// The plan `lucem_pipeline::evaluate` runs: every node with what feeds its ports.
    pub fn plan(&self) -> Plan {
        let mut plan = BTreeMap::new();
        for n in &self.nodes {
            let (op, enabled) = match &n.kind {
                NodeKind::Op { op, enabled } => (Some(op.clone()), *enabled),
                // A view passes its result through: an identity node
                NodeKind::View { .. } => (None, true),
                _ => (None, true),
            };
            plan.insert(n.id.clone(), PlanNode { op, enabled, inputs: self.inputs(&n.id) });
        }
        plan
    }

    /// The processing of channel `channel` of `image`: the steps after its source, in order
    /// (first output at each step), and the view at the end of the chain (when there is one).
    pub fn channel_chain(&self, image: &str, channel: usize) -> (Vec<String>, Option<String>) {
        let Some(source) = self.nodes.iter().find(|n| matches!(&n.kind, NodeKind::Source { image: i, channel: c } if i == image && *c == channel)) else { return (Vec::new(), None) };
        let is_view = |id: &str| matches!(self.node(id).map(|n| &n.kind), Some(NodeKind::View { .. }));
        let (mut steps, mut view, mut cur) = (Vec::new(), None, source.id.clone());
        for _ in 0..64 {
            let outs: Vec<String> = self.outputs(&cur).into_iter().filter(|(_, port)| *port == 0).map(|(to, _)| to.to_string()).collect();
            if let Some(v) = outs.iter().find(|o| is_view(o)) {
                view = Some(v.clone());
            }
            let Some(next) = outs.into_iter().find(|o| !is_view(o)) else { break };
            steps.push(next.clone());
            cur = next;
        }
        (steps, view)
    }

    /// Channels of `image` (of `channels`) shown processed: their chain has a step on and its
    /// view is shown. Non-destructive: the processing replaces how the channel is drawn; every
    /// step off, Raw, or no step, and the channel draws as it is.
    pub fn processed_channels(&self, image: &str, channels: usize) -> std::collections::HashMap<usize, String> {
        let mut out = std::collections::HashMap::new();
        for c in 0..channels {
            let (steps, view) = self.channel_chain(image, c);
            let Some(view) = view else { continue };
            let shown = matches!(self.node(&view).map(|n| &n.kind), Some(NodeKind::View { style, .. }) if style.visible);
            let any_on = steps.iter().any(|id| match self.node(id).map(|n| &n.kind) {
                Some(NodeKind::Op { enabled, .. }) | Some(NodeKind::Model { enabled, .. }) | Some(NodeKind::Script { enabled, .. }) => *enabled,
                _ => false,
            });
            if shown && any_on {
                out.insert(c, view);
            }
        }
        out
    }

    /// View nodes, in node order.
    pub fn views(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes.iter().filter(|n| matches!(n.kind, NodeKind::View { .. }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, kind: NodeKind) -> GraphNode {
        GraphNode { id: id.into(), x: 0.0, y: 0.0, kind }
    }

    fn op(op: Op) -> NodeKind {
        NodeKind::Op { enabled: true, op }
    }

    /// A channel shows processed only while a step of its chain is on and its result is shown;
    /// a step off (or Raw) and the channel draws as it is.
    #[test]
    fn a_channel_is_processed_only_while_a_step_is_on() {
        let mut g = GraphDoc {
            nodes: vec![
                node("src", NodeKind::Source { image: "img".into(), channel: 1 }),
                node("blur", op(Op::GaussianBlur { sigma: 2.0 })),
                node("view", NodeKind::View { name: "Blur".into(), style: ViewStyle::default() }),
            ],
            edges: vec![EdgeDoc { from: "src".into(), to: "blur".into(), port: 0 }, EdgeDoc { from: "blur".into(), to: "view".into(), port: 0 }],
        };
        assert_eq!(g.channel_chain("img", 1), (vec!["blur".to_string()], Some("view".to_string())));
        assert_eq!(g.processed_channels("img", 2).get(&1).map(String::as_str), Some("view"));
        if let Some(NodeKind::Op { enabled, .. }) = g.node_mut("blur").map(|n| &mut n.kind) {
            *enabled = false;
        }
        assert!(g.processed_channels("img", 2).is_empty(), "every step off: raw");
        if let Some(NodeKind::Op { enabled, .. }) = g.node_mut("blur").map(|n| &mut n.kind) {
            *enabled = true;
        }
        if let Some(NodeKind::View { style, .. }) = g.node_mut("view").map(|n| &mut n.kind) {
            style.visible = false;
        }
        assert!(g.processed_channels("img", 2).is_empty(), "Raw chosen: raw");
    }

    /// dapi → blur → otsu → label → measure(labels, dapi) → table
    fn segmentation() -> GraphDoc {
        let mut g = GraphDoc {
            nodes: vec![
                node("dapi", NodeKind::Source { image: "img1".into(), channel: 0 }),
                node("blur", op(Op::GaussianBlur { sigma: 2.0 })),
                node("otsu", op(Op::OtsuThreshold { bins: 256 })),
                node("label", op(Op::Label {})),
                node("measure", op(Op::Measure {})),
                node("table", NodeKind::View { name: "Nuclei".into(), style: ViewStyle::default() }),
            ],
            ..GraphDoc::default()
        };
        for (a, b, p) in [("dapi", "blur", 0), ("blur", "otsu", 0), ("otsu", "label", 0), ("label", "measure", 0), ("dapi", "measure", 1), ("measure", "table", 0)] {
            g.connect(a, b, p).unwrap();
        }
        g
    }

    #[test]
    fn typed_ports_are_checked() {
        let mut g = segmentation();
        assert!(g.connect("dapi", "label", 0).is_err(), "label takes a mask, not an image");
        assert!(g.connect("label", "measure", 1).is_err(), "measure's 2nd input is an image");
        assert!(g.connect("table", "blur", 0).is_err(), "a view has no output");
        assert!(g.connect("measure", "blur", 0).is_err(), "a table is not an image");
        assert!(g.connect("otsu", "blur", 0).is_err(), "a mask is not an image");
        assert_eq!(g.port_for("dapi", "measure"), Some(1));
        assert_eq!(g.images_of("table").into_iter().collect::<Vec<_>>(), ["img1"]);
    }

    #[test]
    fn removing_bridges_where_types_fit() {
        let mut g = segmentation();
        g.remove_bridging("blur");
        assert_eq!(g.input("otsu", 0), Some("dapi"));
        // Removing the threshold would feed an image into label: that edge is not made
        g.remove_bridging("otsu");
        assert_eq!(g.input("label", 0), None);
    }

    #[test]
    fn splice_and_plan() {
        let mut g = segmentation();
        g.nodes.push(node("open", op(Op::Open { radius: 2 })));
        g.splice("label", 0, "open").unwrap();
        assert_eq!(g.input("open", 0), Some("otsu"));
        assert_eq!(g.input("label", 0), Some("open"));
        let plan = g.plan();
        assert_eq!(plan["measure"].inputs, vec![Some("label".to_string()), Some("dapi".to_string())]);
        assert_eq!(lucem_pipeline::halo(&plan, "table"), 8 + 4);
    }
}

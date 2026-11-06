//! What can be put on the Pipeline canvas (the library, the menus, the search), and how a node
//! reads on it: its title, its second line, its icon and colour.

use gpui_kit::assets::IconName;
use gpui_kit::Hsla;
use lumen_pipeline::{catalog as op_catalog, Op, PortType};

use crate::domain::graph::NodeKind;
use crate::domain::project::ProjectDoc;

/// What an item adds.
#[derive(Debug, Clone, PartialEq)]
pub enum Template {
    Op(Op),
    /// A channel of an image.
    Source { image: String, channel: usize },
    /// An image's annotations as a mask.
    Annotations { image: String },
    View,
}

impl Template {
    pub fn kind(&self) -> NodeKind {
        match self {
            Template::Op(op) => NodeKind::Op { enabled: true, op: op.clone() },
            Template::Source { image, channel } => NodeKind::Source { image: image.clone(), channel: *channel },
            Template::Annotations { image } => NodeKind::Annotations { image: image.clone(), class: None },
            Template::View => NodeKind::View { name: "Result".into(), style: Default::default() },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Item {
    pub label: String,
    pub hint: String,
    pub template: Template,
}

#[derive(Debug, Clone)]
pub struct Category {
    pub name: &'static str,
    pub icon: IconName,
    pub items: Vec<Item>,
}

/// The families of operators, in menu order.
pub const FAMILIES: [&str; 7] = ["Filter", "Intensity", "Restore", "Threshold", "Mask", "Segment", "Measure"];

pub fn family_icon(f: &str) -> IconName {
    match f {
        "Filter" => IconName::SlidersHorizontal,
        "Intensity" => IconName::Sun,
        "Restore" => IconName::Sparkles,
        "Threshold" => IconName::Contrast,
        "Mask" => IconName::Shapes,
        "Segment" => IconName::Grid2x2,
        _ => IconName::Table2,
    }
}

/// `image → mask` and the like, for hints.
pub fn signature(op: &Op) -> String {
    let ins: Vec<&str> = op.inputs().iter().map(|(t, _)| t.name()).collect();
    format!("{} → {}", ins.join(" + "), op.output().name())
}

/// Every operator family with its ops.
pub fn operations() -> Vec<Category> {
    let ops = op_catalog();
    FAMILIES
        .iter()
        .map(|f| Category {
            name: f,
            icon: family_icon(f),
            items: ops.iter().filter(|o| o.family() == *f).map(|o| Item { label: o.title().into(), hint: signature(o), template: Template::Op(o.clone()) }).collect(),
        })
        .collect()
}

/// The sources the project offers: each channel of each image, and each image's annotations.
pub fn sources(doc: &ProjectDoc) -> Category {
    let mut items = Vec::new();
    for img in &doc.images {
        for (c, ch) in img.channels.iter().enumerate() {
            items.push(Item { label: format!("{} · {}", img.name, ch.label), hint: "image".into(), template: Template::Source { image: img.id.clone(), channel: c } });
        }
        items.push(Item { label: format!("{} · annotations", img.name), hint: "mask".into(), template: Template::Annotations { image: img.id.clone() } });
    }
    Category { name: "Sources", icon: IconName::Image, items }
}

/// A node's title, second line and icon on the canvas.
pub fn describe(kind: &NodeKind, doc: &ProjectDoc) -> (String, String, IconName) {
    match kind {
        NodeKind::Source { image, channel } => {
            let img = doc.image(image);
            let ch = img.and_then(|i| i.channels.get(*channel)).map_or_else(|| format!("channel {channel}"), |c| c.label.clone());
            (ch, format!("{} · image", img.map_or(image.as_str(), |i| i.name.as_str())), IconName::Image)
        }
        NodeKind::Annotations { image, class } => {
            let img = doc.image(image).map_or(image.as_str(), |i| i.name.as_str()).to_string();
            (class.clone().unwrap_or_else(|| "Annotations".into()), format!("{img} · mask"), IconName::Shapes)
        }
        NodeKind::Op { op, .. } => (op.title().into(), summary(op), family_icon(op.family())),
        NodeKind::View { name, .. } => (name.clone(), "result · shown in Explore".into(), IconName::Eye),
    }
}

/// The op's values in one line (`sigma 2 · amount 1`).
pub fn summary(op: &Op) -> String {
    let v = serde_json::to_value(op).unwrap_or_default();
    let mut parts = Vec::new();
    if let Some(m) = v.as_object() {
        for (k, x) in m {
            if k != "op" {
                parts.push(format!("{} {}", k.replace('_', " "), crate::ui::param_editor::trim_number(x.as_f64().unwrap_or(0.0))));
            }
        }
    }
    if parts.is_empty() { signature(op) } else { parts.join(" · ") }
}

/// The colour of a port type (handles, edges).
pub fn port_color(t: Option<PortType>) -> Hsla {
    match t {
        Some(PortType::Image) => gpui_kit::hsla(210. / 360., 0.70, 0.62, 1.0),
        Some(PortType::Mask) => gpui_kit::hsla(186. / 360., 0.80, 0.50, 1.0),
        Some(PortType::Labels) => gpui_kit::hsla(36. / 360., 0.90, 0.58, 1.0),
        Some(PortType::Table) => gpui_kit::hsla(130. / 360., 0.50, 0.55, 1.0),
        None => gpui_kit::hsla(0., 0., 0.75, 1.0),
    }
}

/// The colour of a node kind (its strip and minimap mark).
pub fn kind_color(kind: &NodeKind) -> Hsla {
    match kind {
        NodeKind::Source { .. } | NodeKind::Annotations { .. } => gpui_kit::hsla(152. / 360., 0.55, 0.45, 1.0),
        NodeKind::Op { op, .. } => port_color(Some(op.output())),
        NodeKind::View { .. } => gpui_kit::hsla(271. / 360., 0.60, 0.65, 1.0),
    }
}

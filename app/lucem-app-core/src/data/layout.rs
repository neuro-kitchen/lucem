//! The views of each workspace, saved with the project (`layout.json`): which views Explore
//! shows, Analyze's plots and panels. View state: not in the undo history, not in the Changes
//! diff. Each workspace writes its part when its views change and restores it when a project
//! opens.

use std::path::Path;

use serde::{Deserialize, Serialize};

pub const LAYOUT_FILE: &str = "layout.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Layout {
    pub lucem_layout: u32,
    pub explore: ExploreLayout,
    pub analyze: AnalyzeLayout,
}

impl Default for Layout {
    fn default() -> Self {
        Self { lucem_layout: 1, explore: ExploreLayout::default(), analyze: AnalyzeLayout::default() }
    }
}

/// Explore's views, in order (their kinds: "Plane", "2D", "3D", "Channels").
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExploreLayout {
    pub views: Vec<String>,
}

/// Analyze: the table shown, its plots, the side panels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalyzeLayout {
    pub table: bool,
    pub plots: Vec<PlotLayout>,
    pub hierarchy: bool,
    pub side: bool,
}

impl Default for AnalyzeLayout {
    fn default() -> Self {
        Self { table: true, plots: Vec::new(), hierarchy: true, side: true }
    }
}

/// A plot view: its kind ("Histogram", "Scatter", "Box plot", "Counts") and columns.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PlotLayout {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
}

impl Layout {
    /// The layout saved in a project folder (the default when there is none or it does not read).
    pub fn load(folder: &Path) -> Self {
        std::fs::read_to_string(folder.join(LAYOUT_FILE)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self, folder: &Path) -> std::io::Result<()> {
        let text = serde_json::to_string_pretty(self).expect("a layout always serializes");
        crate::domain::project::write_atomic(&folder.join(LAYOUT_FILE), &text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layout_reads_back_and_a_missing_one_is_the_default() {
        let dir = std::env::temp_dir().join(format!("lucem-layout-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(Layout::load(&dir), Layout::default());
        let mut l = Layout::default();
        l.explore.views = vec!["Plane".into(), "3D".into()];
        l.analyze.plots.push(PlotLayout { kind: "Scatter".into(), x: Some("area".into()), y: Some("perimeter".into()), group: None });
        l.analyze.side = false;
        l.save(&dir).unwrap();
        assert_eq!(Layout::load(&dir), l);
        std::fs::remove_dir_all(&dir).ok();
    }
}

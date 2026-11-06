//! The workspaces of the title bar (Blender style): each is a stage of the work on the project.

use gpui_kit::assets::IconName;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Workspace {
    /// The image: view, annotate, measure, apply operations.
    #[default]
    Explore,
    /// The processing graph on a canvas.
    Pipeline,
    /// Tables of measurements.
    Analyze,
}

impl Workspace {
    pub const ALL: [Workspace; 3] = [Workspace::Explore, Workspace::Pipeline, Workspace::Analyze];

    pub fn title(self) -> &'static str {
        match self {
            Workspace::Explore => "Explore",
            Workspace::Pipeline => "Pipeline",
            Workspace::Analyze => "Analyze",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Workspace::Explore => "The image: view, annotate, measure, apply operations (1)",
            Workspace::Pipeline => "The processing graph: sources, operations and results on a canvas (2)",
            Workspace::Analyze => "Measurements of annotations and objects as tables (3)",
        }
    }

    pub fn icon(self) -> IconName {
        match self {
            Workspace::Explore => IconName::Microscope,
            Workspace::Pipeline => IconName::Workflow,
            Workspace::Analyze => IconName::Table2,
        }
    }
}

//! Blender-style top-bar workspace modes (`Craft`, `Pipeline`, `Analytics`).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkspaceMode {
    /// Main ArtCraft × VectorCraft × Blender interactive canvas with Layers & Modifier stack.
    CraftStudio,
    /// Node-graph / operator pipeline focus with expanded modifier & node inspector.
    NodePipeline,
    /// Scientific Polars & Arrow analytics workspace with full RegionProps table & histograms.
    AnalyticsLab,
}

impl WorkspaceMode {
    pub const ALL: &'static [Self] = &[
        Self::CraftStudio,
        Self::NodePipeline,
        Self::AnalyticsLab,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::CraftStudio => "Craft & Explore",
            Self::NodePipeline => "GPU Pipeline",
            Self::AnalyticsLab => "Polars Analytics",
        }
    }
}

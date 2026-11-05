//! Blender-style top-bar workspace modes (`Compositing`, `Vector & Paint`, `RegionProps`).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkspaceMode {
    /// Main interactive compositing & OME-Zarr multiscale studio.
    CraftStudio,
    /// Vector paths, ROI annotation, and brush painting workspace.
    NodePipeline,
    /// Scientific Polars & Arrow analytics workspace with RegionProps table & histograms.
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
            Self::CraftStudio => "Compositing",
            Self::NodePipeline => "Vector & Paint",
            Self::AnalyticsLab => "RegionProps",
        }
    }
}

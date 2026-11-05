//! Navigation routes and Left Option Rail tabs.

/// Active tab in the persistent Left Option Rail (shared between Startup and Studio).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeftRailTab {
    /// Center VectorCraft-style Project Hub modal over the blank/active canvas.
    ProjectHub,
    /// New Blank Canvas configurator inside the Project Hub.
    NewCanvas,
    /// Built-in synthetic scientific & creative phantoms.
    SamplePhantoms,
    /// Active Studio canvas & Explore viewport.
    StudioCanvas,
    /// Polars & Arrow Analytics / RegionProps table drawer.
    AnalyticsDrawer,
}

impl LeftRailTab {
    pub const ALL: &'static [Self] = &[
        Self::ProjectHub,
        Self::NewCanvas,
        Self::SamplePhantoms,
        Self::StudioCanvas,
        Self::AnalyticsDrawer,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ProjectHub => "Project Hub",
            Self::NewCanvas => "Blank Canvas",
            Self::SamplePhantoms => "Sample Lab",
            Self::StudioCanvas => "Studio Canvas",
            Self::AnalyticsDrawer => "Polars Table",
        }
    }
}

/// Top-level route state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppRoute {
    /// Startup overlay displayed in the center of the blank/live canvas (VectorCraft style).
    StartupHub,
    /// Full interactive ArtCraft × VectorCraft × Blender studio canvas.
    StudioWorkspace,
}

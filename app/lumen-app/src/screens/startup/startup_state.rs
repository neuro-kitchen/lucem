//! Immutable UI state for the Startup screen (`StartupUiState`).

use std::path::PathBuf;

use crate::domain::{BlankCanvasSpec, ProjectPreset};
use crate::navigation::LeftRailTab;

/// UI state rendered by [`super::StartupScreen`].
#[derive(Debug, Clone)]
pub struct StartupUiState {
    pub active_rail_tab: LeftRailTab,
    pub blank_spec: BlankCanvasSpec,
    pub presets: Vec<ProjectPreset>,
    pub recent_files: Vec<PathBuf>,
    pub compute_backend: String,
    pub status_banner: Option<String>,
}

//! ViewModel for the Startup screen (`StartupViewModel`).

use crate::di::AppContainer;
use crate::domain::BlankCanvasSpec;
use crate::navigation::LeftRailTab;

use super::startup_state::StartupUiState;

pub struct StartupViewModel {
    state: StartupUiState,
}

impl StartupViewModel {
    pub fn new(container: &AppContainer) -> Self {
        let state = StartupUiState {
            active_rail_tab: LeftRailTab::ProjectHub,
            blank_spec: BlankCanvasSpec::default(),
            presets: container.project_repo.presets().to_vec(),
            recent_files: container.project_repo.recent_files(),
            compute_backend: container.device_service.backend_label().to_string(),
            status_banner: None,
        };
        Self { state }
    }

    pub fn state(&self) -> &StartupUiState {
        &self.state
    }

    pub fn set_rail_tab(&mut self, tab: LeftRailTab) {
        self.state.active_rail_tab = tab;
    }

    pub fn set_blank_size(&mut self, width: usize, height: usize) {
        self.state.blank_spec.width = width;
        self.state.blank_spec.height = height;
    }

    pub fn set_blank_background(&mut self, intensity: f32) {
        self.state.blank_spec.background_intensity = intensity.clamp(0.0, 1.0);
    }

    pub fn set_status(&mut self, msg: Option<String>) {
        self.state.status_banner = msg;
    }

    pub fn refresh_recents(&mut self, container: &AppContainer) {
        self.state.recent_files = container.project_repo.recent_files();
    }
}

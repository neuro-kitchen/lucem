//! Repository for curated presets, synthetic phantoms, and recent files.

use std::path::PathBuf;
use std::sync::Mutex;

use crate::domain::ProjectPreset;

pub struct ProjectRepository {
    presets: Vec<ProjectPreset>,
    recent_files: Mutex<Vec<PathBuf>>,
}

impl ProjectRepository {
    pub fn new() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let initial_files = lumen_io::discover_workspace_data_paths(&cwd);
        Self {
            presets: ProjectPreset::curated_list(),
            recent_files: Mutex::new(initial_files),
        }
    }

    pub fn presets(&self) -> &[ProjectPreset] {
        &self.presets
    }

    pub fn record_recent(&self, path: PathBuf) {
        if let Ok(mut list) = self.recent_files.lock() {
            list.retain(|p| p != &path);
            list.insert(0, path);
            list.truncate(10);
        }
    }

    pub fn recent_files(&self) -> Vec<PathBuf> {
        self.recent_files
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}

impl Default for ProjectRepository {
    fn default() -> Self {
        Self::new()
    }
}

//! Dependency Injection container (`AppContainer`) holding the active CubeCL compute runtime and
//! application repositories.

use std::sync::Arc;

use lumen_core::cubecl::prelude::Client;
use lumen_core::ComputeTarget;

use crate::data::device_service::DeviceService;
use crate::data::project_repository::ProjectRepository;

/// Shared application container injected into KMP-style ViewModels.
#[derive(Clone)]
pub struct AppContainer {
    pub device_service: Arc<DeviceService>,
    pub project_repo: Arc<ProjectRepository>,
}

impl AppContainer {
    pub fn new() -> Self {
        let device_service = Arc::new(DeviceService::discover());
        let project_repo = Arc::new(ProjectRepository::new());
        Self {
            device_service,
            project_repo,
        }
    }

    pub fn client(&self) -> &Client {
        self.device_service.client()
    }

    pub fn target(&self) -> ComputeTarget {
        self.device_service.target()
    }
}

impl Default for AppContainer {
    fn default() -> Self {
        Self::new()
    }
}

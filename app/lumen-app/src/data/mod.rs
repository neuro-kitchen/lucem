//! Data repositories and CubeCL compute device discovery services.

pub mod device_service;
pub mod project_repository;

pub use device_service::DeviceService;
pub use project_repository::ProjectRepository;

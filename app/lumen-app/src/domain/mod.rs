//! Domain models for `lumen-app` (`ProjectTemplate`, `Layer`, `ActiveTool`, `WorkspaceMode`).

pub mod layer;
pub mod project;
pub mod tool;
pub mod workspace_mode;

pub use layer::{LayerKind, StudioLayer};
pub use project::{BlankCanvasSpec, ProjectPreset};
pub use tool::ActiveTool;
pub use workspace_mode::WorkspaceMode;

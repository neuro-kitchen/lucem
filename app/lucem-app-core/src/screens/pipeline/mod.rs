//! Pipeline: the project's graph on a canvas (sources, typed operators, result views), the
//! library and the inspector.

pub mod canvas;
pub mod catalog;
pub mod pipeline_screen;

pub use pipeline_screen::PipelineView;

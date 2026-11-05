//! Pipeline engine of `lumen`: Blender-style non-destructive GPU modifier stack and live Polars/Arrow
//! analytics extraction on the active CubeCL device.

pub mod executor;
pub mod modifier;

pub use executor::{PipelineExecutionResult, PipelineExecutor};
pub use modifier::{ModifierKind, ModifierNode};

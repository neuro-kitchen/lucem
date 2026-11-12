//! Pipeline engine of `lucem`: operators as typed graph nodes ([`ops`]), a graph evaluator over
//! image regions ([`eval`]), your own functions over images of any size ([`custom`]), pipelines as
//! data for headless runs ([`spec`]), and the older single-input modifier stack with live analytics
//! ([`modifier`], [`executor`]).

pub mod custom;
pub mod eval;
pub mod executor;
pub mod modifier;
pub mod ops;
pub mod spec;
pub mod stitch;
pub mod table;

pub use eval::{at_downsample, evaluate, evaluate_all, halo, Plan, PlanNode};
pub use executor::{PipelineExecutionResult, PipelineExecutor};
pub use modifier::{ModifierKind, ModifierNode};
pub use custom::{measure, run_spec, segment, transform, BlockContext, RunError, RunOptions, RunSummary, Walk};
pub use ops::{catalog, Op, PortType, Value};
pub use spec::{PipelineSpec, SpecNode};
pub use stitch::{LabelStitcher, Stitched};
pub use table::Table;

//! Pipeline engine of `lumen`: operators as typed graph nodes ([`ops`]), a graph evaluator over
//! image regions ([`eval`]), and the older single-input modifier stack with live analytics
//! ([`modifier`], [`executor`]).

pub mod eval;
pub mod executor;
pub mod modifier;
pub mod ops;

pub use eval::{evaluate, halo, Plan, PlanNode};
pub use executor::{PipelineExecutionResult, PipelineExecutor};
pub use modifier::{ModifierKind, ModifierNode};
pub use ops::{catalog, Op, PortType, Value};

//! lucem: one dependency that re-exports the lucem crates (core, io, ops, ml).

pub use lucem_core as core;
pub use lucem_io as io;
pub use lucem_ml as ml;
pub use lucem_ops as ops;

pub use lucem_core::memory::{DeviceArray, DeviceCache, Scratch, Workspace};
pub use lucem_core::{AxisSplit, ComputeError, ComputeTarget, CoreError, CoreResult, Shape};
pub use lucem_ops::{OpsError, OpsResult};

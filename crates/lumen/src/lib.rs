//! lumen: one dependency that re-exports the lumen crates (core, io, ops, ml).

pub use lumen_core as core;
pub use lumen_io as io;
pub use lumen_ml as ml;
pub use lumen_ops as ops;

pub use lumen_core::memory::{DeviceArray, DeviceCache, Scratch, Workspace};
pub use lumen_core::{AxisSplit, ComputeError, ComputeTarget, CoreError, CoreResult, Shape};
pub use lumen_ops::{OpsError, OpsResult};

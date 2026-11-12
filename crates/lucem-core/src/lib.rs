//! Core of lucem: the one place every crate reaches a GPU (or the CPU runtime) through.
//!
//! - [`device`]: [`ComputeTarget`], the name of a CubeCL runtime across the workspace (CLI flag,
//!   Python argument, app setting, `LUCEM_RUNTIME`). Needs no GPU libraries.
//! - [`shape`]: [`Shape`] of C-contiguous arrays and [`AxisSplit`], the `[outer, len, inner]` view of
//!   one axis that lets a 1-D kernel work along any axis.
//! - [`compute`] (`compute` feature): clients, persisted kernel and autotune caches, device shutdown
//!   at exit, launch geometry, autotune helpers, timing.
//! - [`memory`] (`compute` feature): [`DeviceArray`](memory::DeviceArray), typed buffers,
//!   [`Workspace`](memory::Workspace) and [`DeviceCache`](memory::DeviceCache) for reusing device
//!   memory, and the memory budget.
//! - [`element`] (`compute` feature): the element types kernels are generic over.
//!
//! Ported from dsp-kitchen's `dspk-core` (compute layer) and `dspk-ops::core` (buffers); the reasons
//! behind each rule are in `.knowledge/gpu/`.

pub mod device;
pub mod error;
pub mod shape;
pub mod tiling;

#[cfg(feature = "compute")]
pub mod compute;
#[cfg(feature = "compute")]
pub mod element;
#[cfg(feature = "compute")]
pub mod memory;
#[cfg(feature = "compute")]
#[doc(hidden)]
pub mod test_support;

/// The CubeCL every lucem crate builds on (one version across the workspace; used by
/// [`runtime_test!`]).
#[cfg(feature = "compute")]
pub use cubecl;

pub use device::{ComputeError, ComputeTarget};
pub use error::{CoreError, CoreResult};
pub use shape::{AxisSplit, Shape};
pub use tiling::{Block, Zyx};

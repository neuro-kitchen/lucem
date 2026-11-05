//! The compute layer (`compute` feature): the single entry point through which every crate reaches a
//! CubeCL runtime.
//!
//! - [`ComputeTask`] / [`ComputeTarget::run`](crate::device::ComputeTarget): run generic work on the
//!   selected runtime; [`ComputeTarget::client`](crate::device::ComputeTarget) for a client.
//! - [`configure_runtime`]: compiled kernels and autotune decisions persist across processes.
//! - [`shutdown_devices`]: devices are shut down at exit (GPU driver threads otherwise race the
//!   process teardown).
//! - [`LaunchGeometry`]: cube sizes and counts from the runtime's properties, with the position
//!   helpers kernels index with.
//! - [`tune`]: CubeCL autotuning for settings with no device-independent best value; [`pinning`]
//!   pins result-changing choices for reproducible runs.
//! - [`mod@bench`]: device-synchronised timing.
//!
//! Algorithm crates (`lumen-ops`, `lumen-ml`, …) take a `Client` and use these; they never inspect or
//! special-case the device.

pub mod bench;
pub mod launch;
pub mod limits;
pub mod pinning;
mod special;
mod target;
pub mod tune;

pub use crate::device::{ComputeError, ComputeTarget, RUNTIME_ENV, TEST_RUNTIMES_ENV};
pub use launch::{cube_row, element_position, line_position, row_share, tile_point, unit_col, unit_row, LaunchGeometry, RowShare};
pub use limits::{device_elements, MAX_DEVICE_ELEMENTS};
pub use pinning::{pin_tuned_choices, tuned_choices_pinned, PinnedChoices};
pub use special::{negative_infinity, positive_infinity, NEGATIVE_INFINITY_BITS, POSITIVE_INFINITY_BITS};
pub use target::{configure_runtime, open_device, shutdown_devices, ComputeTask, CACHE_DIR_ENV, KERNEL_CACHE_ENV};

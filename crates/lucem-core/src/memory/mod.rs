//! Device memory: arrays that stay on the device, and reuse of what was allocated.
//!
//! On a discrete GPU a transfer costs far more than most kernels (dsp-kitchen: ~20 ms to move a
//! 1080p frame each way vs 0.2–0.6 ms of kernel), so the rules here are what makes lucem fast:
//!
//! - [`DeviceArray`]: every operation takes and returns these; transfers happen only where the caller
//!   asks ([`DeviceArray::from_host`], [`DeviceArray::to_host`]).
//! - [`buffer`]: typed allocation, owned uploads (`client.create`, 2–3× faster than
//!   `create_from_slice`), writes into existing buffers, partial downloads.
//! - [`Scratch`] and [`Workspace`]: buffers that only grow and are reused across passes and calls.
//! - [`DeviceCache`]: constant data (filter taps, footprints, look-up tables) uploaded once per
//!   device.
//! - [`budget`]: large allocations are checked against a memory limit (CubeCL 0.11 can run out of
//!   memory silently and return wrong results).

mod array;
pub mod budget;
pub mod buffer;
mod cache;
mod kernels;
mod scratch;
mod workspace;

pub use array::DeviceArray;
pub use budget::{check_allocation, device_memory_limit, memory_usage, set_device_memory_limit, DEVICE_MEMORY_LIMIT_ENV};
pub use cache::DeviceCache;
pub use scratch::Scratch;
pub use workspace::Workspace;

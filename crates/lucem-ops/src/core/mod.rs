//! Shared device primitives: the single home for every building block reused across `lucem-ops`.
//!
//! - [`pad`]: [`PadMode`] and device boundary indexing ([`kernels::pad_index`], [`kernels::read_padded`]).
//! - [`check`]: shape, aliasing, axis, origin, and weight validation.
//! - [`separable`]: multi-axis 1-D pass runner over [`Workspace`](lucem_core::memory::Workspace) ping-pong buffers.
//! - [`structuring`]: [`StructuringElement`] (boolean footprints, separable rectangle detection, device-cached offsets).
//! - [`elementwise`]: copy, affine scale/offset, clip, rescale, cast, LUT, and threshold comparison.
//! - [`reduce`]: global and axis reductions (`sum`, `min`, `max`, Welford `mean`/`var`, `argmin`, `argmax`).
//! - [`select`]: exact $k$-th element and percentile selection via IEEE-754 radix histograms.
//! - [`histogram`]: global, ranged, and integer histograms.
//! - [`scan`]: axis prefix sums and N-D integral images.
//! - [`compact`]: stream compaction of non-zero indices and coordinates in raster order.
//! - [`components`]: connected-component labeling, sequential relabeling, and deterministic per-region reductions.
//! - [`distance`]: exact separable Euclidean distance transform (EDT), feature transform, and chamfer distances.
//! - [`spline`]: B-spline prefiltering and N-D coordinate interpolation (orders 0–5).
//! - [`fft`]: N-D forward and inverse Fourier transforms on split complex buffers.
//! - [`eigen`]: per-pixel symmetric $2 \times 2$ and $3 \times 3$ eigenvalues.
//! - [`random`]: counter-based uniform and Gaussian RNG on the device.

pub mod check;
pub mod compact;
pub mod components;
pub mod distance;
pub mod eigen;
pub mod elementwise;
pub mod fft;
pub mod histogram;
pub mod kernels;
pub mod pad;
pub mod random;
pub mod reduce;
pub mod scan;
pub mod select;
pub mod separable;
pub mod spline;
pub mod structuring;

pub use pad::{PadMode, PAD_CONSTANT, PAD_MIRROR, PAD_NEAREST, PAD_REFLECT, PAD_WRAP};
pub use structuring::StructuringElement;

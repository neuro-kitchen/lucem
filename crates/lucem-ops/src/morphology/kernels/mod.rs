//! `#[cube]` kernels for `lucem_ops::morphology`.

pub(crate) mod binary;
pub(crate) mod geodesic;
pub(crate) mod grayscale;
pub(crate) mod skeleton;

pub use binary::{binary_1d_kernel, binary_2d_footprint_kernel, hit_or_miss_combine_kernel};
pub use geodesic::geodesic_step_2d_kernel;
pub use grayscale::grey_weighted_2d_kernel;
pub use skeleton::zhang_suen_subpass_kernel;

//! `#[cube]` kernels for `lucem_ops::metric`.

pub(crate) mod fidelity;
pub(crate) mod segmentation;

pub use fidelity::{mul_kernel, sq_diff_kernel, ssim_map_kernel};
pub use segmentation::{invert_binary_mask_kernel, mask_at_edt_kernel};

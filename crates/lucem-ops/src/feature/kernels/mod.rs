//! `#[cube]` kernels for `lucem_ops::feature`.

pub(crate) mod blobs;
pub(crate) mod canny;
pub(crate) mod corners;
pub(crate) mod peaks;
pub(crate) mod structure;
pub(crate) mod texture;

pub use blobs::doh_scale_kernel;
pub use canny::{canny_grad_dir_kernel, canny_nms_kernel};
pub use corners::{
    fast9_score_kernel, foerstner_2d_kernel, harris_2d_kernel, kitchen_rosenfeld_2d_kernel,
    moravec_2d_kernel, shi_tomasi_2d_kernel,
};
pub use peaks::peak_candidates_2d_kernel;
pub use structure::{hessian_det_2d_kernel, outer_product_2d_kernel, shape_index_2d_kernel};
pub use texture::{lbp_2d_kernel, ncc_template_2d_kernel};

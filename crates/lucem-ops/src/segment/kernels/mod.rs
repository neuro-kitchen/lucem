//! `#[cube]` kernels for `lucem_ops::segment`.

pub(crate) mod active_contour;
pub(crate) mod labels;
pub(crate) mod superpixel;
pub(crate) mod watershed;

pub use active_contour::{
    chan_vese_step_2d_kernel, checkerboard_level_set_2d_kernel, disk_level_set_2d_kernel,
    morph_contour_force_2d_kernel, morph_smooth_curv_2d_kernel,
};
pub use labels::{
    expand_labels_gather_kernel, find_boundaries_2d_kernel, join_pair_key_kernel,
    mark_boundaries_gray_kernel, mark_boundaries_rgb_kernel,
};
pub use superpixel::slic_assign_2d_kernel;
pub use watershed::watershed_steepest_init_2d_kernel;

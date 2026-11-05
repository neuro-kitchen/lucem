//! Shared `#[cube]` kernels for `lumen_ops::core`.

pub(crate) mod compact;
pub(crate) mod components;
pub(crate) mod distance;
pub(crate) mod eigen;
pub(crate) mod elementwise;
pub(crate) mod fft;
pub(crate) mod histogram;
pub(crate) mod pad;
pub(crate) mod random;
pub(crate) mod reduce;
pub(crate) mod scan;
pub(crate) mod select;
pub(crate) mod spline;

pub use compact::{flag_nonzero_kernel, scatter_indices_kernel};
pub use components::{cc_init_labels_kernel, cc_propagate_2d_kernel};
pub use distance::{edt_axis_1d_kernel, edt_init_kernel, sqrt_kernel};
pub use eigen::{sym_eigvals_2x2_kernel, sym_eigvals_3x3_kernel};
pub use elementwise::{
    affine_kernel, cast_kernel, clip_kernel, copy_kernel, float_to_uint_kernel, hypot_kernel,
    linear_comb_kernel, lut_kernel, rescale_clip_kernel, threshold_device_kernel,
    threshold_scalar_kernel, uint_to_float_kernel,
};
pub use fft::{complex_mul_kernel, dft_axis_kernel};
pub use histogram::{histogram_float_kernel, histogram_uint_kernel};
pub use pad::{pad_index, read_padded};
pub use random::{pcg_hash, random_normal_kernel, random_uniform_kernel};
pub use reduce::{
    reduce_axis_kernel, row_argmin_argmax_kernel, row_mean_var_kernel, row_sum_min_max_kernel,
};
pub use scan::{scan_axis_kernel, scan_u32_add_offsets_kernel, scan_u32_blocks_kernel};
pub use select::{key_to_f32_kernel, radix_pass_hist_kernel};
pub use spline::{cubic_bspline_weight, sample_2d_kernel, spline_pole_1d_kernel};

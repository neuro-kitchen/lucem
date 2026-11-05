//! Image segmentation: watershed, label post-processing, superpixels, random walker, and active contours.
//!
//! - [`watershed`]: marker-controlled flooding [`watershed`] and Image Foresting Transform [`watershed_ift`].
//! - [`labels`]: [`expand_labels`], [`clear_border`], [`find_boundaries`], [`mark_boundaries`], and [`join_segmentations`].
//! - [`superpixel`]: [`slic`], [`felzenszwalb`], [`quickshift`], and [`random_walker`].
//! - [`active_contour`]: [`chan_vese`], [`morph_chan_vese`], [`morph_geodesic_contour`], [`checkerboard_level_set`], and [`disk_level_set`].

pub mod active_contour;
pub mod kernels;
pub mod labels;
pub mod superpixel;
pub mod watershed;

pub use active_contour::{
    chan_vese, checkerboard_level_set, disk_level_set, morph_chan_vese, morph_geodesic_contour,
    ChanVeseOptions,
};
pub use labels::{
    clear_border, expand_labels, find_boundaries, join_segmentations, mark_boundaries,
    BoundaryMode,
};
pub use superpixel::{felzenszwalb, quickshift, random_walker, slic, SlicOptions};
pub use watershed::{watershed, watershed_ift, WatershedOptions};

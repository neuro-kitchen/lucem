//! Quantitative image and region measurements.
//!
//! - [`regions`]: connected-component labeling, bounding boxes, and region feature extraction (`regionprops`).
//! - [`moments`]: raw, central, scale-normalized, and Hu rotation-invariant moments, plus inertia tensors.
//! - [`topology`]: perimeter, Crofton perimeter, Euler characteristic, block reduction, line profiles, Shannon entropy, and blur score.
//! - [`colocalize`]: fluorescence colocalization coefficients (Pearson's $r$, Manders' $M_1, M_2$, overlap, intersection).
//! - [`contour`]: 2-D marching squares and 3-D marching cubes isosurface extraction.

pub mod colocalize;
pub mod contour;
pub mod kernels;
pub mod moments;
pub mod regions;
pub mod topology;

pub use colocalize::{intersection_coeff, manders_coeff, manders_overlap, pearson_r};
pub use contour::{marching_cubes, marching_squares, mesh_surface_area, Contour2d, SurfaceMesh};
pub use moments::{
    central_moments, hu_moments, inertia_tensor, inertia_tensor_eigvals, moments,
    normalized_moments,
};
pub use regions::{bounding_boxes, label_components, region_features, region_stats, RegionFeatures};
pub use topology::{
    block_reduce, blur_score, crofton_perimeter, euler_number, perimeter, profile_line,
    shannon_entropy,
};

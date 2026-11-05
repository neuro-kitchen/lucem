//! Feature detection and local descriptors: local maxima, multi-scale blobs (LoG, DoG, DoH),
//! Hessian & structure tensors, Canny edges, corner detectors, template matching, LBP, GLCM, and HOG.

pub mod blobs;
pub mod canny;
pub mod corners;
pub mod kernels;
pub mod peaks;
pub mod structure;
pub mod texture;

pub use blobs::{blobs_dog, blobs_doh, blobs_log, Blob, BlobDetector};
pub use canny::{canny, Canny};
pub use corners::{
    corner_subpix, fast, foerstner, harris, kitchen_rosenfeld, moravec, shi_tomasi, Harris,
    HarrisMethod,
};
pub use peaks::{local_peaks, LocalPeaks};
pub use structure::{
    hessian_det, hessian_eigvals, hessian_matrix, shape_index, structure_tensor,
    structure_tensor_eigvals,
};
pub use texture::{
    glcm, glcm_props, hog, lbp, match_template, multiscale_features, GlcmProps, Hog, LbpMethod,
};

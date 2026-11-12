//! Geometric image transforms, multi-resolution pyramids, integral images, and projection transforms
//! (Hough & Radon).

pub mod integral;
pub mod kernels;
pub mod projection;
pub mod pyramid;
pub mod warp;

pub use integral::{integral_image, integral_image_into, integrate_box};
pub use projection::{
    hough_circles, hough_ellipses, hough_line_peaks, hough_lines, iradon, radon,
    BackprojectionFilter, HoughEllipse, HoughLinePeak, HoughLines,
};
pub use pyramid::{
    downsample_mean, gaussian_pyramid, laplacian_pyramid, pyramid_expand, pyramid_reduce,
    resize_mean, Pyramid,
};
pub use warp::{
    affine_warp, polar_warp, rescale, resize, rotate, sample_coords, shift, swirl, zoom,
    WarpSettings,
};

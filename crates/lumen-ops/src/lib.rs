//! Image operations on any GPU or the CPU, written once in CubeCL.
//!
//! # Modules
//!
//! | Domain | Module | What it provides |
//! |---|---|---|
//! | Shared primitives | [`core`] | Padding modes, argument checks, separable passes, structuring elements, reductions, histograms, scans, compaction, connected components, distance transforms, splines, FFT, eigenvalues, RNG |
//! | Spatial & frequency filters | [`filter`] | Correlation/convolution, box blur, Gaussian smoothing & derivatives, edge stencils, order-statistic filters, vessel/neurite ridge filters, spectral filters, sliding-histogram filters |
//! | Binarization | [`threshold`] | Global histogram thresholds (Otsu, Multi-Otsu, Li, Yen, ISODATA, mean, minimum, triangle), local adaptive thresholds (Niblack, Sauvola), hysteresis thresholding |
//! | Mathematical morphology | [`morphology`] | Grayscale, binary, isotropic (EDT), geodesic reconstruction, regional extrema, hole/object cleaning, skeletonization, medial axis, convex hull |
//! | Dynamic range & exposure | [`intensity`] | Pixel type scaling & conversion, contrast stretching, gamma/log/sigmoid curves, global & CLAHE histogram equalization, histogram matching, synthetic noise |
//! | Color & stains | [`color`] | Color space conversions, optical-density stain unmixing/mixing (H&E, H-DAB), perceptual color difference ($\Delta E$), label overlay rendering |
//! | Quantitative measurements | [`measure`] | Connected-component region properties, raw/central/normalized/Hu moments, inertia tensors, perimeter, Euler number, colocalization metrics, marching squares/cubes |
//! | Segmentation | [`segment`] | Marker-controlled watershed, label expansion, boundary extraction, SLIC, Felzenszwalb, Quickshift, random walker, Chan-Vese & morphological snakes |
//! | Feature detection | [`feature`] | Local peak detection, multi-scale blobs (LoG, DoG, DoH), Hessian & structure tensors, Canny edges, corner detectors, template matching, LBP, GLCM, HOG |
//! | Geometric transforms | [`transform`] | Coordinate sampling, affine/projective warps, shift/zoom/rotate/resize/rescale, polar/swirl warps, Gaussian/Laplacian pyramids, integral images, Hough & Radon transforms |
//! | Image restoration | [`restore`] | Total-variation, bilateral & non-local means denoising, rolling-ball background subtraction, Richardson-Lucy & Wiener deconvolution, biharmonic inpainting, phase unwrapping |
//! | Registration & flow | [`register`] | Subpixel Fourier phase cross-correlation, dense optical flow (ILK, TV-L1) |
//! | Evaluation metrics | [`metric`] | MSE, NRMSE, PSNR, SSIM, Hausdorff distance, adapted Rand error, variation of information, normalized mutual information |

#![allow(unused_unsafe, float_literal_f32_fallback)]

#[macro_use]
pub mod test_support;

pub mod analytics;
pub mod color;
pub mod core;
pub mod error;
pub mod feature;
pub mod filter;
pub mod intensity;
pub mod measure;
pub mod metric;
pub mod morphology;
pub mod register;
pub mod restore;
pub mod segment;
pub mod threshold;
pub mod transform;

pub use error::{OpsError, OpsResult};

//! Full-reference image fidelity and segmentation evaluation metrics: Mean Squared Error (`mse`),
//! Normalized Root Mean Squared Error (`nrmse`), Peak Signal-to-Noise Ratio (`psnr`), Structural
//! Similarity Index (`ssim`, `ssim_with_map`), Hausdorff distance (`hausdorff`, `hausdorff_pair`),
//! Adapted Rand Error (`adapted_rand`), Variation of Information (`variation_of_info`), and
//! Normalized Mutual Information (`mutual_info`).

pub mod fidelity;
pub mod kernels;
pub mod segmentation;

pub use fidelity::{mse, nrmse, psnr, ssim, ssim_with_map, NrmseNormalization, Ssim};
pub use segmentation::{
    adapted_rand, hausdorff, hausdorff_pair, mutual_info, variation_of_info, AdaptedRandResult,
    HausdorffMethod,
};

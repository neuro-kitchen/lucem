---
title: "lumen_ops"
description: "Image operations on the device: filters, thresholds, morphology, segmentation, measurements."
sidebar:
  label: "Overview"
  order: 0
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Image operations on any GPU or the CPU, written once in CubeCL.

#### Modules

| Domain | Module | What it provides |
|---|---|---|
| Shared primitives | `core` | Padding modes, argument checks, separable passes, structuring elements, reductions, histograms, scans, compaction, connected components, distance transforms, splines, FFT, eigenvalues, RNG |
| Spatial & frequency filters | `filter` | Correlation/convolution, box blur, Gaussian smoothing & derivatives, edge stencils, order-statistic filters, vessel/neurite ridge filters, spectral filters, sliding-histogram filters |
| Binarization | `threshold` | Global histogram thresholds (Otsu, Multi-Otsu, Li, Yen, ISODATA, mean, minimum, triangle), local adaptive thresholds (Niblack, Sauvola), hysteresis thresholding |
| Mathematical morphology | `morphology` | Grayscale, binary, isotropic (EDT), geodesic reconstruction, regional extrema, hole/object cleaning, skeletonization, medial axis, convex hull |
| Dynamic range & exposure | `intensity` | Pixel type scaling & conversion, contrast stretching, gamma/log/sigmoid curves, global & CLAHE histogram equalization, histogram matching, synthetic noise |
| Color & stains | `color` | Color space conversions, optical-density stain unmixing/mixing (H&E, H-DAB), perceptual color difference ($\Delta E$), label overlay rendering |
| Quantitative measurements | `measure` | Connected-component region properties, raw/central/normalized/Hu moments, inertia tensors, perimeter, Euler number, colocalization metrics, marching squares/cubes |
| Segmentation | `segment` | Marker-controlled watershed, label expansion, boundary extraction, SLIC, Felzenszwalb, Quickshift, random walker, Chan-Vese & morphological snakes |
| Feature detection | `feature` | Local peak detection, multi-scale blobs (LoG, DoG, DoH), Hessian & structure tensors, Canny edges, corner detectors, template matching, LBP, GLCM, HOG |
| Geometric transforms | `transform` | Coordinate sampling, affine/projective warps, shift/zoom/rotate/resize/rescale, polar/swirl warps, Gaussian/Laplacian pyramids, integral images, Hough & Radon transforms |
| Image restoration | `restore` | Total-variation, bilateral & non-local means denoising, rolling-ball background subtraction, Richardson-Lucy & Wiener deconvolution, biharmonic inpainting, phase unwrapping |
| Registration & flow | `register` | Subpixel Fourier phase cross-correlation, dense optical flow (ILK, TV-L1) |
| Evaluation metrics | `metric` | MSE, NRMSE, PSNR, SSIM, Hausdorff distance, adapted Rand error, variation of information, normalized mutual information |

## At the crate root

### `OpsError`

```rust
pub enum OpsError
```

**Variants**

- `Core(CoreError)`: Error from `lumen-core` (shapes, axes, element limits, device memory budget, runtime).
- `AliasedBuffers`: Input and output arrays alias the same underlying device buffer.
- `ShapeMismatch { first_name: &'static str, first: Vec<usize>, second_name: &'static str, second: Vec<usize> }`: Two arrays that must have identical shapes disagree.
- `InvalidDimension { what: &'static str, expected: &'static str, actual: usize, shape: Vec<usize> }`: Unexpected array dimensionality (`ndim`).
- `UnknownPadMode(String)`: Unknown boundary padding mode name.
- `InvalidOrigin { origin: i32, size: usize, min: i32, max: i32 }`: Invalid filter origin offset for a window of given size.
- `InvalidWeights { what: &'static str, reason: &'static str }`: Empty or non-finite filter weights or structuring element.
- `InvalidParameter { name: &'static str, reason: String }`: Invalid parameter value (e.g. negative sigma, zero window size, percentile outside `0..=100`).
- `EmptyInput(&'static str)`: Operation is not supported on an empty array (e.g. global min/max or threshold without elements).
- `Analytics(String)`: Error building an Apache Arrow `RecordBatch` or Polars `DataFrame` in analytics.

Implements `Clone`, `Debug`, `Display`, `Error`, `From`, `PartialEq`.

### `OpsResult`

```rust
pub type OpsResult<T> = Result<T, OpsError>;
```

## Modules

| Module | What it holds |
|---|---|
| [`lumen_ops::analytics`](/lumen/reference/rust/lumen-ops/analytics/) | Columnar & tabular analytics powered by **Apache Arrow (`arrow`)** and **Polars (`polars`)**. |
| [`lumen_ops::color`](/lumen/reference/rust/lumen-ops/color/) | Color space conversions, Beer-Lambert histological stain unmixing, perceptual color difference metrics (CIE76, CIE94, CIEDE2000, CMC), and label map false-color overlays. |
| [`lumen_ops::core`](/lumen/reference/rust/lumen-ops/core/) | Shared device primitives: the single home for every building block reused across `lumen-ops`. |
| [`lumen_ops::error`](/lumen/reference/rust/lumen-ops/error/) | Errors of `lumen-ops`. |
| [`lumen_ops::feature`](/lumen/reference/rust/lumen-ops/feature/) | Feature detection and local descriptors: local maxima, multi-scale blobs (LoG, DoG, DoH), Hessian & structure tensors, Canny edges, corner detectors, template matching, LBP, GLCM, and HOG. |
| [`lumen_ops::filter`](/lumen/reference/rust/lumen-ops/filter/) | Spatial, derivative, order-statistic, ridge, spectral, and windowed-histogram filters. |
| [`lumen_ops::intensity`](/lumen/reference/rust/lumen-ops/intensity/) | Intensity conversions, contrast stretching, equalization, and stochastic noise synthesis. |
| [`lumen_ops::measure`](/lumen/reference/rust/lumen-ops/measure/) | Quantitative image and region measurements. |
| [`lumen_ops::metric`](/lumen/reference/rust/lumen-ops/metric/) | Full-reference image fidelity and segmentation evaluation metrics: Mean Squared Error (`mse`), Normalized Root Mean Squared Error (`nrmse`), Peak Signal-to-Noise Ratio (`psnr`), Structural Similarity Index (`ssim`, `ssim_with_map`), Hausdorff distance (`hausdorff`, `hausdorff_pair`), Adapted Rand Error (`adapted_rand`), Variation of Information (`variation_of_info`), and Normalized Mutual Information (`mutual_info`). |
| [`lumen_ops::morphology`](/lumen/reference/rust/lumen-ops/morphology/) | Mathematical morphology: grayscale, binary, exact-Euclidean isotropic, geodesic reconstruction, and skeletonization. |
| [`lumen_ops::register`](/lumen/reference/rust/lumen-ops/register/) | Image registration & dense optical flow: subpixel Fourier phase cross-correlation (`phase_correlation`) and dense 2-D optical flow (`ilk_flow`, `tvl1_flow`). |
| [`lumen_ops::restore`](/lumen/reference/rust/lumen-ops/restore/) | Image restoration: total-variation (`tv_chambolle`, `tv_bregman`), bilateral (`bilateral`), and non-local means (`non_local_means`) denoising, robust noise estimation (`estimate_noise`), rolling-ball background estimation (`rolling_ball`), Richardson-Lucy & Wiener deconvolution (`richardson_lucy`, `wiener_deconvolve`, `unsupervised_wiener`), biharmonic PDE inpainting (`biharmonic_inpaint`), and reliability-guided phase unwrapping (`unwrap_phase`). |
| [`lumen_ops::segment`](/lumen/reference/rust/lumen-ops/segment/) | Image segmentation: watershed, label post-processing, superpixels, random walker, and active contours. |
| [`lumen_ops::threshold`](/lumen/reference/rust/lumen-ops/threshold/) | Global, adaptive local, and hysteresis thresholding. |
| [`lumen_ops::transform`](/lumen/reference/rust/lumen-ops/transform/) | Geometric image transforms, multi-resolution pyramids, integral images, and projection transforms (Hough & Radon). |

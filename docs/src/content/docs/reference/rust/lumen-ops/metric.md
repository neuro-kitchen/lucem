---
title: "lumen_ops::metric"
description: "Full-reference image fidelity and segmentation evaluation metrics: Mean Squared Error (`mse`), Normalized Root Mean Squared Error (`nrmse`), Peak Signal-to-Noise Ratio (`psnr`), Structural Similarity Index (`ssim`, `ssim_with_map`), Hausdorff distance (`hausdorff`, `hausdorff_pair`), Adapted Rand Error (`adapted_rand`), Variation of Information (`variation_of_info`), and Normalized Mutual Information (`mutual_info`)."
sidebar:
  label: "metric"
  order: 9
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Full-reference image fidelity and segmentation evaluation metrics: Mean Squared Error (`mse`),
Normalized Root Mean Squared Error (`nrmse`), Peak Signal-to-Noise Ratio (`psnr`), Structural
Similarity Index (`ssim`, `ssim_with_map`), Hausdorff distance (`hausdorff`, `hausdorff_pair`),
Adapted Rand Error (`adapted_rand`), Variation of Information (`variation_of_info`), and
Normalized Mutual Information (`mutual_info`).

## Types

### `Ssim`

```rust
pub struct Ssim
```

Settings for the Structural Similarity Index (`ssim`).

**Fields**

- `win_size: Option<usize>`: Sliding window side length (must be odd; defaults to `7` for box windows or `11` for Gaussian).
- `gaussian_weights: bool`: When `true`, uses separable Gaussian weighting with standard deviation `sigma`; when `false`, uses a uniform box window of size `win_size`.
- `sigma: f64`: Gaussian standard deviation when `gaussian_weights = true` (default `1.5`).
- `data_range: Option<f64>`: Explicit dynamic range $L$ of the input images (when `None`, inferred from `max - min` of `image_true`, or `1.0` if constant).
- `use_sample_covariance: bool`: Whether to normalize local covariances by $N / (N - 1)$ (Bessel sample correction).
- `k1: f64`: Luminance stabilization constant $K_1$ (default `0.01`).
- `k2: f64`: Contrast/structure stabilization constant $K_2$ (default `0.03`).

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `AdaptedRandResult`

```rust
pub struct AdaptedRandResult
```

Output of `adapted_rand`.

**Fields**

- `error: f64`: Adapted Rand Error ($1 - F_1$ of pairwise same-cluster precision and recall).
- `precision: f64`: Pairwise precision ($\sum_{i,j} p_{ij}^2 / \sum_j t_j^2$).
- `recall: f64`: Pairwise recall ($\sum_{i,j} p_{ij}^2 / \sum_i s_i^2$).

Implements `Clone`, `Copy`, `Debug`, `PartialEq`.

### `NrmseNormalization`

```rust
pub enum NrmseNormalization
```

Normalization denominator for `nrmse`.

**Variants**

- `Euclidean`: Euclidean norm of `image_true`: $\sqrt{\frac{1}{N}\sum I_{\text{true}}^2}$.
- `MinMax`: Dynamic range of `image_true`: $\max(I_{\text{true}}) - \min(I_{\text{true}})$.
- `Mean`: Mean intensity of `image_true`: $|\overline{I_{\text{true}}}|$.

Implements `Clone`, `Copy`, `Debug`, `Default`, `Eq`, `PartialEq`.

### `HausdorffMethod`

```rust
pub enum HausdorffMethod
```

Variant of Hausdorff distance for `hausdorff`.

**Variants**

- `Standard`: Standard maximum directed Hausdorff distance: $\max\bigl(\max_{a \in A} d(a, B),\; \max_{b \in B} d(b, A)\bigr)$.
- `Modified`: Dubuisson & Jain (1994) modified (mean directed) Hausdorff distance: $\max\bigl(\frac{1}{|A|}\sum_{a \in A} d(a, B),\; \frac{1}{|B|}\sum_{b \in B} d(b, A)\bigr)$.

Implements `Clone`, `Copy`, `Debug`, `Default`, `Eq`, `PartialEq`.

## Functions

### `mse`

```rust
pub fn mse<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
) -> OpsResult<f64>
```

Computes the Mean Squared Error (MSE) between `image_true` and `image_test`:
$$\operatorname{MSE}(I, J) = \frac{1}{N} \sum_{i=1}^{N} (I_i - J_i)^2$$

### `nrmse`

```rust
pub fn nrmse<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
    normalization: NrmseNormalization,
) -> OpsResult<f64>
```

Computes the Normalized Root Mean Squared Error (NRMSE) between `image_true` and `image_test`.

### `psnr`

```rust
pub fn psnr<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
    data_range: Option<f64>,
) -> OpsResult<f64>
```

Computes the Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) between `image_true` and
`image_test`:
$$\operatorname{PSNR} = 10 \log_{10}\!\left(\frac{\text{data\_range}^2}{\operatorname{MSE}}\right)$$

When `data_range` is `None`, defaults to `max(image_true) - min(image_true)` (or `1.0` if
`image_true` is constant).

### `ssim_with_map`

```rust
pub fn ssim_with_map<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
    settings: &Ssim,
) -> OpsResult<(f64, DeviceArray<F>)>
```

Computes both the scalar mean SSIM and the full-resolution per-pixel SSIM map on the device.

### `ssim`

```rust
pub fn ssim<F: ImageFloat>(
    client: &Client,
    image_true: &DeviceArray<F>,
    image_test: &DeviceArray<F>,
    settings: &Ssim,
) -> OpsResult<f64>
```

Computes the mean Structural Similarity Index (SSIM) between `image_true` and `image_test`
(Wang, Bovik, Sheikh & Simoncelli, 2004).

### `hausdorff`

```rust
pub fn hausdorff<A: Numeric + CubeElement, B: Numeric + CubeElement>(
    client: &Client,
    mask_a: &DeviceArray<A>,
    mask_b: &DeviceArray<B>,
    method: HausdorffMethod,
) -> OpsResult<f64>
```

Computes the exact Euclidean Hausdorff distance (`Standard` or `Modified`) between non-zero
foreground elements of `mask_a` and `mask_b` on the device using separable Euclidean distance
transforms.

Returns `0.0` if both masks are empty (all zero), or `f64::INFINITY` if exactly one mask is
empty.

### `hausdorff_pair`

```rust
pub fn hausdorff_pair<A: Numeric + CubeElement, B: Numeric + CubeElement>(
    client: &Client,
    mask_a: &DeviceArray<A>,
    mask_b: &DeviceArray<B>,
) -> OpsResult<Option<(Vec<usize>, Vec<usize>)>>
```

Returns the pair of N-D pixel coordinates `(coord_in_a, coord_in_b)` that realize the standard
Hausdorff distance between `mask_a` and `mask_b`, or `None` if either mask has no non-zero
foreground pixels.

### `adapted_rand`

```rust
pub fn adapted_rand(
    client: &Client,
    labels_true: &DeviceArray<u32>,
    labels_test: &DeviceArray<u32>,
    ignore_labels: Option<&[u32]>,
) -> OpsResult<AdaptedRandResult>
```

Computes the SNEMI3D Adapted Rand Error (`(error, precision, recall)`) between ground-truth
integer segmentation `labels_true` and candidate segmentation `labels_test`.

By default (`ignore_labels = None`), background label `0` in `labels_true` is ignored (pass
`Some(&[])` to include `0`).

### `variation_of_info`

```rust
pub fn variation_of_info(
    client: &Client,
    labels_true: &DeviceArray<u32>,
    labels_test: &DeviceArray<u32>,
    ignore_labels: Option<&[u32]>,
) -> OpsResult<(f64, f64)>
```

Computes Meilă's (2003) Variation of Information `(false_splits, false_merges)` in bits
($\log_2$), where `false_splits = H(labels_test | labels_true)` (over-segmentation conditional
entropy) and `false_merges = H(labels_true | labels_test)` (under-segmentation conditional
entropy).

Labels listed in `ignore_labels` (defaults to `&[]`) are excluded if present in either array.

### `mutual_info`

```rust
pub fn mutual_info<F: ImageFloat>(
    client: &Client,
    image_a: &DeviceArray<F>,
    image_b: &DeviceArray<F>,
    bins: usize,
) -> OpsResult<f64>
```

Computes Studholme, Hill & Hawkes's (1999) Normalized Mutual Information (NMI) between two
images using a `bins x bins` joint intensity histogram:
$$\operatorname{NMI}(A, B) = \frac{H(A) + H(B)}{H(A, B)} \in [1.0, 2.0]$$

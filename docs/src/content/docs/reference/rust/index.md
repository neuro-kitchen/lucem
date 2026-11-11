---
title: Rust API
description: "A map of the lumen crates, and every public function and type with its signature."
sidebar:
  label: Overview
  order: 0
---

lumen is a set of Rust crates. To get the library through one dependency, use `lumen`, which
re-exports `lumen-core`, `lumen-io`, `lumen-ops` and `lumen-ml` as modules. The crates are not on
crates.io, so the pages here stand in for docs.rs: each lists every public function, type and
method with its exact signature and documentation, generated from the code.

```toml
[dependencies]
lumen = { git = "https://github.com/neuro-kitchen/lumen", features = ["wgpu"] }
```

| Crate | What it is |
|---|---|
| [`lumen-core`](/lumen/reference/rust/lumen-core/) | Device arrays and memory, shapes, compute targets (runtimes), tiling |
| [`lumen-io`](/lumen/reference/rust/lumen-io/) | Reading and writing images: TIFF, OME-Zarr, pyramids, projects |
| [`lumen-ops`](/lumen/reference/rust/lumen-ops/) | Image operations on the device: filters, thresholds, morphology, segmentation, measurements |
| [`lumen-pipeline`](/lumen/reference/rust/lumen-pipeline/) | Pipelines as data: the operation catalog, the executor, tables |
| [`lumen-hist`](/lumen/reference/rust/lumen-hist/) | Histology: tissue, cells, classifiers, densities, the analysis engine and sessions |
| [`lumen-ml`](/lumen/reference/rust/lumen-ml/) | The model-plugin protocol: discovery, client, messages |
| [`lumen-hub`](/lumen/reference/rust/lumen-hub/) | Verified downloads of published artifacts (model weights) |

## In `lumen-ops`

| Need | Module | Start with |
|---|---|---|
| Smooth, sharpen, find edges | [`filter`](/lumen/reference/rust/lumen-ops/filter/) | `gaussian_blur`, `median_filter`, `sobel`, `frangi` |
| Turn an image into a mask | [`threshold`](/lumen/reference/rust/lumen-ops/threshold/) | `otsu`, `sauvola`, `hysteresis_threshold` |
| Clean up masks | [`morphology`](/lumen/reference/rust/lumen-ops/morphology/) | `binary_open`, `remove_small_objects`, `remove_small_holes` |
| Split touching objects | [`segment`](/lumen/reference/rust/lumen-ops/segment/) | `watershed`, `slic`, `expand_labels` |
| Measure objects | [`measure`](/lumen/reference/rust/lumen-ops/measure/) | `label_components`, `region_features`, `pearson_r` |
| Contrast and exposure | [`intensity`](/lumen/reference/rust/lumen-ops/intensity/) | `rescale_range`, `clahe`, `equalize_hist` |
| Separate stains | [`color`](/lumen/reference/rust/lumen-ops/color/) | `unmix_stains` |
| Remove background, denoise | [`restore`](/lumen/reference/rust/lumen-ops/restore/) | `rolling_ball`, `tv_chambolle`, `bilateral` |
| Find spots and features | [`feature`](/lumen/reference/rust/lumen-ops/feature/) | `local_peaks`, `blobs_log`, `canny` |
| Warp, resize, pyramids | [`transform`](/lumen/reference/rust/lumen-ops/transform/) | `affine_warp`, `resize`, `gaussian_pyramid` |
| Align images | [`register`](/lumen/reference/rust/lumen-ops/register/) | `phase_correlation` |
| Compare results | [`metric`](/lumen/reference/rust/lumen-ops/metric/) | `ssim`, `psnr`, `adapted_rand` |
| Tables of measurements | [`analytics`](/lumen/reference/rust/lumen-ops/analytics/) | `regions_to_dataframe` |

## Conventions

- **A client first.** Functions on the device take a CubeCL `Client`:
  `ComputeTarget::from_env()?.client()?` picks the runtime named by `LUMEN_RUNTIME` (`wgpu`,
  `cpu`, `cuda`, `hip`), or else the first one compiled in.
- **Data on the device.** Images are `DeviceArray<F>` with a `Shape` (`[y, x]` or `[z, y, x]`,
  row-major). `DeviceArray::from_host` uploads, `to_host` reads back; nothing moves between host
  and device unless a function says so.
- **Three forms of an operation.** `op` returns a new array; `op_into` writes into an output you
  pass (no allocation, for loops and pipelines); `op_device` keeps a scalar result on the device
  (`otsu_device`) where `op` reads it back (`otsu`).
- **Errors** are `OpsResult<T>` (`lumen-ops`), `CoreResult<T>` (`lumen-core`) and `HistResult<T>`
  (`lumen-hist`); shapes are checked before any kernel runs.
- **The same operations by name.** A pipeline (`lumen-pipeline`), the app and `lumen run` call
  these functions through the [operation catalog](/lumen/reference/operations/).

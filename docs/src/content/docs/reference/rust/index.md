---
title: Rust API
description: "A map of the lucem crates, and every public function and type with its signature."
sidebar:
  label: Overview
  order: 0
---

lucem is a set of Rust crates. To get the library through one dependency, use `lucem`, which
re-exports `lucem-core`, `lucem-io`, `lucem-ops` and `lucem-ml` as modules. The crates are not on
crates.io, so the pages here stand in for docs.rs: each lists every public function, type and
method with its exact signature and documentation, generated from the code.

```toml
[dependencies]
lucem = { git = "https://github.com/neuro-kitchen/lucem", features = ["wgpu"] }
```

| Crate | What it is |
|---|---|
| [`lucem-core`](/lucem/reference/rust/lucem-core/) | Device arrays and memory, shapes, compute targets (runtimes), tiling |
| [`lucem-io`](/lucem/reference/rust/lucem-io/) | Reading and writing images: TIFF, OME-Zarr, pyramids, projects |
| [`lucem-ops`](/lucem/reference/rust/lucem-ops/) | Image operations on the device: filters, thresholds, morphology, segmentation, measurements |
| [`lucem-pipeline`](/lucem/reference/rust/lucem-pipeline/) | Pipelines as data: the operation catalog, the executor, tables |
| [`lucem-hist`](/lucem/reference/rust/lucem-hist/) | Histology: tissue, cells, classifiers, densities, the analysis engine and sessions |
| [`lucem-ml`](/lucem/reference/rust/lucem-ml/) | The model-plugin protocol: discovery, client, messages |
| [`lucem-hub`](/lucem/reference/rust/lucem-hub/) | Verified downloads of published artifacts (model weights) |

## In `lucem-ops`

| Need | Module | Start with |
|---|---|---|
| Smooth, sharpen, find edges | [`filter`](/lucem/reference/rust/lucem-ops/filter/) | `gaussian_blur`, `median_filter`, `sobel`, `frangi` |
| Turn an image into a mask | [`threshold`](/lucem/reference/rust/lucem-ops/threshold/) | `otsu`, `sauvola`, `hysteresis_threshold` |
| Clean up masks | [`morphology`](/lucem/reference/rust/lucem-ops/morphology/) | `binary_open`, `remove_small_objects`, `remove_small_holes` |
| Split touching objects | [`segment`](/lucem/reference/rust/lucem-ops/segment/) | `watershed`, `slic`, `expand_labels` |
| Measure objects | [`measure`](/lucem/reference/rust/lucem-ops/measure/) | `label_components`, `region_features`, `pearson_r` |
| Contrast and exposure | [`intensity`](/lucem/reference/rust/lucem-ops/intensity/) | `rescale_range`, `clahe`, `equalize_hist` |
| Separate stains | [`color`](/lucem/reference/rust/lucem-ops/color/) | `unmix_stains` |
| Remove background, denoise | [`restore`](/lucem/reference/rust/lucem-ops/restore/) | `rolling_ball`, `tv_chambolle`, `bilateral` |
| Find spots and features | [`feature`](/lucem/reference/rust/lucem-ops/feature/) | `local_peaks`, `blobs_log`, `canny` |
| Warp, resize, pyramids | [`transform`](/lucem/reference/rust/lucem-ops/transform/) | `affine_warp`, `resize`, `gaussian_pyramid` |
| Align images | [`register`](/lucem/reference/rust/lucem-ops/register/) | `phase_correlation` |
| Compare results | [`metric`](/lucem/reference/rust/lucem-ops/metric/) | `ssim`, `psnr`, `adapted_rand` |
| Tables of measurements | [`analytics`](/lucem/reference/rust/lucem-ops/analytics/) | `regions_to_dataframe` |

## Conventions

- **A client first.** Functions on the device take a CubeCL `Client`:
  `ComputeTarget::from_env()?.client()?` picks the runtime named by `LUCEM_RUNTIME` (`wgpu`,
  `cpu`, `cuda`, `hip`), or else the first one compiled in.
- **Data on the device.** Images are `DeviceArray<F>` with a `Shape` (`[y, x]` or `[z, y, x]`,
  row-major). `DeviceArray::from_host` uploads, `to_host` reads back; nothing moves between host
  and device unless a function says so.
- **Three forms of an operation.** `op` returns a new array; `op_into` writes into an output you
  pass (no allocation, for loops and pipelines); `op_device` keeps a scalar result on the device
  (`otsu_device`) where `op` reads it back (`otsu`).
- **Errors** are `OpsResult<T>` (`lucem-ops`), `CoreResult<T>` (`lucem-core`) and `HistResult<T>`
  (`lucem-hist`); shapes are checked before any kernel runs.
- **The same operations by name.** A pipeline (`lucem-pipeline`), the app and `lucem run` call
  these functions through the [operation catalog](/lucem/reference/operations/).

---
title: Course
description: "Worked microscopy and histology examples, from a first GPU call to tiled @lucem.op plugins and ML models."
sidebar:
  label: Overview
  order: 0
---

Each page corresponds to a runnable script in the repository's `playground/` folder, building step
by step from a first GPU call to tiled custom Python operators and model plugins.

Every script synthesizes its own fluorescence / histology test data with known ground truth — no
external downloads or real patient slides required — so you can verify every assertion numerically
on any machine.

| Chapter | What you learn |
|---|---|
| Basics | Runtimes (`cuda`, `wgpu`, `cpu`) and numerical agreement; writing and reading multi-channel OME-Zarr |
| Filtering | Removing uneven illumination with rolling ball, Gaussian denoising, and CLAHE local contrast |
| Segmentation | Otsu auto-thresholding and connected-component nuclei counting |
| Pipelines | On-device `Pipeline` chains and shareable `.lucem-recipe.json` recipes |
| Custom ops | Writing `@lucem.op(family="Filter", halo=12)` tiled Python ops with zero seam artifacts |
| Histology & ML | Installing and discovering `lucem[hist]` and `lucem[ml]` (`stardist`, `sam`, `instanseg`) |

## Run them yourself

From the repository root:

```sh
uv sync
cd bindings/python && uv run --project ../.. maturin develop --profile validate && cd ../..
uv run python playground/01_basics/01_runtimes.py
```

`LUCEM_RUNTIME=cpu` works on any machine without a GPU. Every script uses `# %%` cell markers so it
also opens directly as a notebook in VS Code, Positron, or Jupyter.

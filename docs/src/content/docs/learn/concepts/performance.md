---
title: Architecture & performance
description: How Lucem stays 60 fps across gigapixel OME-Zarr slides, GPU pipelines, warm Python workers, and 1M+ detected cells.
sidebar:
  order: 5
---

Lucem is engineered around four architectural invariants so whole-slide microscopy stays interactive on both desktop and browser (`wasm32`):

## 1. Pyramid-first chunked I/O (`lucem-io`)

Slides are read lazily from multi-resolution **OME-Zarr v0.5** and **OME-TIFF** pyramids. Only the tiles intersecting your current viewport (at the LOD matching your zoom) are decoded and uploaded. Coarser LOD tiles act as instant fallbacks during fast panning and zooming, and mid-flight tile requests are cancelled as soon as you zoom past them.

## 2. Unified GPU compute (`CubeCL`)

All 52 built-in image processing operations (`lucem-ops`) are written once in Rust with **CubeCL** and run on **CUDA**, **wgpu / WebGPU**, or **CPU** without rewriting kernels. The exact same pipeline runs interactively in **Explore**, in the **Pipeline** graph, in headless CLI batches, and from Python.

## 3. Persistent warm `/dev/shm` Python workers

When `lucem serve` runs a custom `@lucem.op` Python script, it does **not** spawn a fresh Python interpreter per tile. Instead, a persistent warm `uv` worker pool exchanges tile buffers over shared memory (`/dev/shm` `np.memmap`), keeping per-tile overhead in the low milliseconds while supporting full NumPy / SciPy / scikit-image ecosystems.

## 4. Server-side viewport rasterization for 100k–1M+ cells

Following the architecture pioneered by **QuPath**, **Cytomine**, and **Vitessce**, Lucem's browser client never downloads 100,000+ full-resolution JSON cell polygons onto the single-threaded WASM main thread:

- CUDA **StarDist** and **InstanSeg** run on the server GPU (`lucem-ml`), keeping the full object table in server memory.
- As you pan or zoom in **Explore**, the browser requests a single ~15–30 KB viewport PNG overlay (`hist/raster`) rasterized natively on the server in milliseconds.
- The **Analyze** workspace fetches compact, simplified summaries on demand.

## 5. GPU VRAM & Batched-Volume ML (`LUCEM_ML_BATCH`)

StarDist and InstanSeg tile any ROI into canonical **`512×512`** (`tile: 384, halo: 64`) squares so CubeCL autotunes once and reuses cached CUDA kernels across every future ROI (~28–32 ms per tile on an RTX 2070).

- **Default (`LUCEM_ML_BATCH=1`):** Keeps peak VRAM around **~250–350 MiB** so shared 6–8 GB laptop/desktop GPUs (running a compositor + browser WebGPU canvas) never hit PCIe VRAM swapping. Switching between StarDist and InstanSeg automatically evicts the idle model plugin from VRAM first.
- **High-VRAM workstations & servers (`LUCEM_ML_BATCH=4` or `8`):** On dedicated 16–80 GB GPUs (RTX 4090, A6000, L40S, H100), set `LUCEM_ML_BATCH=4` (or `8`) before starting `lucem serve` or `lucem hist`. Lucem stacks `B` canonical tiles into a single 4D volume `[B, C, 512, 512]` and evaluates them in one parallel GPU forward pass — saturating GPU SMs with one kernel-launch and readback cycle per batch (padding partial tail batches to `B` so CubeCL never re-autotunes).

```bash
# Dedicated 24 GB+ GPU server: 4-tile batched volume inference
LUCEM_ML_BATCH=4 lucem serve --data ./slides
```

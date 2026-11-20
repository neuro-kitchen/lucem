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

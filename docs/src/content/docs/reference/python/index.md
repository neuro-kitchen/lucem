---
title: Python API
description: "Reference for the lucem Python package (lucem, lucem[web], lucem[app], lucem[hist], lucem[ml])."
sidebar:
  label: Overview
  order: 0
---

Install `lucem` with `uv` or `pip`, picking only the optional extras your project needs:

```sh
uv add lucem                         # Core: runtime, io, ops, custom (@lucem.op), ml client
uv add "lucem[web]"                  # + Static WASM browser viewer (`lucem web`)
uv add "lucem[app]"                  # + Native desktop workstation (`lucem app`)
uv add "lucem[hist]"                 # + Histology: stains, tissue, cells, sessions
uv add "lucem[stardist]"             # + StarDist nuclei plugin
uv add "lucem[ml]"                   # + StarDist + SAM + InstanSeg plugins
```

## Module map

| Module | What it provides |
|---|---|
| [`lucem.runtime`](/lucem/reference/python/runtime/) | Query/select `cuda`, `wgpu`, or `cpu` and run `autotune()` |
| [`lucem.io`](/lucem/reference/python/io/) | Lazy OME-Zarr / OME-TIFF `Pyramid` reading and block-wise `ZarrWriter` |
| [`lucem.ops`](/lucem/reference/python/ops/) | All 52 GPU operators plus on-device `Pipeline` chains |
| [`lucem.custom`](/lucem/reference/python/custom/) | `@lucem.op`, `@lucem.transform`, `blocks()`, `measure()`, `segment()` |
| [`lucem.web` / `app` / `ml` / `hist`](/lucem/reference/python/app-web-ml/) | Launch browser/desktop UIs, discover ML plugins, run histology |

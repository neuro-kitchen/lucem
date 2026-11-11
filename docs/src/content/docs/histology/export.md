---
title: Export
description: Write the analysis next to the image, in open formats.
sidebar:
  order: 7
---

**Export** writes the shown image's analysis to its output store, in the background:

| Path | What it holds |
|---|---|
| `labels/cells`, `labels/nuclei` | Label images: each object's id per pixel (OME-Zarr labels) |
| `lumen/cells.parquet`, `lumen/tiles.parquet` | One row per object with its measurements and region |
| `lumen/shapes.parquet` | Every outline, as GeoParquet |
| `lumen/provenance.json` | The session, the pipeline and the lumen version that made it |

The store is the image's own OME-Zarr when lumen can write to it, else
`<project>/analysis/<image>.lumen.zarr`. See [Output store](/lumen/reference/files/output-store/).

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
| `lucem/cells.parquet`, `lucem/tiles.parquet` | One row per object with its measurements and region |
| `lucem/shapes.parquet` | Every outline, as GeoParquet |
| `lucem/provenance.json` | The session, the pipeline and the lucem version that made it |

The store is the image's own OME-Zarr when lucem can write to it, else
`<project>/analysis/<image>.lucem.zarr`. See [Output store](/lucem/reference/files/output-store/).

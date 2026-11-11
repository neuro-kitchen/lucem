---
title: Output store
description: Where an analysis writes its results.
sidebar:
  order: 5
---

An image's results go in its output store: the image's own OME-Zarr when it can be written,
else a store beside it (`<image>.lumen.zarr`).

```text
slide.ome.zarr/
  labels/
    cells/          each cell's id per pixel (OME-Zarr label image, a pyramid)
    nuclei/         each nucleus' id per pixel
  lumen/
    cells.parquet   one row per cell: id, class, region, centroid (µm), measurements
    tiles.parquet   one row per superpixel tile
    shapes.parquet  every outline (GeoParquet: cells, tiles, regions)
    provenance.json the session, the pipeline, the lumen version
```

Parquet reads directly in Python (`polars.read_parquet`, `pandas.read_parquet`) and R
(`arrow::read_parquet`); GeoParquet in GeoPandas.

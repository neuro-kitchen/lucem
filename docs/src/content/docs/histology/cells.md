---
title: Count cells
description: Find nuclei, grow cells around them, and measure each.
sidebar:
  order: 2
---

Cells are found inside the selected regions, tile by tile on the GPU. Each nucleus is found,
grown into its cell, and measured: area, shape, and the intensity of every channel (or stain) in
the nucleus, the cytoplasm and the whole cell.

## With a threshold

| Setting | What it does |
|---|---|
| Background radius | Rolling-ball background subtracted first (µm; 0: none) |
| Median radius | A median filter against noise (µm; 0: none) |
| Smoothing | Gaussian smoothing of the nuclear channel (µm) |
| Threshold | Of the smoothed channel: an optical density (brightfield) or the channel's units. Fluorescence can take it from each image (Otsu's method) |
| Nucleus area | Smallest and largest nucleus kept (µm²) |
| Split touching nuclei | **At dark gaps** (seeds at each nucleus' brightest point, boundaries along the valleys; best in dense tissue) or **by shape** |
| Cell expansion | How far a nucleus grows into its cell (µm; 0: nuclei only) |

## With a model

Choose a nucleus model (for example **StarDist H&E** or **StarDist fluorescence**) instead of the
threshold. The model reads the image at about its own scale and finds the nuclei; lucem grows and
measures the cells the same way. See [Models](/lucem/models/).

## On the command line

```sh
lucem hist cells slide.ome.zarr --region 0,0,4096,4096 --stains h-dab --out cells.tsv
lucem hist cells slide.ome.zarr --within tissue.geojson --model stardist-he --out cells.geojson
```

New cells replace the cells inside the same regions; the rest are kept.

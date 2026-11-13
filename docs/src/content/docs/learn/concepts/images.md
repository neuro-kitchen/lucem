---
title: Images and pyramids
description: "How lucem reads large images: levels, channels, planes and pixel size."
sidebar:
  order: 1
---

A whole-slide image can be 100,000 pixels across, far more than fits on a screen or in a GPU's
memory. lucem reads images as **pyramids**: the full-resolution image (level 0) and smaller
copies of it (levels 1, 2, …, each a fraction of the one before), cut into tiles. The app draws
the level that matches the zoom, and reads only the tiles on screen.

| Term | Meaning |
|---|---|
| Level | One resolution of the image; level 0 is full resolution |
| Downsample | How many level-0 pixels one pixel of a level covers |
| Channel | One stain or fluorophore (RGB images: red, green, blue) |
| Plane | One z slice of a volume |
| Pixel size | Micrometres per level-0 pixel; measurements in µm need it |

Positions, outlines and measurements are always given in **level-0 pixels** (and in µm when the
pixel size is known), whatever level was read to compute them.

## OME-Zarr

OME-Zarr is lucem's home format: a folder of chunked arrays with the pyramid and its metadata. Label
images (cells, nuclei) are written inside it, under `labels/`, and open as images of their own.
Make one from any image with `lucem run` or the app's export.

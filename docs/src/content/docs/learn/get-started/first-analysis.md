---
title: Your first analysis
description: Open an image, count cells, read the table, and run the same analysis on more images.
sidebar:
  order: 2
---

## 1. Open an image

**Lucem → Open image or OME-Zarr…** (<kbd>Ctrl</kbd>+<kbd>O</kbd>), or start the app with a path:

```sh
cargo run -p lucem-app -- slide.ome.zarr
```

lucem reads OME-Zarr, OME-TIFF and common image files. No image at hand? Write a synthetic one:

```sh
lucem sample sample.ome.zarr            # a 2D image
lucem sample volume.ome.zarr --depth 64 # a volume
```

## 2. Count the cells in a region

1. Draw a rectangle (<kbd>U</kbd>) or a polygon (<kbd>P</kbd>) around the area to count.
2. Open the **Histology** panel on the right rail and choose **Count cells**.
3. Check the stains: lucem guesses H-DAB for RGB images and, for fluorescence, the channel named
   like a nuclear stain (DAPI, Hoechst).
4. Press **Run**. Progress shows under the button and on the status bar; **Cancel** stops after
   the current tile.

The cells appear over the image, outlined. Click one to select it; the **Selection** panel shows
its measurements.

## 3. Read the table

Switch to **Analyze** (<kbd>Ctrl</kbd>+<kbd>3</kbd>) and choose **Cells**. Every cell is a row:
filter a column, search, sort, add a histogram or a scatter plot beside the table, and export the
rows as CSV.

## 4. Save, then run it on more images

Save the project in a folder (**Lucem → Save project in folder…**). The steps you ran are in
its `session.json`. Run the same analysis on other images, without the app:

```sh
lucem session run my-project/session.json slides/*.ome.zarr --out results/
```

Each image gets its cells as a Parquet table, its outlines as GeoParquet and label images,
written inside its OME-Zarr when it can be written, else in `results/`. See
[Sessions](/lucem/learn/concepts/sessions/).

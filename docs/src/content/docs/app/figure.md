---
title: Figure composer
description: Compose multi-panel publication figures from channels, composites, pipeline stages, and ROIs with calibrated scale bars and PNG / SVG export.
sidebar:
  order: 5
---

The **Figure** workspace is a built-in publication composer (inspired by Photoshop artboards and QuickFigures) that turns any open slide, channel set, pipeline stage, or drawn ROI into a ready-to-submit multi-panel figure without leaving Lucem.

![Figure composer workspace showing multi-channel panels, calibrated scale bars, and inspector controls](/lucem/screenshots/07_figure-maker.png)

## Quick presets

Use the top toolbar to build a complete figure in one click:

- **All channels + Merge** — creates one panel per channel (`A`, `B`, `C`…) plus a final **Composite** merge panel, inheriting the exact display window, gamma, and colormap LUTs from **Explore**.
- **Pipeline stages** — places the raw input alongside every computed **Pipeline** stage preview so you can showcase preprocessing or segmentation steps side-by-side.
- **Add panel** — appends a new panel that you can point at any channel, composite, or pipeline stage.

## Syncing regions of interest (ROIs)

Every annotation you draw in **Explore** (rectangle, polygon, Magic Wand selection) appears in the left **Regions of Interest** list:

- Click **Sync all → `<ROI>`** to crop every panel on the artboard to the exact same region simultaneously.
- Or select an individual panel and change its **Region** dropdown in the right Inspector to mix overview panels with high-magnification insets.

## Artboard & panel styling

- **Background:** switch between **Dark**, **White** (journal print), and **Black** (fluorescence presentation).
- **Columns:** arrange panels in `1`, `2`, `3`, or `4` columns with consistent gutters.
- **Panel letters:** toggle automatic `A`, `B`, `C`… badges.
- **Titles & subtitles:** edit the Figure title/subtitle and each panel's title/subtitle live in the right Inspector.
- **Calibrated scale bars:** toggle per-panel scale bars, pick any corner (`BL`, `BR`, `TL`, `TR`), and choose **Auto** (computes a clean round length from the slide's `pixel_size_um`) or a manual length (`25 µm` to `1 mm`).

## Exporting

- **Export PNG** — composites the entire multi-panel artboard (background, gutters, crisp rendered panels, and burned-in scale bars) into a high-resolution `.png`.
- **Export SVG** — produces a vector `.svg` with embedded high-resolution panel PNGs plus editable vector `<text>` titles, panel letters, and calibrated scale bars ready for Adobe Illustrator, Inkscape, or Affinity Designer.

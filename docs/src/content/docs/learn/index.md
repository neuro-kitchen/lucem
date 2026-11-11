---
title: Learn
description: What lumen is, and where to begin.
sidebar:
  order: 0
---

lumen analyses large microscopy images on your GPU: whole-slide brightfield (H&E, IHC) and
fluorescence images, 2D and 3D. It comes as three things that share one engine:

| | What it's for |
|---|---|
| The app (`lumen-app`) | Look at images, draw shapes, find tissue and cells, build pipelines, read the results |
| The command line (`lumen`) | Describe images, run pipelines and analyses headless, serve images to a lab |
| Model plugins (`lumen-ml-*`) | Deep-learning models such as StarDist and SAM, run as separate programs |

What you do in the app is recorded as a **session**. The same session runs again, without the
app, over a folder of slides: `lumen session run session.json slides/*.ome.zarr`.

## Start here

1. [Install](/lumen/learn/get-started/install/): build the app and the command line.
2. [Your first analysis](/lumen/learn/get-started/first-analysis/): open an image, count cells,
   read the table, and run the same analysis on more images.
3. [Concepts](/lumen/learn/concepts/images/): images and pyramids, projects, the pipeline,
   sessions.

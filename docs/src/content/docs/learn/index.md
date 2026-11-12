---
title: Learn
description: What lucem is, and where to begin.
sidebar:
  order: 0
---

lucem analyses large microscopy images on your GPU: whole-slide brightfield (H&E, IHC) and
fluorescence images, 2D and 3D. It comes as three things that share one engine:

| | What it's for |
|---|---|
| The app (`lucem-app`) | Look at images, draw shapes, find tissue and cells, build pipelines, read the results |
| The command line (`lucem`) | Describe images, run pipelines and analyses headless, serve images to a lab |
| Model plugins (`lucem-ml-*`) | Deep-learning models such as StarDist and SAM, run as separate programs |

What you do in the app is recorded as a **session**. The same session runs again, without the
app, over a folder of slides: `lucem session run session.json slides/*.ome.zarr`.

## Start here

1. [Install](/lucem/learn/get-started/install/): build the app and the command line.
2. [Your first analysis](/lucem/learn/get-started/first-analysis/): open an image, count cells,
   read the table, and run the same analysis on more images.
3. [Concepts](/lucem/learn/concepts/images/): images and pyramids, projects, the pipeline,
   sessions.

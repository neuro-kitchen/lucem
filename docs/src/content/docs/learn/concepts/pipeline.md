---
title: The pipeline
description: Processing as a graph of typed steps, computed on the GPU and shown as layers.
sidebar:
  order: 3
---

The Pipeline is a graph. Data flows along its edges, and every edge carries one type:

| Type | What it is |
|---|---|
| `image` | Values per pixel (one channel, scaled 0–1) |
| `mask` | On or off per pixel |
| `labels` | An object id per pixel |
| `table` | One row per object |

Nodes are:

- **Sources**: a channel of an image, or the annotations of a class as a mask.
- **Operations**: filters, thresholds, morphology, segmentation, measurement. Each has typed
  inputs and one typed output, so a wrong connection is refused before anything runs. See
  [Operations](/lucem/reference/operations/).
- **Views**: a result drawn over the image in Explore and listed in Analyze.
- **Scripts**: a Python or Rust file as a step; edit it, save it, it runs again.
- **Models**: a model plugin as a step (for example StarDist nuclei → labels), run from the
  Inspector's **Run** button.

## Processing a channel

Adding a step to a channel (Layers → the channel → **Processing** → **Add step**, or the
**Process** menu) builds a chain for it in the graph. The processed result is drawn in the
channel's colour instead of the raw channel; **Raw** shows the original again. Nothing in the
image file changes.

## Headless

The same graph runs without the app over a whole image, block by block:

```sh
lucem run pipeline.json image.ome.zarr result.ome.zarr
```

---
title: Models
description: Deep-learning models as plugins.
sidebar:
  order: 0
---

Models run as separate programs (`lumen-ml-<family>`), one per model family, each under its
model's own licence and with its own runtime. lumen finds them in `$LUMEN_PLUGINS`,
`~/.local/share/lumen/plugins`, next to the app, and on `PATH`.

| Plugin | Models | Task | Licence |
|---|---|---|---|
| `lumen-ml-stardist` | StarDist H&E (`stardist-he`), StarDist fluorescence (`stardist-fluo`) | Nuclei as labels | BSD-3-Clause |
| `lumen-ml-sam` | Segment Anything 2 | An object from a click or a box | Apache-2.0 |

## Installing weights

**Settings → Models** lists each plugin, its runtime and its models. **Install** downloads a
model's weights and checks their SHA-256 before keeping them, in `$LUMEN_MODELS` or
`~/.cache/lumen/models`.

## StarDist

The published StarDist models, run on lumen's GPU stack (Burn on CubeCL). The Keras weights from
Zenodo are converted when installed; the outputs match the published ones.

| Model | Images | Source |
|---|---|---|
| `stardist-he` | Brightfield H&E (RGB), about 0.5 µm per pixel | Zenodo 6338615 |
| `stardist-fluo` | One fluorescence channel (the nuclear one) | Zenodo 6348085 |

Large regions run in tiles of 2,048 pixels with overlaps, so a whole slide fits in GPU memory.

## Segment Anything

The **Segment** tool (<kbd>S</kbd>) outlines what you click or box; it can carry the outline
through the planes of a volume. SAM runs on ONNX Runtime; **Install GPU runtime** adds its CUDA
build where cuDNN is found.

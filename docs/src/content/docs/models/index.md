---
title: Models
description: Deep-learning models as plugins.
sidebar:
  order: 0
---

Models run as separate programs (`lucem-ml-<family>`), one per model family, each under its
model's own licence and with its own runtime. lucem finds them in `$LUCEM_PLUGINS`,
`~/.local/share/lucem/plugins`, next to the app, and on `PATH`.

| Plugin | Models | Task | Licence |
|---|---|---|---|
| `lucem-ml-stardist` | StarDist H&E (`stardist-he`), StarDist fluorescence (`stardist-fluo`) | Nuclei as labels | BSD-3-Clause |
| `lucem-ml-instanseg` | InstanSeg brightfield (`instanseg-brightfield`), InstanSeg single-channel (`instanseg-fluorescence`) | Nuclei as labels | Apache-2.0 |
| `lucem-ml-sam` | Segment Anything 2 | An object from a click or a box | Apache-2.0 |

From Python each is an extra: `pip install "lucem[stardist]"`, `"lucem[instanseg]"`,
`"lucem[sam]"`, or `"lucem[ml]"` for all of them.

## Installing weights

**Settings → Models** lists each plugin, its runtime and its models. **Install** downloads a
model's weights and checks their SHA-256 before keeping them, in `$LUCEM_MODELS` or
`~/.cache/lucem/models`.

## StarDist

The published StarDist models, run on lucem's GPU stack (Burn on CubeCL). The Keras weights from
Zenodo are converted when installed; the outputs match the published ones.

| Model | Images | Source |
|---|---|---|
| `stardist-he` | Brightfield H&E (RGB), about 0.5 µm per pixel | Zenodo 6338615 |
| `stardist-fluo` | One fluorescence channel (the nuclear one) | Zenodo 6348085 |

Large regions run in tiles of 2,048 pixels with overlaps, so a whole slide fits in GPU memory.

## InstanSeg

The published InstanSeg nucleus models (Goldsborough, Bankhead et al. 2024), run on Burn like
StarDist. Each bioimage.io package from InstanSeg's GitHub releases is checked against its
SHA-256, and its TorchScript weights are read without PyTorch and kept as safetensors. The
postprocessing follows the published TorchScript's own: seeds, InstanSeg's pixel classifier
around each, then overlapping seeds joined. Images are resampled to the models' 0.5 µm per pixel
when their pixel size is known.

| Model | Images | Source |
|---|---|---|
| `instanseg-brightfield` | Brightfield H&E and DAB (RGB) | `brightfield_nuclei` v0.1.1 |
| `instanseg-fluorescence` | One fluorescence channel (the nuclear one) | `single_channel_nuclei` v0.1.2 |

The multiplexed "nuclei and cells" model uses another network, the channel-invariant one, and
is not available yet.

## Segment Anything

The **Segment** tool (<kbd>S</kbd>) outlines what you click or box; it can carry the outline
through the planes of a volume. SAM runs on ONNX Runtime; **Install GPU runtime** adds its CUDA
build where cuDNN is found.

---
title: Command line
description: The lucem command.
sidebar:
  order: 0
---

```text
lucem info <image>                     Describe an image
lucem doctor                           Which compute runtimes are usable
lucem autotune [--runtime NAME] [--sides 256,1024,2048] [--operators-only | --models-only]
                                       Tune this device once: the operators and every model
lucem run <spec.json> <image> <output> [--block N] [--runtime NAME]
                                       Run a pipeline over a whole image
lucem sample <output.ome.zarr> [--size N] [--depth N]
                                       Write a synthetic image (a volume with --depth)
lucem ops [--json]                     Every pipeline operation: family, types, parameters
lucem hist stains | tissue | cells <image> …
                                       Histology: stain vectors, tissue, cells
lucem session run <session.json> <image>... [--out DIR] [--runtime NAME]
                                       Run an analysis saved by the app on many images
lucem serve [--data DIR] [--bind ADDR] …
                                       Serve the images under DIR to enrolled clients
lucem clients enroll | list | revoke   Who may connect to `serve`
lucem fetch <host:port> …              List a server's images, or read a tile of one
```

`--runtime` (or `LUCEM_RUNTIME`) chooses `cuda`, `wgpu` or `cpu`; by default the first usable.

## Tuning a device

The first time an operator or a model meets a device and an image size, CubeCL compiles its
kernels and benchmarks the candidates it has (a model's convolutions: every layer, at every tile
shape). The choices are kept in the user cache folder, so it happens once, but that once can take
minutes on a first image. `lucem autotune` does it ahead of time for everything: lucem's operators
over a grid of image sizes, then every model plugin found (each runs `<plugin> tune`: its installed
models on every tile shape), on every usable runtime. The app has the same as **File › Tune this
device**, and Python as `lucem.runtime.autotune()`.

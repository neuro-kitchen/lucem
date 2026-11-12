---
title: Command line
description: The lucem command.
sidebar:
  order: 0
---

```text
lucem info <image>                     Describe an image
lucem doctor                           Which compute runtimes are usable
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

---
title: Command line
description: The lumen command.
sidebar:
  order: 0
---

```text
lumen info <image>                     Describe an image
lumen doctor                           Which compute runtimes are usable
lumen run <spec.json> <image> <output> [--block N] [--runtime NAME]
                                       Run a pipeline over a whole image
lumen sample <output.ome.zarr> [--size N] [--depth N]
                                       Write a synthetic image (a volume with --depth)
lumen ops [--json]                     Every pipeline operation: family, types, parameters
lumen hist stains | tissue | cells <image> …
                                       Histology: stain vectors, tissue, cells
lumen session run <session.json> <image>... [--out DIR] [--runtime NAME]
                                       Run an analysis saved by the app on many images
lumen serve [--data DIR] [--bind ADDR] …
                                       Serve the images under DIR to enrolled clients
lumen clients enroll | list | revoke   Who may connect to `serve`
lumen fetch <host:port> …              List a server's images, or read a tile of one
```

`--runtime` (or `LUMEN_RUNTIME`) chooses `cuda`, `wgpu` or `cpu`; by default the first usable.

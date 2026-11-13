---
title: lucem run
description: A pipeline over a whole image, headless.
sidebar:
  order: 3
---

```text
lucem run <spec.json> <image> <output> [--block N] [--runtime NAME]
```

Runs a pipeline spec over the whole image, block by block (`--block`: the block side in pixels),
and writes the result as OME-Zarr. `lucem ops --json` lists the operations a spec can use.

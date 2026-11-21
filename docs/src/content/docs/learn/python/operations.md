---
title: Operations & pipelines
description: "Run GPU operations on NumPy arrays, one by one or chained on the device."
sidebar:
  order: 2
---

## One operation at a time

Each call uploads the NumPy array to the active CubeCL runtime (`cuda`, `wgpu`, or `cpu`), runs the kernel, and returns a NumPy array:

```python
import lucem

smooth = lucem.ops.gaussian_blur(image, sigma=1.5)
flat = lucem.ops.rolling_ball(smooth, radius=32.0)
mask = lucem.ops.otsu_threshold(flat)
```

## Chained on the device (`Pipeline`)

Calling three operators separately transfers pixels to and from the GPU three times. A `lucem.ops.Pipeline` keeps intermediate buffers on the device for one upload and one download, and automatically computes the combined `halo` for tiled execution:

```python
from lucem.ops import Pipeline, stage

pipe = Pipeline([
    stage.gaussian_blur(sigma=1.5),
    stage.rolling_ball(radius=32.0),
    stage.otsu_threshold(),
])

print("halo (px):", pipe.halo)
mask = pipe.process(image)
```

## Sharing pipelines as `.lucem-recipe.json`

In the Lucem **Pipeline** workspace, click **Make recipe** in the top floating toolbar (or unselected Inspector) to export the enabled GPU op chain as a `.lucem-recipe.json` file stored in `~/.config/lucem/recipes/` (or copied to your clipboard in standalone browser mode). Anyone can drop that JSON file into their `recipes/` folder and apply the exact same GPU chain with one click.

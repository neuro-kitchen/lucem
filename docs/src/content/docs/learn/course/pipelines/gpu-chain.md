---
title: "On-device GPU chains & recipes"
description: "Run multi-step chains in one GPU pass and export shareable .lucem-recipe.json files."
editUrl: https://github.com/neuro-kitchen/lucem/edit/main/playground/04_pipelines/01_gpu_chain.py
sidebar:
  order: 1
---

`lucem.ops.Pipeline` keeps intermediate buffers on the GPU so a multi-step chain does one upload,
all stages on device, and one download — and reports the combined `halo` needed for tiled whole-slide
runs.

```python
from lucem.ops import Pipeline, stage

pipe = Pipeline([
    stage.gaussian_blur(sigma=1.5),
    stage.rolling_ball(radius=30.0),
    stage.otsu_threshold(),
])
print("combined halo (px):", pipe.halo)

mask = pipe.process(image)
```

In the Lucem **Pipeline** workspace, click **Make recipe** to export the same chain as a
`.lucem-recipe.json` file that colleagues can drop into `~/.config/lucem/recipes/` and reuse with
one click.

The whole script is [`playground/04_pipelines/01_gpu_chain.py`](https://github.com/neuro-kitchen/lucem/blob/main/playground/04_pipelines/01_gpu_chain.py). Run it with `uv run python playground/04_pipelines/01_gpu_chain.py`.

---
title: lucem.ops
description: "Run any of Lucem's 52 CubeCL GPU operators individually or chained on-device."
sidebar:
  order: 3
---

Every operator in [Operations](/lucem/reference/operations/) is exposed as both a direct function (`lucem.ops.<name>`) and a `Pipeline` stage (`lucem.ops.stage.<name>`).

```python
import lucem
from lucem.ops import Pipeline, stage

# Direct call (uploads -> runs -> downloads)
blurred = lucem.ops.gaussian_blur(tile, sigma=2.0)

# On-device chain (one upload -> all stages on GPU -> one download)
pipe = Pipeline([
    stage.gaussian_blur(sigma=1.5),
    stage.rolling_ball(radius=30.0),
    stage.otsu_threshold(),
])
mask = pipe.process(tile)
print("combined halo (px):", pipe.halo)
```

Every direct call and `Pipeline.process(..., runtime=...)` accepts an optional `runtime="cuda" | "wgpu" | "cpu"` override.

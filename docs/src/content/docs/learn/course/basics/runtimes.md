---
title: "Runtimes"
description: "Run the same CubeCL operator on cuda, wgpu, and cpu and verify numerical agreement."
editUrl: https://github.com/neuro-kitchen/lucem/edit/main/playground/01_basics/01_runtimes.py
sidebar:
  order: 1
---

lucem runs every operator on a *runtime*: `cuda` (NVIDIA), `wgpu` (Vulkan / Metal / DirectX, any
GPU), or `cpu`. The same call produces the same numerical result on each; only the execution time
differs.

**Which runtimes this build has, and which one is active**

```python
import time
import numpy as np
import lucem

print("available:", lucem.runtime.available())
print("in use:   ", lucem.runtime.current())
```

**Run `gaussian_blur` across every usable runtime and compare**

```python
rng = np.random.default_rng(42)
yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
image = np.zeros((512, 512), dtype=np.float32)
for cy, cx, r in [(120, 140, 22), (240, 310, 18), (380, 190, 26), (340, 390, 20)]:
    image += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * r**2))
image = np.clip(image + rng.normal(0.0, 0.04, image.shape).astype(np.float32), 0.0, 1.0)

results = {}
for rt in lucem.runtime.available():
    try:
        lucem.ops.gaussian_blur(image, sigma=2.5, runtime=rt)  # warm kernel cache
    except RuntimeError as exc:
        print(f"{rt}: not usable here ({exc})")
        continue
    t0 = time.perf_counter()
    results[rt] = lucem.ops.gaussian_blur(image, sigma=2.5, runtime=rt)
    print(f"{rt:6s}: {(time.perf_counter() - t0) * 1000:.2f} ms")
```

The whole script is [`playground/01_basics/01_runtimes.py`](https://github.com/neuro-kitchen/lucem/blob/main/playground/01_basics/01_runtimes.py). Run it with `uv run python playground/01_basics/01_runtimes.py`.

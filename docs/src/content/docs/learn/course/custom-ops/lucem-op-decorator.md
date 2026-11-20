---
title: "Writing @lucem.op tiled filters"
description: "Decorate a typed Python function once and run it seamlessly over tiled OME-Zarr slides and in the Lucem app."
editUrl: https://github.com/neuro-kitchen/lucem/edit/main/playground/05_custom_ops/01_lucem_op_decorator.py
sidebar:
  order: 1
---

`@lucem.op(family="Filter", halo=12)` turns a standard NumPy / SciPy function into a halo-aware tiled
operator that works identically in Python scripts and inside the Lucem app's **Pipeline** and
**Code** workspaces:

```python
import numpy as np
from scipy.ndimage import gaussian_filter
import lucem


@lucem.op(family="Filter", halo=12)
def local_contrast_boost(
    image: np.ndarray,
    sigma: float = 3.0,
    strength: float = 1.8,
) -> np.ndarray:
    """Enhance fine cellular structures by subtracting a blurred local background."""
    img = image.astype(np.float32, copy=False)
    bg = gaussian_filter(img, sigma=max(float(sigma), 0.2))
    enhanced = img + float(strength) * (img - bg)
    return np.clip(enhanced, float(img.min()), float(img.max())).astype(np.float32)
```

**Verify tiled execution matches whole-image execution across tile borders**

```python
img = lucem.io.open(zarr_path)
tiled = local_contrast_boost.apply(img, sigma=3.0, strength=1.8)
whole = local_contrast_boost(src, sigma=3.0, strength=1.8)

assert np.max(np.abs(tiled[16:-16, 16:-16] - whole[16:-16, 16:-16])) < 1e-3
```

The whole script is [`playground/05_custom_ops/01_lucem_op_decorator.py`](https://github.com/neuro-kitchen/lucem/blob/main/playground/05_custom_ops/01_lucem_op_decorator.py). Run it with `uv run python playground/05_custom_ops/01_lucem_op_decorator.py`.

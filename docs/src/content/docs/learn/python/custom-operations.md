---
title: Custom operations (@lucem.op)
description: "Write a typed Python function once and run it tiled in Python or live inside the Lucem app."
sidebar:
  order: 3
---

`@lucem.op` lets you write an ordinary NumPy / SciPy function that operates on a single 2D tile while Lucem handles everything else: reading OME-Zarr / TIFF tiles, adding `halo` border overlap, zero-copy `/dev/shm` transfer, cropping the halo back off, and stitching tiles cleanly.

## One function, two places to run

```python
# /// script
# dependencies = ["numpy", "scipy"]
# ///
import numpy as np
from scipy.ndimage import gaussian_filter
import lucem


@lucem.op(family="Filter", halo=12)
def local_contrast_boost(
    image: np.ndarray,
    sigma: float = 3.0,
    strength: float = 1.8,
) -> np.ndarray:
    """Enhances fine cellular structures by subtracting a blurred local background."""
    img = image.astype(np.float32, copy=False)
    bg = gaussian_filter(img, sigma=max(float(sigma), 0.2))
    enhanced = img + float(strength) * (img - bg)
    return np.clip(enhanced, float(img.min()), float(img.max())).astype(np.float32)
```

### 1. Run it in Python over gigapixel slides

```python
img = lucem.io.open("slide.ome.zarr")

# One channel into memory (tiled with 12 px halo overlap)
result = local_contrast_boost.apply(img, sigma=3.0, strength=1.8)

# Or stream the entire slide directly to a new OME-Zarr store
local_contrast_boost.write(img, "boosted.ome.zarr", sigma=3.0, strength=1.8)
```

### 2. Run it live inside the Lucem app & `Code` tab

Save the same file under `~/.config/lucem/plugins/ops/` (or create it in the **Code** tab):

- **Typed array args** (`image: np.ndarray`, `Mask`, `Labels`) automatically become **Pipeline input ports**.
- **Scalar args with defaults** (`sigma: float = 3.0`, `strength: float = 1.8`) automatically become **click-to-type / drag-to-scrub Inspector controls**.
- **Warm `/dev/shm` worker pool:** `lucem serve` keeps a persistent Python worker alive (`uv run --no-project`) reading `/dev/shm` `np.memmap` buffers zero-copy and hot-reloading your script by SHA-256 on every save.

:::caution
Set `halo` to at least the spatial radius of your filter (for a Gaussian blur, `ceil(3 * sigma)`). A halo that is too small produces visible seams at tile boundaries.
:::

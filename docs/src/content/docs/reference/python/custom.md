---
title: lucem.custom & @lucem.op
description: "Decorate typed Python functions for halo-padded tiled execution in Python and in the Lucem app."
sidebar:
  order: 4
---

`@lucem.op` (also available as `lucem.custom.op` and `@lucem.transform`) wraps any NumPy / SciPy tile function so Lucem handles tile scheduling, `halo` border overlap, zero-copy `/dev/shm` transfer, and halo cropping automatically.

```python
import numpy as np
from scipy.ndimage import gaussian_filter
import lucem

@lucem.op(family="Filter", halo=12)
def local_contrast_boost(image: np.ndarray, sigma: float = 3.0, strength: float = 1.8) -> np.ndarray:
    img = image.astype(np.float32, copy=False)
    bg = gaussian_filter(img, sigma=max(float(sigma), 0.2))
    return np.clip(img + float(strength) * (img - bg), float(img.min()), float(img.max())).astype(np.float32)

# Run on a single NumPy array
out = local_contrast_boost(tile, sigma=4.0, strength=2.0)

# Run tiled over an OME-Zarr pyramid (in memory or streaming to disk)
img = lucem.io.open("slide.ome.zarr")
full = local_contrast_boost.apply(img, sigma=4.0, strength=2.0)
local_contrast_boost.write(img, "boosted.ome.zarr", sigma=4.0, strength=2.0)
```

| Symbol | Description |
|---|---|
| `@lucem.op(family=..., halo=...)` | Decorator returning a `Transform` that also registers ports & Inspector parameters in the Lucem app. |
| `lucem.custom.blocks(image, *, halo=0, ...)` | Prefetching iterator yielding `(tile_array, Tile)` across an image of any size. |
| `lucem.custom.measure(image, fn, ...)` | Calls `fn(x, tile)` per block and gathers scalar results into a dict of NumPy arrays. |
| `lucem.custom.segment(image, fn, output, ...)` | Runs an instance-labeling function per block, stitches objects across tile borders, and returns a measurement table. |

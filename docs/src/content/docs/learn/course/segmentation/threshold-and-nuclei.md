---
title: "Thresholding & nuclei counting"
description: "Denoise, auto-threshold with Otsu, and count objects with connected components."
editUrl: https://github.com/neuro-kitchen/lucem/edit/main/playground/03_segmentation/01_threshold_and_nuclei.py
sidebar:
  order: 1
---

Classic fluorescence nuclei segmentation pairs Gaussian denoising with automatic thresholding
(`otsu_threshold`, `li_threshold`, or `triangle_threshold`) and connected-component labeling.

```python
import numpy as np
import lucem

yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
centers = [(100, 100), (120, 320), (260, 200), (300, 400), (420, 140), (410, 340)]
image = np.zeros((512, 512), dtype=np.float32)
for cy, cx in centers:
    image += 0.9 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 14.0**2))
image = np.clip(image, 0.0, 1.0)

denoised = lucem.ops.gaussian_blur(image, sigma=1.5)
mask = lucem.ops.otsu_threshold(denoised)
labels = lucem.ops.connected_components(mask)

assert int(labels.max()) == len(centers)
```

The whole script is [`playground/03_segmentation/01_threshold_and_nuclei.py`](https://github.com/neuro-kitchen/lucem/blob/main/playground/03_segmentation/01_threshold_and_nuclei.py). Run it with `uv run python playground/03_segmentation/01_threshold_and_nuclei.py`.

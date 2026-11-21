---
title: "Background subtraction & filtering"
description: "Remove uneven illumination with rolling ball, denoise with Gaussian blur, and boost local contrast with CLAHE."
editUrl: https://github.com/neuro-kitchen/lucem/edit/main/playground/02_filtering/01_background_and_blur.py
sidebar:
  order: 1
---

Fluorescence and brightfield images frequently combine uneven illumination (vignetting or tissue
autofluorescence) with shot noise. A classic GPU pre-processing sequence smooths noise, subtracts
slow-varying background with a rolling ball, and normalizes local contrast with CLAHE.

```python
import numpy as np
import lucem

rng = np.random.default_rng(7)
yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)

illumination = 0.35 * (xx / 512.0) + 0.25 * (yy / 512.0)
nuclei = np.zeros((512, 512), dtype=np.float32)
for cy in range(64, 512, 96):
    for cx in range(64, 512, 96):
        nuclei += 0.65 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 12.0**2))

raw = np.clip(illumination + nuclei + rng.normal(0.0, 0.03, (512, 512)).astype(np.float32), 0.0, 1.0)

smooth = lucem.ops.gaussian_blur(raw, sigma=1.5)
flat = lucem.ops.rolling_ball(smooth, radius=32.0)
boosted = lucem.ops.clahe(flat, clip_limit=2.0)
```

The whole script is [`playground/02_filtering/01_background_and_blur.py`](https://github.com/neuro-kitchen/lucem/blob/main/playground/02_filtering/01_background_and_blur.py). Run it with `uv run python playground/02_filtering/01_background_and_blur.py`.

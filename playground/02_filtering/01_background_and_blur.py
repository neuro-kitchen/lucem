# %% [markdown]
# # Filtering and Background Subtraction
#
# Fluorescence and brightfield microscopy images commonly suffer from uneven illumination
# (vignetting or autofluorescence haze) plus shot noise. Lucem provides GPU-accelerated
# Gaussian smoothing, rolling-ball background subtraction, and CLAHE local contrast enhancement.

# %%
import numpy as np

import lucem

rng = np.random.default_rng(7)
yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)

# Smooth uneven illumination ramp + bright synthetic nuclei + shot noise
illumination = 0.35 * (xx / 512.0) + 0.25 * (yy / 512.0)
nuclei = np.zeros((512, 512), dtype=np.float32)
for cy in range(64, 512, 96):
    for cx in range(64, 512, 96):
        nuclei += 0.65 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 12.0**2))

raw = np.clip(illumination + nuclei + rng.normal(0.0, 0.03, (512, 512)).astype(np.float32), 0.0, 1.0)

# %% [markdown]
# ## Denoise → Rolling ball → CLAHE

# %%
smooth = lucem.ops.gaussian_blur(raw, sigma=1.5)
flat = lucem.ops.rolling_ball(smooth, radius=32.0)
boosted = lucem.ops.clahe(flat, clip_limit=2.0)

print(f"raw     mean={raw.mean():.3f}  std={raw.std():.3f}")
print(f"flat    mean={flat.mean():.3f}  std={flat.std():.3f}  (illumination ramp removed)")
print(f"boosted mean={boosted.mean():.3f}  std={boosted.std():.3f}")

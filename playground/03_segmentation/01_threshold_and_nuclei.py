# %% [markdown]
# # Thresholding and Nuclei Segmentation
#
# Classic fluorescence nuclei segmentation combines smoothing, automatic thresholding
# (Otsu or Li), and connected-component labeling.

# %%
import numpy as np

import lucem

yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
centers = [(100, 100), (120, 320), (260, 200), (300, 400), (420, 140), (410, 340)]
image = np.zeros((512, 512), dtype=np.float32)
for cy, cx in centers:
    image += 0.9 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 14.0**2))
image = np.clip(image, 0.0, 1.0)

# %% [markdown]
# ## Smooth → Otsu threshold → Connected components

# %%
denoised = lucem.ops.gaussian_blur(image, sigma=1.5)
mask = lucem.ops.otsu_threshold(denoised)
labels = lucem.ops.connected_components(mask)

n_objects = int(labels.max())
print(f"expected {len(centers)} nuclei, found {n_objects}")
assert n_objects == len(centers)

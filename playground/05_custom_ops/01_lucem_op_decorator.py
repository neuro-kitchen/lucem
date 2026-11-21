# %% [markdown]
# # Custom Tiled Python Ops with `@lucem.op`
#
# Decorate any typed Python function with `@lucem.op(family="Filter", halo=12)`:
#
# - **In Python / Jupyter:** call `local_contrast_boost.apply(img, sigma=3.0)` or `.write(...)`
#   and Lucem walks the image tile by tile with `halo=12` pixels of overlap and trims borders back.
# - **In the Lucem app / `Code` tab:** drop the same `.py` file into `~/.config/lucem/plugins/ops/`
#   and Lucem automatically turns `image: np.ndarray` into an input port and `sigma` / `strength`
#   into live Inspector controls backed by a warm `/dev/shm` Python worker.

# %%
from pathlib import Path
import tempfile

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


# %% [markdown]
# ## Verify tiled execution matches whole-image execution with zero seams

# %%
work = Path(tempfile.mkdtemp(prefix="lucem-custom-"))
zarr_path = work / "tile_test.ome.zarr"

yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
src = np.clip(0.5 + 0.3 * np.sin(xx / 18.0) * np.cos(yy / 18.0), 0.0, 1.0)

w = lucem.io.ZarrWriter(zarr_path, ["DAPI"], (1, 512, 512), (1, 256, 256), dtype="float32")
w.write(0, (0, 0, 0), src[None, :, :])
w.finish()

img = lucem.io.open(zarr_path)
tiled = local_contrast_boost.apply(img, sigma=3.0, strength=1.8)
whole = local_contrast_boost(src, sigma=3.0, strength=1.8)

# Interior pixels (away from outer image boundary) agree closely across tile boundaries
max_diff = float(np.max(np.abs(tiled[16:-16, 16:-16] - whole[16:-16, 16:-16])))
print(f"tiled vs whole-image max diff across tile seams: {max_diff:.2e}")
assert max_diff < 1e-3

# /// script
# dependencies = ["numpy", "scipy"]
# ///
"""Example drop-in Python filter for Lucem using the `@lucem.op` decorator.

Just decorate a normal typed Python function with `@lucem.op(...)`:
- Array inputs (`image: np.ndarray`) become Pipeline input ports automatically.
- Scalar arguments with defaults (`sigma: float = 3.0`, `strength: float = 1.8`)
  become live Inspector parameters automatically.
- Lucem handles OME-Zarr reading, halo-padded tiling (`halo=12`), `/dev/shm`
  zero-copy memory sharing, and tile reassembly automatically.
"""

import lucem
import numpy as np
from scipy.ndimage import gaussian_filter


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


if __name__ == "__main__":
    y, x = np.mgrid[0:64, 0:64].astype(np.float32)
    spot = np.exp(-((x - 32.0) ** 2 + (y - 32.0) ** 2) / (2.0 * 4.0**2))
    out = local_contrast_boost(spot, sigma=3.0, strength=1.8)
    print(f"input range:  [{spot.min():.4f}, {spot.max():.4f}]")
    print(f"output range: [{out.min():.4f}, {out.max():.4f}]  shape={out.shape} dtype={out.dtype}")


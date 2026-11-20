# /// script
# dependencies = ["numpy", "scipy"]
# ///
# lucem: name = "Local Contrast Boost"
# lucem: family = "Filter"
# lucem: description = "Enhances fine cellular structures by subtracting a blurred local background"
# lucem: halo = 12
# lucem: input image: image
# lucem: param sigma = 3.0
# lucem: param strength = 1.8
# lucem: output image

"""Example drop-in Python filter for Lucem.

How Lucem reads this file
-------------------------
1. The `# /// script ... # ///` block (PEP 723) tells Lucem which Python packages
   this op needs. Lucem auto-provisions a cached `uv` venv and keeps a warm Python
   worker alive between tile runs so repeated edits execute in milliseconds.
2. The `# lucem: ...` header lines declare the node's UI metadata and graph ports:
   - `name`        — Display title in Pipeline › Script and on the node card.
   - `family`      — Menu category (`Filter`, `Restore`, `Threshold`, `Segment`, …).
   - `description` — Tooltip / Inspector blurb.
   - `halo`        — Border overlap in pixels so tiled execution has no seam artifacts.
   - `input`       — Named input port and its type (`image`, `mask`, or `labels`).
   - `param`       — Editable numeric parameter shown as a scrubber in the Inspector.
   - `output`      — Output port type (`image`, `mask`, or `labels`).
3. Lucem calls `run(...)` with one argument per `input` (as a 2D `np.ndarray`:
   `float32` for `image`, `uint32` for `mask`/`labels`) followed by each `param`.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def run(image: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """Boost local high-frequency detail relative to a Gaussian background."""
    img = image.astype(np.float32, copy=False)
    sigma = max(float(sigma), 0.2)
    strength = float(strength)

    # Smooth estimate of the local illumination / background
    background = gaussian_filter(img, sigma=sigma)

    # Add scaled local detail back onto the original image
    detail = img - background
    enhanced = img + strength * detail

    # Keep output finite and in the same general intensity range
    lo, hi = float(img.min()), float(img.max())
    return np.clip(enhanced, lo, hi).astype(np.float32)


if __name__ == "__main__":
    # Quick standalone smoke test outside Lucem
    y, x = np.mgrid[0:64, 0:64].astype(np.float32)
    spot = np.exp(-((x - 32.0) ** 2 + (y - 32.0) ** 2) / (2.0 * 4.0**2))
    out = run(spot, sigma=3.0, strength=1.8)
    print(f"input range:  [{spot.min():.4f}, {spot.max():.4f}]")
    print(f"output range: [{out.min():.4f}, {out.max():.4f}]  shape={out.shape} dtype={out.dtype}")

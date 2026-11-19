# /// script
# dependencies = ["numpy", "scipy"]
# ///
# lucem: name = "Membrane Enhance"
# lucem: family = "Filter"
# lucem: description = "Band-pass membrane enhancement using Difference of Gaussians"
# lucem: halo = 12
# lucem: input image: image
# lucem: param sigma = 1.5
# lucem: param alpha = 1.5
# lucem: output image

import numpy as np
from scipy.ndimage import gaussian_filter


def run(image, sigma, alpha):
    """Enhance bright thin membranes while suppressing diffuse background."""
    sigma = max(float(sigma), 0.2)
    fine = gaussian_filter(image.astype(np.float32), sigma=sigma)
    coarse = gaussian_filter(image.astype(np.float32), sigma=sigma * 2.5)
    out = np.clip((fine - coarse) * float(alpha), 0.0, 1.0)
    return out.astype(np.float32)

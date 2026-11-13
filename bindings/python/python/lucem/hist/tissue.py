"""Tissue on a low-resolution image of the slide: a smoothed value per pixel (how much stain for
brightfield, how bright for fluorescence) above a threshold (Otsu's by default), small pieces and
holes dropped and filled, then outlined.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from lucem._native import hist as _hist

__all__ = ["Tissue", "defaults", "detect"]


def defaults() -> dict:
    """The default settings: ``brightfield``, ``threshold`` (None: Otsu's), ``sigma_um``, ``min_area_um2``, ``min_hole_um2``."""
    return json.loads(_hist.defaults())["tissue"]


@dataclass
class Tissue:
    #: uint8 ``[h, w]``, 1 on tissue.
    mask: np.ndarray
    threshold: float
    #: The regions as objects: level-0 outlines, ``Area µm^2``.
    regions: list[dict]


def detect(image, pixel_um: float, *, downsample: float = 1.0, origin: tuple[float, float] = (0.0, 0.0), **settings) -> Tissue:
    """The tissue of a low-resolution image.

    Parameters
    ----------
    image : array
        ``[h, w]`` or ``[h, w, c]``; brightfield RGB in 0–255.
    pixel_um : float
        µm per level-0 pixel of the slide.
    downsample : float
        Level-0 pixels per pixel of ``image`` (its pyramid level's downsample).
    origin : (x, y)
        Level-0 position of ``image``'s first pixel.
    **settings
        Any of :func:`defaults` (``brightfield=False`` for fluorescence).
    """
    s = defaults()
    unknown = set(settings) - set(s)
    if unknown:
        raise TypeError(f"unknown setting(s) {sorted(unknown)}; there are {sorted(s)}")
    s.update(settings)
    mask, threshold, regions = _hist.detect_tissue(np.ascontiguousarray(image, dtype=np.float32), pixel_um, json.dumps(s), downsample, tuple(origin))
    return Tissue(mask, threshold, json.loads(regions))

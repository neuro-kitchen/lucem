"""Brightfield stains: optical density, stain vectors and colour deconvolution.

Presets are Ruifrok & Johnston's (2001); :func:`estimate` fits two stains to a region of the slide
itself (Macenko et al., 2009), which follows the actual staining better.

>>> s = lucem.hist.stains.preset("h_dab")                      # doctest: +SKIP
>>> hematoxylin, dab, residual = lucem.hist.stains.deconvolve(rgb, s)  # doctest: +SKIP
"""

from __future__ import annotations

import json

import numpy as np

from lucem._native import hist as _hist

__all__ = ["PRESETS", "background", "deconvolve", "estimate", "preset"]

#: The presets by name: ``h_dab``, ``h_e``, ``h_e_dab``.
PRESETS: dict[str, dict] = json.loads(_hist.stain_presets())


def preset(name: str) -> dict:
    """A preset's stain vectors (``h_dab``, ``h_e``, ``h_e_dab``; ``-`` and case ignored)."""
    key = name.lower().replace("-", "_")
    if key not in PRESETS:
        raise ValueError(f"unknown stains {name!r}: {sorted(PRESETS)}")
    return PRESETS[key]


def _rgb(rgb) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8)


def background(rgb) -> tuple[float, float, float]:
    """The background (white) RGB of a region, uint8 ``[h, w, 3]``."""
    return tuple(_hist.background_rgb(_rgb(rgb)))


def estimate(rgb, names: tuple[str, str] = ("Hematoxylin", "DAB"), *, min_od: float = 0.15, alpha: float = 1.0) -> dict:
    """Two stains fitted to a region holding both (uint8 ``[h, w, 3]``), and the residual."""
    x = _rgb(rgb)
    return json.loads(_hist.estimate_stains(x, list(background(x)), names[0], names[1], min_od, alpha))


def deconvolve(rgb, stains: dict | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The amount of each of the three stains per pixel (float32 ``[h, w]`` optical densities)."""
    s = preset(stains) if isinstance(stains, str) else stains
    a, b, c = _hist.deconvolve_rgb(_rgb(rgb), json.dumps(s))
    return a, b, c

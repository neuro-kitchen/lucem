"""Cells: nuclei found on a detection channel (background, smoothing, threshold, split), grown
into cells, every channel measured per compartment (on the GPU); or cells grown from a model's
nucleus labels (:func:`from_nuclei`, e.g. :meth:`lucem.ml.Plugin.instances`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from lucem._native import hist as _hist

__all__ = ["Cells", "defaults", "detect", "from_nuclei"]


def defaults() -> dict:
    """The default settings (µm, µm²): ``background_radius_um``, ``median_radius_um``, ``sigma_um``,
    ``min_area_um2``, ``max_area_um2``, ``threshold``, ``split_by_shape``, ``split_by_intensity``,
    ``cell_expansion_um``."""
    return json.loads(_hist.defaults())["cells"]


@dataclass
class Cells:
    #: One dict per cell: id, outlines (level-0 pixels), measurements.
    objects: list[dict]
    #: uint32 ``[h, w]`` nucleus and cell labels; a label is its object's ``id``.
    nuclei: np.ndarray
    cells: np.ndarray

    def __len__(self) -> int:
        return len(self.objects)


def _settings(settings: dict) -> str:
    s = defaults()
    unknown = set(settings) - set(s)
    if unknown:
        raise TypeError(f"unknown setting(s) {sorted(unknown)}; there are {sorted(s)}")
    s.update(settings)
    return json.dumps(s)


def _channels(channels: Mapping[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    return [(name, np.ascontiguousarray(v, dtype=np.float32)) for name, v in channels.items()]


def detect(detection, channels: Mapping[str, np.ndarray], pixel_um: float, *, downsample: float = 1.0, origin: tuple[float, float] = (0.0, 0.0), **settings) -> Cells:
    """Finds and measures the cells of a region.

    Parameters
    ----------
    detection : array
        float32 ``[h, w]``, nuclei bright (a hematoxylin optical density, a DAPI channel).
    channels : mapping of name to array
        The channels measured, each ``[h, w]``.
    pixel_um : float
        µm per level-0 pixel.
    downsample, origin
        Where the region lies on the slide (level-0 pixels per pixel; level-0 position).
    **settings
        Any of :func:`defaults`.
    """
    objects, nuclei, cells = _hist.detect_cells_in(np.ascontiguousarray(detection, dtype=np.float32), _channels(channels), pixel_um, _settings(settings), downsample, tuple(origin), None)
    return Cells(json.loads(objects), nuclei, cells)


def from_nuclei(nuclei, detection, channels: Mapping[str, np.ndarray], pixel_um: float, *, downsample: float = 1.0, origin: tuple[float, float] = (0.0, 0.0), **settings) -> Cells:
    """Cells grown from nucleus labels (uint32 ``[h, w]``, e.g. a model's) and measured."""
    labels = np.ascontiguousarray(nuclei, dtype=np.uint32)
    objects, nuc, cells = _hist.detect_cells_in(np.ascontiguousarray(detection, dtype=np.float32), _channels(channels), pixel_um, _settings(settings), downsample, tuple(origin), labels)
    return Cells(json.loads(objects), nuc, cells)

"""Measurements out: one row per object, a summary per region, GeoJSON."""

from __future__ import annotations

import json

from lucem._native import hist as _hist

__all__ = ["geojson", "summary", "table"]


def table(objects: list[dict], image: str, pixel_um: float) -> dict[str, list]:
    """One row per object as columns (``Image``, ``Object ID``, …, every measurement; text cells,
    numbers where they read as such): ready for ``polars.DataFrame`` or ``pandas.DataFrame``."""
    header, rows = _hist.objects_table(json.dumps(objects), image, pixel_um)

    def cell(v: str):
        try:
            return float(v)
        except ValueError:
            return v if v else None

    return {name: [cell(r[i]) if i >= 4 else (r[i] or None) for r in rows] for i, name in enumerate(header)}


def geojson(objects: list[dict]) -> dict:
    """The objects as a GeoJSON FeatureCollection (level-0 pixels)."""
    return json.loads(_hist.objects_geojson(json.dumps(objects)))


def summary(region, objects: list[dict], pixel_um: float) -> dict[str, float]:
    """What a region (an outline in level-0 pixels) reports about the objects in it: counts per
    class, positive %, H-score, densities per mm², area."""
    return json.loads(_hist.summarize_region([list(p) for p in region], json.dumps(objects), pixel_um))

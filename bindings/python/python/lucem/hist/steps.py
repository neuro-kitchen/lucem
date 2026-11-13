"""The app's steps, run from Python: build them here or read them from a ``session.json`` the app
saved, then :func:`run` them on an image through the same engine the app and ``lucem session
run`` use. A step is a dict, the session's own form, so it goes both ways.

>>> image = lucem.io.open("slide.ome.zarr")                         # doctest: +SKIP
>>> result = lucem.hist.steps.run(image, [
...     lucem.hist.steps.tissue(),
...     lucem.hist.steps.cells(stains="h_dab", within="tissue"),
... ])                                                               # doctest: +SKIP
>>> lucem.hist.steps.run(image, lucem.hist.steps.recipe("session.json"))  # doctest: +SKIP
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from lucem._native import hist as _hist
from lucem.hist import cells as _cells
from lucem.hist import stains as _stains
from lucem.hist import tissue as _tissue

__all__ = ["Result", "cells", "check", "export", "recipe", "run", "tissue"]


@dataclass
class Result:
    #: Tissue outlines (level-0 pixels).
    tissue: list = field(default_factory=list)
    #: Cells and superpixel tiles, as objects.
    cells: list[dict] = field(default_factory=list)
    tiles: list[dict] = field(default_factory=list)
    #: What each step did, one line each.
    notes: list[str] = field(default_factory=list)


def check(step: dict) -> dict:
    """The step with every default filled in (raises ``ValueError`` if it does not read)."""
    full, _ = _hist.check_step(json.dumps(step))
    return json.loads(full)


def recipe(session: str | os.PathLike) -> list[dict]:
    """The steps of a ``session.json`` to run on another image: the last of each kind, in order."""
    return json.loads(_hist.session_recipe(os.fspath(session)))


def _within(within) -> dict:
    if within in ("image", None):
        return {"in": "image"}
    if within == "tissue":
        return {"in": "tissue"}
    return {"in": "regions", "outlines": [list(map(list, o)) for o in within]}


def tissue(**settings) -> dict:
    """Find tissue (settings as :func:`lucem.hist.tissue.defaults`)."""
    s = _tissue.defaults()
    s.update(settings)
    return {"action": "tissue", "settings": s}


def cells(*, stains: str | dict | None = None, nuclear: int = 0, within="image", model: str | None = None, auto_threshold: bool = False, **settings) -> dict:
    """Find cells: brightfield with ``stains`` (a preset name or vectors), else fluorescence with
    nuclei on channel ``nuclear``; ``within`` is ``"image"``, ``"tissue"`` or a list of outlines;
    ``model`` a nucleus model of a plugin (``stardist-he``) instead of the threshold."""
    staining = {"staining": "fluorescence", "nuclear": nuclear} if stains is None else {"staining": "brightfield", "stains": _stains.preset(stains) if isinstance(stains, str) else stains}
    detection = _cells.defaults()
    detection.update(settings)
    step = {"action": "cells", "within": _within(within), "staining": staining, "detection": detection, "auto_threshold": auto_threshold}
    if model is not None:
        step["model"] = model
    return step


def export() -> dict:
    """Write everything to the output store (in the app and ``lucem session run``)."""
    return {"action": "export"}


def run(image, steps: list[dict]) -> Result:
    """Runs ``steps`` on an image (:func:`lucem.io.open`) and returns what they found."""
    native = getattr(image, "_native", image)
    out = json.loads(_hist.run_steps(native, [json.dumps(s) for s in steps]))
    return Result(out["tissue"], out["cells"], out["tiles"], out["notes"])

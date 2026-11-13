"""Where operators run: ``wgpu``, ``cuda``, ``cpu`` (or ``LUCEM_RUNTIME`` in the environment),
and tuning this machine once (:func:`autotune`).

>>> lucem.runtime.available()   # doctest: +SKIP
['wgpu', 'cuda', 'cpu']
>>> lucem.runtime.use("cpu")    # doctest: +SKIP
>>> lucem.runtime.autotune()    # doctest: +SKIP
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from lucem import _native
from lucem._native import current_runtime as current
from lucem._native import runtimes as available
from lucem._native import set_runtime as use

__all__ = ["autotune", "available", "current", "use"]


def autotune(runtime: str | None = None, *, sides: Sequence[int] | None = None, operators: bool = True, models: bool = True) -> dict:
    """Tunes this device once, as ``lucem autotune``: lucem's operators over a grid of image sides,
    then every model plugin found (each installed model on every tile shape). The choices and the
    compiled kernels are kept in lucem's cache folder, so later runs start at full speed.

    Parameters
    ----------
    runtime : str, optional
        ``wgpu``, ``cuda`` or ``cpu``; by default every runtime usable here.
    sides : sequence of int, optional
        Image sides for the operators (default 256, 1024, 2048).
    operators, models : bool
        What to tune.

    Returns
    -------
    dict
        Per runtime: ``operator_runs``, ``operator_failures``, ``seconds`` and ``plugins``
        (``{name: "tuned" | "skipped" | why it failed}``).
    """
    return json.loads(_native.autotune(runtime, list(sides) if sides is not None else None, operators, models))

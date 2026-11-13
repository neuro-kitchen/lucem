"""Where operators run: ``wgpu``, ``cuda``, ``cpu`` (or ``LUCEM_RUNTIME`` in the environment).

>>> lucem.runtime.available()   # doctest: +SKIP
['wgpu', 'cuda', 'cpu']
>>> lucem.runtime.use("cpu")    # doctest: +SKIP
"""

from lucem._native import current_runtime as current
from lucem._native import runtimes as available
from lucem._native import set_runtime as use

__all__ = ["available", "current", "use"]

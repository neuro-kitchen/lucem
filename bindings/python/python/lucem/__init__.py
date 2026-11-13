"""lucem: GPU image processing for biology, written once in CubeCL.

- :mod:`lucem.runtime` picks where operators run (``wgpu``, ``cuda``, ``cpu``).
- :mod:`lucem.io` opens images (2D or 3D, OME-Zarr, TIFF, image files) and writes OME-Zarr block by block.
- :mod:`lucem.ops` runs lucem's operators on NumPy arrays, alone or chained on the device.
- :mod:`lucem.custom` runs your own functions over an image of any size, block by block.
- :mod:`lucem.ml` finds the model plugins (``pip install "lucem[stardist]"``, ``"lucem[sam]"``) and talks to them.
- :mod:`lucem.hist` adds histology: stains, tissue, cells, the app's recorded steps (``pip install "lucem[hist]"``).
- :mod:`lucem.app` opens the desktop app (``pip install "lucem[app]"``).
"""

import importlib
import importlib.util
import sys

# `lucem[hist]` installs `lucem-hist`: this package's extension built with histology in. Only one
# extension is ever loaded, so there is one GPU device per runtime and one image type.
if importlib.util.find_spec("lucem_hist") is not None:
    sys.modules["lucem._native"] = importlib.import_module("lucem_hist._native")

from lucem import app, custom, io, ml, ops, runtime  # noqa: E402
from lucem._native import __version__  # noqa: E402

__all__ = ["__version__", "app", "custom", "io", "ml", "ops", "runtime"]
if hasattr(sys.modules["lucem._native"], "hist"):
    __all__.append("hist")


def __getattr__(name: str):
    if name == "hist":
        return importlib.import_module("lucem.hist")
    raise AttributeError(f"module 'lucem' has no attribute {name!r}")

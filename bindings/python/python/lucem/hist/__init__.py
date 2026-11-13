"""lucem.hist: histology on top of lucem (``pip install "lucem[hist]"``).

- :mod:`lucem.hist.stains`: stain vectors (presets, fitted to a region) and colour deconvolution.
- :mod:`lucem.hist.tissue`: tissue on a low-resolution image of the slide.
- :mod:`lucem.hist.cells`: nuclei and cells found and measured (GPU), or grown from a model's nuclei.
- :mod:`lucem.hist.steps`: the app's recorded steps (tissue, cells, classifiers, intensity, …), built
  in Python or read from a ``session.json``, run on an image through the engine the app runs.
- :mod:`lucem.hist.export`: measurement tables, region summaries, GeoJSON.

Objects (cells, tissue regions) are dicts with level-0 pixel outlines and measurements in µm, the
same as the app's.
"""

import lucem._native as _native

if not hasattr(_native, "hist"):
    raise ImportError('lucem.hist is not installed: pip install "lucem[hist]"')

from lucem.hist import cells, export, stains, steps, tissue  # noqa: E402

__all__ = ["cells", "export", "stains", "steps", "tissue"]

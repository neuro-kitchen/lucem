"""lucem's operators on arrays, run on the chosen runtime (see :mod:`lucem.runtime`).

One-shot functions upload their input, run and return NumPy: ``lucem.ops.gaussian_blur(x, sigma=2)``.
A :class:`Pipeline` chains :mod:`lucem.ops.stage` stages on the device: one upload, every stage, one
download, which is what makes it fast on a GPU::

    pipe = lucem.ops.Pipeline([stage.gaussian_blur(sigma=2), stage.threshold(value=0.2), stage.open(radius=2)])
    mask = pipe(image)

Images are float32; masks and labels are uint32; a table is a dict of columns.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from lucem import _native
from lucem.ops import stage

_DTYPE = {"image": np.float32, "mask": np.uint32, "labels": np.uint32}


class Pipeline:
    """Stages run on the device: one upload, every stage, one download.

    Parameters
    ----------
    stages : sequence of Stage
        One-input operators, each taking what the previous one gives (:mod:`lucem.ops.stage`).
    """

    def __init__(self, stages: Sequence[stage.Stage]):
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        self.stages = list(stages)
        self._specs = [s.spec for s in self.stages]
        self._input = _DTYPE[self.stages[0].input_type]

    @property
    def halo(self) -> int:
        """Pixels of context the chain may look at around a pixel: pass it to ``lucem.custom``."""
        return sum(s.halo for s in self.stages)

    def __call__(self, x: np.ndarray):
        return _native.run_chain(self._specs, np.asarray(x, dtype=self._input))

    def __repr__(self) -> str:
        return "Pipeline(" + ", ".join(s.tag for s in self.stages) + ")"


def _function(entry: dict):
    tag, inputs, defaults = entry["tag"], entry["inputs"], entry["params"]

    def call(*arrays, **params):
        if len(arrays) != len(inputs):
            raise TypeError(f"{tag} takes {len(inputs)} array(s), {len(arrays)} given")
        unknown = set(params) - set(defaults)
        if unknown:
            raise TypeError(f"{tag}: unknown parameter(s) {sorted(unknown)}; it takes {sorted(defaults)}")
        spec = json.dumps({"op": tag, **defaults, **params})
        typed = [np.asarray(a, dtype=_DTYPE[p["type"]]) for a, p in zip(arrays, inputs)]
        return _native.run_op(spec, typed)

    call.__name__ = tag
    ports = ", ".join(f"{p['name']} ({p['type']})" for p in inputs)
    settings = ", ".join(f"{k}={v!r}" for k, v in defaults.items())
    call.__doc__ = f"{entry['title']}: takes {ports}, gives {entry['output']}." + (f"\n\nDefaults: {settings}." if settings else "")
    return call


__all__ = ["Pipeline", "stage"]
for _entry in stage.CATALOG:
    globals()[_entry["tag"]] = _function(_entry)
    __all__.append(_entry["tag"])
del _entry

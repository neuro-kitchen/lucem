"""Stages of a :class:`lucem.ops.Pipeline`: one per one-input operator of the catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass

from lucem import _native

CATALOG: list[dict] = json.loads(_native.op_catalog())


@dataclass(frozen=True)
class Stage:
    """One operator with its parameters."""

    tag: str
    params: tuple
    input_type: str
    output_type: str

    @property
    def spec(self) -> str:
        return json.dumps({"op": self.tag, **dict(self.params)})

    @property
    def halo(self) -> int:
        """Pixels of context the stage reads around a pixel."""
        return _native.op_halo(self.spec)


def _stage(entry: dict):
    tag, defaults = entry["tag"], entry["params"]

    def make(**params) -> Stage:
        unknown = set(params) - set(defaults)
        if unknown:
            raise TypeError(f"{tag}: unknown parameter(s) {sorted(unknown)}; it takes {sorted(defaults)}")
        return Stage(tag, tuple({**defaults, **params}.items()), entry["inputs"][0]["type"], entry["output"])

    make.__name__ = tag
    make.__doc__ = entry["title"]
    return make


__all__ = ["CATALOG", "Stage"]
for _entry in CATALOG:
    if len(_entry["inputs"]) == 1:
        globals()[_entry["tag"]] = _stage(_entry)
        __all__.append(_entry["tag"])
del _entry

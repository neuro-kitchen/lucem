"""Model plugins: separate programs named ``lucem-ml-<family>``, each with its models, licence and
runtime (``pip install "lucem[stardist]"``, ``"lucem[sam]"``, or ``"lucem[ml]"`` for all).

lucem never links a model: a plugin runs as its own process and answers JSON requests. Weights and
runtimes are downloaded (and checked) the first time a model is used, into lucem's cache.

>>> for p in lucem.ml.plugins():          # doctest: +SKIP
...     print(p["name"], [m["id"] for m in p["models"]])
>>> with lucem.ml.Plugin("lucem-ml-stardist") as p:      # doctest: +SKIP
...     labels = p.instances("stardist-he", rgb, pixel_um=0.5)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from lucem import _native

__all__ = ["Plugin", "plugin_dirs", "plugins", "problems"]


def _found() -> list[dict]:
    return json.loads(_native.ml_plugins())


def plugins() -> list[dict]:
    """The plugins found, each its manifest: ``name``, ``version``, ``licence``, ``runtime``, ``models``, ``program``."""
    return [entry["ok"] for entry in _found() if "ok" in entry]


def problems() -> list[str]:
    """Why each plugin program that was found but did not answer could not be used."""
    return [entry["error"] for entry in _found() if "error" in entry]


def plugin_dirs() -> list[Path]:
    """The folders searched for plugin programs, in order (then ``PATH``)."""
    return [Path(p) for p in _native.ml_plugin_dirs()]


class Plugin:
    """A plugin program in server mode: its model stays loaded between requests.

    Parameters
    ----------
    program : str
        A path, or a name found on ``PATH`` (``lucem-ml-stardist``).
    """

    def __init__(self, program: str):
        self._native = _native.Plugin(program)

    @property
    def program(self) -> str:
        return self._native.program

    def alive(self) -> bool:
        """Whether the program is still running."""
        return self._native.alive()

    def request(self, request: dict) -> dict:
        """Sends a request of the plugin protocol and returns its answer."""
        return json.loads(self._native.request(json.dumps(request)))

    def info(self) -> dict:
        """The plugin's manifest: its models, licence and runtime."""
        return self.request({"request": "info"})

    def install(self, model: str) -> dict:
        """Downloads and verifies a model's weights (done by itself on first use otherwise)."""
        return self.request({"request": "install", "model": model})

    def instances(self, model: str, image: np.ndarray, *, pixel_um: float | None = None) -> np.ndarray:
        """Every object of an image as labels (uint32 ``[h, w]``, 0: background).

        Parameters
        ----------
        model : str
            A model id of this plugin (``stardist-he``, ``stardist-fluo``).
        image : ndarray
            uint8 ``[h, w]`` (one channel) or ``[h, w, 3]`` (RGB). Scale other types to 0–255 first.
        pixel_um : float, optional
            µm per pixel: the model resamples to the scale it was trained at.
        """
        image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            raise TypeError(f"expected a uint8 image, got {image.dtype}: scale it to 0–255 first")
        if image.ndim == 2:
            (h, w), channels = image.shape, 1
        elif image.ndim == 3 and image.shape[2] == 3:
            h, w, channels = image.shape
        else:
            raise ValueError(f"expected [h, w] or [h, w, 3], got shape {image.shape}")
        with tempfile.TemporaryDirectory(prefix="lucem-ml-") as tmp:
            raw = os.path.join(tmp, "image.raw")
            image.tofile(raw)
            answer = self.request({
                "request": "instances",
                "model": model,
                "image": {"path": raw, "width": w, "height": h, "channels": channels},
                "pixel_um": pixel_um,
                "out_dir": tmp,
            })
            out = answer.get("labels") or {}
            path = out.get("path")
            if not path or not os.path.exists(path):
                return np.zeros((h, w), dtype=np.uint32)
            labels = np.fromfile(path, dtype="<u4")
        return labels.reshape(int(out["height"]), int(out["width"])).astype(np.uint32, copy=False)

    def close(self) -> None:
        """Stops the program."""
        self._native = None

    def __enter__(self) -> Plugin:
        return self

    def __exit__(self, *_) -> None:
        self.close()

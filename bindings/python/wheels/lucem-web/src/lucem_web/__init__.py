"""Bundled static WebAssembly assets for ``lucem[web]``."""

from __future__ import annotations

from pathlib import Path

__all__ = ["dist_dir"]


def dist_dir() -> Path | None:
    """Return the bundled ``dist/`` directory if populated, else ``None``."""
    d = Path(__file__).resolve().parent / "dist"
    return d if (d / "index.html").is_file() else None

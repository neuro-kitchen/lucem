"""The desktop app, opened from Python (``pip install "lucem[app]"``)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["open", "program"]

_NAME = "lucem-app"


def program() -> Path | None:
    """Where the ``lucem-app`` program is, or ``None`` when it is not installed.

    Returns
    -------
    pathlib.Path or None
        Next to this Python first (``lucem[app]`` puts it there), else from ``PATH``.
    """
    beside = Path(sys.executable).parent / (_NAME + (".exe" if os.name == "nt" else ""))
    if beside.is_file():
        return beside
    found = shutil.which(_NAME)
    return Path(found) if found else None


def open(path: str | os.PathLike | None = None) -> subprocess.Popen:  # noqa: A001 (as lucem.io.open)
    """Opens the desktop app, optionally on an image or a project, and returns without waiting for it.

    Parameters
    ----------
    path : str or os.PathLike, optional
        The image or project folder to open; by default the app starts as usual.

    Returns
    -------
    subprocess.Popen
        The app's process.
    """
    exe = program()
    if exe is None:
        raise FileNotFoundError('the lucem app is not installed: pip install "lucem[app]"')
    args = [str(exe)] + ([os.fspath(path)] if path is not None else [])
    return subprocess.Popen(args)

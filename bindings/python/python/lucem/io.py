"""Opening images (2D or 3D) and writing results block by block."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np

from lucem._native import Image, ZarrStore

__all__ = ["Image", "ZarrWriter", "open", "sample_volume"]


def open(path: str | PathLike) -> Image:  # noqa: A001
    """Opens an OME-Zarr container, a TIFF stack or an image file.

    Nothing is decoded until it is read: ``image.read_block(level, channel, origin, size)`` reads any
    box and ``image.read_plane(level, channel, axis, index)`` a plane along any axis.
    """
    return Image(path)


def sample_volume(size: int = 128, depth: int = 64) -> Image:
    """A synthetic two-channel volume, for trying things out."""
    return Image.sample_volume(size, depth)


class ZarrWriter:
    """Writes an OME-Zarr image chunk by chunk, without holding it in memory.

    Parameters
    ----------
    path : path
        The store to create. It must not exist.
    channels : sequence of str
        A name per channel.
    shape : (depth, height, width)
        Depth 1 is a 2D image.
    chunks : (z, y, x)
        The chunk size: every :meth:`write` is one whole chunk.
    dtype : {"float32", "uint32"}
    pixel_size_um, z_spacing_um : float, optional
    """

    def __init__(self, path, channels: Sequence[str], shape, chunks, dtype: str = "float32", name: str = "image", pixel_size_um=None, z_spacing_um=None):
        self._store = ZarrStore(path, list(channels), list(shape), list(chunks), dtype, name, pixel_size_um, z_spacing_um)
        self.dtype = np.dtype(dtype)

    def write(self, channel: int, origin, data: np.ndarray) -> None:
        """Stores a whole chunk (2D or 3D) of ``channel`` at ``origin`` ``(z, y, x)``."""
        self._store.write(channel, list(origin), np.asarray(data, dtype=self.dtype))

    def remap_labels(self, mapping: np.ndarray) -> None:
        """Replaces each label ``l`` by ``mapping[l]`` (labels stores only)."""
        self._store.remap_labels(np.asarray(mapping, dtype=np.uint32))

    def finish(self):
        """Builds the coarser levels, writes the metadata and returns the store's path."""
        return self._store.finish()

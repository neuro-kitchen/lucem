"""Custom operations: your own Python functions over an image of any size, block by block.

lucem reads the image in blocks (the next ones are read on another thread while your function
runs), so an image of any size goes through a function that only ever sees one block:

- :func:`measure` calls ``fn(x, tile)`` on each block and gathers what it returns into a table.
- :func:`transform` wraps ``fn(x)`` (or ``fn(x, tile)``) that returns the block processed, same
  shape. Blocks are read with ``halo`` pixels of context and the result is trimmed back, so the
  output equals ``fn`` on the whole array for any ``fn`` that looks at most ``halo`` pixels away.
- :func:`segment` wraps ``fn`` returning labels; objects cut by block borders are joined, and it
  returns one table row per object.

``x`` is float32 NumPy, ``[h, w]`` for a plane or ``[d, h, w]`` with ``volume=True``. A
:class:`lucem.ops.Pipeline` is a valid ``fn``: it stays on the GPU between its stages (pass
``halo=pipe.halo``). Any other Python library works too, more slowly.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import PathLike

import numpy as np

from lucem import _native
from lucem import io as _io

__all__ = ["Tile", "Transform", "blocks", "measure", "segment", "transform"]


@dataclass(frozen=True)
class Tile:
    """Where a block sits (positions are ``(z, y, x)`` at the level walked)."""

    index: int
    channel: int
    level: int
    #: The part this block owns.
    origin: tuple
    size: tuple
    #: The part that was read: the owned part plus the halo, clipped to the image.
    read_origin: tuple
    read_size: tuple
    #: Size of a pixel of this level in µm, if known.
    pixel_size_um: float | None = None

    @property
    def inner(self) -> tuple:
        """Slices of the read block selecting the owned part, for a block of this tile's shape."""
        slices = tuple(slice(o - r, o - r + s) for o, r, s in zip(self.origin, self.read_origin, self.size))
        return slices[1:] if self.read_size[0] == 1 else slices


def _schedule(image, level: int, volume: bool, block: int | None, block_z: int | None, halo: int, halo_z: int | None):
    depth, height, width, tile_d, tile_h, tile_w, _ = image.levels[level]
    by, bx = (block, block) if block else (max(tile_h, 256), max(tile_w, 256))
    if volume and depth > 1:
        grid = [block_z or max(tile_d, 32), by, bx]
        pad = [halo if halo_z is None else halo_z, halo, halo]
    else:
        grid, pad = [1, by, bx], [0, halo, halo]
    return [depth, height, width], grid, _native.schedule_blocks([depth, height, width], grid, pad)


def blocks(image, *, level: int = 0, channels: Sequence[int] | None = None, volume: bool = False, block: int | None = None, block_z: int | None = None, halo: int = 0, halo_z: int | None = None) -> Iterator[tuple[np.ndarray, Tile]]:
    """Yields ``(x, tile)`` for every block of every channel, reading the next one ahead."""
    _, _, plan = _schedule(image, level, volume, block, block_z, halo, halo_z)
    pixel = image.pixel_size_um * image.levels[level][6] if image.pixel_size_um else None
    wanted = range(len(image.channels)) if channels is None else list(channels)
    jobs = [(c, b) for b in plan for c in wanted]

    def read(job):
        c, (_, _, _, read_origin, read_size) = job
        x = image.read_block(level, c, list(read_origin), list(read_size))
        return x[0] if read_size[0] == 1 else x

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for n, job in enumerate(jobs):
            x = pending.result() if pending is not None else read(job)
            pending = pool.submit(read, jobs[n + 1]) if n + 1 < len(jobs) else None
            c, (index, origin, size, read_origin, read_size) = job
            yield x, Tile(index, c, level, tuple(origin), tuple(size), tuple(read_origin), tuple(read_size), pixel)


def _takes_tile(fn: Callable) -> bool:
    try:
        params = [p for p in inspect.signature(fn).parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        return False
    return len(params) >= 2


def measure(image, fn: Callable, *, level: int = 0, channels: Sequence[int] | None = None, volume: bool = False, block: int | None = None, halo: int = 0) -> dict[str, np.ndarray]:
    """Calls ``fn(x, tile)`` on every block and gathers what it returns into a table.

    ``fn`` returns a number or a dict of numbers; each block gives one row, with the block's
    ``channel``, ``z``, ``y`` and ``x``.
    """
    rows: list[dict] = []
    passes_tile = _takes_tile(fn)
    for x, tile in blocks(image, level=level, channels=channels, volume=volume, block=block, halo=halo):
        got = fn(x, tile) if passes_tile else fn(x)
        values = got if isinstance(got, dict) else {"value": got}
        rows.append({"channel": tile.channel, "z": tile.origin[0], "y": tile.origin[1], "x": tile.origin[2], **values})
    names = list(dict.fromkeys(k for r in rows for k in r))
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=np.float64) for k in names}


class Transform:
    """A function of a block, ready to run over a whole image (see :func:`transform`)."""

    def __init__(self, fn: Callable, *, halo: int = 0, halo_z: int | None = None, volume: bool = False, block: int | None = None, block_z: int | None = None):
        self.fn = fn
        self.halo = halo
        self.halo_z = halo_z
        self.volume = volume
        self.block = block
        self.block_z = block_z
        self._passes_tile = _takes_tile(fn)

    def __call__(self, x, tile=None):
        return self.fn(x, tile) if self._passes_tile else self.fn(x)

    def blocks(self, image, *, level: int = 0, channels: Sequence[int] | None = None) -> Iterator[tuple[np.ndarray, Tile]]:
        """Yields each block processed (only the part it owns) and its tile."""
        for x, tile in blocks(image, level=level, channels=channels, volume=self.volume, block=self.block, block_z=self.block_z, halo=self.halo, halo_z=self.halo_z):
            y = np.asarray(self(x, tile))
            if y.shape != x.shape:
                raise ValueError(f"the function returned shape {y.shape} for a block of {x.shape}")
            yield y[tile.inner], tile

    def apply(self, image, *, level: int = 0, channel: int = 0) -> np.ndarray:
        """The whole result of one channel as an array (it must fit in memory)."""
        depth, height, width = image.levels[level][:3]
        out = None
        for y, t in self.blocks(image, level=level, channels=[channel]):
            if out is None:
                out = np.zeros((depth, height, width), dtype=y.dtype)
            z, r, c = t.origin
            sz, sy, sx = t.size
            out[z : z + sz, r : r + sy, c : c + sx] = y
        return out[0] if depth == 1 else out

    def write(self, image, output: str | PathLike, *, level: int = 0, channels: Sequence[int] | None = None):
        """Writes the result to an OME-Zarr store (which must not exist), without holding it in memory."""
        depth, height, width = image.levels[level][:3]
        _, grid, _ = _schedule(image, level, self.volume, self.block, self.block_z, self.halo, self.halo_z)
        wanted = list(range(len(image.channels))) if channels is None else list(channels)
        names = [image.channels[c] for c in wanted]
        chunks = [min(a, b) for a, b in zip(grid, (depth, height, width))]
        px, dz = image.pixel_size_um, image.z_spacing_um
        ds = image.levels[level][6]
        writer = None
        for y, t in self.blocks(image, level=level, channels=wanted):
            if writer is None:
                kind = "uint32" if np.issubdtype(y.dtype, np.integer) or y.dtype == bool else "float32"
                writer = _io.ZarrWriter(output, names, (depth, height, width), chunks, dtype=kind, name=image.name, pixel_size_um=px * ds if px else None, z_spacing_um=dz)
            writer.write(wanted.index(t.channel), t.origin, y)
        return writer.finish()


def transform(fn: Callable | None = None, *, halo: int = 0, halo_z: int | None = None, volume: bool = False, block: int | None = None, block_z: int | None = None):
    """Wraps a function of a block as a :class:`Transform`; also a decorator.

    Parameters
    ----------
    fn : callable, optional
        ``fn(x)`` or ``fn(x, tile)``, returning the block processed, the same shape as ``x``.
    halo : int
        Pixels of context ``fn`` needs on each side in y and x.
    halo_z : int, optional
        Planes of context with ``volume=True`` (default: ``halo``).
    volume : bool
        Give ``fn`` 3D blocks ``[d, h, w]`` instead of one plane at a time.
    block, block_z : int, optional
        Block side and depth (memory only, when ``halo`` is enough: results do not depend on them).

    Examples
    --------
    >>> @lucem.custom.transform(halo=16)
    ... def smooth(x):
    ...     return lucem.ops.gaussian_blur(x, sigma=4.0)
    >>> smooth.write(lucem.io.open("slide.ome.zarr"), "smoothed.ome.zarr")  # doctest: +SKIP
    """
    if fn is None:
        return lambda f: Transform(f, halo=halo, halo_z=halo_z, volume=volume, block=block, block_z=block_z)
    return Transform(fn, halo=halo, halo_z=halo_z, volume=volume, block=block, block_z=block_z)


def segment(image, fn: Callable, output: str | PathLike, *, channel: int = 0, level: int = 0, volume: bool = False, block: int | None = None, block_z: int | None = None, halo: int = 0, halo_z: int | None = None) -> dict[str, np.ndarray]:
    """Writes labels from ``fn`` of every block, joining objects cut by block borders.

    ``fn(x)`` or ``fn(x, tile)`` returns integer labels of ``x``'s shape (0: background), numbered
    any way within a block. Returns a table with one row per object (area, centroid, bounding box,
    intensity). With ``volume=False`` each plane's objects are separate.
    """
    depth, height, width = image.levels[level][:3]
    _, grid, _ = _schedule(image, level, volume, block, block_z, halo, halo_z)
    chunks = [min(a, b) for a, b in zip(grid, (depth, height, width))]
    px = image.pixel_size_um
    writer = _io.ZarrWriter(output, [image.channels[channel]], (depth, height, width), chunks, dtype="uint32", name=image.name, pixel_size_um=px * image.levels[level][6] if px else None, z_spacing_um=image.z_spacing_um)
    stitcher = _native.Stitcher(grid, volume and depth > 1)
    passes_tile = _takes_tile(fn)
    for x, tile in blocks(image, level=level, channels=[channel], volume=volume, block=block, block_z=block_z, halo=halo, halo_z=halo_z):
        got = np.asarray(fn(x, tile) if passes_tile else fn(x))
        if got.shape != x.shape:
            raise ValueError(f"the function returned shape {got.shape} for a block of {x.shape}")
        labels = np.asarray(got[tile.inner], dtype=np.uint32)
        ids = stitcher.add(list(tile.origin), labels, x[tile.inner])
        writer.write(0, tile.origin, ids)
    mapping, table = stitcher.finish()
    writer.remap_labels(mapping)
    writer.finish()
    return table

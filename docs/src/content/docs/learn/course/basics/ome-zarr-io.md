---
title: "OME-Zarr I/O"
description: "Write multi-channel OME-Zarr pyramids block by block and read them lazily."
editUrl: https://github.com/neuro-kitchen/lucem/edit/main/playground/01_basics/02_ome_zarr_io.py
sidebar:
  order: 2
---

`lucem.io.ZarrWriter` writes multi-channel OME-Zarr stores block by block without holding a
gigapixel slide in memory. `lucem.io.open` opens OME-Zarr, OME-TIFF, and standard image files
lazily as a pyramid.

**Write a 2-channel synthetic fluorescence slide (DAPI + FITC)**

```python
from pathlib import Path
import tempfile
import numpy as np
import lucem

work = Path(tempfile.mkdtemp(prefix="lucem-playground-"))
out_path = work / "synthetic_nuclei.ome.zarr"

yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
dapi = np.exp(-((yy - 220) ** 2 + (xx - 260) ** 2) / (2.0 * 12.0**2))
fitc = np.exp(-((yy - 220) ** 2 + (xx - 260) ** 2) / (2.0 * 28.0**2))

writer = lucem.io.ZarrWriter(
    out_path,
    channels=["DAPI", "FITC"],
    shape=(1, 512, 512),
    chunks=(1, 256, 256),
    dtype="float32",
    name="synthetic_nuclei",
    pixel_size_um=0.5,
)
writer.write(0, (0, 0, 0), dapi[None, :, :])
writer.write(1, (0, 0, 0), fitc[None, :, :])
writer.finish()
```

**Open lazily and verify exact round-trip**

```python
img = lucem.io.open(out_path)
print("channels:", img.channels, "pixel µm:", img.pixel_size_um)

roundtrip = img.read_block(0, 0, [0, 0, 0], [1, 512, 512])[0]
assert np.allclose(roundtrip, dapi, atol=1e-6)
```

The whole script is [`playground/01_basics/02_ome_zarr_io.py`](https://github.com/neuro-kitchen/lucem/blob/main/playground/01_basics/02_ome_zarr_io.py). Run it with `uv run python playground/01_basics/02_ome_zarr_io.py`.

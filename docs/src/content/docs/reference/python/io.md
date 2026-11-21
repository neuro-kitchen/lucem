---
title: lucem.io
description: "Open OME-Zarr, OME-TIFF, and microscopy slides lazily and write OME-Zarr block by block."
sidebar:
  order: 2
---

```python
import lucem

img = lucem.io.open("slide.ome.zarr")
print(img.name, img.channels, img.pixel_size_um, img.levels)

# Read one block at level 0, channel 0: origin=[z, y, x], size=[d, h, w]
tile = img.read_block(0, 0, [0, 0, 0], [1, 512, 512])[0]
```

## Writing OME-Zarr block by block

```python
writer = lucem.io.ZarrWriter(
    "out.ome.zarr",
    channels=["DAPI", "FITC"],
    shape=(1, 1024, 1024),
    chunks=(1, 256, 256),
    dtype="float32",
    pixel_size_um=0.5,
)
writer.write(0, (0, 0, 0), dapi_block)
writer.write(1, (0, 0, 0), fitc_block)
writer.finish()
```

| Symbol | Description |
|---|---|
| `lucem.io.open(path)` | Opens an OME-Zarr, OME-TIFF, TIFF, or PNG/JPEG lazily as an `Image` pyramid. |
| `lucem.io.ZarrWriter(...)` | Streams multi-channel 2D/3D blocks directly to an OME-Zarr store without holding the whole slide in RAM. |
| `lucem.io.sample_volume(size=128, depth=64)` | Generates a synthetic 3D fluorescence volume in memory for tests and examples. |

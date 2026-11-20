# %% [markdown]
# # Synthetic OME-Zarr I/O
#
# `lucem.io.ZarrWriter` writes multi-channel OME-Zarr pyramids block by block without holding the
# full slide in RAM. `lucem.io.open` opens OME-Zarr, OME-TIFF, and standard microscopy images
# lazily as a `Pyramid`.

# %%
from pathlib import Path
import tempfile

import numpy as np

import lucem

work = Path(tempfile.mkdtemp(prefix="lucem-playground-"))
out_path = work / "synthetic_nuclei.ome.zarr"

# Build a 2-channel 512x512 synthetic fluorescence image (DAPI nuclei + FITC cytoplasm)
yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
dapi = np.zeros((512, 512), dtype=np.float32)
fitc = np.zeros((512, 512), dtype=np.float32)
for cy, cx, r in [(100, 120, 16), (220, 260, 20), (360, 180, 18), (390, 380, 22)]:
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    dapi += np.exp(-dist2 / (2.0 * (r * 0.6) ** 2))
    fitc += np.exp(-dist2 / (2.0 * (r * 1.4) ** 2))

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

# %% [markdown]
# ## Open lazily and verify exact round-trip

# %%
img = lucem.io.open(out_path)
print("name:    ", img.name)
print("channels:", img.channels)
print("levels:  ", img.levels)
print("pixel µm:", img.pixel_size_um)

roundtrip = img.read_block(0, 0, [0, 0, 0], [1, 512, 512])[0]
assert np.allclose(roundtrip, dapi, atol=1e-6)
print("exact round-trip verified:", out_path)

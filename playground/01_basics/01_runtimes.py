# %% [markdown]
# # Runtimes
#
# lucem runs every operator on a *runtime*: `cuda` (NVIDIA), `wgpu` (Vulkan / Metal / DX12),
# or `cpu`. Every call returns the same numerical result across runtimes — only the wall time
# changes.

# %%
import time

import numpy as np

import lucem

print("available:", lucem.runtime.available())
print("in use:   ", lucem.runtime.current())

# %% [markdown]
# ## Synthetic fluorescence field across every available runtime

# %%
rng = np.random.default_rng(42)
yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
image = np.zeros((512, 512), dtype=np.float32)
for cy, cx, r in [(120, 140, 22), (240, 310, 18), (380, 190, 26), (340, 390, 20)]:
    image += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * r**2))
image = np.clip(image + rng.normal(0.0, 0.04, image.shape).astype(np.float32), 0.0, 1.0)

results: dict[str, np.ndarray] = {}
for rt in lucem.runtime.available():
    try:
        # First call compiles and caches the CubeCL kernel on disk
        lucem.ops.gaussian_blur(image, sigma=2.5, runtime=rt)
    except RuntimeError as exc:
        print(f"{rt}: not usable on this machine ({exc})")
        continue
    t0 = time.perf_counter()
    out = lucem.ops.gaussian_blur(image, sigma=2.5, runtime=rt)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    results[rt] = out
    print(f"{rt:6s}: {dt_ms:6.2f} ms  shape={out.shape}  range=[{out.min():.3f}, {out.max():.3f}]")

# %% [markdown]
# ## Verify numerical agreement across runtimes

# %%
names = list(results)
if len(names) >= 2:
    ref = results[names[0]]
    for other in names[1:]:
        diff = float(np.max(np.abs(results[other] - ref)))
        print(f"max |{other} - {names[0]}| = {diff:.2e}")
        assert diff < 1e-4

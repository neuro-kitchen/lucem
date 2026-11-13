# lucem (Python)

GPU image processing for biology, written once in CubeCL: the same operators, histology steps and
models as the [lucem](https://github.com/neuro-kitchen/lucem) app, on NumPy arrays and on images
of any size, on any GPU (wgpu, CUDA) or the CPU.

## Install

```bash
pip install lucem                  # io, ops, custom walks, the model plugin client
pip install "lucem[hist]"          # + histology: stains, tissue, cells, the app's steps
pip install "lucem[app]"           # + the desktop app (`lucem app`)
pip install "lucem[stardist]"      # + the StarDist nucleus models (BSD-3-Clause)
pip install "lucem[sam]"           # + Segment Anything (Apache-2.0)
pip install "lucem[ml]"            # + every model plugin
pip install "lucem[all]"           # everything
```

Wheels for CPython 3.12+ on Linux x86_64 (glibc 2.34+), Windows x64 and macOS arm64 (no CUDA there).
Model weights (and SAM's ONNX Runtime) download and are checked against their SHA-256 the first
time a model is used.

## Use

```python
import lucem
from lucem.ops import Pipeline, stage

image = lucem.io.open("slide.ome.zarr")             # nothing read yet
x = image.read_region(0, 0, 0, 0, 1024, 1024)        # level, channel, x, y, width, height
mask = Pipeline([stage.gaussian_blur(sigma=2), stage.threshold(value=0.2), stage.open(radius=2)])(x)

# Your own function over an image of any size, block by block
smooth = lucem.custom.transform(lambda b: lucem.ops.gaussian_blur(b, sigma=4.0), halo=16)
smooth.write(image, "smoothed.ome.zarr")

# Histology (lucem[hist]): the steps the app records, run from Python
from lucem.hist import steps
result = steps.run(image, [steps.tissue(), steps.cells(stains="h_dab", within="tissue")])
table = lucem.hist.export.table(result.cells, image.name, image.pixel_size_um)

# Models (lucem[stardist]): a plugin program, found by itself
with lucem.ml.Plugin("lucem-ml-stardist") as p:
    labels = p.instances("stardist-he", rgb, pixel_um=0.5)
```

The `lucem` command of the package opens the app (`lucem app [path]`) and lists the model plugins
(`lucem plugins`). The full command line (operators, pipelines, histology batches, the server, MCP)
is the Rust program: `cargo install --git https://github.com/neuro-kitchen/lucem lucem-cli`.

## How it is built

The `lucem-py` crate is the one extension (`lucem._native`); `lucem-py-core` and `lucem-py-hist`
register into it. The `lucem` wheel is built without histology, the `lucem-hist` wheel with it
(`wheels/lucem-hist`), and only one is loaded: when `lucem-hist` is installed, `lucem` uses its
extension, so there is one GPU device per runtime. `wheels/lucem-app` and `wheels/lucem-ml-*` put
the app and the model plugins (programs, never linked) into wheels.

```bash
uv sync                                   # from the repository root: builds and installs lucem (editable, with hist)
uv run pytest -q bindings/python/tests
```

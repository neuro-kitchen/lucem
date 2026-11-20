# lucem playground

Runnable `# %%` cell scripts that mirror the **Learn › Course** chapters in the documentation.
Every script generates its own synthetic microscopy / histology data on the fly (no external
datasets required) so results can be verified numerically on any machine.

## Running scripts with `uv`

From the repository root:

```sh
uv sync
cd bindings/python && uv run --project ../.. maturin develop --profile validate && cd ../..
uv run python playground/01_basics/01_runtimes.py
```

Set `LUCEM_RUNTIME=cpu` if you want to force CPU execution. Every file uses `# %%` cell markers so
it opens directly as an interactive notebook in VS Code, Positron, or Jupyter.

## Chapters

| Folder | What you learn |
|---|---|
| `01_basics/` | Runtimes (`cuda`, `wgpu`, `cpu`) and OME-Zarr I/O on synthetic multi-channel images |
| `02_filtering/` | Gaussian blur, rolling-ball background subtraction, and CLAHE local contrast |
| `03_segmentation/` | Auto-thresholding (Otsu, Li), morphological cleanup, and connected-component labeling |
| `04_pipelines/` | Chaining GPU operators on-device (`lucem.ops.Pipeline`) and `.lucem-recipe.json` sharing |
| `05_custom_ops/` | Writing `@lucem.op(family="Filter", halo=12)` tiled Python ops for both Python and the app |
| `06_histology_and_ml/` | Discovering `lucem[ml]` plugins (`stardist`, `sam`, `instanseg`) and `lucem[hist]` workflows |

---
title: Installing with uv & Python
description: "Install lucem and optional extras (web, app, hist, ml) in any uv or pyproject.toml project."
sidebar:
  order: 1
---

`lucem` is packaged as a lean core wheel (`numpy` + CubeCL GPU operators + OME-Zarr I/O + `@lucem.op` tiled walker), with heavy features split into optional companion extras so your `pyproject.toml` stays clean and fast to resolve with `uv`.

## Pick only what you need

```toml
# pyproject.toml
[project]
name = "my-histology-study"
requires-python = ">=3.12"
dependencies = [
    "lucem[web,hist,stardist]",
]
```

Or add from the terminal with `uv`:

```sh
uv add lucem                         # Core GPU ops, OME-Zarr I/O, @lucem.op
uv add "lucem[web]"                  # + Browser WASM workstation (`lucem web`)
uv add "lucem[app]"                  # + Native desktop workstation (`lucem app`)
uv add "lucem[hist]"                 # + Histology: stain deconvolution, tissue, cells
uv add "lucem[stardist]"             # + StarDist nuclei model plugin
uv add "lucem[ml]"                   # + StarDist + SAM + InstanSeg plugins
uv add "lucem[all]"                  # Everything
```

## Ephemeral runs with `uv run`

Launch the browser viewer or inspect installed model plugins without creating a project:

```sh
uv run --with "lucem[web]" lucem web --open
uv run --with "lucem[ml]" lucem plugins
```

## Launching the UI from Python

```python
import lucem

# Browser WASM app (pip/uv install "lucem[web]")
lucem.web.serve(data="./slides", port=8080, open_browser=True)

# Native desktop workstation (pip/uv install "lucem[app]")
lucem.app.open("./slides/sample.ome.zarr")
```

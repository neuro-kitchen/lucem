---
title: web, app, ml & hist
description: "Launch the browser or desktop UI, discover ML model plugins, and run histology from Python."
sidebar:
  order: 5
---

## `lucem.web` (`lucem[web]`)

```python
import lucem

lucem.web.dist_dir()                                       # Path to bundled WASM dist/
lucem.web.serve(data="./slides", port=8080, open_browser=True)
```

CLI shortcut: `uv run lucem web --data ./slides --open`

## `lucem.app` (`lucem[app]`)

```python
import lucem

lucem.app.program()                      # Path to lucem-app binary
lucem.app.open("slide.ome.zarr")         # Launches the desktop workstation
```

CLI shortcut: `uv run lucem app slide.ome.zarr`

## `lucem.ml` (`lucem[stardist]`, `lucem[sam]`, `lucem[instanseg]`, `lucem[ml]`)

```python
import lucem

lucem.ml.plugin_dirs()                   # Directories scanned for lucem-ml-* programs
lucem.ml.plugins()                       # Discovered plugins, runtimes, and model manifests
```

CLI shortcut: `uv run lucem plugins`

## `lucem.hist` (`lucem[hist]`)

Installs the histology-enabled native extension so stain deconvolution, tissue detection, cell counting, and session replay run directly from Python on the same GPU context.

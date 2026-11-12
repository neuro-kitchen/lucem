---
title: Sessions
description: The analysis as data, so it runs again anywhere.
sidebar:
  order: 4
---

Every histology step you run (find tissue, find cells, classify, score intensity, class pixels,
make superpixels or a density map, export) is recorded in the project's `session.json` with its
settings. The History panel lists it.

A step is data: an **action** such as

```json
{ "action": "cells", "within": { "in": "tissue" }, "model": "stardist-he", "staining": { … }, "detection": { … } }
```

The app, the command line and (in time) an agent run steps the same way, through one engine,
so a step gives the same result wherever it runs.

## Running a session on other images

```sh
lucem session run session.json slides/*.ome.zarr --out results/ [--runtime cuda]
```

For each image, the last step of each kind runs, in order: a step you ran again with other
settings replaces the earlier one. Regions you drew on the image the session was recorded on
belong to that image, so on other images the step looks in the tissue found instead.

Each image's results are written to its output store: see
[Output store](/lucem/reference/files/output-store/).

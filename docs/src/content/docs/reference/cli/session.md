---
title: lucem session
description: Run an analysis saved by the app on many images.
sidebar:
  order: 2
---

```text
lucem session run <session.json> <image>... [--out DIR] [--runtime NAME]
```

Runs the session's last step of each kind on each image, through the same engine as the app,
and writes each image's [output store](/lucem/reference/files/output-store/): inside its OME-Zarr
when it can be written, else `DIR/<image>.lucem.zarr`. It prints each step's outcome and fails
(after the others) for images that could not be analysed.

```text
$ lucem session run session.json he-test.ome.zarr
2 step(s): Find cells → Export
he-test.ome.zarr: 2539 cells, 0 tiles in 17.1 s → he-test.ome.zarr
  2539 cell(s) found
  Exported
```

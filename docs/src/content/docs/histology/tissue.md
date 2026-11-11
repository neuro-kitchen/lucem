---
title: Find tissue
description: The tissue of a slide as regions, found at a coarse level.
sidebar:
  order: 1
---

**Find tissue** looks at a coarse level of the whole image (at most 4,096 pixels across) and
keeps what differs from the background: on brightfield slides, the stained areas against the
bright glass; on fluorescence, the signal against the dark. The regions become shapes of class
**Tissue**, undoable as one change.

Later steps can look in the tissue instead of a region you draw: a recorded session run on other
images does so.

```sh
lumen hist tissue slide.ome.zarr --out tissue.geojson           # brightfield
lumen hist tissue slide.ome.zarr --fluorescence --out tissue.geojson
```

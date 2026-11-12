---
title: layout.json
description: The views of each workspace.
sidebar:
  order: 4
---

View state, saved so a project opens as you left it. It is not undone and not part of the
analysis.

```json
{
  "lucem_layout": 1,
  "explore": { "views": ["Plane", "3D"] },
  "analyze": {
    "table": true,
    "plots": [ { "kind": "Scatter", "x": "area_um2", "y": "perimeter_um" } ],
    "hierarchy": true,
    "side": true
  }
}
```

---
title: session.json
description: The analysis steps run, as data.
sidebar:
  order: 3
---

```json
{
  "lucem_session": 1,
  "steps": [
    { "at": 1762870000, "image": "slide-1", "action": "tissue", "settings": { "brightfield": true } },
    { "at": 1762870042, "image": "slide-1", "action": "cells", "within": { "in": "tissue" },
      "staining": { "staining": "brightfield", "stains": { … } }, "detection": { … }, "model": "stardist-he" },
    { "at": 1762870120, "image": "slide-1", "action": "export" }
  ]
}
```

| `action` | Fields |
|---|---|
| `tissue` | `settings` |
| `cells` | `within`, `staining`, `detection`, `auto_threshold`, `model` |
| `classify_objects` | `classifier`, `tiles` |
| `intensity` | `thresholds` |
| `pixel_classes` | `classifier`, `pixel_um`, `within` |
| `superpixels` | `within`, `settings`, `pixel_um` |
| `density` | `settings` |
| `export` | |

`within` is `{"in": "image"}`, `{"in": "tissue"}` or `{"in": "regions", "outlines": [[[x, y], …], …]}`
(level-0 pixels).

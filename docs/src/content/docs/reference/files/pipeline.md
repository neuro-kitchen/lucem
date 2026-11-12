---
title: pipeline.json
description: The processing graph.
sidebar:
  order: 2
---

```json
{
  "lucem_pipeline": 1,
  "graph": {
    "nodes": [
      { "id": "n1", "x": 0, "y": 0, "kind": { "type": "source", "image": "img1", "channel": 0 } },
      { "id": "n2", "x": 240, "y": 0, "kind": { "type": "op", "op": { "op": "gaussian_blur", "sigma": 2.0 } } }
    ],
    "edges": [ { "from": "n1", "to": "n2", "port": 0 } ]
  }
}
```

Node kinds: `source` (a channel of an image), `annotations` (a class's shapes as a mask), `op`
(an [operation](/lucem/reference/operations/)), `view`, `script` and `model`.

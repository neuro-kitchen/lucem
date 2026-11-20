---
title: "Histology & ML model plugins"
description: "Install and discover lucem[hist] and lucem[ml] (StarDist, SAM, InstanSeg) via uv extras."
editUrl: https://github.com/neuro-kitchen/lucem/edit/main/playground/06_histology_and_ml/01_stardist_and_hist.py
sidebar:
  order: 1
---

Histology workflows and deep-learning model plugins are distributed as optional companion wheels so
core `lucem` stays small and `pyproject.toml` projects using `uv` only pull what they need:

```sh
uv add "lucem[hist,stardist]"
# or all model plugins + browser app:
uv add "lucem[web,hist,ml]"
```

**Discover installed model plugins and check `lucem.hist`**

```python
import lucem

for p in lucem.ml.plugins():
    print(f"{p['name']} {p['version']} ({p['program']})")
    for m in p["models"]:
        print(f"  - {m['id']}: {m['name']}")
```

The whole script is [`playground/06_histology_and_ml/01_stardist_and_hist.py`](https://github.com/neuro-kitchen/lucem/blob/main/playground/06_histology_and_ml/01_stardist_and_hist.py). Run it with `uv run python playground/06_histology_and_ml/01_stardist_and_hist.py`.

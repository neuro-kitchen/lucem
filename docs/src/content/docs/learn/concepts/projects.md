---
title: Projects
description: What a project folder holds, and what changes when.
sidebar:
  order: 2
---

A project is a folder. Every part is a plain file you can read, diff and share:

| File | What it holds | Changes when |
|---|---|---|
| `project.lucem.json` | The images (paths and how each channel is shown), annotations, classes, notes | You edit anything; undo works on it |
| `pipeline.json` | The processing graph | You change the Pipeline |
| `session.json` | The analysis steps run, with their settings | A histology step runs |
| `layout.json` | The views of each workspace | You add, close or arrange views |
| `histology/` | Histology settings, and each image's cells and tiles (`<image>.objects.json.gz`) | A histology step runs |

Image paths inside the project folder are stored relative to it, so the folder can move. Images
are never copied into the project.

Projects saved before format 2 kept the graph inside `project.lucem.json`; they open as they are
and are written as format 2 on the next save. See [Files](/lucem/reference/files/project/).

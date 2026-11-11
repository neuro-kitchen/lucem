---
title: project.lumen.json
description: The project document.
sidebar:
  order: 1
---

The project's images, annotations, classes and notes, as one versioned JSON document
(`"lumen_project": 2`). It changes only through commands, which is what undo works on and what
an agent will call.

| Field | Holds |
|---|---|
| `lumen_project` | The format version (2; version 1 opens and is migrated) |
| `name` | The project's name |
| `images` | Each image: `id` (`img1`…), `path` (relative to the project folder when inside it), and how each channel is shown (colour, window, gamma, shown) |
| `active_image` | The image shown in Explore |
| `annotations` | Each shape: `id`, `image`, `shape` (rect, ellipse, polygon, freehand, point, line, stroke), `class`, `note`, `z` (plane) |
| `classes` | Each class: name, colour, shown, opacity |
| `notes` | Notes, optionally pinned to a point |

The graph lives in [`pipeline.json`](/lumen/reference/files/pipeline/).

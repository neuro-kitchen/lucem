---
title: Pipeline
description: Build the processing graph on a canvas.
sidebar:
  order: 2
---

The Pipeline workspace is the graph on a canvas. Its menu row: **Sources**, **Edit**,
**Process** (every operation, a submenu per family), **Output**, **Script**, **Model** and
**View**.

- Add a node from a menu, the search, or by dragging; connect an output port to an input port of
  the same type (the port colours: image, mask, labels, table).
- Select a node to edit its settings in the Inspector. Model nodes run only when you press
  **Run** there.
- The bar at the bottom left zooms, fits the graph (<kbd>F</kbd>), and turns the dot grid and the
  minimap on or off.

The graph is saved in the project's `pipeline.json` and runs headless with `lucem run`.

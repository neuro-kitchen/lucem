---
title: Explore
description: Views of the image, tools to draw and select, and the panels beside them.
sidebar:
  order: 1
---

## Views

The centre holds views of the image, each a tab you can drag into a split. The buttons in a
view's tab switch what it shows; **+** adds a view or arranges them all.

| View | Shows |
|---|---|
| Plane | One plane, full canvas; it can show another image than the selected one (its tab menu) |
| 2D | The XY, XZ and YZ planes through the 3D cursor (volumes) |
| 3D | The volume rendered, with the annotations (volumes) |
| Channels | Each channel on its own and all together, moving together |

**Add view or container** (<kbd>+</kbd> in a tab) arranges every view as tabs, side by side,
stacked or in a grid. Views of one image can move and zoom together.

## Tools

The tool column on the left; the options bar above the views shows the active tool's settings.

| Tool | Key | Does |
|---|---|---|
| Select | <kbd>V</kbd> | Click a shape or a cell to select it, drag to move; <kbd>Shift</kbd> adds |
| Region select | <kbd>M</kbd> | Drag a box: every shape and cell inside it is selected |
| Hand | <kbd>H</kbd> | Move the image (or hold <kbd>Space</kbd> with any tool) |
| Zoom | <kbd>Z</kbd> | Click to zoom in, <kbd>Alt</kbd>+click to zoom out |
| Rectangle, Ellipse | <kbd>U</kbd>, <kbd>O</kbd> | Drag a shape |
| Polygon, Lasso | <kbd>P</kbd>, <kbd>L</kbd> | Click corners, or drag around a region |
| Points | <kbd>N</kbd> | Click to place a point |
| Brush, Eraser | <kbd>B</kbd>, <kbd>E</kbd> | Paint a class, or erase paint |
| Wand | <kbd>W</kbd> | Paint that follows similar intensity |
| Threshold | <kbd>T</kbd> | A live threshold of a channel, turned into shapes |
| Segment (SAM) | <kbd>S</kbd> | A model outlines the object you click or box |
| Ruler, Pixel probe | <kbd>R</kbd>, <kbd>I</kbd> | Measure a length; read pixel values |

With two or more shapes selected, the options bar offers **Add**, **Subtract**, **Intersect** and
**Xor**: the selected shapes become one (one <kbd>Ctrl</kbd>+<kbd>Z</kbd> undoes it).

## Panels

The rail on the right opens one panel at a time: **Layers**, **Histology**, **Metadata**, then
**Selection**, **Notes** and **History**.

- **Layers**: channels, results, shapes and detections. Select a channel to edit its colour
  (wheel) and levels (histogram, black and white points, gamma); its processing is listed under
  them. The Navigator sits at the bottom.
- **Histology**: the recipes; see [Histology](/lumen/histology/).
- **Selection**: what is selected, with its measurements (the mean of each, for many).
- **History**: every change, to step back to; then the session's analysis steps.

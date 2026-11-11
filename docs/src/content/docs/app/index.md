---
title: App
description: "The desktop app: look at images, build pipelines, read the results."
sidebar:
  order: 0
---

`lumen-app` is a workstation for large microscopy images. It draws whole slides and volumes from
their pyramids and runs processing on your GPU as you change it.

## Workspaces

The workspaces sit at the top left, after the **Lumen** and **Help** menus. Each has its own
menu row under the title bar.

| Workspace | What it's for | Keys |
|---|---|---|
| [Explore](/lumen/app/explore/) | Look at images, draw and edit shapes, run histology | <kbd>Ctrl</kbd>+<kbd>1</kbd> |
| [Pipeline](/lumen/app/pipeline/) | The processing graph on a canvas | <kbd>Ctrl</kbd>+<kbd>2</kbd> |
| [Analyze](/lumen/app/analyze/) | Tables and plots of what was measured | <kbd>Ctrl</kbd>+<kbd>3</kbd> |

## Finding things

<kbd>Ctrl</kbd>+<kbd>K</kbd> opens the search: every command, tool and operation by name, with
what it does. <kbd>↑</kbd> <kbd>↓</kbd> move, <kbd>Enter</kbd> runs, <kbd>Esc</kbd> closes.
**Process → Find an operation…** lists the operations only.

## What runs in the background

Long steps (finding cells, a model, an export) run in the background. The status bar shows each
one with how far it is; the Histology panel's **Cancel** stops a step after its current tile.
Only one step that drives the GPU runs at a time.

See also [Keyboard and mouse](/lumen/app/keyboard/).

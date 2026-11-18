---
title: In your browser
description: "The viewer at /lucem/viewer/: open an image from your disk, nothing to install."
sidebar:
  order: 1
---

**[Open the viewer](/lucem/viewer/)**: the same app as the desktop's, built for the browser.
Nothing is installed and nothing leaves your computer: the files you open are read in the page.

## Open an image

**Drop it on the page**, or use the File menu:

- **Open file…** (<kbd>Ctrl</kbd>+<kbd>O</kbd>): the file chooser, as for a PDF (no question from
  the browser): a TIFF or a picture (`.png`, `.jpg`, `.bmp`, `.webp`; read whole, so keep them
  small).
- **Open project…** (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>O</kbd>): a project, or an
  **OME-Zarr** (`.ome.zarr`, as `lucem convert` writes them). An OME-Zarr is stored as a folder,
  so your browser asks once to let the page read it ("View files"): nothing is uploaded, the page
  reads it where it is. Dropping it on the page skips that question. Only the tiles a view needs
  are read, so a whole slide larger than the browser's memory opens too.
- **The samples** (File › Samples): a 3D volume and the test patterns.

What you pick opens at once (up to eight images; more, and the page asks which).

Your work (open images, views, shapes) is kept in the browser and comes back after a reload. Files
you picked are not: a browser forgets them, so pick them again.

Not in the viewer yet: processing steps and histology (tissue, cells, classifiers), and the
tables. Use [the container](/lucem/deploy/#the-app-in-your-browser), where the server computes
them, or the [desktop app](/lucem/app/).

## What your browser needs

The page checks before it downloads the app (about 9 MB), and says what is missing:

| | Needed for | Where |
|---|---|---|
| Hardware graphics | drawing (in software the page would crawl) | every recent browser, with graphics acceleration on |
| Threads | reading tiles on several cores | recent Chrome, Edge, Firefox, Safari (the page turns them on itself, after one reload) |
| WebGPU or WebGL2 | drawing | WebGPU in Chrome and Edge; WebGL2 everywhere else |

## The browser, the desktop and a server

| | Images | Processing |
|---|---|---|
| This page | files on your computer | none yet |
| [The container](/lucem/deploy/#the-app-in-your-browser) | the server's | the server's GPU: processing steps and histology |
| The desktop app | files, servers | your GPU, every runtime |

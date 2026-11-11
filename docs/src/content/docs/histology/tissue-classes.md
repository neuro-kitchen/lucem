---
title: Tissue classes
description: Class every pixel from painted examples, and measure the areas.
sidebar:
  order: 4
---

The pixel classifier learns from regions or brush strokes of at least two classes and classes
every pixel inside the selected regions (or where it was trained). It works at the resolution set
in the recipe (µm per pixel), so a whole slide stays fast.

- The class map is drawn over the image; its areas (mm² and % per class) are measured.
- **To regions** turns the map into shapes of each class (pieces under 100 µm² dropped, holes
  under 100 µm² filled).

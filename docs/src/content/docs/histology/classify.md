---
title: Classify cells
description: Class cells from your examples, and score them by one stain.
sidebar:
  order: 3
---

## Cell classes

1. Find the cells.
2. Draw shapes or points of each class (Tumor, Stroma, Immune…) over cells of that class.
3. **Train**: a Random Trees classifier learns from the cells under your examples and classes
   every cell. Superpixel tiles are classed the same way, by a classifier of their own.

The classifier is kept with the project and classes new cells as they are found. Shapes of the
classes Tissue, Region, Measurement and Hotspot are never examples.

## Positive % and H-score

Cells are scored by one measurement (for example the nucleus' mean DAB optical density) against
three thresholds: negative, 1+, 2+, 3+. The outcome gives the positive percentage and the H-score
(0–300).

---
title: Superpixels
description: Tiles that follow edges, measured and classed like cells.
sidebar:
  order: 5
---

Superpixels (SLIC) cut the selected regions into pieces that follow the image's edges. Each is
measured like a cell and can be classed by its own classifier: a way to class tissue without
finding cells.

| Setting | What it does |
|---|---|
| Spacing | About how wide a superpixel is (µm) |
| Compactness | Higher: more square pieces; lower: closer to the edges |

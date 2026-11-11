---
title: Histology
description: Find tissue and cells, class them, score stains, map densities.
sidebar:
  order: 0
---

The **Histology** panel (right rail in Explore) holds six recipes. Open one, check its settings,
press **Run**. A recipe works inside the region shapes you select (draw one first), except
finding tissue, which looks at the whole image.

| Recipe | Gives |
|---|---|
| [Count cells](/lumen/histology/cells/) | Every cell in the selected regions, measured |
| [Positive % and H-score](/lumen/histology/classify/#positive--and-h-score) | Cells scored negative, 1+, 2+, 3+ by one stain |
| [Cell classes](/lumen/histology/classify/) | Cells classed like the shapes you class over them |
| [Tissue classes](/lumen/histology/tissue-classes/) | Every pixel classed from painted examples; areas measured |
| [Superpixel classes](/lumen/histology/superpixels/) | Tiles that follow edges, classed without finding cells |
| [Density and hotspots](/lumen/histology/density/) | Cells per area as a map, and the densest spots |

Every step that ran is recorded in the project's `session.json`, so the same analysis runs on
other images without the app: see [Sessions](/lumen/learn/concepts/sessions/).

## Stains

How nuclei show in the image decides how they are found:

- **Brightfield**: H-DAB, H&E or H-E-DAB stain vectors, or vectors **fitted** to a region you
  select (it should hold both stains and some background). The hematoxylin channel finds nuclei.
- **Fluorescence**: the nuclear channel (DAPI, Hoechst). lumen picks the channel named like one.

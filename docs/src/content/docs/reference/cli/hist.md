---
title: lucem hist
description: Histology from the command line.
sidebar:
  order: 1
---

### `lucem hist stains`

```text
lucem hist stains <image> --region X,Y,W,H [--stains h-dab|h-e] [--out FILE.json]
```

Stain vectors fitted to a region (level-0 pixels). The file is used by `--stains FILE.json`.

### `lucem hist tissue`

```text
lucem hist tissue <image> [--threshold T] [--fluorescence] --out FILE.geojson
```

### `lucem hist cells`

```text
lucem hist cells <image> [--region X,Y,W,H | --within FILE.geojson] [--level N]
    [--stains h-dab|h-e|h-e-dab|FILE.json | --nuclear CHANNEL]
    [--threshold T] [--sigma UM] [--expansion UM] [--min-area UM2] [--max-area UM2]
    [--background UM] [--split gaps|shape|none] [--tile PX] [--classifier FILE.json]
    [--intensity "MEASUREMENT=T1[,T2,T3]"] [--out FILE.tsv|.csv|.geojson]
    [--preview FILE.png] [--model ID] [--runtime NAME]
```

Cells detected, measured and classified, then a summary. `--preview` writes the region's nuclear
channel with the outlines; `--model` finds the nuclei with a model plugin (`stardist-he`,
`stardist-fluo`).

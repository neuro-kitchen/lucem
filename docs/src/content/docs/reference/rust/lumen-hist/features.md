---
title: "lumen_hist::features"
description: "Smoothed features: each object gets, for every measurement, the mean of its neighbours' values weighted by a Gaussian of their centroid distance (FWHM in µm), and how many neighbours are near. A cell then carries what its surroundings look like, which helps a classifier tell regions apart (tumour vs stroma) where single cells are ambiguous."
sidebar:
  label: "features"
  order: 8
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Smoothed features: each object gets, for every measurement, the mean of its neighbours'
values weighted by a Gaussian of their centroid distance (FWHM in µm), and how many
neighbours are near. A cell then carries what its surroundings look like, which helps a
classifier tell regions apart (tumour vs stroma) where single cells are ambiguous.

## Functions

### `add_smoothed`

```rust
pub fn add_smoothed(objects: &mut [HistObject], fwhm_um: f64, pixel_um: f64)
```

Adds `Smoothed: {fwhm} µm: {name}` for every measurement, and `Smoothed: {fwhm} µm: Nearby
detection counts` (the sum of the weights). `pixel_um`: µm per level-0 pixel.

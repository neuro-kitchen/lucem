# lucem-ml-stardist

[lucem](https://github.com/neuro-kitchen/lucem)'s StarDist plugin: nuclei as star-convex polygons
(Schmidt et al. 2018, Weigert et al. 2020), run on Burn (CUDA, wgpu or the CPU). It is a program
(`lucem-ml-stardist`) that lucem, the app and `lucem.ml` start when a StarDist model is used.

Install it with `pip install "lucem[stardist]"`. The published models (H&E nuclei, fluorescence
nuclei) are downloaded, checked against their SHA-256 and converted once, the first time they are
used. BSD-3-Clause, as StarDist.

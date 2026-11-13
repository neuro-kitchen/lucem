# lucem-ml-instanseg

[lucem](https://github.com/neuro-kitchen/lucem)'s InstanSeg plugin: nuclei as pixel embeddings
(Goldsborough, Bankhead et al. 2024), run on Burn (CUDA, wgpu or the CPU). It is a program
(`lucem-ml-instanseg`) that lucem, the app and `lucem.ml` start when an InstanSeg model is used.

Install it with `pip install "lucem[instanseg]"`. The published brightfield and single-channel
nucleus models are downloaded, checked against their SHA-256 and converted once (read from their
TorchScript archives without PyTorch), the first time they are used. Apache-2.0, as InstanSeg.

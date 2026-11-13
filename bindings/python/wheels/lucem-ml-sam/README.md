# lucem-ml-sam

[lucem](https://github.com/neuro-kitchen/lucem)'s Segment Anything plugin: an object's mask from a
click or a box, run on ONNX Runtime. It is a program (`lucem-ml-sam`) that the app and `lucem.ml`
start when SAM is used.

Install it with `pip install "lucem[sam]"`. The weights and ONNX Runtime are downloaded and checked
the first time they are used (Linux x86_64 and macOS arm64; on Windows, install the `onnxruntime`
Python package next to it). Apache-2.0, as Segment Anything.

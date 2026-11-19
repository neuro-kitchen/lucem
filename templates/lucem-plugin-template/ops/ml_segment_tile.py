# /// script
# dependencies = ["numpy", "torch"]
# ///
# lucem: name = "PyTorch Tile Inference"
# lucem: family = "Segment"
# lucem: description = "Warm PyTorch GPU tile inference example (loads model once in init())"
# lucem: halo = 16
# lucem: input image: image
# lucem: param threshold = 0.5
# lucem: output mask

"""Example ML script op for Lucem's warm Python worker.

- `init()` runs ONCE when the script is loaded (or edited), keeping PyTorch and
  model weights resident on CUDA / MPS / CPU across every viewport pan/zoom.
- On Python 3.15+, `lazy import torch` can be used at the top of the file to
  defer framework imports until first use.
"""

import numpy as np

_DEVICE = None
_MODEL = None


def init():
    """Load your TorchScript / PyTorch / ONNX checkpoint once on the best device."""
    global _DEVICE, _MODEL
    import torch

    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    else:
        _DEVICE = torch.device("cpu")

    # Replace this toy 3x3 smoothing/edge head with your own `torch.jit.load(...)`
    # or `torch.load(...)` model checkpoint.
    conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1.0 / 9.0)
    _MODEL = conv.to(_DEVICE).eval()


def run(image, threshold):
    """Run fast GPU inference on a 2-D float32 tile (`[H, W]`, values 0..1)."""
    import torch

    if _MODEL is None:
        init()
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(_DEVICE)
        prob = torch.sigmoid((_MODEL(x) - float(threshold)) * 12.0)[0, 0]
        mask = (prob > 0.5).to(torch.uint32).cpu().numpy()
    return mask

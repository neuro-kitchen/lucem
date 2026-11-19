# Lucem Plugin Template

Starter repository for building and sharing **Lucem** microscopy plugins — modeled on the simplicity of Fiji scripts/update sites and QuPath extensions, without ABI breakage or dependency conflicts.

## What You Can Build

| Tier | Format | Best For | Zero-Setup Runtime |
|---|---|---|---|
| **1. Recipes** | `recipes/*.lucem-recipe.json` | Reusable multi-step GPU pipelines (no code) | Built into Lucem |
| **2. Script Ops** | `ops/*.py`, `ops/*.rs` | Custom filters, channel math, masks, measurements | Python via `uv` (PEP 723 inline deps) or Rust via `rustc`/`cargo` |
| **3. ML / Native Plugins** | `lucem-ml-<name>` binary | Deep-learning models (PyTorch, ONNX, Burn, Cellpose, StarDist) | Out-of-process JSON-lines protocol (`lucem-ml`) |

---

## Quick Start (Test Locally in 30 Seconds)

1. Clone this template (or click **Use this template** on GitHub):
   ```bash
   git clone https://github.com/<you>/<your-lucem-plugin>.git
   cd <your-lucem-plugin>
   ```
2. Point Lucem at your plugin folder (or symlink `ops/` into `~/.config/lucem/plugins/ops/`):
   ```bash
   LUCEM_PLUGINS="$PWD" lucem
   ```
3. Open the **Pipeline** workspace → **Script** menu: your ops in `ops/` appear automatically!
4. Edit any `.py` or `.rs` file and save — Lucem reloads the header and re-runs the node live.

---

## Writing a Drop-In Script Op (`ops/`)

### Python (with automatic `uv` dependency isolation)

Declare dependencies with standard [PEP 723](https://packaging.python.org/en/latest/specifications/inline-script-metadata/) `# /// script` headers, and declare Lucem ports/parameters with `# lucem:` lines:

```python
# /// script
# dependencies = ["numpy", "scipy"]
# ///
# lucem: name = "Membrane Enhance"
# lucem: family = "Filter"
# lucem: description = "Enhances bright cell membranes using DoG + positive clip"
# lucem: halo = 12
# lucem: input image: image
# lucem: param sigma = 1.5
# lucem: param alpha = 1.0
# lucem: output image

import numpy as np
from scipy.ndimage import gaussian_filter


def run(image, sigma, alpha):
    fine = gaussian_filter(image, sigma=sigma)
    coarse = gaussian_filter(image, sigma=sigma * 2.5)
    return np.clip((fine - coarse) * alpha, 0.0, 1.0).astype(np.float32)
```

When `uv` is installed, Lucem runs your script in an isolated, cached environment with the exact packages declared in `dependencies`.

### Rust (compiled once & cached)

```rust
// lucem: name = "Channel Ratio"
// lucem: family = "Intensity"
// lucem: description = "Computes A / (B + eps) normalized ratio"
// lucem: input a: image
// lucem: input b: image
// lucem: param eps = 0.01
// lucem: output image

pub fn run(inputs: &Inputs, params: &Params) -> Result<Array, String> {
    let a = inputs.get("a").ok_or("missing input a")?;
    let b = inputs.get("b").ok_or("missing input b")?;
    let eps = params.get("eps").copied().unwrap_or(0.01) as f32;
    let data = a.data.iter().zip(&b.data).map(|(&x, &y)| x / (y + eps)).collect();
    Ok(Array { width: a.width, height: a.height, data })
}
```

You can pull any crate from crates.io by adding `// lucem: crate rayon = "1"`.

---

## Repository Structure

```text
.
├── lucem-plugin.toml                       # Plugin metadata & entrypoints
├── ops/
│   ├── membrane_enhance.py                 # Example Python op (uv + scipy)
│   └── channel_ratio.rs                    # Example Rust op (two-image math)
├── recipes/
│   └── nuclei_segmentation.lucem-recipe.json  # Example no-code GPU recipe
├── registry-entry.json                     # Example submission for a community registry
└── .github/workflows/validate.yml          # Automated validation on PRs
```

## Publishing to a Community Registry

1. Tag a GitHub release (e.g. `v0.1.0`).
2. Update `registry-entry.json` with your repo URL, tag, and SHA-256 checksum.
3. Open a PR against the Lucem Community Registry (or host your own lab `registry.json` URL).

## License

Dual-licensed under MIT or Apache-2.0 (change `lucem-plugin.toml` to match your project's license).

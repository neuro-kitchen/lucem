# lumen

An application for fluorescence brain-section analysis, written in Rust: GPU image processing with
CubeCL, models with Burn, and a desktop UI with GPUI (gpui-kit).

## Status
Skeleton workspace (2026-11-05): every crate below exists and builds, none has code yet.

## Layout (as in dsp-kitchen)
All operations are general purpose, so there is no domain folder like dsp-kitchen's `synapse/`.

| Path | Purpose |
|---|---|
| `crates/lumen` | Umbrella crate: re-exports core, io, ops, ml |
| `crates/lumen-core` | Core types (images, channels, pixel types) and the compute layer (runtime selection, launch geometry, device buffers) |
| `crates/lumen-io` | Readers and writers: OME-Zarr, OME-TIFF, vendor microscopy formats |
| `crates/lumen-ops` | Image operations in CubeCL: filters, morphology, thresholds, statistics |
| `crates/lumen-ml` | Models on Burn: nuclei and cell segmentation, classifiers |
| `engine/lumen-pipeline` | Chains operations on the device over tiles of large images |
| `serve/lumen-view` | Multiscale pyramids and display tiles |
| `app/lumen-cli` | Command-line interface (binary `lumen`) |
| `app/lumen-app` | Desktop application (GPUI) |
| `bindings/python/` | Python package: `lumen-py` crate (native module `lumen._native`), `python/lumen`, `tests/` |
| `playground/` | Experiments and small test programs |
| `docs/` | Documentation |
| `.knowledge/` | Notes, decisions, references (not in git) |
| `assets/` | Icons, fonts, themes |

## Toolchain
Rust 1.99 stable (minimum 1.95 for burn 0.22), edition 2024. CUDA 13.4 is available for CubeCL's CUDA backend.

## Python (maturin + uv)
As in dsp-kitchen: `bindings/python/pyproject.toml` is the `lumen` package (maturin, native module
`lumen._native`, abi3 for CPython ≥ 3.12). The root `pyproject.toml` is the development environment
(`lumen-dev`, not a package) and installs `lumen` from `bindings/python` in editable mode
(Python 3.13, `.python-version`).

```bash
uv sync                       # builds the Rust extension via maturin and installs it (editable)
uv run python -c "import lumen; print(lumen.__version__)"
uv run maturin develop --uv -m bindings/python/lumen-py/Cargo.toml   # rebuild after Rust changes (add --release for speed)
uv run pytest -q bindings/python/tests
cargo test --workspace        # Rust side (extension-module feature off)
```

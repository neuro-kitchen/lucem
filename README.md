# lumen

An application for fluorescence brain-section analysis, written in Rust: GPU image processing with
CubeCL, models with Burn, and a desktop UI with GPUI (gpui-kit).

## Status
Empty Cargo workspace (2026-10-14). Crates are added under `crates/lumen-<part>`.

## Layout (as in dsp-kitchen)
| Path | Purpose |
|---|---|
| `crates/` | Workspace crates `lumen-*` |
| `lumen_py/` | Python SDK: PyO3 crate (`src/`, native module `lumen_bindings`) + Python package `lumen/`, tests |
| `playground/` | Experiments and small test programs |
| `docs/` | Documentation |
| `.knowledge/` | Notes, decisions, references |
| `assets/` | Icons, fonts, themes |

## Toolchain
Rust 1.99 stable (minimum 1.95 for burn 0.22), edition 2024. CUDA 13.4 is available for CubeCL's CUDA backend.

## Python (maturin + uv)
`pyproject.toml` builds `lumen_py` with maturin (as in dsp-kitchen): native module `lumen_bindings.lumen_bindings`,
Python package `lumen` (Python 3.13, `.python-version`).

```bash
uv sync                       # builds the Rust extension via maturin and installs it (editable)
uv run python -c "import lumen; print(lumen.__version__)"
uv run maturin develop --uv   # rebuild after Rust changes (add --release for speed)
uv run pytest -q lumen_py/tests
cargo test --workspace        # Rust side (extension-module feature off)
```

## Related
Python reference implementation of the current pipeline: `~/dev/pythonProjects/tdtni`
(preprocessing, InstanSeg nuclei, RFP soma model).

# lumen (Python)

GPU image processing for biology, written once in CubeCL. This is the Python package of the
[lumen](https://github.com/neuro-kitchen/lumen) workspace: the `lumen-py` crate builds the native
module `lumen._native`, and `python/lumen` is the package around it.

```bash
uv sync                                   # from the repository root: builds and installs lumen (editable)
uv run python -c "import lumen; print(lumen.__version__)"
uv run pytest -q bindings/python/tests
```

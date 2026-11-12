# lucem (Python)

GPU image processing for biology, written once in CubeCL. This is the Python package of the
[lucem](https://github.com/neuro-kitchen/lucem) workspace: the `lucem-py` crate builds the native
module `lucem._native`, and `python/lucem` is the package around it.

```bash
uv sync                                   # from the repository root: builds and installs lucem (editable)
uv run python -c "import lucem; print(lucem.__version__)"
uv run pytest -q bindings/python/tests
```

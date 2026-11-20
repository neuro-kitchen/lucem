# lucem-web

Platform-independent (`py3-none-any`) companion package bundling the static Lucem WebAssembly app (`index.html`, `loader.js`, `isolate-sw.js`, `mt/`).

Install through the extra on `lucem`:

```sh
uv add "lucem[web]"
# or: pip install "lucem[web]"
```

Then serve it locally:

```sh
uv run lucem web --open
```

Or from Python:

```python
import lucem
lucem.web.serve(port=8080, open_browser=True)
```

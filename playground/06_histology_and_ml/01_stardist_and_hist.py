# %% [markdown]
# # Histology and ML Model Plugins (`lucem[hist]`, `lucem[ml]`)
#
# Heavy optional domains live in companion packages installed via `uv` / `pip` extras:
#
# - `uv add "lucem[hist]"` — stain deconvolution (H&E, H-DAB), tissue masks, cell measurement
# - `uv add "lucem[stardist]"` / `"lucem[sam]"` / `"lucem[instanseg]"` / `"lucem[ml]"` — ML model plugins
# - `uv add "lucem[web]"` / `"lucem[app]"` — browser WASM viewer or native desktop workstation

# %%
import lucem

print("lucem version:", lucem.__version__)
print("ML plugin search dirs:", [str(d) for d in lucem.ml.plugin_dirs()])

plugins = lucem.ml.plugins()
if not plugins:
    print('No ML plugins found yet. Install with: uv add "lucem[stardist]" or "lucem[ml]"')
else:
    for p in plugins:
        print(f"Plugin: {p['name']} {p['version']} ({p['program']})")
        for m in p["models"]:
            state = "installed" if m.get("installed") else "downloads on first use"
            print(f"  - {m['id']}: {m['name']} [{state}]")

# %% [markdown]
# ## Optional histology module (`lucem[hist]`)

# %%
try:
    hist = lucem.hist
    print("lucem.hist is available:", dir(hist)[:8])
except ImportError:
    print('lucem.hist not installed. Install with: uv add "lucem[hist]"')

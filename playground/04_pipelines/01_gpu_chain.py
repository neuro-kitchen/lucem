# %% [markdown]
# # On-Device GPU Pipelines and Recipes
#
# Calling operators one at a time uploads and downloads the array at every step.
# `lucem.ops.Pipeline` executes the entire chain on the GPU in a single upload/download pass
# and tracks the combined `halo` automatically for tiled whole-slide processing.

# %%
import json
from pathlib import Path
import tempfile

import numpy as np

import lucem
from lucem.ops import Pipeline, stage

yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
image = np.clip(
    0.2 * (xx / 512.0) + 0.8 * np.exp(-((yy - 256) ** 2 + (xx - 256) ** 2) / (2.0 * 24.0**2)),
    0.0,
    1.0,
)

# %% [markdown]
# ## Run a multi-stage chain in one GPU pass

# %%
pipe = Pipeline([
    stage.gaussian_blur(sigma=1.5),
    stage.rolling_ball(radius=30.0),
    stage.otsu_threshold(),
])
print("combined halo (px):", pipe.halo)

mask = pipe.process(image)
print("foreground pixels:", int((mask > 0).sum()))

# %% [markdown]
# ## Save as a shareable `.lucem-recipe.json`
#
# Recipes use the same JSON format exported by **Make recipe** in the Lucem Pipeline toolbar.

# %%
recipe = {
    "schema": 1,
    "name": "nuclei_mask_recipe",
    "family": "Recipe",
    "description": "Smooth, subtract background, and auto-threshold nuclei",
    "steps": [
        {"op": "gaussian_blur", "sigma": 1.5},
        {"op": "rolling_ball", "radius": 30.0},
        {"op": "otsu_threshold"},
    ],
}
work = Path(tempfile.mkdtemp(prefix="lucem-recipe-"))
recipe_path = work / "nuclei_mask.lucem-recipe.json"
recipe_path.write_text(json.dumps(recipe, indent=2))
print("wrote recipe:", recipe_path)

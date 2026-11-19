//! Reusable pipeline recipes (`.lucem-recipe.json`, Tier 1 zero-code sharing) and one-click
//! Macro Recorder export of an op chain to a reproducible Python / CLI script.

use std::path::{Path, PathBuf};

use lucem_pipeline::Op;
use serde::{Deserialize, Serialize};

/// A saved, reusable sequence of GPU pipeline operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    #[serde(default = "schema_v1")]
    pub schema: u32,
    pub name: String,
    #[serde(default = "default_family")]
    pub family: String,
    #[serde(default)]
    pub description: String,
    pub steps: Vec<Op>,
}

fn schema_v1() -> u32 {
    1
}

fn default_family() -> String {
    "Recipe".to_string()
}

/// Folders scanned for `.lucem-recipe.json` files (`$LUCEM_PLUGINS/recipes` and `<config>/lucem/recipes`).
pub fn recipes_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(env_dirs) = std::env::var("LUCEM_PLUGINS") {
        for p in std::env::split_paths(&env_dirs) {
            dirs.push(p.join("recipes"));
        }
    }
    if let Some(cfg) = crate::data::prefs::Prefs::dir() {
        dirs.push(cfg.join("recipes"));
        dirs.push(cfg.join("plugins").join("recipes"));
    }
    dirs
}

/// Loads all valid `.lucem-recipe.json` recipes from [`recipes_dirs`].
pub fn discover_recipes() -> Vec<(PathBuf, Recipe)> {
    let mut out = Vec::new();
    for dir in recipes_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_recipe = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".lucem-recipe.json"));
            if !is_recipe {
                continue;
            }
            if let Ok(recipe) = load_recipe(&path) {
                out.push((path, recipe));
            }
        }
    }
    out.sort_by(|a, b| a.1.family.cmp(&b.1.family).then_with(|| a.1.name.cmp(&b.1.name)));
    out
}

/// Reads a single `.lucem-recipe.json` file (tolerates optional `id` keys on step objects).
pub fn load_recipe(path: &Path) -> Result<Recipe, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut val: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("invalid recipe JSON: {e}"))?;
    // Strip optional decorative `id` keys from step entries so `Op`'s `deny_unknown_fields` parses cleanly
    if let Some(steps) = val.get_mut("steps").and_then(|s| s.as_array_mut()) {
        for step in steps {
            if let Some(obj) = step.as_object_mut() {
                obj.remove("id");
            }
        }
    }
    serde_json::from_value(val).map_err(|e| format!("invalid recipe schema: {e}"))
}

/// Saves a recipe into `<config>/lucem/recipes/<slug>.lucem-recipe.json`.
pub fn save_recipe(recipe: &Recipe) -> Result<PathBuf, String> {
    let dir = crate::data::prefs::Prefs::dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("recipes");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let slug: String = recipe
        .name
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let path = dir.join(format!("{}.lucem-recipe.json", slug.trim_matches('_')));
    let json = serde_json::to_string_pretty(recipe).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Exports a sequence of Lucem ops as a reproducible Python script (Fiji Macro Recorder style).
pub fn export_chain_as_python(name: &str, steps: &[Op]) -> String {
    let json_steps = serde_json::to_string_pretty(steps).unwrap_or_else(|_| "[]".into());
    format!(
        r#"#!/usr/bin/env python3
"""Reproducible Lucem pipeline macro: {name}"""
import json
import subprocess
import sys
from pathlib import Path

PIPELINE_STEPS = json.loads(r'''{json_steps}''')


def run(input_image: str, output_dir: str = "out") -> None:
    spec_path = Path(output_dir) / "pipeline.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps({{"steps": PIPELINE_STEPS}}, indent=2))
    print(f"Wrote {{len(PIPELINE_STEPS)}} step(s) to {{spec_path}}")
    subprocess.run(["lucem", "run", str(spec_path), input_image], check=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python macro.py <image.ome.zarr> [out_dir]")
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out")
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_round_trips_and_exports_macro() {
        let recipe = Recipe {
            schema: 1,
            name: "Nuclei Clean".into(),
            family: "Segment".into(),
            description: "Denoise and label nuclei".into(),
            steps: vec![
                Op::GaussianBlur { sigma: 1.5 },
                Op::LiThreshold { bins: 256 },
                Op::FillHoles {},
                Op::Label {},
            ],
        };
        let text = serde_json::to_string_pretty(&recipe).unwrap();
        let parsed: Recipe = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed, recipe);
        let py = export_chain_as_python(&recipe.name, &recipe.steps);
        assert!(py.contains("gaussian_blur"));
        assert!(py.contains("li_threshold"));
        assert!(py.contains("fill_holes"));
    }
}

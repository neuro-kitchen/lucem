//! What is remembered between runs: recent images and projects, panel state. Stored as JSON in
//! the user's configuration folder (`$XDG_CONFIG_HOME/lumen` or `~/.config/lumen`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prefs {
    #[serde(default)]
    pub recent: Vec<PathBuf>,
    #[serde(default = "yes")]
    pub sidebar_open: bool,
    /// Rulers on the canvas edges (off by default: the image fills the canvas).
    #[serde(default)]
    pub rulers: bool,
    /// Explore's side panels: the tab shown on each side (`None`: the side is closed).
    #[serde(default = "left_tab")]
    pub left_tab: Option<String>,
    #[serde(default)]
    pub right_tab: Option<String>,
    /// The studio on the right (Affinity style): the tab shown in each group, and which groups
    /// are collapsed to their tab strip.
    #[serde(default)]
    pub studio_tabs: Vec<String>,
    #[serde(default)]
    pub studio_collapsed: Vec<bool>,
    /// Explore's side docks as the user arranged them (which panels, grouped how, active tabs,
    /// sizes): the dock's own state of each side, `{"left": …, "right": …}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dock_sides: Option<serde_json::Value>,
    /// The Pipeline docked under the views in Explore.
    #[serde(default)]
    pub pipeline_dock: bool,
    /// Views of the same image move and zoom together.
    #[serde(default)]
    pub link_views: bool,
    /// Explore's side docks shown (left: Layers; right: the studio).
    #[serde(default = "yes")]
    pub left_open: bool,
    #[serde(default = "yes")]
    pub right_open: bool,
    /// Appearance: dark (`Some(true)`), light (`Some(false)`) or the system's (`None`).
    #[serde(default = "dark")]
    pub theme: Option<bool>,
    /// The accent the user picked (`None`: lumen's blue). Saved as `"#rrggbb"`; a number from
    /// before (one of the old preset colours) still reads.
    #[serde(default, deserialize_with = "accent_or_old")]
    pub accent: Option<kitchen_ui::Rgb>,
    /// The programs scripts use (Python, uv, Ruff, Cargo, rustc), when not found on the `PATH`.
    #[serde(default)]
    pub tools: crate::data::tools::ToolPaths,
    /// What is around the image: 0 grey, 1 dark, 2 light.
    #[serde(default)]
    pub pasteboard: u8,
    /// Interface text size (px).
    #[serde(default = "font_size")]
    pub font_size: f32,
    /// Threads reading tiles of the first view (0: the machine's cores, at most 8).
    #[serde(default)]
    pub tile_threads: usize,
    /// The compute backend (`wgpu`, `cuda`, `cpu`; `None`: automatic).
    #[serde(default)]
    pub runtime: Option<String>,
    /// The model the Segment tool uses (a plugin's model id; `None`: the plugin's default).
    #[serde(default)]
    pub ml_model: Option<String>,
}

fn dark() -> Option<bool> {
    Some(true)
}

/// `"#rrggbb"`, `null`, or an old preset as a number (the old default blue is lumen's own, `None`).
fn accent_or_old<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<kitchen_ui::Rgb>, D::Error> {
    Ok(match serde_json::Value::deserialize(d)? {
        serde_json::Value::String(t) => kitchen_ui::Rgb::parse(&t),
        serde_json::Value::Number(n) => n.as_u64().filter(|&v| v != 0x3d65a8).map(|v| kitchen_ui::Rgb(v as u32 & 0xffffff)),
        _ => None,
    })
}

fn font_size() -> f32 {
    15.0
}

fn left_tab() -> Option<String> {
    Some("display".into())
}

fn yes() -> bool {
    true
}

impl Default for Prefs {
    fn default() -> Self {
        Self { recent: Vec::new(), sidebar_open: true, rulers: false, left_tab: left_tab(), right_tab: None, studio_tabs: Vec::new(), studio_collapsed: Vec::new(), pipeline_dock: false, link_views: false, left_open: true, right_open: true, theme: dark(), accent: None, tools: Default::default(), pasteboard: 0, font_size: font_size(), tile_threads: 0, runtime: None, ml_model: None, dock_sides: None }
    }
}

/// How many recent entries are kept.
const RECENT: usize = 12;

impl Prefs {
    /// The configuration folder.
    pub fn dir() -> Option<PathBuf> {
        if let Some(x) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
            return Some(PathBuf::from(x).join("lumen"));
        }
        if let Some(a) = std::env::var_os("APPDATA").filter(|v| !v.is_empty()) {
            return Some(PathBuf::from(a).join("lumen"));
        }
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config").join("lumen"))
    }

    pub fn path() -> Option<PathBuf> {
        Self::dir().map(|d| d.join("prefs.json"))
    }

    pub fn load(path: Option<&Path>) -> Self {
        path.and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self, path: Option<&Path>) {
        let Some(p) = path else { return };
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = std::fs::write(p, serde_json::to_string_pretty(self).unwrap_or_default());
    }

    /// Puts `path` first in the recent list.
    pub fn push_recent(&mut self, path: &Path) {
        self.recent.retain(|p| p != path);
        self.recent.insert(0, path.to_path_buf());
        self.recent.truncate(RECENT);
    }
}

#[cfg(test)]
mod accent_tests {
    use super::*;

    #[test]
    fn the_accent_reads_old_numbers_and_new_colours() {
        let read = |json: &str| serde_json::from_str::<Prefs>(&format!("{{\"accent\": {json}}}")).unwrap().accent;
        assert_eq!(read("4023720"), None, "the old default blue is lumen's own accent now");
        assert_eq!(read("2850694"), Some(kitchen_ui::Rgb(0x2b7f86)), "another old preset is kept");
        assert_eq!(read("\"#b0285f\""), Some(kitchen_ui::Rgb(0xb0285f)));
        assert_eq!(read("null"), None);
        let p = Prefs { accent: Some(kitchen_ui::Rgb(0x2b7f86)), ..Prefs::default() };
        assert_eq!(serde_json::from_str::<Prefs>(&serde_json::to_string(&p).unwrap()).unwrap(), p);
    }
}

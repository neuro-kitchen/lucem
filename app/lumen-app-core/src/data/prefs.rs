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
    #[serde(default = "yes")]
    pub rulers: bool,
    /// Explore's side panels: the tab shown on each side (`None`: the side is closed).
    #[serde(default = "left_tab")]
    pub left_tab: Option<String>,
    #[serde(default)]
    pub right_tab: Option<String>,
}

fn left_tab() -> Option<String> {
    Some("display".into())
}

fn yes() -> bool {
    true
}

impl Default for Prefs {
    fn default() -> Self {
        Self { recent: Vec::new(), sidebar_open: true, rulers: true, left_tab: left_tab(), right_tab: None }
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

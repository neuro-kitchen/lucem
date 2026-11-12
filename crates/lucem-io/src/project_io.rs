//! Serialization for `.lucem` non-destructive studio projects.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::IoResult;

/// A saved modifier node inside a `.lucem` project file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SavedModifier {
    pub kind: String,
    pub enabled: bool,
    pub primary_param: f64,
    pub secondary_param: f64,
}

/// Top-level `.lucem` project manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectManifest {
    pub name: String,
    pub canvas_width: usize,
    pub canvas_height: usize,
    pub source_path: Option<String>,
    pub phantom_kind: Option<String>,
    pub colormap: String,
    pub exposure: f32,
    pub gamma: f32,
    pub modifiers: Vec<SavedModifier>,
}

impl ProjectManifest {
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> IoResult<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> IoResult<Self> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

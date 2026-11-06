//! The project document: the whole working project as one versioned JSON file, so the app, the
//! command line and an agent (a future MCP server) read and change the same thing.
//!
//! A project holds its images (where they are and how each channel is shown), the processing
//! graph, the annotations and their classes, and notes. It changes only through
//! [`crate::domain::command::Command`]s; the app saves it after each change.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::annotation::{default_classes, Annotation, ClassDoc};
use super::graph::GraphDoc;

/// The document version this app reads and writes (`lumen_project` field).
pub const PROJECT_VERSION: u32 = 1;

/// File name of a project inside its folder.
pub const PROJECT_FILE: &str = "project.lumen.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectDoc {
    /// Document version; must equal [`PROJECT_VERSION`].
    pub lumen_project: u32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub images: Vec<ImageRef>,
    /// The image shown in Explore.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_image: Option<String>,
    #[serde(default, skip_serializing_if = "GraphDoc::is_empty")]
    pub graph: GraphDoc,
    #[serde(default)]
    pub annotations: Vec<Annotation>,
    #[serde(default = "default_classes")]
    pub classes: Vec<ClassDoc>,
    #[serde(default)]
    pub notes: Vec<NoteDoc>,
}

impl Default for ProjectDoc {
    fn default() -> Self {
        Self {
            lumen_project: PROJECT_VERSION,
            name: "Untitled".into(),
            images: Vec::new(),
            active_image: None,
            graph: GraphDoc::default(),
            annotations: Vec::new(),
            classes: default_classes(),
            notes: Vec::new(),
        }
    }
}

/// An image of the project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageRef {
    /// `img1`, `img2`, …
    pub id: String,
    /// The file or container (relative to the project folder when inside it); `sample:<name>`
    /// for a generated sample.
    pub path: PathBuf,
    pub name: String,
    /// How each channel is shown (filled when the image is first opened).
    #[serde(default)]
    pub channels: Vec<ChannelDoc>,
}

/// How one channel is shown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChannelDoc {
    pub label: String,
    #[serde(default = "yes")]
    pub visible: bool,
    pub color: [u8; 3],
    /// Raw values shown as black and as full colour.
    pub min: f32,
    pub max: f32,
    #[serde(default = "one")]
    pub gamma: f32,
}

fn yes() -> bool {
    true
}
fn one() -> f32 {
    1.0
}

/// A note: free text, optionally pinned to a place on an image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoteDoc {
    #[serde(default)]
    pub id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// Level-0 position on the image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<[f64; 2]>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("project file: {0}")]
    Parse(String),
    #[error("project version {0} is not supported (this app reads version {PROJECT_VERSION})")]
    Version(u32),
    #[error("{0}")]
    Io(String),
}

impl ProjectDoc {
    pub fn from_json(text: &str) -> Result<Self, ProjectError> {
        let doc: Self = serde_json::from_str(text).map_err(|e| ProjectError::Parse(e.to_string()))?;
        if doc.lumen_project != PROJECT_VERSION {
            return Err(ProjectError::Version(doc.lumen_project));
        }
        Ok(doc)
    }

    pub fn load(path: &Path) -> Result<Self, ProjectError> {
        let text = std::fs::read_to_string(path).map_err(|e| ProjectError::Io(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a project always serializes")
    }

    /// Writes the JSON through a temporary file, so a reader never sees half of it.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, self.to_json())?;
        std::fs::rename(tmp, path)
    }

    /// JSON Schema of the document (for agents and editors).
    pub fn json_schema() -> String {
        serde_json::to_string_pretty(&schemars::schema_for!(ProjectDoc)).expect("a schema always serializes")
    }

    pub fn image(&self, id: &str) -> Option<&ImageRef> {
        self.images.iter().find(|i| i.id == id)
    }

    pub fn image_mut(&mut self, id: &str) -> Option<&mut ImageRef> {
        self.images.iter_mut().find(|i| i.id == id)
    }

    pub fn active(&self) -> Option<&ImageRef> {
        self.active_image.as_deref().and_then(|id| self.image(id))
    }

    pub fn class(&self, name: &str) -> Option<&ClassDoc> {
        self.classes.iter().find(|c| c.name == name)
    }

    /// Annotations of image `id`.
    pub fn annotations_of<'a>(&'a self, id: &'a str) -> impl Iterator<Item = &'a Annotation> + 'a {
        self.annotations.iter().filter(move |a| a.image == id)
    }

    pub fn new_image_id(&self) -> String {
        fresh("img", |id| self.image(id).is_some())
    }

    pub fn new_annotation_id(&self) -> String {
        fresh("a", |id| self.annotations.iter().any(|a| a.id == id))
    }

    pub fn new_note_id(&self) -> String {
        fresh("n", |id| self.notes.iter().any(|n| n.id == id))
    }

    /// `path` as stored in the project: relative to `folder` when inside it.
    pub fn stored_path(path: &Path, folder: Option<&Path>) -> PathBuf {
        match folder.and_then(|f| path.strip_prefix(f).ok()) {
            Some(rel) if !rel.as_os_str().is_empty() => rel.to_path_buf(),
            _ => path.to_path_buf(),
        }
    }

    /// A stored path resolved against the project folder.
    pub fn resolve(path: &Path, folder: Option<&Path>) -> PathBuf {
        if path.is_absolute() || path.to_string_lossy().starts_with("sample:") {
            return path.to_path_buf();
        }
        folder.map_or_else(|| path.to_path_buf(), |f| f.join(path))
    }
}

fn fresh(prefix: &str, taken: impl Fn(&str) -> bool) -> String {
    (1..).map(|i| format!("{prefix}{i}")).find(|id| !taken(id)).expect("ids are unbounded")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_version() {
        let d = ProjectDoc::default();
        assert_eq!(ProjectDoc::from_json(&d.to_json()).unwrap(), d);
        assert!(matches!(ProjectDoc::from_json(r#"{"lumen_project": 9}"#), Err(ProjectError::Version(9))));
        assert!(ProjectDoc::from_json(r#"{"lumen_project": 1, "imagez": []}"#).is_err());
        assert!(ProjectDoc::json_schema().contains("lumen_project"));
    }

    #[test]
    fn paths_inside_the_project_are_relative() {
        let f = Path::new("/p");
        assert_eq!(ProjectDoc::stored_path(Path::new("/p/data/x.ome.zarr"), Some(f)), PathBuf::from("data/x.ome.zarr"));
        assert_eq!(ProjectDoc::stored_path(Path::new("/q/x.tif"), Some(f)), PathBuf::from("/q/x.tif"));
        assert_eq!(ProjectDoc::resolve(Path::new("data/x.ome.zarr"), Some(f)), PathBuf::from("/p/data/x.ome.zarr"));
    }
}

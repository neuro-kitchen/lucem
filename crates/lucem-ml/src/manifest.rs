//! What a plugin offers, as data: `<program> info` prints a [`PluginInfo`] as JSON.

use serde::{Deserialize, Serialize};

/// What a model does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    /// A mask from prompts (points, boxes): SAM.
    InteractiveSegmentation,
    /// Every object of a region as labels: nuclei, cells.
    Segmentation,
    /// Points or boxes of objects.
    Detection,
    /// A class per object or region.
    Classification,
    /// A feature vector per patch (a foundation model's embedding), for classifiers downstream.
    Embedding,
}

impl Task {
    pub fn title(self) -> &'static str {
        match self {
            Task::InteractiveSegmentation => "Interactive segmentation",
            Task::Segmentation => "Segmentation",
            Task::Detection => "Detection",
            Task::Classification => "Classification",
            Task::Embedding => "Embedding",
        }
    }
}

/// A file a model needs, verified by hash before use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Weight {
    pub file: String,
    pub url: String,
    /// Lowercase hex SHA-256 (a file without one is refused).
    pub sha256: String,
    pub bytes: u64,
}

/// One model of a plugin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique among all plugins (`sam2-hiera-tiny`).
    pub id: String,
    pub name: String,
    /// The upstream family (`sam`, `stardist`, …).
    pub family: String,
    pub task: Task,
    /// The images it is made for (`histology`, `fluorescence`, `natural`): the app lists a
    /// histology model where histology is analysed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domains: Vec<String>,
    pub description: String,
    /// The licence of the model and its weights (SPDX).
    pub licence: String,
    /// Whom to credit: the paper and the upstream code.
    #[serde(default)]
    pub paper: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
    pub weights: Vec<Weight>,
    /// Its weights are downloaded and verified.
    #[serde(default)]
    pub installed: bool,
    /// The plugin's suggestion when the user has not chosen.
    #[serde(default)]
    pub default: bool,
    /// Channels its images have: 3 (RGB, e.g. H&E) or 1 (one fluorescence channel); 0: not
    /// said (RGB).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub channels: u32,
}

fn is_zero(v: &u32) -> bool {
    *v == 0
}

impl ModelInfo {
    pub fn download_bytes(&self) -> u64 {
        self.weights.iter().map(|w| w.bytes).sum()
    }
}

/// Whether the plugin's runtime can run here, and why not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub ready: bool,
    /// What runs the models (`ONNX Runtime 1.22.1 (CUDA, CPU)`), or what is missing and how to
    /// get it.
    pub detail: String,
    /// A GPU build of the runtime the plugin can install (`install runtime-cuda`), when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<GpuRuntime>,
}

/// A plugin's optional GPU runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuRuntime {
    pub installed: bool,
    /// Installed and everything it needs is there (a driver, libraries such as cuDNN).
    pub usable: bool,
    pub bytes: u64,
    /// What it is, or what it still needs.
    pub detail: String,
}

/// A plugin program and what it offers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    /// The program's licence (it may differ from lucem's).
    pub licence: String,
    pub runtime: RuntimeStatus,
    pub models: Vec<ModelInfo>,
    /// Where it was found (set by [`crate::discover`], not by the plugin).
    #[serde(default)]
    pub program: String,
}

impl PluginInfo {
    pub fn model(&self, id: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.id == id)
    }
}

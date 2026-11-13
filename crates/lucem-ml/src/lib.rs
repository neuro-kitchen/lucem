//! `lucem-ml`: how lucem talks to machine-learning models, without containing any.
//!
//! Models come as **plugins**: separate programs named `lucem-ml-<family>` (as dsp-kitchen's
//! sorters), each built from one crate per upstream model family under `models/`, each under its
//! upstream's licence, with its own runtime (ONNX Runtime, Burn, …). The app and the command line
//! never link them, so a model never adds weight, licences or a runtime to lucem itself, and a
//! plugin can be installed, updated or removed on its own.
//!
//! - [`manifest`]: what a plugin offers — its models, their task (interactive segmentation,
//!   segmentation, detection, classification), weights, licence and provenance — as data.
//! - [`protocol`]: the requests and events, JSON lines over the program's stdin and stdout; images
//!   and masks travel as raw files in a scratch folder.
//! - [`discover`]: finding the plugin programs (`$LUCEM_PLUGINS`, the user's plugin folder, next
//!   to the app) and asking each for its manifest.
//! - [`client`]: a plugin running in server mode (`<program> serve`), kept alive between requests
//!   so a model stays loaded (and SAM keeps its image embedding between clicks).
//!
//! Workspace layout (see `.tasks/2026-11-10_round5-tasks.md`):
//! ```text
//! crates/lucem-ml            this contract (the app depends on it; serde only)
//! hub/lucem-hub              verified downloads and the model cache (all network access)
//! models/<family>            one crate per upstream family (models/sam: SAM 2, Apache-2.0)
//! plugins/ml/lucem-ml-<fam>  the program the app runs, over the protocol
//! ```

/// The id `install` takes (and `Request::Install` carries) for a plugin's runtime itself rather
/// than a model: a plugin whose runtime is not ready installs it under this id.
pub const RUNTIME_ID: &str = "runtime";
/// The id for a plugin's GPU build of its runtime (see [`manifest::GpuRuntime`]).
pub const RUNTIME_CUDA_ID: &str = "runtime-cuda";

pub mod client;
pub mod discover;
pub mod legacy;
pub mod manifest;
pub mod protocol;
pub mod tune;

pub use client::{set_plugin_runtime, PluginProcess};
pub use discover::{discover, plugin_dirs, PROGRAM_PREFIX};
pub use manifest::{GpuRuntime, ModelInfo, PluginInfo, RuntimeStatus, Task, Weight};
pub use tune::{tune_plugins, PluginTune, TuneOutcome};
pub use protocol::{Event, InstanceClass, InstancesResult, MaskOut, Prompt, RasterRef, Request, SegmentResult};

//! User intents/events emitted by the Startup screen (`StartupEvent`).

use std::path::PathBuf;

use lumen_io::PhantomKind;

use crate::navigation::LeftRailTab;

/// Intents dispatched from the Startup screen to [`super::StartupViewModel`].
#[derive(Debug, Clone)]
pub enum StartupEvent {
    /// Switch the active tab in the Left Option Rail.
    SelectRailTab(LeftRailTab),
    /// Open native OS file dialog to pick an image (`.png`, `.tif`, `.tiff`, `.jpg`, `.webp`, etc.).
    PromptOpenImageFile,
    /// Open native OS folder dialog to pick an `.ome.zarr` container directory.
    PromptOpenZarrFolder,
    /// Load a specific image file path (from recent files or drag-and-drop).
    OpenImagePath(PathBuf),
    /// Create a blank canvas using the current `BlankCanvasSpec` dimensions and background.
    CreateBlankCanvas,
    /// Cycle the blank-canvas resolution preset (`512×512`, `1024×768`, `1920×1080`, `2048×2048`).
    SelectBlankSize(usize, usize),
    /// Adjust blank-canvas background intensity (`0.0`, `0.08`, `0.5`, `1.0`).
    SelectBlankBackground(f32),
    /// Launch one of the curated ArtCraft × VectorCraft × Blender project presets.
    LaunchPreset(usize),
    /// Load one of the built-in synthetic scientific/creative phantoms directly.
    LoadPhantom(PhantomKind),
    /// Dismiss the center Project Hub modal and enter the blank/current canvas directly.
    EnterBlankCanvasNow,
}

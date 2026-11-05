//! User intent events emitted by the Studio & Explore Workspace (`StudioEvent`).

use std::path::PathBuf;

use lumen_pipeline::ModifierKind;
use lumen_view::ColormapLut;

use crate::domain::{ActiveTool, WorkspaceMode};

#[derive(Debug, Clone)]
pub enum StudioEvent {
    /// Return to the Startup Blank Canvas + Center Project Hub.
    OpenProjectHub,
    /// Open the native OS file picker to load a new image into the Studio.
    PromptOpenImage,
    /// Load a dropped or selected file path directly into the Studio.
    LoadImageFile(PathBuf),
    /// Switch the Blender-style top workspace mode tab (`CraftStudio`, `NodePipeline`, `AnalyticsLab`).
    SwitchWorkspaceMode(WorkspaceMode),
    /// Switch the active left-shelf viewport tool (`ExplorePan`, `ZoomLens`, `PixelProbe`, `VectorContour`).
    SelectTool(ActiveTool),
    /// Toggle the bottom Polars/Arrow Analytics Drawer open or closed.
    ToggleAnalyticsDrawer,
    /// Zoom in by a discrete step around the viewport center.
    ZoomIn,
    /// Zoom out by a discrete step around the viewport center.
    ZoomOut,
    /// Reset zoom to `100%` (`1:1` native pixel resolution) and center pan.
    ZoomOneToOne,
    /// Fit the active image inside the viewport with margin.
    FitToViewport,
    /// Scroll-wheel or trackpad zoom anchored at cursor offset `(rel_x, rel_y)` from viewport center.
    ZoomAtCursor {
        factor: f32,
        rel_x: f32,
        rel_y: f32,
    },
    /// Begin click-and-drag panning at viewport coordinates `(x, y)`.
    BeginPan {
        x: f32,
        y: f32,
    },
    /// Continue click-and-drag panning at viewport coordinates `(x, y)` and update pixel probe.
    PointerMove {
        x: f32,
        y: f32,
        view_w: f32,
        view_h: f32,
    },
    /// End click-and-drag panning.
    EndPan,
    /// Click on the viewport canvas at `(x, y)` (used when `ZoomLens` or `VectorContour` tool is active).
    CanvasClick {
        x: f32,
        y: f32,
        view_w: f32,
        view_h: f32,
    },
    /// Toggle dark checkerboard/grid background under the image.
    ToggleGrid,
    /// Toggle VectorCraft Marching Squares iso-contour overlay on the canvas.
    ToggleContours,
    /// Select a scientific/creative display colormap LUT.
    SelectColormap(ColormapLut),
    /// Adjust display exposure by `delta`.
    AdjustExposure(f32),
    /// Adjust display gamma by `delta`.
    AdjustGamma(f32),
    /// Toggle display tonal inversion.
    ToggleInvert,
    /// Reset display tonemap settings (`exposure = 1.0`, `gamma = 1.0`, `invert = false`).
    ResetTonemap,
    /// Toggle visibility of a layer in the ArtCraft layer stack.
    ToggleLayerVisibility(u64),
    /// Append a new non-destructive GPU modifier to the Blender-style modifier stack.
    AddModifier(ModifierKind),
    /// Enable or disable a modifier node in the stack.
    ToggleModifier(u64),
    /// Adjust a modifier's primary parameter by `delta`.
    AdjustModifierPrimary {
        id: u64,
        delta: f64,
    },
    /// Remove a modifier node from the stack.
    RemoveModifier(u64),
    /// Clear all modifiers from the stack and restore the base raster image.
    ClearModifiers,
}

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
    /// Open the native OS folder picker to load an `.ome.zarr` container into the Studio.
    PromptOpenZarrFolder,
    /// Load a dropped or selected file path directly into the Studio.
    LoadImageFile(PathBuf),
    /// Toggle the top-bar `Operations ▾` dropdown menu open or closed.
    ToggleOperationsMenu,
    /// Close the top-bar `Operations ▾` dropdown menu.
    CloseOperationsMenu,
    /// Switch the Blender-style top workspace mode tab (`Compositing`, `Vector & Paint`, `RegionProps`).
    SwitchWorkspaceMode(WorkspaceMode),
    /// Switch the active right-shelf viewport tool.
    SelectTool(ActiveTool),
    /// Adjust the active brush radius in pixels.
    AdjustBrushRadius(f32),
    /// Adjust the active brush strength / opacity (`0.05..=1.0`).
    AdjustBrushStrength(f32),
    /// Adjust the active vector stroke width in pixels.
    AdjustStrokeWidth(f32),
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
    /// Begin click-and-drag panning or painting at viewport coordinates `(x, y)`.
    BeginPan {
        x: f32,
        y: f32,
    },
    /// Continue click-and-drag panning or painting at viewport coordinates `(x, y)` and update pixel probe.
    PointerMove {
        x: f32,
        y: f32,
        view_w: f32,
        view_h: f32,
    },
    /// End click-and-drag panning or painting.
    EndPan,
    /// Click on the viewport canvas at `(x, y)`.
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
    /// Toggle visibility of a layer in the layer stack.
    ToggleLayerVisibility(u64),
    /// Append a new non-destructive GPU modifier to the Operations Tree.
    AddModifier(ModifierKind),
    /// Enable or disable a modifier node in the Operations Tree.
    ToggleModifier(u64),
    /// Adjust a modifier's primary parameter by `delta`.
    AdjustModifierPrimary {
        id: u64,
        delta: f64,
    },
    /// Remove a modifier node from the Operations Tree.
    RemoveModifier(u64),
    /// Clear all modifiers from the Operations Tree and restore the base raster image.
    ClearModifiers,
    /// Toggle automatic zoom-based pyramid level switching (`Auto-LOD`) on an `.ome.zarr` container.
    ToggleOmeZarrAutoPyramid,
    /// Switch active pyramid level (`s0`, `s1`, `s2`, `s3`) inside an opened `.ome.zarr` container.
    SelectOmeZarrLevel(String),
    /// Switch active OME-Zarr channel (`None` = Composite, `Some(c)` = channel `c`).
    SelectOmeZarrChannel(Option<usize>),
    /// Switch between primary image pyramid (`None`) and segmentation label group (`Some("cells")`, `Some("nuclei")`).
    SelectOmeZarrLabelGroup(Option<String>),
}

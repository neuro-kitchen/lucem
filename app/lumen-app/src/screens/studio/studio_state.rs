//! Immutable UI state for the Studio & Explore Workspace (`StudioUiState`).

use std::sync::Arc;

use gpui_kit::RenderImage;
use lumen_io::LoadedImage;
use lumen_pipeline::{ModifierNode, PipelineExecutionResult};
use lumen_view::{TonemapSettings, ViewportCamera};

use crate::domain::{ActiveTool, LayerKind, StudioLayer, WorkspaceMode};

#[derive(Clone)]
pub struct StudioUiState {
    pub active_mode: WorkspaceMode,
    pub active_tool: ActiveTool,
    pub source_image: LoadedImage,
    pub render_image: Arc<RenderImage>,
    pub camera: ViewportCamera,
    pub tonemap: TonemapSettings,
    pub layers: Vec<StudioLayer>,
    pub modifiers: Vec<ModifierNode>,
    pub next_modifier_id: u64,
    pub pipeline_result: Option<PipelineExecutionResult>,
    pub show_analytics_drawer: bool,
    pub is_panning: bool,
    pub last_mouse_pos: Option<(f32, f32)>,
    pub cursor_img_yx: Option<(usize, usize)>,
    pub cursor_intensity: Option<f32>,
    pub viewport_size: (f32, f32),
    pub compute_target_name: String,
    pub status_banner: String,
}

impl StudioUiState {
    pub fn default_layers_for(image_name: &str) -> Vec<StudioLayer> {
        vec![
            StudioLayer {
                id: 1,
                name: format!("{image_name} (Base Raster)"),
                kind: LayerKind::RasterImage,
                visible: true,
                opacity: 1.0,
            },
            StudioLayer {
                id: 2,
                name: "Marching Squares Contours".to_string(),
                kind: LayerKind::VectorContours,
                visible: false,
                opacity: 0.9,
            },
            StudioLayer {
                id: 3,
                name: "Connected Regions (Otsu)".to_string(),
                kind: LayerKind::RegionOverlay,
                visible: true,
                opacity: 0.85,
            },
        ]
    }
}

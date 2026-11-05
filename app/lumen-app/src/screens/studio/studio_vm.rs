//! ViewModel for the Studio & Explore Workspace (`StudioViewModel`).
//!
//! Coordinates GPU device uploads (`DeviceArray<f32>`), non-destructive modifier evaluation via
//! [`PipelineExecutor`], live Polars/Arrow [`AnalyticsViewModel`] updates, BGRA8 [`RenderImage`]
//! tonemapping, and interactive pan/zoom/explore camera transformations.

use lumen_core::memory::DeviceArray;
use lumen_io::{generate_phantom, LoadedImage, PhantomKind};
use lumen_pipeline::{ModifierKind, ModifierNode, PipelineExecutor};
use lumen_view::{build_render_image, ColormapLut, TonemapSettings, ViewportCamera};

use super::studio_event::StudioEvent;
use super::studio_state::StudioUiState;
use crate::data::DeviceService;
use crate::domain::{ActiveTool, LayerKind, WorkspaceMode};
use crate::screens::analytics::analytics_vm::AnalyticsViewModel;

pub struct StudioViewModel {
    pub state: StudioUiState,
    /// Cached base image on the CubeCL device so modifier stack adjustments don't re-upload from host.
    base_device_image: Option<DeviceArray<f32>>,
}

impl StudioViewModel {
    pub fn new(device_service: &DeviceService, analytics_vm: &mut AnalyticsViewModel) -> Self {
        let initial_img = generate_phantom(PhantomKind::FluorescenceNuclei, 512, 512);
        let tonemap = TonemapSettings::default();
        let render_image = build_render_image(
            initial_img.width,
            initial_img.height,
            &initial_img.luminance_f32,
            &tonemap,
            None,
        );

        let mut vm = Self {
            state: StudioUiState {
                active_mode: WorkspaceMode::CraftStudio,
                active_tool: ActiveTool::ExplorePan,
                layers: StudioUiState::default_layers_for(&initial_img.name),
                source_image: initial_img,
                render_image,
                camera: ViewportCamera::default(),
                tonemap,
                modifiers: Vec::new(),
                next_modifier_id: 1,
                pipeline_result: None,
                show_analytics_drawer: true,
                is_panning: false,
                last_mouse_pos: None,
                cursor_img_yx: None,
                cursor_intensity: None,
                viewport_size: (920.0, 620.0),
                compute_target_name: device_service.backend_label().to_string(),
                status_banner: "Ready — Scroll to zoom, drag to pan, or add GPU modifiers".to_string(),
            },
            base_device_image: None,
        };
        vm.load_image(vm.state.source_image.clone(), device_service, analytics_vm);
        vm
    }

    /// Loads a new [`LoadedImage`] into the Studio, uploads its `f32` buffer to the CubeCL device,
    /// fits the camera to the viewport, and runs the initial GPU pipeline + Polars analytics pass.
    pub fn load_image(
        &mut self,
        image: LoadedImage,
        device_service: &DeviceService,
        analytics_vm: &mut AnalyticsViewModel,
    ) {
        let w = image.width;
        let h = image.height;
        self.state.layers = StudioUiState::default_layers_for(&image.name);
        self.state.camera.fit_to_viewport(
            w,
            h,
            self.state.viewport_size.0,
            self.state.viewport_size.1,
            36.0,
        );
        self.state.cursor_img_yx = None;
        self.state.cursor_intensity = None;

        let client = device_service.client();
        self.base_device_image = image.to_device_luminance(client).ok();
        self.state.source_image = image;
        self.recompute_pipeline(device_service, analytics_vm);
    }

    /// Re-runs the non-destructive GPU modifier stack on `base_device_image` and refreshes both the
    /// viewport `RenderImage` and the Polars/Arrow `AnalyticsViewModel`.
    pub fn recompute_pipeline(
        &mut self,
        device_service: &DeviceService,
        analytics_vm: &mut AnalyticsViewModel,
    ) {
        let client = device_service.client();
        if self.base_device_image.is_none() {
            self.base_device_image = self.state.source_image.to_device_luminance(client).ok();
        }

        if let Some(ref dev_img) = self.base_device_image {
            match PipelineExecutor::execute(client, dev_img, &self.state.modifiers, true) {
                Ok(res) => {
                    analytics_vm.ingest_pipeline_result(&res);
                    self.state.status_banner = format!(
                        "GPU Pipeline ({}) evaluated {} modifier(s) in {:.2} ms — {} regions in Polars DataFrame",
                        self.state.compute_target_name,
                        self.state.modifiers.iter().filter(|m| m.enabled).count(),
                        res.elapsed_ms,
                        res.regions.len()
                    );
                    self.state.pipeline_result = Some(res);
                }
                Err(err) => {
                    self.state.status_banner = format!("Pipeline error: {err}");
                }
            }
        }
        self.rebuild_display_image();
    }

    /// Re-tonemaps the current processed host buffer into an `Arc<RenderImage>` without re-running
    /// the GPU kernels (used when changing LUT, exposure, gamma, or contour visibility).
    pub fn rebuild_display_image(&mut self) {
        let (w, h) = (self.state.source_image.width, self.state.source_image.height);
        let luminance: &[f32] = self
            .state
            .pipeline_result
            .as_ref()
            .map(|r| r.output_host.as_slice())
            .unwrap_or(self.state.source_image.luminance_f32.as_slice());

        let contour_points: Option<Vec<Vec<[f64; 2]>>> = if self.state.camera.show_contours {
            self.state.pipeline_result.as_ref().map(|r| r.contours.clone())
        } else {
            None
        };

        // Syncblack/white points with current image dynamic range if available
        if let Some(ref res) = self.state.pipeline_result {
            let min_f = res.min_val as f32;
            let max_f = res.max_val as f32;
            if (max_f - min_f) > 1e-5 {
                self.state.tonemap.black_point = min_f;
                self.state.tonemap.white_point = max_f;
            }
        }

        self.state.render_image = build_render_image(
            w,
            h,
            luminance,
            &self.state.tonemap,
            contour_points.as_deref(),
        );
    }

    /// Handles a [`StudioEvent`] emitted by any Studio screen component.
    pub fn on_event(
        &mut self,
        event: StudioEvent,
        device_service: &DeviceService,
        analytics_vm: &mut AnalyticsViewModel,
    ) {
        match event {
            StudioEvent::OpenProjectHub | StudioEvent::PromptOpenImage | StudioEvent::LoadImageFile(_) => {
                // Handled by root LumenApp coordinator
            }
            StudioEvent::SwitchWorkspaceMode(mode) => {
                self.state.active_mode = mode;
                if mode == WorkspaceMode::AnalyticsLab {
                    self.state.show_analytics_drawer = true;
                }
            }
            StudioEvent::SelectTool(tool) => {
                self.state.active_tool = tool;
                if tool == ActiveTool::VectorContour && !self.state.camera.show_contours {
                    self.state.camera.show_contours = true;
                    self.sync_contour_layer_flag();
                    self.rebuild_display_image();
                }
            }
            StudioEvent::ToggleAnalyticsDrawer => {
                self.state.show_analytics_drawer = !self.state.show_analytics_drawer;
            }
            StudioEvent::ZoomIn => {
                self.state.camera.zoom_step(1.25);
            }
            StudioEvent::ZoomOut => {
                self.state.camera.zoom_step(1.0 / 1.25);
            }
            StudioEvent::ZoomOneToOne => {
                self.state.camera.reset_one_to_one();
            }
            StudioEvent::FitToViewport => {
                self.state.camera.fit_to_viewport(
                    self.state.source_image.width,
                    self.state.source_image.height,
                    self.state.viewport_size.0,
                    self.state.viewport_size.1,
                    36.0,
                );
            }
            StudioEvent::ZoomAtCursor {
                factor,
                rel_x,
                rel_y,
            } => {
                self.state.camera.zoom_at_cursor_offset(factor, rel_x, rel_y);
            }
            StudioEvent::BeginPan { x, y } => {
                self.state.is_panning = true;
                self.state.last_mouse_pos = Some((x, y));
            }
            StudioEvent::PointerMove {
                x,
                y,
                view_w,
                view_h,
            } => {
                if view_w > 64.0 && view_h > 64.0 {
                    self.state.viewport_size = (view_w, view_h);
                }
                if self.state.is_panning {
                    if let Some((lx, ly)) = self.state.last_mouse_pos {
                        let dx = x - lx;
                        let dy = y - ly;
                        self.state.camera.pan_by(dx, dy);
                    }
                    self.state.last_mouse_pos = Some((x, y));
                }
                let img_w = self.state.source_image.width;
                let img_h = self.state.source_image.height;
                let coords = self.state.camera.screen_to_image(x, y, img_w, img_h, view_w, view_h);
                self.state.cursor_img_yx = coords;
                self.state.cursor_intensity = coords.and_then(|(py, px_i)| {
                    let buf = self
                        .state
                        .pipeline_result
                        .as_ref()
                        .map(|r| r.output_host.as_slice())
                        .unwrap_or(self.state.source_image.luminance_f32.as_slice());
                    buf.get(py * img_w + px_i).copied()
                });
            }
            StudioEvent::EndPan => {
                self.state.is_panning = false;
                self.state.last_mouse_pos = None;
            }
            StudioEvent::CanvasClick {
                x,
                y,
                view_w,
                view_h,
            } => {
                if self.state.active_tool == ActiveTool::ZoomLens {
                    let rel_x = x - 0.5 * view_w;
                    let rel_y = y - 0.5 * view_h;
                    self.state.camera.zoom_at_cursor_offset(1.35, rel_x, rel_y);
                }
            }
            StudioEvent::ToggleGrid => {
                self.state.camera.show_grid = !self.state.camera.show_grid;
            }
            StudioEvent::ToggleContours => {
                self.state.camera.show_contours = !self.state.camera.show_contours;
                self.sync_contour_layer_flag();
                self.rebuild_display_image();
            }
            StudioEvent::SelectColormap(lut) => {
                self.state.tonemap.colormap = lut;
                self.rebuild_display_image();
            }
            StudioEvent::AdjustExposure(delta) => {
                self.state.tonemap.exposure = (self.state.tonemap.exposure + delta).clamp(0.15, 5.0);
                self.rebuild_display_image();
            }
            StudioEvent::AdjustGamma(delta) => {
                self.state.tonemap.gamma = (self.state.tonemap.gamma + delta).clamp(0.2, 4.0);
                self.rebuild_display_image();
            }
            StudioEvent::ToggleInvert => {
                self.state.tonemap.invert = !self.state.tonemap.invert;
                self.rebuild_display_image();
            }
            StudioEvent::ResetTonemap => {
                let lut: ColormapLut = self.state.tonemap.colormap;
                self.state.tonemap = TonemapSettings {
                    colormap: lut,
                    ..TonemapSettings::default()
                };
                self.rebuild_display_image();
            }
            StudioEvent::ToggleLayerVisibility(layer_id) => {
                let mut toggled_contour = false;
                for layer in &mut self.state.layers {
                    if layer.id == layer_id {
                        layer.visible = !layer.visible;
                        if layer.kind == LayerKind::VectorContours {
                            self.state.camera.show_contours = layer.visible;
                            toggled_contour = true;
                        }
                    }
                }
                if toggled_contour {
                    self.rebuild_display_image();
                }
            }
            StudioEvent::AddModifier(kind) => {
                let id = self.state.next_modifier_id;
                self.state.next_modifier_id += 1;
                self.state.modifiers.push(ModifierNode::new(id, kind));
                self.recompute_pipeline(device_service, analytics_vm);
            }
            StudioEvent::ToggleModifier(id) => {
                if let Some(m) = self.state.modifiers.iter_mut().find(|m| m.id == id) {
                    m.enabled = !m.enabled;
                }
                self.recompute_pipeline(device_service, analytics_vm);
            }
            StudioEvent::AdjustModifierPrimary { id, delta } => {
                if let Some(m) = self.state.modifiers.iter_mut().find(|m| m.id == id) {
                    let (lo, hi) = m.kind.bounds();
                    m.primary_param = (m.primary_param + delta).clamp(lo, hi);
                }
                self.recompute_pipeline(device_service, analytics_vm);
            }
            StudioEvent::RemoveModifier(id) => {
                self.state.modifiers.retain(|m| m.id != id);
                self.recompute_pipeline(device_service, analytics_vm);
            }
            StudioEvent::ClearModifiers => {
                self.state.modifiers.clear();
                self.recompute_pipeline(device_service, analytics_vm);
            }
        }
    }

    /// Applies an initial set of modifiers when launching a preset from the Startup Project Hub.
    pub fn apply_preset_modifiers(
        &mut self,
        kinds: &[ModifierKind],
        show_contours: bool,
        device_service: &DeviceService,
        analytics_vm: &mut AnalyticsViewModel,
    ) {
        self.state.modifiers.clear();
        for &kind in kinds {
            let id = self.state.next_modifier_id;
            self.state.next_modifier_id += 1;
            self.state.modifiers.push(ModifierNode::new(id, kind));
        }
        self.state.camera.show_contours = show_contours;
        self.sync_contour_layer_flag();
        self.recompute_pipeline(device_service, analytics_vm);
    }

    fn sync_contour_layer_flag(&mut self) {
        for layer in &mut self.state.layers {
            if layer.kind == LayerKind::VectorContours {
                layer.visible = self.state.camera.show_contours;
            }
        }
    }
}

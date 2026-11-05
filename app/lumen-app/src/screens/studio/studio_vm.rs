//! ViewModel for the Studio & Explore Workspace (`StudioViewModel`).
//!
//! Coordinates non-blocking background GPU pipeline evaluation (`AsyncStudioJob`), automatic
//! OME-Zarr multiscale pyramid streaming (`s3 ↔ s2 ↔ s1 ↔ s0`) with atomic generation
//! cancellation, live Polars/Arrow [`AnalyticsViewModel`] updates, raster brush painting, and
//! zero-latency interactive pan/zoom camera transformations on the UI thread.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use gpui_kit::RenderImage;
use lumen_core::cubecl::prelude::Client;
use lumen_core::memory::DeviceArray;
use lumen_io::{generate_phantom, LoadedImage, OmeZarrContainerInfo, PhantomKind};
use lumen_pipeline::{ModifierKind, ModifierNode, PipelineExecutionResult, PipelineExecutor};
use lumen_view::{build_render_image, ColormapLut, TonemapSettings, ViewportCamera};

use super::studio_event::StudioEvent;
use super::studio_state::StudioUiState;
use crate::data::DeviceService;
use crate::domain::{ActiveTool, LayerKind, WorkspaceMode};
use crate::screens::analytics::analytics_vm::AnalyticsViewModel;

type ZarrCacheKey = (PathBuf, String, Option<usize>, Option<String>);

/// Heavy I/O + GPU + Polars + BGRA8 tonemapping job queued by [`StudioViewModel`] to be executed on
/// GPUI's background thread pool (`cx.background_executor()`) off the main UI thread.
#[derive(Clone)]
pub enum AsyncStudioJob {
    /// Evaluate the GPU modifier stack, Polars analytics, and parallel BGRA8 tonemapping on a background thread.
    RecomputePipeline {
        generation: u64,
        client: Client,
        image: LoadedImage,
        modifiers: Vec<ModifierNode>,
        tonemap: TonemapSettings,
        show_contours: bool,
    },
    /// Load an OME-Zarr pyramid level (`s3..s0`), channel, or label mask on a background thread,
    /// then evaluate the GPU modifier stack and parallel BGRA8 tonemapping.
    SwitchOmeZarrView {
        generation: u64,
        client: Client,
        info: OmeZarrContainerInfo,
        cached_image: Option<LoadedImage>,
        preserve_camera_scale: bool,
        modifiers: Vec<ModifierNode>,
        tonemap: TonemapSettings,
        show_contours: bool,
    },
}

/// Output produced on a background worker thread by executing an [`AsyncStudioJob`].
pub struct AsyncStudioJobOutput {
    pub generation: u64,
    pub loaded_image: Option<(LoadedImage, bool)>,
    pub base_device_image: Option<DeviceArray<f32>>,
    pub pipeline_result: PipelineExecutionResult,
    pub render_image: Arc<RenderImage>,
    pub updated_tonemap: TonemapSettings,
}

impl AsyncStudioJob {
    /// Executes the heavy I/O, CubeCL GPU pipeline, Polars/Arrow region feature table construction,
    /// and multi-threaded BGRA8 tonemapping on a background worker thread. Returns `None` if
    /// superseded by a newer `job_generation`.
    pub fn run_on_worker(self, latest_gen: &AtomicU64) -> Result<Option<AsyncStudioJobOutput>, String> {
        match self {
            AsyncStudioJob::RecomputePipeline {
                generation,
                client,
                image,
                modifiers,
                mut tonemap,
                show_contours,
            } => {
                if latest_gen.load(Ordering::Acquire) != generation {
                    return Ok(None);
                }
                let dev_img = image
                    .to_device_luminance(&client)
                    .map_err(|e| format!("Device upload error: {e}"))?;
                if latest_gen.load(Ordering::Acquire) != generation {
                    return Ok(None);
                }
                let res = PipelineExecutor::execute(&client, &dev_img, &modifiers, show_contours)
                    .map_err(|e| format!("Pipeline error: {e}"))?;
                if latest_gen.load(Ordering::Acquire) != generation {
                    return Ok(None);
                }

                let min_f = res.min_val as f32;
                let max_f = res.max_val as f32;
                if (max_f - min_f) > 1e-5 {
                    tonemap.black_point = min_f;
                    tonemap.white_point = max_f;
                }
                let contours_opt = if show_contours {
                    Some(res.contours.as_slice())
                } else {
                    None
                };
                let render_image = build_render_image(
                    res.width,
                    res.height,
                    &res.output_host,
                    &tonemap,
                    contours_opt,
                );

                Ok(Some(AsyncStudioJobOutput {
                    generation,
                    loaded_image: None,
                    base_device_image: Some(dev_img),
                    pipeline_result: res,
                    render_image,
                    updated_tonemap: tonemap,
                }))
            }
            AsyncStudioJob::SwitchOmeZarrView {
                generation,
                client,
                info,
                cached_image,
                preserve_camera_scale,
                modifiers,
                mut tonemap,
                show_contours,
            } => {
                if latest_gen.load(Ordering::Acquire) != generation {
                    return Ok(None);
                }
                let img = match cached_image {
                    Some(mut cached) => {
                        if let Some(ref mut z) = cached.ome_zarr {
                            z.auto_pyramid = info.auto_pyramid;
                        }
                        cached
                    }
                    None => lumen_io::load_ome_zarr_with_options(&info)
                        .map_err(|e| format!("OME-Zarr load error: {e}"))?,
                };
                if latest_gen.load(Ordering::Acquire) != generation {
                    return Ok(None);
                }
                let dev_img = img
                    .to_device_luminance(&client)
                    .map_err(|e| format!("Device upload error: {e}"))?;
                if latest_gen.load(Ordering::Acquire) != generation {
                    return Ok(None);
                }
                let res = PipelineExecutor::execute(&client, &dev_img, &modifiers, show_contours)
                    .map_err(|e| format!("Pipeline error: {e}"))?;
                if latest_gen.load(Ordering::Acquire) != generation {
                    return Ok(None);
                }

                let min_f = res.min_val as f32;
                let max_f = res.max_val as f32;
                if (max_f - min_f) > 1e-5 {
                    tonemap.black_point = min_f;
                    tonemap.white_point = max_f;
                }
                let contours_opt = if show_contours {
                    Some(res.contours.as_slice())
                } else {
                    None
                };
                let render_image = build_render_image(
                    res.width,
                    res.height,
                    &res.output_host,
                    &tonemap,
                    contours_opt,
                );

                Ok(Some(AsyncStudioJobOutput {
                    generation,
                    loaded_image: Some((img, preserve_camera_scale)),
                    base_device_image: Some(dev_img),
                    pipeline_result: res,
                    render_image,
                    updated_tonemap: tonemap,
                }))
            }
        }
    }
}

pub struct StudioViewModel {
    pub state: StudioUiState,
    /// Cached base image on the CubeCL device.
    base_device_image: Option<DeviceArray<f32>>,
    /// Decoded OME-Zarr pyramid level cache so zooming back and forth across `s3..s0` is zero-I/O.
    ome_zarr_cache: HashMap<ZarrCacheKey, LoadedImage>,
    /// Monotonically increasing generation counter used to cancel/ignore stale background jobs.
    job_generation: Arc<AtomicU64>,
    /// Target OME-Zarr pyramid level currently being loaded in the background (`s3..s0`), so repeated
    /// scroll ticks toward the same level don't spawn duplicate jobs.
    in_flight_zarr_level: Option<String>,
    /// Pending background job to be spawned by the root `LumenApp` onto `cx.background_executor()`.
    pending_async_job: Option<AsyncStudioJob>,
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
                show_operations_menu: false,
                brush_radius: 14.0,
                brush_strength: 0.85,
                stroke_width: 1.5,
                painted_strokes: 0,
                is_panning: false,
                last_mouse_pos: None,
                cursor_img_yx: None,
                cursor_intensity: None,
                viewport_size: (860.0, 620.0),
                compute_target_name: device_service.backend_label().to_string(),
                status_banner:
                    "Ready — Multi-threaded GPU & OME-Zarr Auto-LOD active".to_string(),
            },
            base_device_image: None,
            ome_zarr_cache: HashMap::new(),
            job_generation: Arc::new(AtomicU64::new(0)),
            in_flight_zarr_level: None,
            pending_async_job: None,
        };
        vm.load_image(vm.state.source_image.clone(), device_service, analytics_vm);
        vm
    }

    pub fn job_generation_handle(&self) -> Arc<AtomicU64> {
        self.job_generation.clone()
    }

    pub fn next_generation(&self) -> u64 {
        self.job_generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn current_generation(&self) -> u64 {
        self.job_generation.load(Ordering::Acquire)
    }

    pub fn take_pending_async_job(&mut self) -> Option<AsyncStudioJob> {
        self.pending_async_job.take()
    }

    /// Applies the completed output of a background worker job onto the UI state if its
    /// `generation` is still current.
    pub fn apply_async_job_output(
        &mut self,
        output: AsyncStudioJobOutput,
        analytics_vm: &mut AnalyticsViewModel,
    ) {
        if output.generation != self.current_generation() {
            return;
        }
        self.in_flight_zarr_level = None;

        if let Some((img, preserve_camera_scale)) = output.loaded_image {
            if let Some(ref info) = img.ome_zarr {
                let key = (
                    info.root_path.clone(),
                    info.active_level.clone(),
                    info.active_channel,
                    info.active_label_group.clone(),
                );
                self.ome_zarr_cache.insert(key, img.clone());
            }
            if preserve_camera_scale {
                let old_w = self.state.source_image.width.max(1) as f32;
                let new_w = img.width.max(1) as f32;
                let scale_ratio = old_w / new_w;
                self.state.camera.set_zoom(self.state.camera.zoom * scale_ratio);
            }
            self.state.layers = StudioUiState::default_layers_for(&img.name);
            self.sync_contour_layer_flag();
            self.state.source_image = img;
        }

        if let Some(dev_img) = output.base_device_image {
            self.base_device_image = Some(dev_img);
        }

        self.state.tonemap = output.updated_tonemap;
        self.state.render_image = output.render_image;
        analytics_vm.ingest_pipeline_result(&output.pipeline_result);

        let lod_tag = self
            .state
            .source_image
            .ome_zarr
            .as_ref()
            .map(|z| {
                format!(
                    "OME-Zarr {} ({}×{}) · ",
                    z.active_level, self.state.source_image.width, self.state.source_image.height
                )
            })
            .unwrap_or_default();
        self.state.status_banner = format!(
            "{lod_tag}Async GPU ({}) evaluated {} op(s) in {:.1} ms",
            self.state.compute_target_name,
            self.state.modifiers.iter().filter(|m| m.enabled).count(),
            output.pipeline_result.elapsed_ms,
        );
        self.state.pipeline_result = Some(output.pipeline_result);
    }

    /// Loads a new [`LoadedImage`] into the Studio, fits the camera, renders an immediate preview,
    /// and schedules the full GPU pipeline + Polars analytics pass on a background worker thread.
    pub fn load_image(
        &mut self,
        image: LoadedImage,
        device_service: &DeviceService,
        _analytics_vm: &mut AnalyticsViewModel,
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
        self.state.show_operations_menu = false;
        self.in_flight_zarr_level = None;

        if let Some(ref info) = image.ome_zarr {
            let key = (
                info.root_path.clone(),
                info.active_level.clone(),
                info.active_channel,
                info.active_label_group.clone(),
            );
            self.ome_zarr_cache.insert(key, image.clone());
        }

        // Immediate preview tonemap so the image appears on screen in <2ms while the GPU pipeline
        // and Polars analytics run asynchronously in the background.
        self.state.render_image = build_render_image(
            w,
            h,
            &image.luminance_f32,
            &self.state.tonemap,
            None,
        );
        self.state.source_image = image;
        self.queue_recompute_pipeline(device_service);
    }

    /// Queues a non-blocking background GPU pipeline evaluation (`AsyncStudioJob::RecomputePipeline`).
    pub fn queue_recompute_pipeline(&mut self, device_service: &DeviceService) {
        let generation = self.next_generation();
        self.state.status_banner = format!(
            "Evaluating {} GPU op(s) on background worker...",
            self.state.modifiers.iter().filter(|m| m.enabled).count()
        );
        self.pending_async_job = Some(AsyncStudioJob::RecomputePipeline {
            generation,
            client: device_service.client().clone(),
            image: self.state.source_image.clone(),
            modifiers: self.state.modifiers.clone(),
            tonemap: self.state.tonemap,
            show_contours: self.state.camera.show_contours,
        });
    }

    /// Checks whether the current on-screen displayed width (`img_w * camera.zoom`) warrants
    /// pulling a finer or coarser OME-Zarr pyramid level (`s3 ↔ s2 ↔ s1 ↔ s0`) on a background worker.
    fn sync_auto_pyramid_level(&mut self, device_service: &DeviceService) {
        let Some(mut info) = self.state.source_image.ome_zarr.clone() else {
            return;
        };
        if !info.auto_pyramid || info.level_dimensions.len() <= 1 {
            return;
        }
        let old_w = self.state.source_image.width;
        let displayed_w = (old_w as f32) * self.state.camera.zoom;
        let target_level = info.select_level_for_display_width(displayed_w);
        if target_level == info.active_level {
            return;
        }
        if self.in_flight_zarr_level.as_deref() == Some(target_level.as_str()) {
            return;
        }

        info.active_level = target_level.clone();
        self.in_flight_zarr_level = Some(target_level);
        self.queue_ome_zarr_switch(info, true, device_service);
    }

    fn queue_ome_zarr_switch(
        &mut self,
        info: OmeZarrContainerInfo,
        preserve_camera_scale: bool,
        device_service: &DeviceService,
    ) {
        let key = (
            info.root_path.clone(),
            info.active_level.clone(),
            info.active_channel,
            info.active_label_group.clone(),
        );
        let cached_image = self.ome_zarr_cache.get(&key).cloned();
        let generation = self.next_generation();
        self.state.status_banner = format!(
            "Streaming OME-Zarr pyramid level '{}' on background worker...",
            info.active_level
        );
        self.pending_async_job = Some(AsyncStudioJob::SwitchOmeZarrView {
            generation,
            client: device_service.client().clone(),
            info,
            cached_image,
            preserve_camera_scale,
            modifiers: self.state.modifiers.clone(),
            tonemap: self.state.tonemap,
            show_contours: self.state.camera.show_contours,
        });
    }

    /// Re-tonemaps the current processed host buffer into an `Arc<RenderImage>` using the
    /// multi-threaded 1024-entry LUT tonemapper without re-running the GPU kernels.
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

    /// Applies a circular brush or eraser dab at image coordinates `(cy, cx)`.
    fn apply_brush_dab(
        &mut self,
        cy: usize,
        cx: usize,
        erase: bool,
        device_service: &DeviceService,
    ) {
        let w = self.state.source_image.width;
        let h = self.state.source_image.height;
        let r = (self.state.brush_radius.round() as isize).max(1);
        let r2 = (r * r) as f32;
        let strength = self.state.brush_strength.clamp(0.05, 1.0);

        for dy in -r..=r {
            let py = (cy as isize) + dy;
            if py < 0 || py >= (h as isize) {
                continue;
            }
            for dx in -r..=r {
                let px = (cx as isize) + dx;
                if px < 0 || px >= (w as isize) {
                    continue;
                }
                let d2 = ((dy * dy + dx * dx) as f32) / r2;
                if d2 <= 1.0 {
                    let falloff = (1.0 - d2).sqrt() * strength;
                    let idx = (py as usize) * w + (px as usize);
                    let cur = self.state.source_image.luminance_f32[idx];
                    self.state.source_image.luminance_f32[idx] = if erase {
                        (cur * (1.0 - falloff)).clamp(0.0, 1.0)
                    } else {
                        (cur + falloff * (1.0 - cur)).clamp(0.0, 1.0)
                    };
                }
            }
        }

        self.state.painted_strokes += 1;
        // Immediate display update from host buffer, plus background GPU sync
        self.state.render_image = build_render_image(
            w,
            h,
            &self.state.source_image.luminance_f32,
            &self.state.tonemap,
            None,
        );
        self.queue_recompute_pipeline(device_service);
    }

    /// Handles a [`StudioEvent`] emitted by any Studio screen component.
    pub fn on_event(
        &mut self,
        event: StudioEvent,
        device_service: &DeviceService,
        _analytics_vm: &mut AnalyticsViewModel,
    ) {
        match event {
            StudioEvent::OpenProjectHub
            | StudioEvent::PromptOpenImage
            | StudioEvent::PromptOpenZarrFolder
            | StudioEvent::LoadImageFile(_) => {
                // Handled by root LumenApp coordinator
            }
            StudioEvent::ToggleOperationsMenu => {
                self.state.show_operations_menu = !self.state.show_operations_menu;
            }
            StudioEvent::CloseOperationsMenu => {
                self.state.show_operations_menu = false;
            }
            StudioEvent::SwitchWorkspaceMode(mode) => {
                self.state.active_mode = mode;
                self.state.show_operations_menu = false;
                if mode == WorkspaceMode::AnalyticsLab {
                    self.state.show_analytics_drawer = true;
                } else if mode == WorkspaceMode::NodePipeline
                    && self.state.active_tool == ActiveTool::ExplorePan
                {
                    self.state.active_tool = ActiveTool::BrushPaint;
                }
            }
            StudioEvent::SelectTool(tool) => {
                self.state.active_tool = tool;
                self.state.show_operations_menu = false;
                if tool == ActiveTool::VectorContour && !self.state.camera.show_contours {
                    self.state.camera.show_contours = true;
                    self.sync_contour_layer_flag();
                    self.queue_recompute_pipeline(device_service);
                }
            }
            StudioEvent::AdjustBrushRadius(delta) => {
                self.state.brush_radius = (self.state.brush_radius + delta).clamp(1.0, 96.0);
            }
            StudioEvent::AdjustBrushStrength(delta) => {
                self.state.brush_strength = (self.state.brush_strength + delta).clamp(0.05, 1.0);
            }
            StudioEvent::AdjustStrokeWidth(delta) => {
                self.state.stroke_width = (self.state.stroke_width + delta).clamp(0.5, 16.0);
            }
            StudioEvent::ToggleAnalyticsDrawer => {
                self.state.show_analytics_drawer = !self.state.show_analytics_drawer;
            }
            StudioEvent::ZoomIn => {
                self.state.camera.zoom_step(1.25);
                self.sync_auto_pyramid_level(device_service);
            }
            StudioEvent::ZoomOut => {
                self.state.camera.zoom_step(1.0 / 1.25);
                self.sync_auto_pyramid_level(device_service);
            }
            StudioEvent::ZoomOneToOne => {
                self.state.camera.reset_one_to_one();
                self.sync_auto_pyramid_level(device_service);
            }
            StudioEvent::FitToViewport => {
                self.state.camera.fit_to_viewport(
                    self.state.source_image.width,
                    self.state.source_image.height,
                    self.state.viewport_size.0,
                    self.state.viewport_size.1,
                    36.0,
                );
                self.sync_auto_pyramid_level(device_service);
            }
            StudioEvent::ZoomAtCursor {
                factor,
                rel_x,
                rel_y,
            } => {
                self.state.camera.zoom_at_cursor_offset(factor, rel_x, rel_y);
                self.sync_auto_pyramid_level(device_service);
            }
            StudioEvent::BeginPan { x, y } => {
                self.state.show_operations_menu = false;
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
                let img_w = self.state.source_image.width;
                let img_h = self.state.source_image.height;
                let coords = self.state.camera.screen_to_image(x, y, img_w, img_h, view_w, view_h);
                self.state.cursor_img_yx = coords;

                if self.state.is_panning {
                    match self.state.active_tool {
                        ActiveTool::BrushPaint | ActiveTool::Eraser => {
                            if let Some((py, px_i)) = coords {
                                let erase = self.state.active_tool == ActiveTool::Eraser;
                                self.apply_brush_dab(py, px_i, erase, device_service);
                            }
                        }
                        _ => {
                            if let Some((lx, ly)) = self.state.last_mouse_pos {
                                let dx = x - lx;
                                let dy = y - ly;
                                self.state.camera.pan_by(dx, dy);
                            }
                        }
                    }
                    self.state.last_mouse_pos = Some((x, y));
                }

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
                self.state.show_operations_menu = false;
                match self.state.active_tool {
                    ActiveTool::ZoomLens => {
                        let rel_x = x - 0.5 * view_w;
                        let rel_y = y - 0.5 * view_h;
                        self.state.camera.zoom_at_cursor_offset(1.35, rel_x, rel_y);
                        self.sync_auto_pyramid_level(device_service);
                    }
                    ActiveTool::BrushPaint | ActiveTool::Eraser => {
                        let img_w = self.state.source_image.width;
                        let img_h = self.state.source_image.height;
                        if let Some((py, px_i)) =
                            self.state.camera.screen_to_image(x, y, img_w, img_h, view_w, view_h)
                        {
                            let erase = self.state.active_tool == ActiveTool::Eraser;
                            self.apply_brush_dab(py, px_i, erase, device_service);
                        }
                    }
                    _ => {}
                }
            }
            StudioEvent::ToggleGrid => {
                self.state.camera.show_grid = !self.state.camera.show_grid;
            }
            StudioEvent::ToggleContours => {
                self.state.camera.show_contours = !self.state.camera.show_contours;
                self.sync_contour_layer_flag();
                if self.state.camera.show_contours {
                    self.queue_recompute_pipeline(device_service);
                } else {
                    self.rebuild_display_image();
                }
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
                    if self.state.camera.show_contours {
                        self.queue_recompute_pipeline(device_service);
                    } else {
                        self.rebuild_display_image();
                    }
                }
            }
            StudioEvent::AddModifier(kind) => {
                let id = self.state.next_modifier_id;
                self.state.next_modifier_id += 1;
                self.state.modifiers.push(ModifierNode::new(id, kind));
                self.state.show_operations_menu = false;
                self.queue_recompute_pipeline(device_service);
            }
            StudioEvent::ToggleModifier(id) => {
                if let Some(m) = self.state.modifiers.iter_mut().find(|m| m.id == id) {
                    m.enabled = !m.enabled;
                }
                self.queue_recompute_pipeline(device_service);
            }
            StudioEvent::AdjustModifierPrimary { id, delta } => {
                if let Some(m) = self.state.modifiers.iter_mut().find(|m| m.id == id) {
                    let (lo, hi) = m.kind.bounds();
                    m.primary_param = (m.primary_param + delta).clamp(lo, hi);
                }
                self.queue_recompute_pipeline(device_service);
            }
            StudioEvent::RemoveModifier(id) => {
                self.state.modifiers.retain(|m| m.id != id);
                self.queue_recompute_pipeline(device_service);
            }
            StudioEvent::ClearModifiers => {
                self.state.modifiers.clear();
                self.queue_recompute_pipeline(device_service);
            }
            StudioEvent::ToggleOmeZarrAutoPyramid => {
                if let Some(ref mut info) = self.state.source_image.ome_zarr {
                    info.auto_pyramid = !info.auto_pyramid;
                }
                self.sync_auto_pyramid_level(device_service);
            }
            StudioEvent::SelectOmeZarrLevel(level) => {
                if let Some(mut info) = self.state.source_image.ome_zarr.clone() {
                    info.auto_pyramid = false;
                    info.active_level = level.clone();
                    self.in_flight_zarr_level = Some(level);
                    self.queue_ome_zarr_switch(info, true, device_service);
                }
            }
            StudioEvent::SelectOmeZarrChannel(channel) => {
                if let Some(mut info) = self.state.source_image.ome_zarr.clone() {
                    info.active_channel = channel;
                    info.active_label_group = None;
                    self.queue_ome_zarr_switch(info, false, device_service);
                }
            }
            StudioEvent::SelectOmeZarrLabelGroup(label_group) => {
                if let Some(mut info) = self.state.source_image.ome_zarr.clone() {
                    info.active_label_group = label_group;
                    self.queue_ome_zarr_switch(info, false, device_service);
                }
            }
        }
    }

    /// Applies an initial set of modifiers when launching a preset from the Startup Project Hub.
    pub fn apply_preset_modifiers(
        &mut self,
        kinds: &[ModifierKind],
        show_contours: bool,
        device_service: &DeviceService,
        _analytics_vm: &mut AnalyticsViewModel,
    ) {
        self.state.modifiers.clear();
        for &kind in kinds {
            let id = self.state.next_modifier_id;
            self.state.next_modifier_id += 1;
            self.state.modifiers.push(ModifierNode::new(id, kind));
        }
        self.state.camera.show_contours = show_contours;
        self.sync_contour_layer_flag();
        self.queue_recompute_pipeline(device_service);
    }

    fn sync_contour_layer_flag(&mut self) {
        for layer in &mut self.state.layers {
            if layer.kind == LayerKind::VectorContours {
                layer.visible = self.state.camera.show_contours;
            }
        }
    }
}

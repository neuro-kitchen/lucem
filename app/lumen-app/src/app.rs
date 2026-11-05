//! Root GPUI application coordinator (`LumenApp`).
//!
//! Owns the DI [`AppContainer`], top-level [`AppRoute`] (`StartupHub` vs `StudioWorkspace`), and
//! the three KMP-style ViewModels ([`StartupViewModel`], [`StudioViewModel`], [`AnalyticsViewModel`]).
//! Dispatches heavy file I/O, OME-Zarr pyramid decoding, CubeCL GPU pipeline execution, and
//! BGRA8 tonemapping onto GPUI's background worker pool (`cx.background_executor()`).

use std::path::PathBuf;
use std::rc::Rc;

use gpui_kit::{div, prelude::*, App, Context, IntoElement, PathPromptOptions, Render, Window};
use lumen_io::{generate_phantom, load_image_file, LoadedImage};

use crate::di::AppContainer;
use crate::navigation::{AppRoute, LeftRailTab};
use crate::screens::analytics::analytics_event::AnalyticsEvent;
use crate::screens::analytics::analytics_vm::AnalyticsViewModel;
use crate::screens::startup::startup_event::StartupEvent;
use crate::screens::startup::startup_screen::StartupScreen;
use crate::screens::startup::startup_vm::StartupViewModel;
use crate::screens::studio::studio_event::StudioEvent;
use crate::screens::studio::studio_screen::StudioScreen;
use crate::screens::studio::studio_vm::StudioViewModel;

pub struct LumenApp {
    pub container: AppContainer,
    pub route: AppRoute,
    pub startup_vm: StartupViewModel,
    pub studio_vm: StudioViewModel,
    pub analytics_vm: AnalyticsViewModel,
}

impl LumenApp {
    pub fn new(_window: &mut Window, cx: &mut Context<Self>) -> Self {
        let container = AppContainer::new();
        let startup_vm = StartupViewModel::new(&container);
        let mut analytics_vm = AnalyticsViewModel::new();
        let studio_vm = StudioViewModel::new(&container.device_service, &mut analytics_vm);

        let mut app = Self {
            container,
            route: AppRoute::StartupHub,
            startup_vm,
            studio_vm,
            analytics_vm,
        };
        app.dispatch_pending_studio_job(cx);
        app
    }

    /// Spawns any queued [`crate::screens::studio::studio_vm::AsyncStudioJob`] onto GPUI's
    /// background worker thread pool (`cx.background_executor()`), leaving the main UI thread
    /// completely free for 60-120 FPS pan/zoom/input.
    pub fn dispatch_pending_studio_job(&mut self, cx: &mut Context<Self>) {
        let Some(job) = self.studio_vm.take_pending_async_job() else {
            return;
        };
        let gen_handle = self.studio_vm.job_generation_handle();
        let bg_task = cx
            .background_executor()
            .spawn(async move { job.run_on_worker(&gen_handle) });

        cx.spawn(async move |this, cx| {
            let result = bg_task.await;
            let _ = this.update(cx, |app, cx| {
                match result {
                    Ok(Some(output)) => {
                        app.studio_vm
                            .apply_async_job_output(output, &mut app.analytics_vm);
                        cx.notify();
                    }
                    Ok(None) => {
                        // Superseded by a newer generation job; ignore stale result.
                    }
                    Err(err_msg) => {
                        app.studio_vm.state.status_banner = err_msg;
                        cx.notify();
                    }
                }
            });
        })
        .detach();
    }

    /// Asynchronously loads an image file or `.ome.zarr` container from disk on a background
    /// worker thread and transitions to `StudioWorkspace` when ready.
    pub fn open_image_from_disk(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let msg = format!("Loading '{}' on background worker...", path.display());
        self.startup_vm.set_status(Some(msg.clone()));
        self.studio_vm.state.status_banner = msg;
        cx.notify();

        let path_for_worker = path.clone();
        let bg_load = cx
            .background_executor()
            .spawn(async move { load_image_file(&path_for_worker) });

        cx.spawn(async move |this, cx| {
            let load_res = bg_load.await;
            let _ = this.update(cx, |app, cx| {
                match load_res {
                    Ok(img) => {
                        let recorded_path = img
                            .ome_zarr
                            .as_ref()
                            .map(|z| z.root_path.clone())
                            .unwrap_or_else(|| path.clone());
                        app.container.project_repo.record_recent(recorded_path);
                        app.startup_vm.refresh_recents(&app.container);
                        app.startup_vm.set_status(None);
                        app.studio_vm.load_image(
                            img,
                            &app.container.device_service,
                            &mut app.analytics_vm,
                        );
                        app.route = AppRoute::StudioWorkspace;
                        app.dispatch_pending_studio_job(cx);
                    }
                    Err(err) => {
                        let err_msg = format!("Could not load image '{}': {err}", path.display());
                        app.startup_vm.set_status(Some(err_msg.clone()));
                        app.studio_vm.state.status_banner = err_msg;
                    }
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Opens the native OS file picker dialog and loads the selected image (or `.ome.zarr` file)
    /// into the Studio.
    pub fn prompt_open_image(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open Image File in Lumen Studio".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await {
                if let Some(first_path) = paths.into_iter().next() {
                    let _ = this.update(cx, |app, cx| {
                        app.open_image_from_disk(first_path, cx);
                    });
                }
            }
        })
        .detach();
    }

    /// Opens the native OS directory picker dialog to select an `.ome.zarr` container directory.
    pub fn prompt_open_zarr_folder(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Select .ome.zarr Container Folder".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await {
                if let Some(first_path) = paths.into_iter().next() {
                    let _ = this.update(cx, |app, cx| {
                        app.open_image_from_disk(first_path, cx);
                    });
                }
            }
        })
        .detach();
    }

    pub fn handle_startup_event(&mut self, event: StartupEvent, cx: &mut Context<Self>) {
        match event {
            StartupEvent::SelectRailTab(tab) => {
                self.startup_vm.set_rail_tab(tab);
                match tab {
                    LeftRailTab::StudioCanvas => {
                        self.route = AppRoute::StudioWorkspace;
                    }
                    LeftRailTab::AnalyticsDrawer => {
                        self.studio_vm.state.show_analytics_drawer = true;
                        self.route = AppRoute::StudioWorkspace;
                    }
                    _ => {}
                }
            }
            StartupEvent::PromptOpenImageFile => {
                self.prompt_open_image(cx);
            }
            StartupEvent::PromptOpenZarrFolder => {
                self.prompt_open_zarr_folder(cx);
            }
            StartupEvent::OpenImagePath(path) => {
                self.open_image_from_disk(path, cx);
            }
            StartupEvent::CreateBlankCanvas => {
                let spec = self.startup_vm.state().blank_spec;
                if let Ok(blank) = LoadedImage::blank_canvas(
                    format!("Blank Canvas {}×{}", spec.width, spec.height),
                    spec.width,
                    spec.height,
                    spec.background_intensity,
                ) {
                    self.studio_vm.load_image(
                        blank,
                        &self.container.device_service,
                        &mut self.analytics_vm,
                    );
                    self.route = AppRoute::StudioWorkspace;
                    self.dispatch_pending_studio_job(cx);
                }
            }
            StartupEvent::SelectBlankSize(w, h) => {
                self.startup_vm.set_blank_size(w, h);
            }
            StartupEvent::SelectBlankBackground(bg) => {
                self.startup_vm.set_blank_background(bg);
            }
            StartupEvent::LaunchPreset(idx) => {
                if let Some(preset) = self.container.project_repo.presets().get(idx).cloned() {
                    let img = if let Some(phantom_kind) = preset.phantom {
                        generate_phantom(phantom_kind, 512, 512)
                    } else {
                        LoadedImage::blank_canvas("Preset Canvas", 1024, 768, 0.08)
                            .expect("valid blank canvas dimensions")
                    };
                    self.studio_vm.state.tonemap.colormap = preset.colormap;
                    self.studio_vm.load_image(
                        img,
                        &self.container.device_service,
                        &mut self.analytics_vm,
                    );
                    self.studio_vm.apply_preset_modifiers(
                        &preset.initial_modifiers,
                        preset.show_contours,
                        &self.container.device_service,
                        &mut self.analytics_vm,
                    );
                    self.route = AppRoute::StudioWorkspace;
                    self.dispatch_pending_studio_job(cx);
                }
            }
            StartupEvent::LoadPhantom(kind) => {
                let img = generate_phantom(kind, 512, 512);
                self.studio_vm
                    .load_image(img, &self.container.device_service, &mut self.analytics_vm);
                self.route = AppRoute::StudioWorkspace;
                self.dispatch_pending_studio_job(cx);
            }
            StartupEvent::EnterBlankCanvasNow => {
                self.route = AppRoute::StudioWorkspace;
            }
        }
        cx.notify();
    }

    pub fn handle_studio_event(&mut self, event: StudioEvent, cx: &mut Context<Self>) {
        match event {
            StudioEvent::OpenProjectHub => {
                self.startup_vm.set_rail_tab(LeftRailTab::ProjectHub);
                self.route = AppRoute::StartupHub;
            }
            StudioEvent::PromptOpenImage => {
                self.prompt_open_image(cx);
            }
            StudioEvent::PromptOpenZarrFolder => {
                self.prompt_open_zarr_folder(cx);
            }
            StudioEvent::LoadImageFile(path) => {
                self.open_image_from_disk(path, cx);
            }
            other => {
                self.studio_vm.on_event(
                    other,
                    &self.container.device_service,
                    &mut self.analytics_vm,
                );
                self.dispatch_pending_studio_job(cx);
            }
        }
        cx.notify();
    }

    pub fn handle_analytics_event(&mut self, event: AnalyticsEvent, cx: &mut Context<Self>) {
        match event {
            AnalyticsEvent::ToggleDrawer => {
                self.studio_vm.state.show_analytics_drawer =
                    !self.studio_vm.state.show_analytics_drawer;
            }
            AnalyticsEvent::SortBy(col) => {
                self.analytics_vm.set_sort(col);
            }
            AnalyticsEvent::CycleMinAreaFilter => {
                self.analytics_vm.cycle_min_area_filter();
            }
        }
        cx.notify();
    }
}

impl Render for LumenApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        match self.route {
            AppRoute::StartupHub => {
                let entity = cx.entity();
                let on_startup: Rc<dyn Fn(StartupEvent, &mut Window, &mut App)> =
                    Rc::new(move |ev, _window, app| {
                        entity.update(app, |this, cx| {
                            this.handle_startup_event(ev, cx);
                        });
                    });
                div()
                    .size_full()
                    .child(StartupScreen::render(self.startup_vm.state(), on_startup))
            }
            AppRoute::StudioWorkspace => {
                let entity_studio = cx.entity();
                let on_studio: Rc<dyn Fn(StudioEvent, &mut Window, &mut App)> =
                    Rc::new(move |ev, _window, app| {
                        entity_studio.update(app, |this, cx| {
                            this.handle_studio_event(ev, cx);
                        });
                    });

                let entity_analytics = cx.entity();
                let on_analytics: Rc<dyn Fn(AnalyticsEvent, &mut Window, &mut App)> =
                    Rc::new(move |ev, _window, app| {
                        entity_analytics.update(app, |this, cx| {
                            this.handle_analytics_event(ev, cx);
                        });
                    });

                div().size_full().child(StudioScreen::render(
                    &self.studio_vm.state,
                    self.analytics_vm.state(),
                    on_studio,
                    on_analytics,
                ))
            }
        }
    }
}

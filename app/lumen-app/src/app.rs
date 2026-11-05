//! Root GPUI application coordinator (`LumenApp`).
//!
//! Owns the DI [`AppContainer`], top-level [`AppRoute`] (`StartupHub` vs `StudioWorkspace`), and
//! the three KMP-style ViewModels ([`StartupViewModel`], [`StudioViewModel`], [`AnalyticsViewModel`]).

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
    pub fn new(_window: &mut Window, _cx: &mut Context<Self>) -> Self {
        let container = AppContainer::new();
        let startup_vm = StartupViewModel::new(&container);
        let mut analytics_vm = AnalyticsViewModel::new();
        let studio_vm = StudioViewModel::new(&container.device_service, &mut analytics_vm);

        Self {
            container,
            // Phase 1 Startup: Opens onto a blank canvas with the Left Option Rail and Center
            // VectorCraft Project Hub modal.
            route: AppRoute::StartupHub,
            startup_vm,
            studio_vm,
            analytics_vm,
        }
    }

    /// Loads an image file from disk into the Studio workspace and transitions to `StudioWorkspace`.
    pub fn open_image_from_disk(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        match load_image_file(&path) {
            Ok(img) => {
                self.container.project_repo.record_recent(path);
                self.startup_vm.refresh_recents(&self.container);
                self.startup_vm.set_status(None);
                self.studio_vm
                    .load_image(img, &self.container.device_service, &mut self.analytics_vm);
                self.route = AppRoute::StudioWorkspace;
            }
            Err(err) => {
                let msg = format!("Could not load image '{}': {err}", path.display());
                self.startup_vm.set_status(Some(msg.clone()));
                self.studio_vm.state.status_banner = msg;
            }
        }
        cx.notify();
    }

    /// Opens the native OS file picker dialog and loads the selected image into the Studio.
    pub fn prompt_open_image(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open Image in Lumen Studio".into()),
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
                }
            }
            StartupEvent::LoadPhantom(kind) => {
                let img = generate_phantom(kind, 512, 512);
                self.studio_vm
                    .load_image(img, &self.container.device_service, &mut self.analytics_vm);
                self.route = AppRoute::StudioWorkspace;
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
            StudioEvent::LoadImageFile(path) => {
                self.open_image_from_disk(path, cx);
            }
            other => {
                self.studio_vm.on_event(
                    other,
                    &self.container.device_service,
                    &mut self.analytics_vm,
                );
            }
        }
        cx.notify();
    }

    pub fn handle_analytics_event(&mut self, event: AnalyticsEvent, cx: &mut Context<Self>) {
        match event {
            AnalyticsEvent::ToggleDrawer => {
                self.studio_vm.state.show_analytics_drawer = !self.studio_vm.state.show_analytics_drawer;
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

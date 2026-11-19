//! The root view: kitchen-shell's frame (title bar with the Lucem menu, Help, the workspaces, the
//! project and where lucem runs; the workspace's menu row; the workspace as a card; the status
//! bar with the jobs and this computer's load; Settings; the search) around lucem's three
//! workspaces. Every action is a command of one registry (`crate::commands`), so the menus, the
//! keys, the search and an agent see the same list. Without an image Explore shows the home view.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, App, AppContext as _, Context, Entity, ExternalPaths, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};
use kitchen_shell::{Registry, RemoteSpec, SettingsSection, Shell, ShellEvent, ShellSpec, WorkspaceSpec};

use crate::commands::{self, Ctx};
use crate::data::store::{AppEvent, Store, ViewRequest};
use crate::navigation::actions::{ToggleSidebar, HELP};
use crate::navigation::route::Workspace;
#[cfg(not(target_family = "wasm"))]
use crate::screens::analyze::AnalyzeView;
use crate::screens::explore::ExploreView;
use crate::screens::pipeline::PipelineView;
use crate::screens::settings::{Section, SettingsView};
use crate::ui::theme::{self, color};

/// lucem's readings in the frame's status bar: the focused view's position and values, its level
/// and zoom (draggable), Fit and 1:1, the cache, model installs and results being computed, the
/// device. Its own entity, so the pointer moving over the image redraws only this.
pub struct StatusReadings {
    store: Entity<Store>,
    explore: Entity<ExploreView>,
    /// The canvas followed, and the subscription that follows it.
    watched: Option<(gpui_kit::EntityId, Subscription)>,
    _subs: Vec<Subscription>,
}

impl StatusReadings {
    fn new(store: Entity<Store>, explore: Entity<ExploreView>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify()), cx.observe(&explore, |_, _, cx| cx.notify())];
        Self { store, explore, watched: None, _subs: subs }
    }
}

/// A status bar reading: an icon and its value, muted.
fn meter(icon: IconName, text: String, tip: &'static str) -> impl IntoElement {
    h_flex().id(tip).gap(px(theme::XS)).items_center().text_color(color::text_muted()).child(Icon::new(icon).small()).child(text).tooltip(move |w, cx| gpui_kit::component::tooltip::Tooltip::new(tip).build(w, cx))
}

impl Render for StatusReadings {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let canvas = self.explore.read(cx).canvas.clone();
        if self.watched.as_ref().map(|w| w.0) != Some(canvas.entity_id()) {
            let sub = cx.observe(&canvas, |_, _, cx| cx.notify());
            self.watched = Some((canvas.entity_id(), sub));
        }
        let s = self.store.read(cx);
        let gpu_mb = lucem_core::memory::tracked_gpu_bytes() / (1024 * 1024);
        let device = match (&s.device, gpu_mb) {
            (Some(d), mb) if mb > 0 => format!("{} · {mb} MB", d.label),
            (Some(d), _) => d.label.clone(),
            (None, mb) if mb > 0 => format!("GPU · {mb} MB"),
            (None, _) => "GPU idle".to_string(),
        };
        let ram_limit_gb = lucem_core::memory::ram_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
        let explore = s.workspace == Workspace::Explore && !s.doc.images.is_empty();
        let st = canvas.read(cx).status(cx);
        let (c1, c2, c3) = (canvas.clone(), canvas.clone(), canvas.clone());
        // Model installs (the jobs themselves are in the frame's status bar)
        let pct = |f: Option<f32>| f.map_or(String::new(), |f| format!(" {:.0} %", f * 100.0));
        let busy = s.ml.busy.as_ref().map(|(what, f)| format!("{what}{}", pct(*f)));
        h_flex()
            .gap(px(theme::LG))
            .items_center()
            .text_xs()
            .when(explore, |d| d.child(div().max_w(px(420.)).truncate().font_family("monospace").text_color(color::text_muted()).child([st.position.clone(), st.values.clone()].into_iter().flatten().collect::<Vec<_>>().join("   "))))
            .children(busy.map(|b| div().text_color(color::accent_mark()).child(b)))
            .when(s.computing && crate::data::jobs::of("model").is_none(), |d| d.child(div().text_color(color::warning()).child("computing results…")))
            .when(explore, |d| {
                d.children(st.level.clone().map(|l| div().font_family("monospace").text_color(color::text_muted()).child(l)))
                    .child(div().font_family("monospace").child(crate::ui::widgets::scrub_label("status-zoom", format!("{:.1} %", st.zoom * 100.0), move |d, _, cx| c1.update(cx, |c, cx| c.zoom_at(if d > 0.0 { 1.05 } else { 1.0 / 1.05 }, None, cx)))))
                    .child(Button::new("status-fit").ghost().xsmall().label("Fit").tooltip("Fit the image (F · Ctrl+0)").on_click(move |_, _, cx| c2.update(cx, |c, cx| c.fit(cx))))
                    .child(Button::new("status-100").ghost().xsmall().label("1:1").tooltip("Actual pixels (Ctrl+1)").on_click(move |_, _, cx| c3.update(cx, |c, cx| c.actual_size(cx))))
                    .when(st.cached_mb > 0, |d| {
                        d.child(meter(IconName::HardDrive, format!("{} MB / {:.1} GB", st.cached_mb, ram_limit_gb), "Decoded tile & view cache vs. RAM memory budget (Settings › Compute)"))
                    })
            })
            .child(meter(IconName::Gpu, device, "GPU device and active compute cache"))
            .child({
                let st_bell = self.store.clone();
                let unread = s.unread_notifications;
                let open = s.notifications_open;
                Button::new("status-notifications")
                    .ghost()
                    .xsmall()
                    .selected(open)
                    .icon(Icon::new(IconName::Bell))
                    .when(unread > 0, |b| b.label(unread.to_string()))
                    .tooltip(if unread > 0 { "Notifications (unread)" } else { "Notifications" })
                    .on_click(move |_, _, cx| st_bell.update(cx, |s, cx| s.toggle_notifications(cx)))
            })
    }
}

/// Explore's body: the home view until an image is open, then Explore.
pub struct ExploreBody {
    store: Entity<Store>,
    explore: Entity<ExploreView>,
    _subs: Vec<Subscription>,
}

impl Render for ExploreBody {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.store.read(cx).doc.images.is_empty() {
            let (a, b) = (self.store.clone(), self.store.clone());
            crate::screens::home::home(&self.store, move |_, cx| commands::open_image(&a, cx), move |_, cx| commands::open_project(&b, cx), cx).into_any_element()
        } else {
            self.explore.clone().into_any_element()
        }
    }
}

/// The Keyboard & mouse sheet (F1), shown over the frame, with live search and keycap badges.
pub struct HelpSheet {
    close: Rc<dyn Fn(&mut Window, &mut App)>,
    search: Entity<gpui_kit::component::input::InputState>,
    _sub: Subscription,
}

impl HelpSheet {
    fn new(close: Rc<dyn Fn(&mut Window, &mut App)>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        use gpui_kit::component::input::{InputEvent, InputState};
        let search = cx.new(|cx| InputState::new(window, cx).placeholder("Search shortcuts, tools, or keys…"));
        let _sub = cx.subscribe(&search, |_, _, e: &InputEvent, cx| {
            if matches!(e, InputEvent::Change) {
                cx.notify();
            }
        });
        Self { close, search, _sub }
    }
}

impl Render for HelpSheet {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (c1, c2) = (self.close.clone(), self.close.clone());
        let query = self.search.read(cx).value().trim().to_ascii_lowercase();
        let groups: Vec<(&str, Vec<(&str, &str)>)> = HELP
            .iter()
            .filter_map(|(group, rows)| {
                let matched: Vec<(&str, &str)> = rows
                    .iter()
                    .copied()
                    .filter(|(k, w)| {
                        query.is_empty()
                            || group.to_ascii_lowercase().contains(&query)
                            || k.to_ascii_lowercase().contains(&query)
                            || w.to_ascii_lowercase().contains(&query)
                    })
                    .collect();
                (!matched.is_empty()).then_some((*group, matched))
            })
            .collect();
        let total: usize = groups.iter().map(|(_, r)| r.len()).sum();
        div()
            .id("help-overlay")
            .absolute()
            // Takes the pointer: a click here never reaches the buttons under it
            .occlude()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui_kit::hsla(0., 0., 0., 0.55))
            .on_click(move |_, w, cx| c1(w, cx))
            .child(
                v_flex()
                    .id("help")
                    .w(px(640.))
                    .max_h(px(640.))
                    .p(px(theme::LG))
                    .gap(px(theme::MD))
                    .rounded(px(theme::RADIUS_LG))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::raised())
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(
                        h_flex()
                            .gap(px(theme::SM))
                            .items_center()
                            .child(div().flex_1().text_lg().font_weight(FontWeight::SEMIBOLD).child("Keyboard & mouse shortcuts"))
                            .child(div().text_xs().text_color(color::text_muted()).child(format!("{total} shortcut(s)")))
                            .child(Button::new("help-close-x").ghost().xsmall().icon(Icon::new(IconName::Close)).on_click(move |_, w, cx| c2(w, cx))),
                    )
                    .child(gpui_kit::component::input::Input::new(&self.search).small())
                    .child(
                        v_flex()
                            .id("help-scroll")
                            .flex_1()
                            .min_h_0()
                            .max_h(px(460.))
                            .overflow_y_scroll()
                            .gap(px(theme::MD))
                            .when(groups.is_empty(), |d| {
                                d.child(div().py(px(theme::LG)).text_sm().text_color(color::text_muted()).child(format!("No shortcuts matching \"{query}\".")))
                            })
                            .children(groups.into_iter().map(|(group, rows)| {
                                v_flex()
                                    .gap(px(theme::XS))
                                    .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child(group.to_uppercase()))
                                    .children(rows.into_iter().map(|(keys, what)| {
                                        h_flex()
                                            .gap(px(theme::MD))
                                            .py(px(2.))
                                            .items_center()
                                            .child(
                                                div().w(px(220.)).flex_none().flex().child(
                                                    div()
                                                        .px(px(6.))
                                                        .py(px(2.))
                                                        .rounded(px(4.))
                                                        .border_1()
                                                        .border_color(color::border())
                                                        .bg(color::control())
                                                        .text_xs()
                                                        .font_family("monospace")
                                                        .child(keys),
                                                ),
                                            )
                                            .child(div().flex_1().text_sm().child(what))
                                    }))
                            })),
                    ),
            )
    }
}

pub struct LucemApp {
    store: Entity<Store>,
    /// The frame: menus, search, status bar, Settings.
    pub shell: Entity<Shell>,
    /// Every command (menus, keys, search, agents).
    pub registry: Registry,
    pub explore: Entity<ExploreView>,
    pub pipeline: Entity<PipelineView>,
    #[cfg(not(target_family = "wasm"))]
    pub analyze: Entity<AnalyzeView>,
    op_hover_preview: Entity<crate::ui::op_gallery::OpHoverPreview>,
    last_toast_id: u64,
    toast_timer: Option<gpui_kit::Task<()>>,
    _subs: Vec<Subscription>,
}

impl LucemApp {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        {
            let prefs = &store.read(cx).prefs;
            theme::configure(prefs, window);
            crate::data::script::set_warm_workers_enabled(prefs.warm_python_worker);
        }
        theme::apply(window, cx);
        let pipeline = cx.new(|cx| PipelineView::new(store.clone(), window, cx));
        // Explore can dock the same Pipeline under its views
        let explore = cx.new(|cx| ExploreView::new(store.clone(), pipeline.clone(), window, cx));
        #[cfg(not(target_family = "wasm"))]
        let analyze = cx.new(|cx| AnalyzeView::new(store.clone(), window, cx));
        // Model plugins are looked for once, in the background
        store.update(cx, |s, cx| s.scan_plugins(cx));

        // The commands, then the frame that draws them
        let registry = Registry::default();
        let shell_cell: Rc<std::cell::OnceCell<gpui_kit::WeakEntity<Shell>>> = Rc::default();
        let help = {
            let cell = shell_cell.clone();
            let close: Rc<dyn Fn(&mut Window, &mut App)> = {
                let cell = cell.clone();
                Rc::new(move |window, cx| {
                    if let Some(s) = cell.get().and_then(|w| w.upgrade()) {
                        s.update(cx, |s, cx| s.set_overlay(None, window, cx));
                    }
                })
            };
            let sheet = cx.new(|cx| HelpSheet::new(close, window, cx));
            let toggle: Rc<dyn Fn(&mut Window, &mut App)> = Rc::new(move |window, cx| {
                if let Some(s) = cell.get().and_then(|w| w.upgrade()) {
                    s.update(cx, |s, cx| {
                        let view = (!s.has_overlay()).then(|| sheet.clone().into());
                        s.set_overlay(view, window, cx)
                    });
                }
            });
            toggle
        };
        let ctx = Ctx {
            store: store.clone(),
            shell: shell_cell.clone(),
            pipeline: pipeline.downgrade(),
            #[cfg(not(target_family = "wasm"))]
            analyze: analyze.downgrade(),
            toggle_help: help,
        };
        commands::register(&registry, &ctx);
        registry.extend(PipelineView::commands(pipeline.downgrade(), store.clone()));
        PipelineView::provide(&registry, store.clone());
        #[cfg(not(target_family = "wasm"))]
        {
            registry.extend(AnalyzeView::commands(analyze.downgrade()));
            AnalyzeView::provide(&registry, analyze.downgrade());
        }
        pipeline.update(cx, |p, _| p.set_registry(registry.clone()));

        let explore_body = cx.new(|cx| ExploreBody { store: store.clone(), explore: explore.clone(), _subs: vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify())] });
        let workspace = |w: Workspace, view: gpui_kit::AnyView, menus| WorkspaceSpec::new(commands::workspace_id(w), w.title(), w.icon(), view).hint(w.hint()).menus(menus);
        #[allow(unused_mut)]
        let mut workspaces = vec![workspace(Workspace::Explore, explore_body.into(), commands::explore_menus()), workspace(Workspace::Pipeline, pipeline.clone().into(), PipelineView::menu_specs())];
        #[cfg(not(target_family = "wasm"))]
        workspaces.push(workspace(Workspace::Analyze, analyze.clone().into(), AnalyzeView::menu_specs()));
        let settings_view = cx.new(|_| SettingsView::new(store.clone()));
        let sections = Section::ALL
            .into_iter()
            .map(|sec| {
                let view = settings_view.clone();
                SettingsSection::new(sec.title(), sec.icon(), move |_, cx| view.update(cx, |v, cx| v.body(sec, cx)))
            })
            .collect::<Vec<_>>();
        let readings = cx.new(|cx| StatusReadings::new(store.clone(), explore.clone(), cx));
        let mut spec = ShellSpec::new("Lucem", registry.clone(), workspaces);
        spec.app_menu = commands::app_menu();
        spec.help_menu = vec![kitchen_shell::MenuItem::command("help.keys")];
        spec.settings = sections;
        spec.layout_toggles = vec![];
        spec.remote = Some(RemoteSpec { label: crate::platform::host_label().into(), icon: IconName::Server, menu: vec![kitchen_shell::MenuItem::command("remote.this"), kitchen_shell::MenuItem::Separator, kitchen_shell::MenuItem::command("remote.connect"), kitchen_shell::MenuItem::provided("remote.images")] });
        spec.status_items = Some(Rc::new(move |_, _| readings.clone().into_any_element()));
        let shell = cx.new(|cx| Shell::new(spec, window, cx));
        let _ = shell_cell.set(shell.downgrade());

        // Keys: every command's, as `keybindings.toml` says in `[commands]` (the canvas binds its
        // own, `[canvas]`); the file is written with every default when it is missing
        for bad in kitchen_shell::keys::bind_commands(&registry, cx) {
            eprintln!("key binding: {bad}");
        }
        if let Some(path) = crate::data::prefs::Prefs::keybindings_path()
            && let Err(e) = kitchen_shell::keys::write_default_keybindings(&path, "lucem-app")
        {
            tracing::warn!("could not write {}: {e}", path.display());
        }
        // A page on its own: what is dropped on the window opens (no dialog, no browser prompt)
        #[cfg(target_family = "wasm")]
        {
            let (drops, store) = (crate::data::remote::drops(), store.clone());
            cx.spawn(async move |_, cx| {
                while let Ok(files) = drops.recv().await {
                    if crate::data::remote::runs_alone() {
                        let taken = crate::data::remote::take_files(files).await;
                        let _ = store.update(cx, |s, cx| s.open_picked(taken, cx));
                    } else {
                        let _ = store.update(cx, |s, cx| s.set_status("This page opens its server's images: File › Open file…", cx));
                    }
                }
            })
            .detach();
        }
        // settings.toml edited outside the app: applied while it runs (the look now)
        #[cfg(not(target_family = "wasm"))]
        if let Some(path) = crate::data::prefs::Prefs::path().map(|p| crate::data::prefs::Prefs::settings_path(&p)) {
            let mut seen = std::fs::read_to_string(&path).unwrap_or_default();
            cx.spawn_in(window, async move |this, cx| loop {
                cx.background_executor().timer(std::time::Duration::from_secs(2)).await;
                let text = std::fs::read_to_string(&path).unwrap_or_default();
                if text.is_empty() || text == seen {
                    continue;
                }
                let applied = this.update_in(cx, |this, window, cx| {
                    match crate::data::settings::Settings::parse(&text) {
                        Ok((settings, notices)) => {
                            let prefs = this.store.update(cx, |s, cx| {
                                s.update_prefs(|p| settings.apply_to(p), cx);
                                s.set_status(notices.into_iter().next().unwrap_or_else(|| "Settings applied from settings.toml".into()), cx);
                                s.prefs.clone()
                            });
                            theme::configure(&prefs, window);
                            theme::apply(window, cx);
                        }
                        Err(e) => this.store.update(cx, |s, cx| s.set_status(format!("Settings not applied: {e}"), cx)),
                    }
                    // What the app wrote back (the same values): not an edit
                    std::fs::read_to_string(&path).unwrap_or_default()
                });
                match applied {
                    Ok(written) => seen = written,
                    Err(_) => break,
                }
            })
            .detach();
        }
        let mut notices = crate::data::settings::take_notices();
        notices.extend(kitchen_shell::keys::take_notices());
        for notice in &notices {
            eprintln!("{notice}");
        }
        if let Some(first) = notices.into_iter().next() {
            store.update(cx, |s, cx| s.set_status(first, cx));
        }

        // The frame and the store follow each other: workspace, status line, project, look
        let subs = vec![
            cx.subscribe(&shell, |this, _, e: &ShellEvent, cx| match e {
                ShellEvent::WorkspaceChanged(id) => {
                    if let Some(w) = Workspace::ALL.into_iter().find(|w| commands::workspace_id(*w) == *id) {
                        this.store.update(cx, |s, cx| s.set_workspace(w, cx));
                    }
                }
                ShellEvent::LookChanged(look) => {
                    let look = *look;
                    this.store.update(cx, |s, cx| {
                        s.update_prefs(
                            |p| {
                                p.theme = Some(look.dark);
                                p.accent = look.accent;
                                p.font_size = look.font_size;
                            },
                            cx,
                        )
                    });
                }
            }),
            // A page's "Open image…": the server's images, over the frame; or Visual Op Gallery
            cx.subscribe_in(&store, window, |this, _, e: &AppEvent, window, cx| match e {
                AppEvent::ServerImages => {
                    let shell = this.shell.downgrade();
                    let close: crate::screens::server_images::Close = Rc::new(move |window, cx| {
                        if let Some(s) = shell.upgrade() {
                            s.update(cx, |s, cx| s.set_overlay(None, window, cx));
                        }
                    });
                    let list = cx.new(|cx| crate::screens::server_images::ServerImages::new(this.store.clone(), close, cx));
                    this.shell.update(cx, |s, cx| s.set_overlay(Some(list.into()), window, cx));
                }
                AppEvent::OpenOpGallery => {
                    this.store.update(cx, |s, cx| {
                        let open = !s.visual_strip_open;
                        s.set_visual_strip(open, cx);
                        if open && s.workspace != Workspace::Explore {
                            s.set_workspace(Workspace::Explore, cx);
                        }
                    });
                }
                _ => {}
            }),
            cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
                this.sync_shell(e, cx);
                if matches!(e, AppEvent::Status)
                    && let Some(toast) = this.store.read(cx).active_toast.clone()
                    && toast.id != this.last_toast_id
                {
                    let id = toast.id;
                    let delay = if toast.is_error { 8 } else { 5 };
                    this.last_toast_id = id;
                    let store = this.store.clone();
                    this.toast_timer = Some(cx.spawn(async move |_, cx| {
                        cx.background_executor().timer(std::time::Duration::from_secs(delay)).await;
                        let _ = store.update(cx, |s, cx| s.dismiss_toast(Some(id), cx));
                    }));
                }
                cx.notify();
            }),
            // A theme that follows the system follows it while running
            cx.observe_window_appearance(window, |this, window, cx| {
                let prefs = this.store.read(cx).prefs.clone();
                if prefs.theme.is_none() {
                    theme::configure(&prefs, window);
                    theme::apply(window, cx);
                }
            }),
        ];
        let op_hover_preview = cx.new(|cx| crate::ui::op_gallery::OpHoverPreview::new(store.clone(), cx));
        let this = Self {
            store,
            shell,
            registry,
            explore,
            pipeline,
            #[cfg(not(target_family = "wasm"))]
            analyze,
            op_hover_preview,
            last_toast_id: 0,
            toast_timer: None,
            _subs: subs,
        };
        this.sync_shell(&AppEvent::Workspace, cx);
        this.sync_shell(&AppEvent::Status, cx);
        this
    }

    /// Tells the frame what changed in the store: the workspace, the status line, the project.
    fn sync_shell(&self, e: &AppEvent, cx: &mut App) {
        let s = self.store.read(cx);
        let workspace = commands::workspace_id(s.workspace);
        // Notifications now appear as a toast connected to the bottom-bar Bell icon;
        // keep the bottom-left status strip quiet and uncluttered.
        let status: String = s
            .doc
            .active()
            .map(|img| img.name.clone())
            .unwrap_or_default();
        let project: SharedString = if s.project_file.is_some() { s.doc.name.clone().into() } else { format!("{} · not saved", s.doc.name).into() };
        let workspace_changed = matches!(e, AppEvent::Workspace);
        self.shell.update(cx, |sh, cx| {
            if workspace_changed && sh.workspace_id() != workspace {
                sh.set_workspace(&workspace, cx);
            }
            if sh.status().as_ref() != status.as_str() {
                sh.set_status(status, cx);
            }
            sh.set_project(Some(project), cx);
        });
    }

    pub fn open_path(&mut self, path: std::path::PathBuf, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.open(path, cx));
    }
}

impl Render for LucemApp {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let s = self.store.read(cx);
        let notifications_open = s.notifications_open;
        let notifications = s.notifications.clone();
        let active_toast = s.active_toast.clone();
        let st_close = self.store.clone();
        let st_clear = self.store.clone();
        let st_dismiss = self.store.clone();
        let st_open = self.store.clone();

        div()
            .id("lucem-app")
            .relative()
            .size_full()
            .on_action(cx.listener(|this, _: &ToggleSidebar, _, cx| this.store.update(cx, |s, cx| s.request_view(ViewRequest::ToggleSides, cx))))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                if let Some(p) = paths.paths().first().cloned() {
                    this.open_path(p, cx);
                }
            }))
            .child(self.shell.clone())
            // Floating Before <-> After animated hover preview card for any hovered op
            .child(self.op_hover_preview.clone())
            // Notification popover anchored above the bottom-bar Bell icon
            .when(notifications_open, |d| {
                d.child(
                    v_flex()
                        .id("notifications-popover")
                        .occlude()
                        .absolute()
                        .bottom(px(36.))
                        .right(px(12.))
                        .w(px(360.))
                        .max_h(px(340.))
                        .rounded(px(theme::RADIUS_LG))
                        .border_1()
                        .border_color(color::border())
                        .bg(color::raised())
                        .p(px(theme::MD))
                        .gap(px(theme::SM))
                        .child(
                            h_flex()
                                .items_center()
                                .gap(px(theme::XS))
                                .child(Icon::new(IconName::Bell).small().text_color(color::text_muted()))
                                .child(div().flex_1().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text()).child("Notifications"))
                                .when(!notifications.is_empty(), |h| {
                                    h.child(
                                        Button::new("notif-clear")
                                            .ghost()
                                            .xsmall()
                                            .label("Clear")
                                            .on_click(move |_, _, cx| st_clear.update(cx, |s, cx| s.clear_notifications(cx))),
                                    )
                                })
                                .child(
                                    Button::new("notif-close")
                                        .ghost()
                                        .xsmall()
                                        .icon(Icon::new(IconName::Close))
                                        .on_click(move |_, _, cx| st_close.update(cx, |s, cx| s.toggle_notifications(cx))),
                                ),
                        )
                        .child(
                            v_flex()
                                .id("notifications-list")
                                .flex_1()
                                .min_h_0()
                                .max_h(px(260.))
                                .overflow_y_scroll()
                                .gap(px(theme::XS))
                                .when(notifications.is_empty(), |v| {
                                    v.child(div().py(px(theme::MD)).text_xs().text_color(color::text_muted()).child("No notifications yet."))
                                })
                                .children(notifications.iter().rev().map(|n| {
                                    h_flex()
                                        .gap(px(theme::SM))
                                        .p(px(theme::SM))
                                        .rounded(px(theme::RADIUS_SM))
                                        .bg(color::panel())
                                        .items_start()
                                        .child(
                                            div()
                                                .mt(px(5.))
                                                .size(px(6.))
                                                .rounded(px(3.))
                                                .flex_none()
                                                .bg(if n.is_error { color::danger() } else { color::accent_mark() }),
                                        )
                                        .child(div().flex_1().min_w_0().text_xs().text_color(color::text()).child(n.message.clone()))
                                })),
                        ),
                )
            })
            // Auto-dismissing toast anchored above the bottom-bar Bell icon
            .when(!notifications_open && active_toast.is_some(), |d| {
                let toast = active_toast.unwrap();
                let toast_id = toast.id;
                let stripe = if toast.is_error { color::danger() } else { color::accent_mark() };
                d.child(
                    h_flex()
                        .id("status-toast")
                        .occlude()
                        .absolute()
                        .bottom(px(36.))
                        .right(px(12.))
                        .max_w(px(400.))
                        .rounded(px(theme::RADIUS_LG))
                        .border_1()
                        .border_color(color::border())
                        .bg(color::raised())
                        .overflow_hidden()
                        .items_center()
                        .child(div().w(px(3.)).h_full().flex_none().bg(stripe))
                        .child(
                            h_flex()
                                .flex_1()
                                .min_w_0()
                                .px(px(theme::MD))
                                .py(px(theme::SM))
                                .gap(px(theme::SM))
                                .items_center()
                                .child(Icon::new(IconName::Bell).small().text_color(stripe))
                                .child(
                                    div()
                                        .id("toast-body")
                                        .flex_1()
                                        .min_w_0()
                                        .cursor_pointer()
                                        .text_xs()
                                        .text_color(color::text())
                                        .on_click(move |_, _, cx| st_open.update(cx, |s, cx| s.toggle_notifications(cx)))
                                        .child(toast.message),
                                )
                                .child(
                                    Button::new("toast-dismiss")
                                        .ghost()
                                        .xsmall()
                                        .icon(Icon::new(IconName::Close))
                                        .tooltip("Dismiss")
                                        .on_click(move |_, _, cx| st_dismiss.update(cx, |s, cx| s.dismiss_toast(Some(toast_id), cx))),
                                ),
                        ),
                )
            })
    }
}

/// Headless UI tests: the real app in a test window (layout, hit testing, no pixels).
#[cfg(test)]
#[path = "app_tests.rs"]
mod tests;

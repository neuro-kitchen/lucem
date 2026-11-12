//! lumen's sections of Settings (Ctrl+,): the frame (kitchen-shell) draws Settings, with its own
//! Appearance first (theme, accent, text size), then these. Every setting says what it does and
//! what its default is, and takes effect at once.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::{div, px, AnyElement, App, Context, Entity, Subscription, FontWeight, IntoElement, ParentElement as _, SharedString, Styled as _, Window};

use crate::data::prefs::Prefs;
use crate::data::tools::{self, Found, Tool, ToolPaths};
use crate::data::store::Store;
use crate::ui::theme::{self, color, PASTEBOARDS};
use crate::ui::widgets::{stepper, MenuSelect};
use gpui_kit::prelude::FluentBuilder as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Display,
    Views,
    Compute,
    Models,
    Tools,
    Performance,
    About,
}

impl Section {
    pub const ALL: [Section; 7] = [Section::Display, Section::Views, Section::Compute, Section::Models, Section::Tools, Section::Performance, Section::About];

    pub fn title(self) -> &'static str {
        match self {
            Section::Display => "Display",
            Section::Views => "Views",
            Section::Compute => "Compute",
            Section::Models => "Models",
            Section::Tools => "Environment",
            Section::Performance => "Performance",
            Section::About => "About",
        }
    }

    pub fn icon(self) -> IconName {
        match self {
            Section::Display => IconName::Monitor,
            Section::Views => IconName::LayoutGrid,
            Section::Compute => IconName::Cpu,
            Section::Models => IconName::Sparkles,
            Section::Tools => IconName::Wrench,
            Section::Performance => IconName::Gauge,
            Section::About => IconName::Info,
        }
    }
}

/// A setting's action.
type Run = Rc<dyn Fn(&mut Window, &mut App)>;

pub struct SettingsView {
    store: Entity<Store>,
    /// Each tool's `--version` (Tools), and the paths it was read for.
    versions: Vec<(Tool, Result<String, String>)>,
    versions_for: Option<ToolPaths>,
    /// What the last environment action said (sync, pin), and whether one runs.
    env_note: Option<Result<String, String>>,
    env_busy: bool,
    _subs: Vec<Subscription>,
}

impl SettingsView {
    pub fn new(store: Entity<Store>) -> Self {
        Self { store, versions: Vec::new(), versions_for: None, env_note: None, env_busy: false, _subs: Vec::new() }
    }

    /// The settings of `section`.
    pub fn body(&mut self, section: Section, cx: &mut Context<Self>) -> AnyElement {
        let prefs = self.store.read(cx).prefs.clone();
        match section {
            Section::Display => self.display(&prefs),
            Section::Views => self.views(&prefs, cx),
            Section::Compute => self.compute(&prefs, cx),
            Section::Models => self.models(&prefs, cx),
            Section::Tools => self.tools(&prefs, cx),
            Section::Performance => self.performance(&prefs),
            Section::About => self.about(&prefs),
        }
    }

    /// Changes the preferences; the look follows at once.
    fn set(store: &Entity<Store>, f: impl FnOnce(&mut Prefs), window: &mut Window, cx: &mut App) {
        store.update(cx, |s, cx| s.update_prefs(f, cx));
        let prefs = store.read(cx).prefs.clone();
        theme::configure(&prefs, window);
        theme::apply(window, cx);
    }

    /// A setting: its name, what it does and its default on the left, the control on the right.
    fn row(title: &str, what: &str, default: &str, control: impl IntoElement) -> impl IntoElement {
        h_flex()
            .w_full()
            .py(px(theme::MD))
            .gap(px(theme::LG))
            .border_b_1()
            .border_color(color::border())
            .child(
                v_flex()
                    .flex_1()
                    .min_w(px(180.))
                    .gap(px(2.))
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(title.to_string()))
                    .child(div().text_xs().text_color(color::text_muted()).child(what.to_string()))
                    // Only a real default is worth a line
                    .when(!default.is_empty() && default != "—", |d| d.child(div().text_xs().text_color(color::text_disabled()).child(format!("Default: {default}")))),
            )
            // Wide controls (a row of choices) wrap instead of squeezing the text
            .child(div().flex_shrink_0().max_w(px(420.)).child(control))
    }

    /// Mutually exclusive choices as one row of buttons.
    fn choices(id: &'static str, options: &[&str], current: usize, on_pick: impl Fn(usize, &mut Window, &mut App) + 'static) -> impl IntoElement {
        let on_pick = Rc::new(on_pick);
        h_flex().gap(px(2.)).p(px(2.)).rounded(px(6.)).bg(color::control()).children(options.iter().enumerate().map(|(i, o)| {
            let f = on_pick.clone();
            Button::new(SharedString::from(format!("{id}-{i}"))).ghost().small().label(o.to_string()).selected(i == current).on_click(move |_, w, cx| f(i, w, cx))
        }))
    }

    /// What the frame's Appearance does not have: a theme that follows the system, the pasteboard.
    fn display(&self, p: &Prefs) -> AnyElement {
        let st = self.store.clone();
        let theme_now = match p.theme {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        };
        let s1 = st.clone();
        let theme = Self::choices("set-theme", &["System", "Dark", "Light"], theme_now, move |i, w, cx| Self::set(&s1, |p| p.theme = [None, Some(true), Some(false)][i], w, cx));
        let s3 = st.clone();
        let paste = Self::choices("set-paste", &PASTEBOARDS, p.pasteboard as usize, move |i, w, cx| Self::set(&s3, |p| p.pasteboard = i as u8, w, cx));
        v_flex()
            .child(Self::row("Theme", "Dark or light, or as the system is set (follows it while running). Appearance has the accent and the text size.", "Dark", theme))
            .child(Self::row("Pasteboard", "What is around the image. Grey shows the edges of dark images.", "Grey", paste))
            .into_any_element()
    }

    fn views(&self, p: &Prefs, cx: &App) -> AnyElement {
        let st = self.store.clone();
        let tool = self.store.read(cx).tool.clone();
        let toggle = |id: &'static str, on: bool, f: Run| Self::choices(id, &["Off", "On"], usize::from(on), move |i, w, cx| {
            if (i == 1) != on {
                f(w, cx)
            }
        });
        let (s1, s2, s3, s4) = (st.clone(), st.clone(), st.clone(), st.clone());
        v_flex()
            .child(Self::row("Rulers", "Rulers along the edges of a plane (Ctrl+R).", "Off", toggle("set-rulers", tool.rulers, Rc::new(move |_, cx| s1.update(cx, |s, cx| s.update_tool(|t| t.rulers = !t.rulers, cx))))))
            .child(Self::row("Crosshair", "The 3D cursor drawn across planes (the Probe tool moves it).", "Off", toggle("set-cross", tool.crosshair, Rc::new(move |_, cx| s2.update(cx, |s, cx| s.update_tool(|t| t.crosshair = !t.crosshair, cx))))))
            .child(Self::row("Link views", "Views of the same image move and zoom together.", "Off", toggle("set-link", p.link_views, Rc::new(move |_, cx| s3.update(cx, |s, cx| s.update_prefs(|p| p.link_views = !p.link_views, cx))))))
            .child(Self::row("Results over the image", "The pipeline's results drawn on the planes.", "On", toggle("set-results", tool.show_results, Rc::new(move |_, cx| s4.update(cx, |s, cx| s.update_tool(|t| t.show_results = !t.show_results, cx))))))
            .into_any_element()
    }

    /// The compute backend: chosen here, switched while running, tested with a checked read.
    fn compute(&self, p: &Prefs, cx: &App) -> AnyElement {
        let s = self.store.read(cx);
        let targets: Vec<lumen_core::ComputeTarget> = lumen_core::ComputeTarget::available();
        // A dropdown with short names; a runtime that cannot run here says why in its entry
        let mut names: Vec<SharedString> = vec!["Automatic".into()];
        names.extend(targets.iter().map(|t| {
            let name = match t.name() {
                "wgpu" => "GPU · Vulkan / Metal",
                "cuda" => "GPU · CUDA",
                "cpu" => "CPU",
                "hip" => "GPU · ROCm",
                other => other,
            };
            SharedString::from(match t.unusable_reason() {
                Some(why) => format!("{name} ({why})"),
                None => name.to_string(),
            })
        }));
        let current = p.runtime.as_deref().and_then(|r| targets.iter().position(|t| t.name() == r)).map_or(0, |i| i + 1);
        let (st, keys) = (self.store.clone(), targets.iter().map(|t| t.name().to_string()).collect::<Vec<_>>());
        let pick = MenuSelect::new("set-runtime", names, Some(current), move |i, _, cx| {
            let r = if i == 0 { None } else { keys.get(i - 1).cloned() };
            st.update(cx, |s, cx| s.set_runtime(r, cx));
        });
        // The chosen one cannot run here: said right under it (`proximity`)
        let chosen_problem = p.runtime.as_deref().and_then(|r| lumen_core::ComputeTarget::parse(r).ok()).and_then(|t| t.unusable_reason().map(|why| format!("{} cannot run here: {why}. The first runtime that runs is used instead.", t.name().to_uppercase())));
        // The device as a status: a dot and words (the dot never alone)
        let (dot, words) = match (&s.device, s.prefs.runtime.as_deref()) {
            (Some(d), _) => (color::success(), d.label.clone()),
            (None, _) => (color::text_muted(), "Idle · starts when a result or the 3D view needs it".to_string()),
        };
        let st = self.store.clone();
        let st2 = self.store.clone();
        let test = h_flex()
            .gap(px(theme::SM))
            .child(Button::new("set-test-device").outline().small().icon(Icon::new(IconName::Play)).label("Test").on_click(move |_, _, cx| st.update(cx, |s, cx| s.test_device(cx))))
            .child(Button::new("set-restart-device").ghost().small().icon(Icon::new(IconName::RotateCcw)).label("Restart").tooltip("Close the device and open it again (after it stopped responding)").on_click(move |_, _, cx| st2.update(cx, |s, cx| {
                s.drop_device(cx);
                s.ensure_device(cx);
            })));
        v_flex()
            .child(
                v_flex()
                    .child(Self::row("Backend", "Where images are computed. Switching applies at once: results are computed again.", "Automatic", pick))
                    .children(chosen_problem.map(|t| div().pb(px(theme::SM)).text_xs().text_color(color::warning()).child(t))),
            )
            .child(Self::row("Device", "", "", h_flex().gap(px(theme::SM)).child(div().size(px(8.)).rounded_full().bg(dot)).child(div().text_sm().child(words))))
            .child(Self::row("Check", "Run a quick check on the device.", "", test))
            .into_any_element()
    }

    /// The model plugins found, their models, installing them and choosing the one used.
    fn models(&self, p: &Prefs, cx: &App) -> AnyElement {
        let s = self.store.read(cx);
        let st = self.store.clone();
        let rescan = Button::new("set-ml-rescan").ghost().small().icon(Icon::new(IconName::RefreshCw)).label("Look again").on_click(move |_, _, cx| st.update(cx, |s, cx| s.scan_plugins(cx)));
        let chosen = s.ml.segmenter(p.ml_model.as_deref()).map(|(_, id, _)| id);
        let busy = s.ml.busy.clone();
        let mut col = v_flex().gap(px(theme::MD)).child(
            h_flex()
                .gap(px(theme::SM))
                .child(div().flex_1().text_xs().text_color(color::text_muted()).child("Models come as plugins: separate programs (lumen-ml-*), each with its own licence and runtime, so they never weigh on lumen."))
                .child(rescan),
        );
        if let Some((what, f)) = &busy {
            col = col.child(div().text_sm().text_color(color::accent_mark()).child(format!("{what}{}", f.map_or(String::new(), |f| format!(" — {:.0} %", f * 100.0)))));
        }
        if s.ml.scanning || !s.ml.scanned {
            col = col.child(div().text_sm().text_color(color::text_muted()).child("Looking for plugins…"));
        } else if s.ml.plugins.is_empty() {
            col = col.child(div().text_sm().child("No model plugin found.")).child(div().text_xs().text_color(color::text_muted()).child("Build one (cargo build -p lumen-ml-sam: it lands next to lumen-app) or copy a lumen-ml-* program into one of:"));
            for d in lumen_ml::plugin_dirs().into_iter().take(3) {
                col = col.child(div().text_xs().font_family("monospace").text_color(color::text_muted()).child(crate::domain::paths::short_path(&d)));
            }
        }
        for (pi, plugin) in s.ml.plugins.iter().enumerate() {
            let mut card = v_flex()
                .gap(px(theme::SM))
                .p(px(theme::MD))
                .rounded(px(8.))
                .border_1()
                .border_color(color::border())
                .bg(color::raised())
                .child(h_flex().gap(px(theme::SM)).child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(plugin.name.clone())).child(div().text_xs().text_color(color::text_muted()).child(format!("{} · {}", plugin.version, plugin.licence))))
                .child(div().text_xs().text_color(if plugin.runtime.ready { color::text_muted() } else { color::warning() }).child(plugin.runtime.detail.clone()));
            if !plugin.runtime.ready {
                let (st, program) = (self.store.clone(), plugin.program.clone());
                card = card.child(h_flex().child(
                    Button::new(SharedString::from(format!("set-ml-runtime-{pi}"))).small().primary().icon(Icon::new(IconName::Download)).label("Install runtime").disabled(busy.is_some()).on_click(move |_, _, cx| st.update(cx, |s, cx| s.install_model(program.clone(), lumen_ml::RUNTIME_ID.to_string(), cx))),
                ));
            }
            if let Some(g) = &plugin.runtime.gpu {
                let mut line = h_flex().gap(px(theme::SM)).child(div().flex_1().text_xs().text_color(if g.installed && !g.usable { color::warning() } else { color::text_muted() }).child(g.detail.clone()));
                if !g.installed {
                    let (st, program) = (self.store.clone(), plugin.program.clone());
                    line = line.child(Button::new(SharedString::from(format!("set-ml-gpu-{pi}"))).small().outline().icon(Icon::new(IconName::Download)).label(format!("Install GPU runtime ({:.0} MB)", g.bytes as f64 / 1e6)).disabled(busy.is_some()).on_click(move |_, _, cx| st.update(cx, |s, cx| s.install_model(program.clone(), lumen_ml::RUNTIME_CUDA_ID.to_string(), cx))));
                }
                card = card.child(line);
            }
            for (mi, m) in plugin.models.iter().enumerate() {
                let size = if m.download_bytes() > 0 { format!(" · {:.0} MB", m.download_bytes() as f64 / 1e6) } else { String::new() };
                let (st, program, id) = (self.store.clone(), plugin.program.clone(), m.id.clone());
                let action = if m.installed {
                    let on = chosen.as_deref() == Some(m.id.as_str());
                    Button::new(SharedString::from(format!("set-ml-use-{pi}-{mi}"))).small().ghost().label(if on { "In use" } else { "Use" }).selected(on).on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_ml_model(id.clone(), cx)))
                } else {
                    Button::new(SharedString::from(format!("set-ml-install-{pi}-{mi}"))).small().outline().icon(Icon::new(IconName::Download)).label("Install").disabled(busy.is_some() || !plugin.runtime.ready).on_click(move |_, _, cx| st.update(cx, |s, cx| s.install_model(program.clone(), id.clone(), cx)))
                };
                card = card.child(
                    h_flex()
                        .gap(px(theme::MD))
                        .py(px(2.))
                        .child(v_flex().flex_1().min_w_0().child(div().text_sm().child(m.name.clone())).child(div().text_xs().text_color(color::text_muted()).truncate().child(format!("{} · {}{size}", m.task.title(), m.description))))
                        .child(action),
                );
            }
            col = col.child(card);
        }
        for p in &s.ml.problems {
            col = col.child(div().text_xs().text_color(color::warning()).child(p.clone()));
        }
        col.into_any_element()
    }

    /// [`Self::choices`] for labels made at run time.

    /// The programs scripts use: where each is found (Settings, its variable or the `PATH`) and
    /// its version; Choose… sets a path, Use PATH clears it. Nothing is installed by lumen.
    fn tools(&mut self, p: &Prefs, cx: &mut Context<Self>) -> AnyElement {
        if self.versions_for.as_ref() != Some(&p.tools) {
            self.versions_for = Some(p.tools.clone());
            cx.spawn(async move |this, cx| {
                let v = cx.background_executor().spawn(async { Tool::ALL.iter().map(|&t| (t, tools::version(t))).collect::<Vec<_>>() }).await;
                this.update(cx, |s, cx| {
                    s.versions = v;
                    cx.notify();
                })
                .ok();
            })
            .detach();
        }
        let mut col = v_flex().child(self.environment(cx)).child(div().pt(px(theme::LG)).pb(px(theme::SM)).text_xs().text_color(color::text_muted()).child("Programs. lumen uses them as they are on this computer: each is looked for at the path set here, then at its variable (LUMEN_PYTHON…), then on the PATH. A Python set here is used instead of the environment."));
        for tool in Tool::ALL {
            let found = tools::find(tool);
            let source = match &found {
                Found::Settings(p) => format!("{} (set here)", crate::domain::paths::short_path(p)),
                Found::Env(p) => format!("{} (from {})", crate::domain::paths::short_path(p), tool.env()),
                Found::Path(p) => format!("{} (on the PATH)", crate::domain::paths::short_path(p)),
                Found::Missing => "Not found".to_string(),
            };
            let version = self.versions.iter().find(|(t, _)| *t == tool).map(|(_, v)| v.clone());
            let (vtext, ok) = match &version {
                Some(Ok(v)) => (v.clone(), true),
                Some(Err(e)) => (e.clone(), false),
                None => ("Checking…".to_string(), true),
            };
            let st = self.store.clone();
            let choose = Button::new(SharedString::from(format!("tool-choose-{}", tool.env()))).outline().small().label("Choose…").on_click(move |_, _, cx| {
                let rx = cx.prompt_for_paths(gpui_kit::PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some(format!("{} program", tool.title()).into()) });
                let st = st.clone();
                cx.spawn(async move |cx| {
                    if let Ok(Ok(Some(paths))) = rx.await
                        && let Some(path) = paths.into_iter().next()
                    {
                        st.update(cx, |s, cx| s.update_prefs(|p| *tool.slot(&mut p.tools) = Some(path), cx));
                    }
                })
                .detach();
            });
            let st = self.store.clone();
            let set_here = matches!(found, Found::Settings(_));
            let clear = Button::new(SharedString::from(format!("tool-clear-{}", tool.env()))).ghost().small().label("Use PATH").disabled(!set_here).on_click(move |_, _, cx| st.update(cx, |s, cx| s.update_prefs(|p| *tool.slot(&mut p.tools) = None, cx)));
            let control = v_flex()
                .items_end()
                .gap(px(theme::XS))
                .child(div().text_xs().font_family("monospace").text_color(color::text_muted()).child(source))
                .child(div().text_xs().text_color(if ok { color::text() } else { color::danger() }).child(vtext))
                .child(h_flex().gap(px(theme::XS)).child(clear).child(choose));
            col = col.child(Self::row(tool.title(), tool.what(), "", control));
        }
        col.into_any_element()
    }

    /// The project's environments: the Python its scripts run in (the project's own, the active
    /// one or lumen's default), and the Rust toolchain pinned for its Rust scripts.
    fn environment(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let dir = tools::scripts_dir();
        let py = tools::python_env();
        let (py_line, py_detail) = match &py {
            Some(env) => (
                format!("{} · Python {}", env.source.title(), env.version().unwrap_or_else(|| "?".into())),
                format!("{} · {} packages", crate::domain::paths::short_path(&env.root), env.packages().len()),
            ),
            None => ("None yet".to_string(), "Scripts run with the Python found (Programs below). Make one so every computer runs them the same.".to_string()),
        };
        let has_project_env = py.as_ref().is_some_and(|e| e.source == kitchen_shell::env::PythonSource::Project);
        let run = |label: &'static str, id: &'static str, primary: bool, f: fn() -> Result<String, String>, cx: &mut Context<Self>| {
            let b = Button::new(id).small().label(label).disabled(self.env_busy);
            let b = if primary { b.primary() } else { b.outline() };
            b.on_click(cx.listener(move |this, _, _, cx| {
                this.env_busy = true;
                this.env_note = None;
                cx.notify();
                cx.spawn(async move |this, cx| {
                    let r = cx.background_executor().spawn(async move { f() }).await;
                    this.update(cx, |s, cx| {
                        s.env_busy = false;
                        s.env_note = Some(r);
                        cx.notify();
                    })
                    .ok();
                })
                .detach();
            }))
        };
        let python = v_flex()
            .items_end()
            .gap(px(theme::XS))
            .child(div().text_xs().text_color(color::text()).child(py_line))
            .child(div().text_xs().font_family("monospace").text_color(color::text_muted()).max_w(px(400.)).child(py_detail))
            .child(
                h_flex()
                    .gap(px(theme::XS))
                    .child(run(if has_project_env { "Sync the project's" } else { "Make the project's" }, "env-sync-project", !has_project_env, tools::sync_project_env, cx))
                    .child(run("Sync lumen's default", "env-sync-default", false, tools::sync_default_env, cx)),
            );
        let pin = tools::rust_toolchain();
        let rust = v_flex()
            .items_end()
            .gap(px(theme::XS))
            .child(div().text_xs().text_color(color::text()).child(match &pin {
                Some(v) => format!("Pinned: Rust {v}"),
                None => "Not pinned: the default toolchain builds them".to_string(),
            }))
            .child(run(if pin.is_some() { "Pin the compiler found now" } else { "Pin the compiler found" }, "env-pin-rust", pin.is_none(), tools::pin_rust, cx));
        let note = match &self.env_note {
            _ if self.env_busy => Some(div().text_xs().text_color(color::text_muted()).child("Working…")),
            Some(Ok(t)) => Some(div().text_xs().font_family("monospace").text_color(color::text_muted()).child(t.lines().last().unwrap_or("Done").to_string())),
            Some(Err(e)) => Some(div().text_xs().text_color(color::danger()).child(e.clone())),
            None => None,
        };
        v_flex()
            .child(div().pb(px(theme::SM)).text_xs().text_color(color::text_muted()).child(format!(
                "The project's scripts folder ({}) holds its environments: pyproject.toml, uv.lock and .venv for Python, rust-toolchain.toml for Rust. The locks travel with the project, so its scripts run the same on another computer.",
                crate::domain::paths::short_path(&dir)
            )))
            .child(Self::row("Python", "The environment scripts and table code run in. uv makes and syncs it from the lock.", "", python))
            .child(Self::row("Rust", "The compiler the project's Rust scripts are built with (rust-toolchain.toml).", "", rust))
            .children(note)
            .into_any_element()
    }

    fn performance(&self, p: &Prefs) -> AnyElement {
        let st = self.store.clone();
        let auto = lumen_view::TileService::default_threads();
        let threads = if p.tile_threads == 0 { format!("Auto ({auto})") } else { p.tile_threads.to_string() };
        let threads = stepper("set-threads", threads, move |d, _, cx| st.update(cx, |s, cx| s.update_prefs(|p| p.tile_threads = (p.tile_threads as i64 + d as i64).clamp(0, 32) as usize, cx)));
        v_flex()
            .child(Self::row("Tile readers", "Threads reading image tiles for the first view (new views use 2). Applies to views opened after the change; 0 is automatic.", &format!("Auto ({auto})"), threads))
            .into_any_element()
    }

    fn about(&self, p: &Prefs) -> AnyElement {
        let st = self.store.clone();
        let path = Prefs::path().map_or("—".to_string(), |p| crate::domain::paths::short_path(&p));
        let reset = Button::new("set-reset").outline().small().label("Reset all settings").on_click(move |_, w, cx| {
            Self::set(
                &st,
                |p| {
                    let recent = std::mem::take(&mut p.recent);
                    *p = Prefs { recent, ..Prefs::default() };
                },
                w,
                cx,
            )
        });
        let _ = p;
        v_flex()
            .child(Self::row("Lumen", "GPU workstation for histology and radiology images.", env!("CARGO_PKG_VERSION"), div()))
            .child(Self::row("Settings file", &path, "per user", div()))
            .child(Self::row(
                "Compute runtimes",
                &format!("Compiled in: {}. The command line and plugins also read LUMEN_RUNTIME.", lumen_core::ComputeTarget::available().iter().map(|t| t.name()).collect::<Vec<_>>().join(", ")),
                "",
                div(),
            ))
            .child(Self::row("Reset", "Every setting back to its default (recent files stay).", "—", reset))
            .into_any_element()
    }
}

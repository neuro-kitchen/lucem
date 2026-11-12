//! The root view: kitchen-shell's frame (title bar with the Lucem menu, Help, the workspaces, the
//! project and where lucem runs; the workspace's menu row; the workspace as a card; the status
//! bar with the jobs and this computer's load; Settings; the search) around lucem's three
//! workspaces. Every action is a command of one registry (`crate::commands`), so the menus, the
//! keys, the search and an agent see the same list. Without an image Explore shows the home view.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, App, AppContext as _, Context, Entity, ExternalPaths, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};
use kitchen_agent::{AcpClient, AgentSpec, AgentWorkspace, McpHost, McpStdio, Policy};
use kitchen_shell::{CommandSpec, MenuItem, MenuSpec, Registry, RemoteSpec, SettingsSection, Shell, ShellEvent, ShellSpec, WorkspaceSpec};

use crate::commands::{self, Ctx};
use crate::data::store::{AppEvent, Store, ViewRequest};
use crate::navigation::actions::{ToggleSidebar, HELP};
use crate::navigation::route::Workspace;
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
        let device = s.device.as_ref().map_or("GPU idle".to_string(), |d| d.label.clone());
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
                    .when(st.cached_mb > 0, |d| d.child(div().text_color(color::text_muted()).child(format!("{} MB cached", st.cached_mb))))
            })
            .child(meter(IconName::Gpu, device, "GPU: idle, or what runs on it"))
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

/// The Keyboard & mouse sheet (F1), shown over the frame.
pub struct HelpSheet {
    close: Rc<dyn Fn(&mut Window, &mut App)>,
}

impl Render for HelpSheet {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let (c1, c2) = (self.close.clone(), self.close.clone());
        div()
            .id("help-overlay")
            .absolute()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui_kit::hsla(0., 0., 0., 0.55))
            .on_click(move |_, w, cx| c1(w, cx))
            .child(
                v_flex()
                    .id("help")
                    .w(px(600.))
                    .p(px(theme::LG))
                    .gap(px(theme::MD))
                    .rounded(px(theme::RADIUS_LG))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::raised())
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("Keyboard & mouse"))
                    .children(HELP.iter().map(|(group, rows)| {
                        v_flex().gap(px(theme::XS)).child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child(group.to_uppercase())).children(
                            rows.iter().map(|(keys, what)| h_flex().gap(px(theme::MD)).child(div().w(px(250.)).text_sm().font_family("monospace").child(*keys)).child(div().text_sm().child(*what))),
                        )
                    }))
                    .child(h_flex().justify_end().child(Button::new("help-close").outline().small().label("Close").on_click(move |_, w, cx| c2(w, cx)))),
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
    pub analyze: Entity<AnalyzeView>,
    /// The Agents workspace: the chat with the agent, its sessions, what it changed.
    pub agents: Entity<AgentWorkspace>,
    /// lucem's commands served to agents (MCP), and the socket they reach it on.
    pub host: Entity<McpHost>,
    socket: std::path::PathBuf,
    _subs: Vec<Subscription>,
}

/// What an agent is told about lucem when it connects.
const AGENT_INSTRUCTIONS: &str = "lucem is a GPU workstation for histology and microscopy. Its tools are the app's own commands, run in the open window as the person would: open images, find tissue and cells, classify, add pipeline steps (process.op.*), show views, export. Commands that change something wait for the person to allow them. Every call is recorded in the session.";

/// The name the app's tools have in the agent (`mcp_lucem_app_hist_find_cells` in Antigravity).
const AGENT_SERVER: &str = "lucem_app";

/// What the agent is told before each session's first prompt: it works inside the running app,
/// through the `lucem_app` tools, and what the app holds now.
fn agent_preamble(store: &Entity<Store>, cx: &App) -> String {
    let s = store.read(cx);
    let images: Vec<String> = s.doc.images.iter().map(|i| {
        let shown = if s.doc.active_image.as_deref() == Some(i.id.as_str()) { ", shown" } else { "" };
        format!("{} (id {}, {} channels{shown})", i.name, i.id, i.channels.len())
    }).collect();
    let project = if s.project_file.is_some() { s.doc.name.clone() } else { format!("{} (not saved)", s.doc.name) };
    format!(
        "[Context from lucem] You are working inside lucem, a running GPU app for histology and microscopy, beside the person. Act through the `{AGENT_SERVER}` tools: each one is a command of the open app and runs in its window as if the person chose it (open images, find tissue and cells, classify, add pipeline steps with process_op_*, change views, export). Commands that change something wait for the person to allow them. Do not read or write files to find out about lucem; the tools are the way in. Project: {project}. Images: {}. Workspace shown: {}.",
        if images.is_empty() { "none open".to_string() } else { images.join("; ") },
        s.workspace.title()
    )
}

/// The agent the chat starts: the one chosen in Settings, else the first installed.
fn chosen_agent(store: &Entity<Store>, cx: &App) -> AgentSpec {
    let known = kitchen_agent::known_agents();
    match store.read(cx).prefs.agent.clone() {
        Some(cmd) => known.iter().find(|a| a.command == cmd).cloned().unwrap_or_else(|| AgentSpec::new("Agent", cmd)),
        None => known.iter().find(|a| a.available()).cloned().unwrap_or_else(|| known[0].clone()),
    }
}

/// Starts the chosen agent in the chat (replacing one running): it reaches lucem's commands
/// through this program run as an MCP bridge to the app's socket.
fn start_agent(agents: &Entity<AgentWorkspace>, store: &Entity<Store>, socket: &std::path::Path, cx: &mut App) {
    let spec = chosen_agent(store, cx);
    if !spec.available() {
        let text = format!("**{}** is not installed (`{}` was not found). Install it, or choose another agent under Agent ▸ Use.", spec.name, spec.command);
        agents.update(cx, |w, cx| w.chat().update(cx, |c, cx| {
            c.push(kitchen_ui::chat::ChatMessage::Agent { text }, cx);
        }));
        return;
    }
    let Ok(exe) = std::env::current_exe() else { return };
    // Not "lucem": that name is often the headless `lucem mcp` already registered in the agent
    let mcp = vec![McpStdio { name: AGENT_SERVER.into(), command: exe, args: vec!["--mcp-bridge".into(), socket.display().to_string()] }];
    let cwd = store.read(cx).project_file.as_ref().and_then(|f| f.parent().map(|p| p.to_path_buf())).or_else(|| std::env::var_os("HOME").map(Into::into)).unwrap_or_else(std::env::temp_dir);
    let client = AcpClient::start(spec, cwd, mcp);
    let preamble = agent_preamble(store, cx);
    agents.update(cx, |w, cx| {
        w.set_preamble(Some(preamble));
        w.connect(client, cx)
    });
}

/// The Agents workspace's commands: start, stop, a new session, which agent.
fn agent_commands(agents: &Entity<AgentWorkspace>, store: &Entity<Store>, socket: &std::path::Path) -> Vec<CommandSpec> {
    let (a1, s1, k1) = (agents.clone(), store.clone(), socket.to_path_buf());
    let (a2, a3, a4) = (agents.downgrade(), agents.downgrade(), agents.downgrade());
    let (a5, s5, k5) = (agents.clone(), store.clone(), socket.to_path_buf());
    let a6 = agents.downgrade();
    vec![
        CommandSpec::new("agent.start", "Start the agent", IconName::Play, move |_, cx| start_agent(&a1, &s1, &k1, cx)).group("Agents").hint("Starts the chosen agent in the chat (again, if it runs)"),
        CommandSpec::new("agent.stop", "Stop the agent", IconName::CircleStop, move |_, cx| {
            let _ = a2.update(cx, |w, cx| w.disconnect(cx));
        })
        .group("Agents")
        .hint("Closes the agent's program")
        .enabled(move |cx| if a3.upgrade().is_some_and(|w| w.read(cx).agent().is_some()) { Ok(()) } else { Err("No agent is running".into()) }),
        CommandSpec::new("agent.new_session", "New session", IconName::Plus, move |_, cx| {
            let _ = a4.update(cx, |w, cx| w.new_session(cx));
        })
        .group("Agents")
        .hint("An empty conversation (the agent starts afresh)"),
        CommandSpec::with_args::<AgentArgs>("agent.use", "Use an agent", IconName::Bot, move |a, _, cx| {
            s5.update(cx, |s, cx| s.update_prefs(|p| p.agent = Some(a.command.clone()), cx));
            if a6.upgrade().is_some_and(|w| w.read(cx).agent().is_some()) {
                start_agent(&a5, &s5, &k5, cx);
            }
            Ok(serde_json::Value::Null)
        })
        .group("Agents")
        .hint("Which agent the chat starts (an ACP command such as `agy acp`)"),
    ]
}

/// An agent to use.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct AgentArgs {
    /// Its ACP command (`agy acp`, `claude-code-acp`).
    command: String,
}

impl LucemApp {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        theme::configure(&store.read(cx).prefs, window);
        theme::apply(window, cx);
        let pipeline = cx.new(|cx| PipelineView::new(store.clone(), window, cx));
        // Explore can dock the same Pipeline under its views
        let explore = cx.new(|cx| ExploreView::new(store.clone(), pipeline.clone(), window, cx));
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
            let sheet = cx.new(|_| HelpSheet { close });
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
        let ctx = Ctx { store: store.clone(), shell: shell_cell.clone(), pipeline: pipeline.downgrade(), analyze: analyze.downgrade(), toggle_help: help };
        commands::register(&registry, &ctx);
        registry.extend(PipelineView::commands(pipeline.downgrade(), store.clone()));
        PipelineView::provide(&registry, store.clone());
        registry.extend(AnalyzeView::commands(analyze.downgrade()));
        AnalyzeView::provide(&registry, analyze.downgrade());
        pipeline.update(cx, |p, _| p.set_registry(registry.clone()));

        // The Agents workspace: lucem's commands as MCP tools (asking before anything that
        // changes), the chat running an ACP agent, the project's files and what changed
        let socket = kitchen_agent::default_socket("lucem");
        let host = cx.new(|cx| McpHost::new("lucem", AGENT_INSTRUCTIONS, registry.clone(), window.window_handle(), Policy::Ask, cx));
        if !cfg!(test) && let Err(e) = host.read(cx).listen(socket.clone()) {
            eprintln!("MCP socket: {e}");
        }
        let project_dir = store.read(cx).project_file.as_ref().and_then(|f| f.parent().map(|p| p.to_path_buf()));
        let watched: Vec<std::path::PathBuf> = project_dir.iter().flat_map(|d| [crate::domain::project::PROJECT_FILE, crate::domain::project::PIPELINE_FILE, "session.json"].map(|f| d.join(f))).collect();
        let changes = cx.new(|cx| kitchen_shell::ChangesPanel::new(watched, cx));
        let sessions = crate::data::prefs::Prefs::dir().unwrap_or_else(std::env::temp_dir).join("agents");
        let files_root = project_dir.clone().or_else(|| std::env::var_os("HOME").map(Into::into)).unwrap_or_else(std::env::temp_dir);
        let context = kitchen_ui::chat::ChatContext { workspace: store.read(cx).doc.name.clone(), location: kitchen_ui::chat::Location::Local, model: String::new() };
        let agents = cx.new(|cx| AgentWorkspace::new(sessions, files_root, host.clone(), changes, context, window, cx));
        registry.extend(agent_commands(&agents, &store, &socket));
        registry.provide("agent.choices", {
            let store = store.clone();
            move |cx| {
                let current = chosen_agent(&store, cx);
                kitchen_agent::known_agents()
                    .into_iter()
                    .map(|a| {
                        if a.available() {
                            MenuItem::run(a.name.clone(), "agent.use", serde_json::json!({"command": a.command})).icon(IconName::Bot).checked(a.command == current.command)
                        } else {
                            MenuItem::label(format!("{} (not installed: {})", a.name, a.command))
                        }
                    })
                    .collect()
            }
        });

        let explore_body = cx.new(|cx| ExploreBody { store: store.clone(), explore: explore.clone(), _subs: vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify())] });
        let workspace = |w: Workspace, view: gpui_kit::AnyView, menus| WorkspaceSpec::new(commands::workspace_id(w), w.title(), w.icon(), view).hint(w.hint()).menus(menus);
        let workspaces = vec![
            workspace(Workspace::Explore, explore_body.into(), commands::explore_menus()),
            workspace(Workspace::Pipeline, pipeline.clone().into(), PipelineView::menu_specs()),
            workspace(Workspace::Analyze, analyze.clone().into(), AnalyzeView::menu_specs()),
            WorkspaceSpec::new("agents", "Agents", IconName::Bot, agents.clone()).hint("Chat with an agent that works in lucem: its sessions, and what it changed").menus(vec![MenuSpec::new(
                "Agent",
                vec![MenuItem::command("agent.start"), MenuItem::command("agent.stop"), MenuItem::command("agent.new_session"), MenuItem::Separator, MenuItem::label("Use"), MenuItem::provided("agent.choices")],
            )]),
        ];
        let settings_view = cx.new(|_| SettingsView::new(store.clone()));
        let sections = Section::ALL
            .into_iter()
            .map(|sec| {
                let view = settings_view.clone();
                SettingsSection::new(sec.title(), sec.icon(), move |_, cx| view.update(cx, |v, cx| v.body(sec, cx)))
            })
            .collect::<Vec<_>>();
        let mut sections = sections;
        sections.push(kitchen_agent::workspace::settings_section(host.clone(), (!cfg!(test)).then(|| socket.clone())));
        let readings = cx.new(|cx| StatusReadings::new(store.clone(), explore.clone(), cx));
        let mut spec = ShellSpec::new("Lucem", registry.clone(), workspaces);
        spec.app_menu = commands::app_menu();
        spec.help_menu = vec![kitchen_shell::MenuItem::command("help.keys")];
        spec.settings = sections;
        spec.layout_toggles = vec!["view.left".into(), "view.right".into()];
        spec.remote = Some(RemoteSpec { label: "This computer".into(), icon: IconName::Server, menu: vec![kitchen_shell::MenuItem::command("remote.this"), kitchen_shell::MenuItem::Separator, kitchen_shell::MenuItem::command("remote.connect")] });
        spec.status_items = Some(Rc::new(move |_, _| readings.clone().into_any_element()));
        let shell = cx.new(|cx| Shell::new(spec, window, cx));
        let _ = shell_cell.set(shell.downgrade());
        registry.add(kitchen_agent::workspace::open_command(&shell, "agents"));

        // Keys: every command's (the canvas binds its own)
        for bad in registry.bind_keys(None, cx) {
            eprintln!("key binding: {bad}");
        }

        // The frame and the store follow each other: workspace, status line, project, look
        let subs = vec![
            cx.subscribe(&shell, |this, _, e: &ShellEvent, cx| match e {
                ShellEvent::WorkspaceChanged(id) => {
                    // The agent starts the first time its workspace is shown (not in tests)
                    if id.as_ref() == "agents" && !cfg!(test) && this.agents.read(cx).agent().is_none() {
                        start_agent(&this.agents, &this.store, &this.socket, cx);
                    }
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
            cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
                this.sync_shell(e, cx);
                // The agent's note follows what is open
                let preamble = agent_preamble(&this.store, cx);
                this.agents.update(cx, |w, _| w.set_preamble(Some(preamble)));
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
        let this = Self { store, shell, registry, explore, pipeline, analyze, agents, host, socket, _subs: subs };
        this.sync_shell(&AppEvent::Workspace, cx);
        this.sync_shell(&AppEvent::Status, cx);
        this
    }

    /// Tells the frame what changed in the store: the workspace, the status line, the project.
    fn sync_shell(&self, e: &AppEvent, cx: &mut App) {
        let s = self.store.read(cx);
        let workspace = commands::workspace_id(s.workspace);
        let status = s.status.clone();
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
        div()
            .id("lucem-app")
            .size_full()
            .on_action(cx.listener(|this, _: &ToggleSidebar, _, cx| this.store.update(cx, |s, cx| s.request_view(ViewRequest::ToggleSides, cx))))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                if let Some(p) = paths.paths().first().cloned() {
                    this.open_path(p, cx);
                }
            }))
            .child(self.shell.clone())
    }
}

/// Headless UI tests: the real app in a test window (layout, hit testing, no pixels).
#[cfg(test)]
#[path = "app_tests.rs"]
mod tests;

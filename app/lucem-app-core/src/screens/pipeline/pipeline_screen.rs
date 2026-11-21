//! The Pipeline workspace (dsp-kitchen's layout): a menu bar on top (Photoshop style: Sources,
//! then one menu per family of operators, Output; an item is added after the selected node, fed
//! by it when it fits), the canvas with a floating tool bar, the inspector of the selection on the
//! right. Every edit is a command, so Explore, the canvas and an agent see the same graph.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _,
    StyledImage as _, Styled as _, Subscription, WeakEntity, Window,
};

use crate::data::results::COLORMAPS;
use crate::data::store::{AppEvent, Store};
use crate::domain::command::Command;
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::ui::param_editor::{Commit, EditOps as _, ParamEditor};
use crate::ui::scrub_input::{scrub_number, ScrubState};
use crate::ui::theme::{self, color};
use crate::ui::widgets::{divider_v, row, MenuSelect, Section};
use kitchen_shell::{CommandSpec, MenuItem, MenuSpec, Registry};
use kitchen_ui::widgets::{CodeChanged, CodeLanguage, CodeView};
use serde_json::json;

use super::canvas::{CanvasEvent, GraphCanvas};
use super::catalog::{describe, operations, port_color, sources, Template};

const INSPECTOR_W: f32 = 312.;

/// Where the floating tool bar sits on the canvas: along the top or bottom edge (a row), or the
/// left or right edge (a column of icons). The bottom place sits above the canvas' own bar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolbarDock {
    #[default]
    Top,
    Bottom,
    Left,
    Right,
}

impl ToolbarDock {
    const ALL: [ToolbarDock; 4] = [ToolbarDock::Top, ToolbarDock::Bottom, ToolbarDock::Left, ToolbarDock::Right];

    fn label(self) -> &'static str {
        match self {
            ToolbarDock::Top => "Top",
            ToolbarDock::Bottom => "Bottom",
            ToolbarDock::Left => "Left",
            ToolbarDock::Right => "Right",
        }
    }

    fn column(self) -> bool {
        matches!(self, ToolbarDock::Left | ToolbarDock::Right)
    }
}

pub struct PipelineView {
    store: Entity<Store>,
    pub canvas: Entity<GraphCanvas>,
    inspector: Entity<Inspector>,
    /// The app's commands, for the bar drawn when the Pipeline is docked in Explore.
    registry: Option<Registry>,
    /// Where the floating tool bar sits.
    dock: ToolbarDock,
    /// Whether the right-side Inspector pane is open.
    inspector_open: bool,
    _subs: Vec<Subscription>,
}

/// A source node: a channel of an image, or (no channel) the image's shapes as a mask.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct SourceArgs {
    /// The image's id.
    image: String,
    /// The channel (0 is the first); none for the image's shapes as a mask.
    #[serde(default)]
    channel: Option<usize>,
}

/// A model step.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ModelArgs {
    /// The model's id (Settings › Models).
    id: String,
    /// Whether it segments what shapes point at (Segment Anything) rather than every nucleus.
    #[serde(default)]
    interactive: bool,
}

/// A drop-in script plugin file path.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ScriptPathArgs {
    /// Absolute path to a `.py` or `.rs` script op.
    path: String,
}

impl PipelineView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let canvas = cx.new(|cx| GraphCanvas::new(store.clone(), cx));
        let inspector = cx.new(|cx| Inspector::new(store.clone(), canvas.clone(), window, cx));
        let subs = vec![
            cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
                if let AppEvent::RevealNode(id) = e {
                    // A new script node: selected, so its code opens in the Inspector's editor
                    let id = id.clone();
                    this.canvas.update(cx, |c, cx| c.select([id], cx));
                }
                if matches!(e, AppEvent::Document(_) | AppEvent::Results | AppEvent::ImageReady(_) | AppEvent::RevealNode(_)) {
                    cx.notify();
                }
            }),
            cx.subscribe(&canvas, |_, _, _: &CanvasEvent, cx| cx.notify()),
        ];
        Self { store, canvas, inspector, registry: None, dock: ToolbarDock::default(), inspector_open: true, _subs: subs }
    }

    fn canvas_do(canvas: &WeakEntity<GraphCanvas>, cx: &mut App, f: impl FnOnce(&mut GraphCanvas, &mut Context<GraphCanvas>)) {
        let _ = canvas.update(cx, f);
    }

    /// The menu bar: Sources, Edit, one menu per family of operators, Output, View.
    /// The Pipeline's own bar: shown above the canvas when the Pipeline is docked in Explore;
    /// in its workspace the same menus sit in the frame's menu row ([`Self::menu_specs`]).
    fn menu_bar(&self, _cx: &mut Context<Self>) -> impl IntoElement + use<> {
        h_flex()
            .id("pipeline-menu-bar")
            .flex_none()
            .h(px(theme::OPTIONS_BAR_H))
            .px(px(theme::SM))
            .items_center()
            .bg(color::panel())
            .border_b_1()
            .border_color(color::border())
            .child(self.menus())
    }

    /// The Pipeline's commands (kitchen-shell): editing the graph, adding sources, outputs,
    /// scripts and model steps, and the canvas' view. Its operations are the `process.op.*`
    /// commands, which add a node while the Pipeline is shown (or with `{"node": true}`).
    pub fn commands(this: WeakEntity<Self>, store: Entity<Store>) -> Vec<CommandSpec> {
        let canvas_do = |f: fn(&mut GraphCanvas, &mut Context<GraphCanvas>)| {
            let this = this.clone();
            move |_: &mut Window, cx: &mut App| {
                if let Some(p) = this.upgrade() {
                    let canvas = p.read(cx).canvas.clone();
                    canvas.update(cx, f);
                }
            }
        };
        let selected = || {
            let this = this.clone();
            move |cx: &App| match this.upgrade() {
                Some(p) if !p.read(cx).canvas.read(cx).selected.is_empty() => Ok(()),
                _ => Err("Select nodes first".to_string()),
            }
        };
        let canvas_flag = |f: fn(&GraphCanvas) -> bool| {
            let this = this.clone();
            move |cx: &App| this.upgrade().is_some_and(|p| f(p.read(cx).canvas.read(cx)))
        };
        let owner = this.clone();
        let add = move |template: Template| {
            let this = owner.clone();
            move |cx: &mut App| {
                if let Some(p) = this.upgrade() {
                    let canvas = p.read(cx).canvas.clone();
                    canvas.update(cx, |c, cx| c.add(&template, None, cx));
                }
            }
        };
        let (add_source, add_view) = (add.clone(), add(Template::View));
        let (s1, s2, s3) = (store.clone(), store.clone(), store.clone());
        vec![
            CommandSpec::new("pipeline.duplicate", "Duplicate", IconName::Copy, canvas_do(|c, cx| c.duplicate_selected(cx))).group("Pipeline").hint("The selected nodes (Ctrl+D on the canvas)").enabled(selected()),
            CommandSpec::new("pipeline.delete", "Delete, reconnecting", IconName::Trash, canvas_do(|c, cx| c.delete_selected(true, cx))).group("Pipeline").hint("The selected nodes; their neighbours are joined (Del)").enabled(selected()),
            CommandSpec::new("pipeline.delete_alone", "Delete alone", IconName::Delete, canvas_do(|c, cx| c.delete_selected(false, cx))).group("Pipeline").hint("The selected nodes only (Shift+Del)").enabled(selected()),
            CommandSpec::new("pipeline.select_all", "Select all", IconName::SquareDashed, canvas_do(|c, cx| c.select_all(cx))).group("Pipeline").hint("Every node (Ctrl+A on the canvas)"),
            CommandSpec::with_args::<SourceArgs>("pipeline.source", "Add a source", IconName::Image, move |a, _, cx| {
                let template = match a.channel {
                    Some(channel) => Template::Source { image: a.image, channel },
                    None => Template::Annotations { image: a.image },
                };
                add_source(template)(cx);
                Ok(serde_json::Value::Null)
            })
            .group("Pipeline")
            .hint("A channel of an image, or an image's shapes as a mask, as a node"),
            CommandSpec::new("pipeline.output", "Result view", IconName::Eye, move |_, cx| add_view(cx)).group("Pipeline").hint("A result shown in Explore and Analyze"),
            CommandSpec::new("script.from_file", "Script from a file (.py, .rs)…", IconName::FileCode, move |_, cx| {
                #[cfg(target_family = "wasm")]
                {
                    s1.update(cx, |s, cx| {
                        s.set_workspace(crate::navigation::route::Workspace::Code, cx);
                        s.set_status("Browse and open server scripts in the Code workspace", cx);
                    });
                    return;
                }
                #[cfg(not(target_family = "wasm"))]
                {
                    let rx = cx.prompt_for_paths(gpui_kit::PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some("Script file".into()) });
                    let store = s1.clone();
                    cx.spawn(async move |cx| {
                        if let Ok(Ok(Some(paths))) = rx.await
                            && let Some(p) = paths.into_iter().next()
                        {
                            store.update(cx, |s, cx| s.add_script_file(p, cx));
                        }
                    })
                    .detach();
                }
            })
            .group("Automate")
            .hint("A Python or Rust file as a step"),
            CommandSpec::with_args::<ScriptPathArgs>("script.from_plugin", "Add a plugin script", IconName::SquareTerminal, move |a, _, cx| {
                s3.update(cx, |s, cx| s.add_script_file(std::path::PathBuf::from(a.path), cx));
                Ok(serde_json::Value::Null)
            })
            .group("Automate")
            .hint("Add a discovered community or user script op from plugins/ops"),
            {
                let this_r = this.clone();
                CommandSpec::with_args::<ScriptPathArgs>("pipeline.apply_recipe", "Add a recipe", IconName::Package, move |a, _, cx| {
                    let recipe = crate::data::recipes::load_recipe(std::path::Path::new(&a.path))?;
                    if let Some(p) = this_r.upgrade() {
                        let canvas = p.read(cx).canvas.clone();
                        canvas.update(cx, |c, cx| {
                            for op in recipe.steps {
                                c.add(&Template::Op(op), None, cx);
                            }
                        });
                    }
                    Ok(serde_json::Value::Null)
                })
                .group("Pipeline")
                .hint("Add every step of a .lucem-recipe.json to the graph")
            },
            CommandSpec::with_args::<ModelArgs>("pipeline.model", "Add a model step", IconName::BrainCircuit, move |a, _, cx| {
                s2.update(cx, |s, cx| s.add_model_node(a.id, a.interactive, cx));
                Ok(serde_json::Value::Null)
            })
            .group("Pipeline")
            .hint("A model plugin as a step: labels from its answers"),
            CommandSpec::new("pipeline.fit", "Fit the graph", IconName::Maximize, canvas_do(|c, cx| c.fit(cx))).group("Pipeline").hint("Every node in view (F on the canvas)").read_only(),
            CommandSpec::new("pipeline.actual", "Actual size", IconName::Scan, canvas_do(|c, cx| c.zoom_to(1.0, cx))).group("Pipeline").hint("100 % (Ctrl+0 on the canvas)").read_only(),
            CommandSpec::new("pipeline.grid", "Dot grid", IconName::Grip, canvas_do(|c, cx| {
                c.show_grid = !c.show_grid;
                cx.notify();
            }))
            .group("Pipeline")
            .hint("Dots behind the nodes")
            .checked(canvas_flag(|c| c.show_grid))
            .read_only(),
            CommandSpec::new("pipeline.minimap", "Minimap", IconName::Map, canvas_do(|c, cx| {
                c.show_minimap = !c.show_minimap;
                cx.notify();
            }))
            .group("Pipeline")
            .hint("The whole graph in a corner")
            .checked(canvas_flag(|c| c.show_minimap))
            .read_only(),
        ]
    }

    /// The Pipeline's menu providers: its sources, the model steps, every family as nodes.
    pub fn provide(registry: &Registry, store: Entity<Store>) {
        let st = store.clone();
        registry.provide("pipeline.sources", move |cx| {
            let items = sources(&st.read(cx).doc).items;
            if items.is_empty() {
                return vec![MenuItem::label("Open an image first")];
            }
            items
                .into_iter()
                .filter_map(|i| match i.template {
                    Template::Source { image, channel } => Some(MenuItem::run(format!("{}   {}", i.label, i.hint), "pipeline.source", json!({"image": image, "channel": channel}))),
                    Template::Annotations { image } => Some(MenuItem::run(format!("{}   {}", i.label, i.hint), "pipeline.source", json!({"image": image}))),
                    _ => None,
                })
                .collect()
        });
        let st_plugins = store.clone();
        registry.provide("pipeline.models", move |cx| {
            let models: Vec<MenuItem> = store
                .read(cx)
                .ml
                .plugins
                .iter()
                .flat_map(|p| p.models.iter())
                .filter(|m| matches!(m.task, lucem_ml::Task::Segmentation | lucem_ml::Task::InteractiveSegmentation))
                .map(|m| {
                    let interactive = m.task == lucem_ml::Task::InteractiveSegmentation;
                    // Named by purpose: what the step does, then with which model
                    let purpose = if interactive { "Segment what shapes point at" } else { "Detect cells" };
                    let label = if m.installed { format!("{purpose} · {}", m.name) } else { format!("{purpose} · {} (not installed)", m.name) };
                    MenuItem::run(label, "pipeline.model", json!({"id": m.id, "interactive": interactive}))
                })
                .collect();
            if models.is_empty() {
                return vec![MenuItem::label("No model plugins found (Settings › Models)")];
            }
            let mut out = vec![MenuItem::label("A step by a model → labels")];
            out.extend(models);
            out
        });
        registry.provide("pipeline.families", |_| {
            operations()
                .into_iter()
                .map(|c| {
                    let items = c.items.iter().filter(|i| matches!(i.template, Template::Op(_))).map(|i| MenuItem::run(format!("{}   {}", i.label, i.hint), crate::commands::op_id(&i.label), json!({"node": true})).icon(c.icon)).collect();
                    MenuItem::submenu_with_icon(c.name, c.icon, items)
                })
                .collect()
        });
        registry.provide("pipeline.histology", |_| {
            let c = MenuItem::command;
            vec![
                MenuItem::label("Slide & tissue"),
                c("hist.find_tissue"),
                c("hist.superpixels"),
                c("hist.density"),
                c("hist.hotspots"),
                MenuItem::Separator,
                MenuItem::label("Cells & phenotyping"),
                c("hist.find_cells"),
                MenuItem::submenu_with_icon("Find cells with", IconName::Microscope, vec![MenuItem::provided("hist.detectors")]),
                c("hist.intensity"),
                c("hist.train"),
                MenuItem::Separator,
                MenuItem::label("Pixel classification"),
                c("hist.train_pixels"),
                c("hist.class_regions"),
            ]
        });
        registry.provide("pipeline.recipes", |_| {
            let recipes = crate::data::recipes::discover_recipes();
            if recipes.is_empty() {
                return vec![MenuItem::label("No recipes found (~/.config/lucem/recipes)")];
            }
            let items: Vec<MenuItem> = recipes
                .into_iter()
                .map(|(path, r)| {
                    let label = format!("{}   {} step(s)", r.name, r.steps.len());
                    MenuItem::run(label, "pipeline.apply_recipe", json!({"path": path.to_string_lossy()})).icon(IconName::Package)
                })
                .collect();
            vec![MenuItem::submenu_with_icon("Recipes", IconName::Package, items)]
        });
        registry.provide("pipeline.plugin_scripts", move |cx| {
            #[cfg(target_family = "wasm")]
            {
                let ops = st_plugins.read(cx).server_plugin_ops.clone();
                if ops.is_empty() {
                    return vec![MenuItem::label("No server plugin scripts found (~/.config/lucem/plugins/ops)")];
                }
                let mut out = vec![MenuItem::label("Server plugin scripts (plugins/ops)")];
                for sc in ops {
                    let label = if sc.description.is_empty() {
                        format!("{} · {}", sc.name, sc.family)
                    } else {
                        format!("{}   {}", sc.name, sc.description)
                    };
                    out.push(MenuItem::run(label, "script.from_plugin", json!({"path": sc.path})).icon(IconName::SquareTerminal));
                }
                return out;
            }
            #[cfg(not(target_family = "wasm"))]
            {
                let _ = (&st_plugins, cx);
                let scripts = crate::data::script::discover_plugin_ops();
                if scripts.is_empty() {
                    return vec![MenuItem::label("No plugin scripts found (~/.config/lucem/plugins/ops)")];
                }
                let mut out = vec![MenuItem::label("Plugin scripts (plugins/ops)")];
                for sc in scripts {
                    let label = if sc.description.is_empty() {
                        format!("{} · {}", sc.name, sc.family)
                    } else {
                        format!("{}   {}", sc.name, sc.description)
                    };
                    out.push(MenuItem::run(label, "script.from_plugin", json!({"path": sc.path.to_string_lossy()})).icon(IconName::SquareTerminal));
                }
                out
            }
        });
    }

    /// The Pipeline's menus: Edit, Sources, Process, active plugin tabs (e.g. Histology), Script, Model, Output.
    pub fn menu_specs() -> Vec<MenuSpec> {
        let c = MenuItem::command;
        let mut menus = vec![
            MenuSpec::new(
                "Edit",
                vec![
                    c("pipeline.duplicate"),
                    c("pipeline.delete"),
                    c("pipeline.delete_alone"),
                    MenuItem::Separator,
                    c("pipeline.select_all"),
                    MenuItem::Separator,
                    c("edit.undo"),
                    c("edit.redo"),
                    MenuItem::Separator,
                    c("pipeline.fit"),
                    c("pipeline.actual"),
                    c("pipeline.grid"),
                    c("pipeline.minimap"),
                ],
            ),
            MenuSpec::new("Sources", vec![MenuItem::provided("pipeline.sources")]),
            MenuSpec::new("Process", vec![c("process.find"), MenuItem::Separator, MenuItem::label("A node on the canvas"), MenuItem::provided("pipeline.families"), MenuItem::Separator, MenuItem::provided("pipeline.recipes")]),
        ];
        for tab in super::catalog::plugin_tabs() {
            menus.push(MenuSpec::new(tab.title, vec![MenuItem::provided(tab.provider)]));
        }
        menus.extend([
            MenuSpec::new("Script", vec![MenuItem::label("A step written as a file; edit, save, it runs again"), c("script.new_python"), c("script.new_rust"), c("script.from_file"), MenuItem::Separator, MenuItem::provided("pipeline.plugin_scripts")]),
            MenuSpec::new("Model", vec![MenuItem::provided("pipeline.models")]),
            MenuSpec::new("Output", vec![c("pipeline.output")]),
        ]);
        menus
    }

    /// Hands the Pipeline the app's commands (its own bar, docked in Explore, draws them).
    pub fn set_registry(&mut self, registry: Registry) {
        self.registry = Some(registry);
    }

    /// The menus as buttons, and the port colours.
    fn menus(&self) -> AnyElement {
        let buttons = self.registry.as_ref().map(|r| Self::menu_specs().iter().enumerate().map(|(i, m)| r.menu_button(format!("pipeline-menu-{i}"), m)).collect::<Vec<_>>()).unwrap_or_default();
        h_flex()
            .flex_1()
            .gap(px(2.))
            .items_center()
            .children(buttons)
            .child(div().flex_1())
            .children([("image", lucem_pipeline::PortType::Image), ("mask", lucem_pipeline::PortType::Mask), ("labels", lucem_pipeline::PortType::Labels), ("table", lucem_pipeline::PortType::Table)].into_iter().map(|(n, t)| {
                h_flex().gap(px(theme::XS)).px(px(theme::XS)).child(div().size(px(8.)).rounded_full().bg(port_color(Some(t)))).child(div().text_xs().text_color(color::text_muted()).child(n))
            }))
            .into_any_element()
    }

    /// Places the floating tool bar on one of the four canvas edges.
    pub fn set_dock(&mut self, dock: ToolbarDock, cx: &mut Context<Self>) {
        self.dock = dock;
        cx.notify();
    }

    /// The tool bar floating on the canvas: placeable on any edge ([`ToolbarDock`]), and steps
    /// aside (fades out, takes no pointer) while the view moves.
    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let c = self.canvas.read(cx);
        let (hand, selected) = (c.hand, !c.selected.is_empty());
        let dock = self.dock;
        let column = dock.column();
        let canvas = self.canvas.downgrade();
        let tool = |id: &'static str, icon: IconName, tip: &'static str, on: bool, f: fn(&mut GraphCanvas, &mut Context<GraphCanvas>)| {
            let canvas = canvas.clone();
            Button::new(id).ghost().small().icon(Icon::new(icon)).tooltip(tip).selected(on).on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, f))
        };
        let sep = || if column { div().h(px(1.)).w(px(20.)).my(px(theme::XS)).bg(color::border()).into_any_element() } else { divider_v().into_any_element() };
        let me = cx.entity().downgrade();
        let place = Button::new("tool-dock")
            .ghost()
            .small()
            .icon(Icon::new(IconName::Grip))
            .tooltip("Place the tool bar")
            .dropdown_menu(move |m: PopupMenu, _, _| {
                let mut m = m.label("Place the tool bar");
                for d in ToolbarDock::ALL {
                    let me = me.clone();
                    m = m.item(PopupMenuItem::new(d.label()).checked(d == dock).on_click(move |_, _, cx| {
                        let _ = me.update(cx, |p, cx| p.set_dock(d, cx));
                    }));
                }
                m
            });
        let store = self.store.clone();
        let run = {
            let btn = Button::new("tool-run").ghost().small().icon(Icon::new(IconName::Play));
            let btn = if column { btn } else { btn.label("Run on view") };
            btn.tooltip("Compute the results again over what Explore shows").on_click(move |_, _, cx| store.update(cx, |s, cx| s.recompute(cx)))
        };
        let pill = div()
            .id("pipeline-toolbar-pill")
            .flex()
            .when(column, |d| d.flex_col())
            .when(!column, |d| d.flex_row())
            .items_center()
            .gap(px(2.))
            .p(px(theme::XS))
            .rounded(px(8.))
            .border_1()
            .border_color(color::border())
            .bg(color::raised())
            .shadow_lg()
            .occlude()
            .child(place)
            .child(tool("tool-select", IconName::MousePointer2, "Select (V)", !hand, |c, cx| {
                c.hand = false;
                cx.notify();
            }))
            .child(tool("tool-hand", IconName::Hand, "Move the canvas (H, or hold Space)", hand, |c, cx| {
                c.hand = true;
                cx.notify();
            }))
            .child(sep())
            .child(tool("tool-duplicate", IconName::Copy, "Duplicate (Ctrl+D)", false, |c, cx| c.duplicate_selected(cx)))
            .child({
                let canvas = canvas.clone();
                Button::new("tool-delete").ghost().small().icon(Icon::new(IconName::Trash)).tooltip("Delete, reconnecting around it (Del; Shift+Del alone)").disabled(!selected).on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, |c, cx| c.delete_selected(true, cx)))
            })
            .child(sep())
            .child(tool("tool-fit", IconName::Maximize, "Fit the graph (F)", false, |c, cx| c.fit(cx)))
            .child(run)
            .child({
                let st = self.store.clone();
                let mut b = Button::new("tool-make-recipe")
                    .ghost()
                    .small()
                    .icon(Icon::new(IconName::Package))
                    .tooltip("Save this pipeline as a shareable .lucem-recipe.json")
                    .on_click(move |_, _, cx| save_pipeline_recipe(&st, cx));
                if !column {
                    b = b.label("Make recipe");
                }
                b
            });
        // Only the pill takes the pointer: the strip it sits in covers no control. The bottom
        // place clears the canvas' own bar (zoom, fit, grid, minimap) under it.
        let strip = div().id("pipeline-toolbar").absolute().flex();
        let strip = match dock {
            ToolbarDock::Top => strip.top_3().left_0().right_0().justify_center(),
            ToolbarDock::Bottom => strip.bottom(px(52.)).left_0().right_0().justify_center(),
            ToolbarDock::Left => strip.left_3().top_0().bottom_0().items_center(),
            ToolbarDock::Right => strip.right_3().top_0().bottom_0().items_center(),
        };
        strip.child(pill)
    }
}

impl Render for PipelineView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // In its own workspace the menus are in the app's menu row; docked in Explore, here
        let own_bar = (self.store.read(cx).workspace != crate::navigation::route::Workspace::Pipeline).then(|| self.menu_bar(cx));
        let inspector_open = self.inspector_open;
        let reopen_inspector = (!inspector_open).then(|| {
            div()
                .absolute()
                .top_3()
                .right_3()
                .occlude()
                .child(
                    Button::new("pipe-inspector-open")
                        .outline()
                        .small()
                        .icon(Icon::new(IconName::PanelRight))
                        .tooltip("Show Inspector")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.inspector_open = true;
                            cx.notify();
                        })),
                )
        });
        v_flex().id("pipeline-workspace").size_full().children(own_bar).child(
            h_flex()
            .flex_1()
            .min_h_0()
            .w_full()
            // Panes are cards on the chrome, PANE_GAP apart (kitchen-ui)
            .p(px(theme::PANE_GAP))
            .gap(px(theme::PANE_GAP))
            .bg(color::chrome())
            .child(div().relative().flex_1().min_w_0().h_full().rounded(px(theme::RADIUS_LG)).overflow_hidden().child(self.canvas.clone()).child(self.toolbar(cx)).children(reopen_inspector))
            .when(inspector_open, |d| {
                d.child(
                    v_flex()
                        .flex_none()
                        .w(px(INSPECTOR_W))
                        .h_full()
                        .rounded(px(theme::RADIUS_LG))
                        .overflow_hidden()
                        .bg(color::panel())
                        .child(
                            h_flex()
                                .flex_none()
                                .h(px(32.))
                                .px(px(theme::MD))
                                .items_center()
                                .border_b_1()
                                .border_color(color::border())
                                .child(div().flex_1().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child("INSPECTOR"))
                                .child(
                                    Button::new("pipe-inspector-close")
                                        .ghost()
                                        .xsmall()
                                        .icon(Icon::new(IconName::PanelRight))
                                        .tooltip("Hide Inspector")
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.inspector_open = false;
                                            cx.notify();
                                        })),
                                ),
                        )
                        .child(div().flex_1().min_h_0().child(self.inspector.clone())),
                )
            }))
    }
}

/// Shared action for both the floating Pipeline toolbar and the unselected Inspector:
/// packages the enabled GPU op chain as a shareable `.lucem-recipe.json`.
fn save_pipeline_recipe(store: &Entity<Store>, cx: &mut App) {
    let (name, steps) = {
        let s = store.read(cx);
        let img_name = s.doc.active().map_or_else(|| "pipeline".to_string(), |i| i.name.clone());
        let stem = img_name.split('.').next().unwrap_or("pipeline");
        let ops: Vec<lucem_pipeline::Op> = s
            .doc
            .graph
            .nodes
            .iter()
            .filter_map(|n| match &n.kind {
                NodeKind::Op { enabled: true, op } => Some(op.clone()),
                _ => None,
            })
            .collect();
        (format!("{stem}_recipe"), ops)
    };
    if steps.is_empty() {
        store.update(cx, |s, cx| {
            s.set_status("Add at least one enabled GPU operation to save a recipe.", cx);
        });
        return;
    }
    let recipe = crate::data::recipes::Recipe {
        schema: 1,
        name,
        family: "Recipe".to_string(),
        description: format!("Exported pipeline with {} operation(s)", steps.len()),
        steps,
    };
    #[cfg(not(target_family = "wasm"))]
    {
        match crate::data::recipes::save_recipe(&recipe) {
            Ok(path) => store.update(cx, |s, cx| {
                s.set_status(format!("Saved recipe to {}", path.display()), cx);
            }),
            Err(e) => store.update(cx, |s, cx| {
                s.set_status(format!("Could not save recipe: {e}"), cx);
            }),
        }
    }
    #[cfg(target_family = "wasm")]
    {
        if crate::data::remote::runs_alone() {
            if let Ok(json) = serde_json::to_string_pretty(&recipe) {
                cx.write_to_clipboard(gpui_kit::ClipboardItem::new_string(json));
                store.update(cx, |s, cx| {
                    s.set_status("Copied .lucem-recipe.json to clipboard.", cx);
                });
            }
        } else {
            let st = store.clone();
            cx.spawn(async move |cx| {
                match crate::data::server_api::save_server_recipe(&recipe).await {
                    Ok(path) => {
                        let _ = cx.update(|cx| {
                            st.update(cx, |s, cx| s.set_status(format!("Saved recipe to {path}"), cx));
                        });
                    }
                    Err(e) => {
                        let _ = cx.update(|cx| {
                            st.update(cx, |s, cx| s.set_status(format!("Could not save recipe: {e}"), cx));
                        });
                    }
                }
            })
            .detach();
        }
    }
}

/// The selected node: what it is, its values, its result.
pub struct Inspector {
    store: Entity<Store>,
    canvas: Entity<GraphCanvas>,
    params: Entity<ParamEditor>,
    scrub: Entity<ScrubState>,
    name: Entity<InputState>,
    name_for: String,
    /// The code of the selected script node: its file, the editor, the text on disk, and the
    /// error last marked in it.
    code: Option<ScriptCode>,
    _subs: Vec<Subscription>,
}

struct ScriptCode {
    path: String,
    view: Entity<CodeView>,
    saved: String,
    error: Option<String>,
    dirty: bool,
    _sub: Subscription,
}

impl Inspector {
    fn new(store: Entity<Store>, canvas: Entity<GraphCanvas>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let name = cx.new(|cx| InputState::new(window, cx));
        let scrub = cx.new(|cx| ScrubState::new(window, cx));
        let subs = vec![
            cx.observe(&scrub, |_, _, cx| cx.notify()),
            cx.subscribe(&canvas, |_, _, _: &CanvasEvent, cx| cx.notify()),
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::Document(_) | AppEvent::Results) {
                    cx.notify();
                }
            }),
            cx.subscribe(&name, |this, input, e: &InputEvent, cx| {
                if matches!(e, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                    let text = input.read(cx).value().trim().to_string();
                    let id = this.name_for.clone();
                    if text.is_empty() || id.is_empty() {
                        return;
                    }
                    let style = match this.store.read(cx).doc.graph.node(&id).map(|n| &n.kind) {
                        Some(NodeKind::View { style, name }) if *name != text => style.clone(),
                        _ => return,
                    };
                    this.store.update(cx, |s, cx| {
                        let _ = s.execute(Command::SetView { id, name: text, style }, cx);
                    });
                }
            }),
        ];
        Self { store, canvas, params: cx.new(|_| ParamEditor::new()), scrub, name, name_for: String::new(), code: None, _subs: subs }
    }

    fn exec(store: &Entity<Store>, cmd: Command, cx: &mut App) {
        store.update(cx, |s, cx| {
            let _ = s.execute(cmd, cx);
        });
    }

    /// The editor of the script at `path` (made when another script is selected), with the errors
    /// of its last run marked on their lines.
    fn script_code(&mut self, id: &str, path: &str, window: &mut Window, cx: &mut Context<Self>) {
        if self.code.as_ref().is_none_or(|c| c.path != path) {
            #[cfg(not(target_family = "wasm"))]
            let text = std::fs::read_to_string(path).unwrap_or_default();
            #[cfg(target_family = "wasm")]
            let text = String::from("# Loading script from server…\n");
            let ext = std::path::Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or_default();
            let view = cx.new(|cx| CodeView::new(CodeLanguage::from_name(ext), text.clone(), window, cx).editable(true));
            let sub = cx.subscribe(&view, |this, _, e: &CodeChanged, cx| {
                if let Some(c) = &mut this.code {
                    c.dirty = e.0 != c.saved;
                }
                cx.notify();
            });
            self.code = Some(ScriptCode { path: path.to_string(), view, saved: text, error: None, dirty: false, _sub: sub });
            #[cfg(target_family = "wasm")]
            if !crate::data::remote::runs_alone() {
                let p = path.to_string();
                cx.spawn_in(window, async move |this, cx| {
                    if let Ok(content) = crate::data::server_api::read_server_script(&p).await {
                        let _ = this.update_in(cx, |insp, window, cx| {
                            if let Some(c) = &mut insp.code
                                && c.path == p
                            {
                                c.saved = content.clone();
                                c.dirty = false;
                                c.view.update(cx, |v, cx| v.set_text(content, window, cx));
                                cx.notify();
                            }
                        });
                    }
                })
                .detach();
            }
        }
        let error = self.store.read(cx).analysis.as_ref().and_then(|a| a.errors.get(id).cloned());
        if let Some(c) = &mut self.code
            && c.error != error
        {
            let file = std::path::Path::new(path).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
            let marks = error.as_deref().map(|e| crate::data::script::error_lines(&file, e)).unwrap_or_default();
            c.error = error;
            c.view.update(cx, |v, cx| v.set_errors(marks, cx));
        }
    }

    fn body(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let selected = self.canvas.read(cx).selected.clone();
        // The selected script node's code, made ready before the store is read below
        let script = (selected.len() == 1)
            .then(|| selected.iter().next().cloned())
            .flatten()
            .and_then(|id| match self.store.read(cx).doc.graph.node(&id).map(|n| n.kind.clone()) {
                Some(NodeKind::Script { path, .. }) => Some((id, path)),
                _ => None,
            });
        match &script {
            Some((id, path)) => self.script_code(id, path, window, cx),
            None => self.code = None,
        }
        let s = self.store.read(cx);
        let doc = s.doc.clone();
        let g = &doc.graph;
        if selected.len() > 1 {
            let canvas = self.canvas.downgrade();
            return Section::new(format!("{} nodes selected", selected.len()))
                .child(div().text_xs().text_color(color::text_muted()).child("Drag one to move them together; Ctrl+D duplicates them with their edges."))
                .child(Button::new("inspector-delete").outline().small().label("Delete them").on_click(move |_, _, cx| {
                    let _ = canvas.update(cx, |c, cx| c.delete_selected(true, cx));
                }))
                .into_any_element();
        }
        let Some(node) = selected.iter().next().and_then(|id| g.node(id)).cloned() else {
            let active_name = doc.active().map_or("No image open".to_string(), |i| i.name.clone());
            let (n_nodes, n_edges) = (g.nodes.len(), g.edges.len());
            let canvas_add = self.canvas.downgrade();
            let st_add = self.store.clone();
            let store_run = self.store.clone();
            let add_btn = Button::new("inspector-add-op")
                .outline()
                .small()
                .icon(Icon::new(IconName::Plus))
                .label("Add operation")
                .dropdown_menu(move |mut m: PopupMenu, w, cx| {
                    for cat in operations() {
                        let (canvas, st_cat) = (canvas_add.clone(), st_add.clone());
                        let icon = cat.icon;
                        let items: Vec<_> = cat.items.into_iter().filter_map(|it| match it.template.clone() {
                            Template::Op(op) => Some((it.label, it.template, op)),
                            _ => None,
                        }).collect();
                        if items.is_empty() {
                            continue;
                        }
                        m = m.submenu(cat.name, w, cx, move |mut sub, _, _| {
                            for (label, template, op) in &items {
                                let (canvas, template, op_hov, st_item) = (canvas.clone(), template.clone(), op.clone(), st_cat.clone());
                                sub = sub.item(crate::ui::op_gallery::op_menu_item(
                                    format!("insp-op-{}", op_hov.tag()),
                                    label.clone(),
                                    icon,
                                    op_hov,
                                    st_item,
                                    move |_, cx| {
                                        let _ = canvas.update(cx, |c, cx| c.add(&template, None, cx));
                                    },
                                ));
                            }
                            sub
                        });
                    }
                    m
                });
            let run_btn = Button::new("inspector-run")
                .primary()
                .small()
                .icon(Icon::new(IconName::Play))
                .label("Run on view")
                .on_click(move |_, _, cx| store_run.update(cx, |s, cx| s.recompute(cx)));
            let store_recipe = self.store.clone();
            let recipe_btn = Button::new("inspector-make-recipe")
                .outline()
                .small()
                .icon(Icon::new(IconName::Package))
                .label("Make recipe")
                .tooltip("Save this pipeline as a shareable .lucem-recipe.json")
                .on_click(move |_, _, cx| save_pipeline_recipe(&store_recipe, cx));
            let quick_ops: Vec<AnyElement> = ["rolling_ball", "gaussian_blur", "clahe", "otsu_threshold", "li_threshold", "canny"]
                .iter()
                .filter_map(|tag| lucem_pipeline::Op::by_tag(tag))
                .map(|op| {
                    let canvas = self.canvas.downgrade();
                    let st_hov = self.store.clone();
                    let st_clear = self.store.clone();
                    let title = op.title();
                    let op_hov = op.clone();
                    h_flex()
                        .id(SharedString::from(format!("pipe-quick-{}", op.tag())))
                        .px(px(10.))
                        .h(px(26.))
                        .gap(px(5.))
                        .items_center()
                        .rounded(px(999.))
                        .border_1()
                        .border_color(color::border())
                        .bg(color::panel())
                        .hover(|d| d.bg(color::raised()))
                        .cursor_pointer()
                        .text_xs()
                        .text_color(color::text())
                        .child(Icon::new(IconName::Plus).xsmall().text_color(color::accent_mark()))
                        .child(title)
                        .on_hover(move |enter, _, cx| {
                            let op = (*enter).then(|| op_hov.clone());
                            st_hov.update(cx, |s, cx| s.set_hovered_op(op, cx));
                        })
                        .on_click(move |_, _, cx| {
                            st_clear.update(cx, |s, cx| s.set_hovered_op(None, cx));
                            let template = Template::Op(op.clone());
                            let _ = canvas.update(cx, |c, cx| c.add(&template, None, cx));
                        })
                        .into_any_element()
                })
                .collect();
            return v_flex()
                .gap(px(theme::MD))
                .child(
                    Section::new("Pipeline")
                        .child(row("Image", div().min_w_0().truncate().text_sm().child(active_name)))
                        .child(row("Graph", div().text_sm().font_family("monospace").child(format!("{n_nodes} nodes · {n_edges} edges")))),
                )
                .child(h_flex().gap(px(theme::SM)).flex_wrap().child(add_btn).child(run_btn).child(recipe_btn))
                .child(
                    v_flex()
                        .p(px(theme::SM))
                        .gap(px(theme::SM))
                        .rounded(px(12.))
                        .bg(color::control())
                        .child(
                            h_flex()
                                .gap(px(6.))
                                .items_center()
                                .child(Icon::new(IconName::Sparkles).xsmall().text_color(color::accent_mark()))
                                .child(div().text_xs().font_weight(FontWeight::MEDIUM).text_color(color::text()).child("Quick ops — hover to preview")),
                        )
                        .child(h_flex().flex_wrap().gap(px(6.)).children(quick_ops)),
                )
                .child(div().text_xs().text_color(color::text_muted()).child("Select a node to edit its parameters, or drag an output handle onto an input."))
                .into_any_element();
        };
        let id = node.id.clone();
        let (title, sub, icon) = describe(&node.kind, &doc);
        let analysis = s.analysis.clone();
        let layer = s.results.get(&id).cloned();
        let preview = analysis
            .as_ref()
            .and_then(|a| a.previews.get(&id).cloned())
            .or_else(|| layer.as_ref().and_then(|r| r.layer.clone()))
            .or_else(|| {
                g.outputs(&id)
                    .iter()
                    .find_map(|(out_id, _)| s.results.get(*out_id).and_then(|r| r.layer.clone()))
            });
        let header = v_flex()
            .gap(px(2.))
            .child(h_flex().gap(px(theme::SM)).child(Icon::new(icon).small()).child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(title)))
            .child(div().text_xs().font_family("monospace").text_color(color::text_muted()).truncate().child(sub))
            .child(div().text_xs().text_color(color::text_muted()).child(id.clone()));
        // Its inputs and what feeds them
        let inputs = node.kind.inputs();
        let ports = (!inputs.is_empty()).then(|| {
            let mut sec = Section::new("Inputs");
            for (i, (t, name)) in inputs.iter().enumerate() {
                let from = g.input(&id, i).map(str::to_string);
                let (store, id2) = (self.store.clone(), id.clone());
                sec = sec.child(
                    h_flex()
                        .gap(px(theme::SM))
                        .child(div().size(px(8.)).rounded_full().bg(port_color(*t)))
                        .child(div().w(px(80.)).text_sm().child(name.to_string()))
                        .child(div().flex_1().min_w_0().truncate().text_sm().text_color(if from.is_some() { color::text() } else { color::warning() }).child(from.clone().unwrap_or_else(|| "not connected".into())))
                        .when(from.is_some(), |d| {
                            d.child(Button::new(SharedString::from(format!("unlink-{i}"))).ghost().xsmall().icon(Icon::new(IconName::Unlink)).tooltip("Disconnect").on_click(move |_, _, cx| Self::exec(&store, Command::Disconnect { to: id2.clone(), port: i }, cx)))
                        }),
                );
            }
            sec
        });
        let body: AnyElement = match &node.kind {
            NodeKind::Op { enabled, op } => {
                let commit: Commit = {
                    let (store, id) = (self.store.clone(), id.clone());
                    Rc::new(move |op, cx| store.update(cx, |s, cx| s.execute(Command::SetOp { id: id.clone(), op }, cx)).map(|_| ()).map_err(|e| e.to_string()))
                };
                self.params.update(cx, |p, cx| p.set_op(&id, op, commit, window, cx));
                let (store, id2, on) = (self.store.clone(), id.clone(), *enabled);
                let docs_url = op.docs_url();
                let desc = op.description();
                let canvas_next = self.canvas.downgrade();
                let next_prompt = op.next_step_prompt();
                let next_chips: Vec<AnyElement> = op
                    .next_steps()
                    .iter()
                    .filter_map(|tag| lucem_pipeline::Op::by_tag(tag))
                    .take(3)
                    .map(|next_op| {
                        let canvas = canvas_next.clone();
                        let st_hov = self.store.clone();
                        let st_clear = self.store.clone();
                        let title = next_op.title();
                        let op_hov = next_op.clone();
                        h_flex()
                            .id(SharedString::from(format!("insp-next-{}", next_op.tag())))
                            .px(px(10.))
                            .h(px(26.))
                            .gap(px(5.))
                            .items_center()
                            .rounded(px(999.))
                            .border_1()
                            .border_color(color::border())
                            .bg(color::panel())
                            .hover(|d| d.bg(color::raised()))
                            .cursor_pointer()
                            .text_xs()
                            .text_color(color::text())
                            .child(Icon::new(IconName::Plus).xsmall().text_color(color::accent_mark()))
                            .child(title)
                            .on_hover(move |enter, _, cx| {
                                let op = (*enter).then(|| op_hov.clone());
                                st_hov.update(cx, |s, cx| s.set_hovered_op(op, cx));
                            })
                            .on_click(move |_, _, cx| {
                                st_clear.update(cx, |s, cx| s.set_hovered_op(None, cx));
                                let template = Template::Op(next_op.clone());
                                let _ = canvas.update(cx, |c, cx| c.add(&template, None, cx));
                            })
                            .into_any_element()
                    })
                    .collect();
                v_flex()
                    .gap(px(theme::MD))
                    .child(div().text_xs().text_color(color::text_muted()).child(desc))
                    .child(Checkbox::new("inspector-enabled").checked(on).label("On (off: passes its input through)").on_click(move |v: &bool, _, cx| Self::exec(&store, Command::EnableNode { id: id2.clone(), enabled: *v }, cx)))
                    .child(Section::new("Parameters").child(self.params.clone()))
                    .when(!next_chips.is_empty(), |d| {
                        d.child(
                            v_flex()
                                .p(px(theme::SM))
                                .gap(px(theme::SM))
                                .rounded(px(12.))
                                .bg(color::control())
                                .child(
                                    h_flex()
                                        .gap(px(6.))
                                        .items_center()
                                        .child(Icon::new(IconName::Sparkles).xsmall().text_color(color::accent_mark()))
                                        .child(div().text_xs().font_weight(FontWeight::MEDIUM).text_color(color::text()).child(next_prompt)),
                                )
                                .child(h_flex().flex_wrap().gap(px(6.)).children(next_chips))
                                .child(div().text_size(px(10.)).text_color(color::text_muted()).child("Hover any chip to preview Before ↔ After")),
                        )
                    })
                    .child(
                        h_flex()
                            .items_center()
                            .gap(px(theme::SM))
                            .child(div().flex_1().text_xs().text_color(color::text_muted()).child(format!("{} · margin {} px", super::catalog::signature(op), op.halo())))
                            .child(
                                Button::new("inspector-docs")
                                    .ghost()
                                    .xsmall()
                                    .icon(Icon::new(IconName::ExternalLink))
                                    .label("Docs")
                                    .tooltip(format!("Open GPU kernel source ({})", op.source_rel_path()))
                                    .on_click(move |_, _, cx| cx.open_url(&docs_url)),
                            ),
                    )
                    .into_any_element()
            }
            NodeKind::Source { image, channel } => {
                let img = doc.image(image);
                let names: Vec<SharedString> = img.map(|i| i.channels.iter().map(|c| SharedString::from(c.label.clone())).collect()).unwrap_or_default();
                let store = self.store.clone();
                let id2 = id.clone();
                v_flex()
                    .gap(px(theme::SM))
                    .child(row("Image", div().text_sm().truncate().child(img.map_or(image.clone(), |i| i.name.clone()))))
                    .child(row("Channel", MenuSelect::new("inspector-channel", names, Some(*channel), move |i, _, cx| Self::exec(&store, Command::SetSourceChannel { id: id2.clone(), channel: i }, cx))))
                    .child(div().text_xs().text_color(color::text_muted()).child("Values scaled to 0–1 by the type's range (16-bit: ÷ 65535)."))
                    .into_any_element()
            }
            NodeKind::Annotations { image, class } => {
                let mut names: Vec<SharedString> = vec!["All classes".into()];
                names.extend(doc.classes.iter().map(|c| SharedString::from(c.name.clone())));
                let raw: Vec<String> = doc.classes.iter().map(|c| c.name.clone()).collect();
                let current = class.as_ref().and_then(|c| raw.iter().position(|r| r == c)).map_or(0, |i| i + 1);
                let (store, id2, img, x, y) = (self.store.clone(), id.clone(), image.clone(), node.x, node.y);
                v_flex()
                    .gap(px(theme::SM))
                    .child(row(
                        "Class",
                        MenuSelect::new("inspector-class", names, Some(current), move |i, _, cx| {
                            // Swap the node for one of the chosen class, keeping its edges
                            let class = if i == 0 { None } else { Some(raw[i - 1].clone()) };
                            let kind = NodeKind::Annotations { image: img.clone(), class };
                            let id = id2.clone();
                            store.update(cx, |s, cx| {
                                let outs: Vec<(String, usize)> = s.doc.graph.outputs(&id).into_iter().map(|(o, p)| (o.to_string(), p)).collect();
                                let fresh = s.doc.graph.fresh_id("annotations");
                                if s.execute(Command::AddNode { kind, x, y, id: Some(fresh.clone()), after: None }, cx).is_ok() {
                                    for (o, p) in outs {
                                        let _ = s.execute(Command::Connect { from: fresh.clone(), to: o, port: p }, cx);
                                    }
                                    let _ = s.execute(Command::RemoveNodes { nodes: vec![id] }, cx);
                                }
                            });
                        }),
                    ))
                    .child(div().text_xs().text_color(color::text_muted()).child("Regions and brush strokes fill the mask; eraser strokes clear it."))
                    .into_any_element()
            }
            NodeKind::View { name, style } => {
                if self.name_for != id {
                    self.name_for = id.clone();
                    let name = name.clone();
                    self.name.update(cx, |i, cx| i.set_value(name, window, cx));
                }
                let set = {
                    let (store, id, name, style) = (self.store.clone(), id.clone(), name.clone(), style.clone());
                    Rc::new(move |f: &dyn Fn(&mut ViewStyle), cx: &mut App| {
                        let mut st = style.clone();
                        f(&mut st);
                        Self::exec(&store, Command::SetView { id: id.clone(), name: name.clone(), style: st }, cx);
                    })
                };
                let (s1, s2, s3) = (set.clone(), set.clone(), set);
                let lut_idx = COLORMAPS.iter().position(|m| *m == style.colormap);
                v_flex()
                    .gap(px(theme::SM))
                    .child(row("Name", div().w_full().child(Input::new(&self.name).small())))
                    .child(Checkbox::new("view-visible").checked(style.visible).label("Drawn over the image").on_click(move |v: &bool, _, cx| {
                        let v = *v;
                        s1(&move |st: &mut ViewStyle| st.visible = v, cx)
                    }))
                    .child(row("Opacity", {
                        let s2_set = s2.clone();
                        let pct = style.opacity * 100.0;
                        scrub_number(
                            &self.scrub,
                            "view-opacity",
                            format!("{pct:.0} %"),
                            format!("{pct:.0}"),
                            "Drag to scrub opacity, or click to type a percentage",
                            move |d, _, cx| s2(&move |st: &mut ViewStyle| st.opacity = (st.opacity + d * 0.05).clamp(0.0, 1.0), cx),
                            move |v, _, cx| {
                                let clamped = (v as f32 / 100.0).clamp(0.0, 1.0);
                                s2_set(&move |st: &mut ViewStyle| st.opacity = clamped, cx);
                            },
                            cx,
                        )
                    }))
                    .child(row(
                        "Colours",
                        MenuSelect::new("view-lut", COLORMAPS.iter().map(|m| SharedString::from(*m)).collect(), lut_idx, move |i, _, cx| s3(&move |st: &mut ViewStyle| st.colormap = COLORMAPS[i].to_string(), cx)),
                    ))
                    .into_any_element()
            }
            NodeKind::Model { image, channels, model, roi, prompts, enabled } => {
                let info = crate::data::ml::find_model(model).map(|(_, m)| m);
                let interactive = info.as_ref().is_some_and(|m| m.task == lucem_ml::Task::InteractiveSegmentation);
                let (store, id2, kind) = (self.store.clone(), id.clone(), node.kind.clone());
                let mut col = v_flex()
                    .gap(px(theme::MD))
                    .child(Checkbox::new("inspector-model-on").checked(*enabled).label("On").on_click(move |v: &bool, _, cx| {
                        let mut k = kind.clone();
                        if let NodeKind::Model { enabled, .. } = &mut k {
                            *enabled = *v;
                        }
                        Self::exec(&store, Command::SetScript { id: id2.clone(), kind: k }, cx)
                    }))
                    .child(row("Model", div().text_sm().child(info.as_ref().map_or_else(|| format!("{model} (no plugin offers it)"), |m| m.name.clone()))));
                // Runs only when asked: over the view shown in Explore (or its region of interest)
                {
                    let (st, id_run) = (self.store.clone(), id.clone());
                    let has_run = { let sig = format!("{id}|{}", serde_json::to_string(&Some(&node.kind)).unwrap_or_default()); crate::data::model_node::last_run(&sig).is_some() };
                    col = col.child(
                        v_flex()
                            .gap(px(theme::XS))
                            .child(Button::new("inspector-model-run").primary().small().icon(Icon::new(IconName::Play)).label(if has_run { "Run again" } else { "Run" }).on_click(move |_, _, cx| {
                                crate::data::model_node::arm(&id_run);
                                st.update(cx, |s, cx| {
                                    // Its labels need a view to be drawn (and run): one is added when there is none
                                    let has_view = s.doc.graph.outputs(&id_run).iter().any(|(to, _)| matches!(s.doc.graph.node(to).map(|n| &n.kind), Some(NodeKind::View { .. })));
                                    if !has_view {
                                        let (x, y) = s.doc.graph.node(&id_run).map_or((0.0, 0.0), |n| (n.x + 240.0, n.y));
                                        let kind = NodeKind::View { name: "Nuclei".into(), style: crate::domain::graph::ViewStyle::default() };
                                        let _ = s.execute(Command::AddNode { kind, x, y, id: None, after: Some(id_run.clone()) }, cx);
                                    }
                                    s.set_status("Running the model over the view…", cx);
                                    s.recompute(cx);
                                });
                            }))
                            .child(div().text_xs().text_color(color::text_muted()).child(if has_run { "Shown from its last run; changing a setting needs Run again." } else { "Not run yet: runs over the view in Explore, or inside its region." })),
                    );
                }
                // What the model sees: the colour image, or one channel
                let names: Vec<String> = doc.image(image).map(|i| i.channels.iter().map(|c| c.label.clone()).collect()).unwrap_or_default();
                let mut options: Vec<(SharedString, Vec<usize>)> = Vec::new();
                if names.len() == 3 {
                    options.push(("Colour (all three channels)".into(), Vec::new()));
                }
                let processed_map = doc.graph.processed_channels(image, names.len());
                options.extend(names.iter().enumerate().map(|(i, n)| {
                    let label = if processed_map.contains_key(&i) {
                        format!("Channel {}: {n} (preprocessed)", i + 1)
                    } else {
                        format!("Channel {}: {n}", i + 1)
                    };
                    (SharedString::from(label), vec![i])
                }));
                let current = options.iter().position(|(_, c)| c == channels).or_else(|| channels.is_empty().then_some(0));
                let (store_c, id_c, kind_c) = (self.store.clone(), id.clone(), node.kind.clone());
                let picks: Vec<Vec<usize>> = options.iter().map(|(_, c)| c.clone()).collect();
                // Where it runs: the whole region asked, a class' regions, or one annotation
                {
                    use crate::domain::graph::Roi;
                    let regions: Vec<&crate::domain::annotation::Annotation> = doc.annotations_of(image).filter(|a| a.shape.is_region()).collect();
                    let classes: Vec<String> = regions.iter().map(|a| a.class.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
                    let mut opts: Vec<(SharedString, Option<Roi>)> = vec![("Whole image".into(), None)];
                    opts.extend(classes.iter().map(|c| (SharedString::from(format!("Every {c} region")), Some(Roi::Class { class: c.clone() }))));
                    opts.extend(regions.iter().take(60).map(|a| {
                        let name = if a.note.is_empty() { format!("{} {} ({})", a.shape.kind(), a.id, a.class) } else { format!("{} ({})", a.note.lines().next().unwrap_or_default(), a.class) };
                        (SharedString::from(name), Some(Roi::Annotation { id: a.id.clone() }))
                    }));
                    let cur = opts.iter().position(|(_, r)| r == roi);
                    let (store_r, id_r, kind_r) = (self.store.clone(), id.clone(), node.kind.clone());
                    let picks: Vec<Option<Roi>> = opts.iter().map(|(_, r)| r.clone()).collect();
                    col = col.child(row(
                        "Region",
                        MenuSelect::new("inspector-model-roi", opts.iter().map(|(l, _)| l.clone()).collect(), cur, move |i, _, cx| {
                            let mut k = kind_r.clone();
                            if let NodeKind::Model { roi, .. } = &mut k {
                                *roi = picks[i].clone();
                            }
                            Self::exec(&store_r, Command::SetScript { id: id_r.clone(), kind: k }, cx)
                        }),
                    ));
                }
                col = col.child(row(
                    "Reads",
                    MenuSelect::new("inspector-model-channels", options.iter().map(|(l, _)| l.clone()).collect(), current, move |i, _, cx| {
                        let mut k = kind_c.clone();
                        if let NodeKind::Model { channels, .. } = &mut k {
                            *channels = picks[i].clone();
                        }
                        Self::exec(&store_c, Command::SetScript { id: id_c.clone(), kind: k }, cx)
                    }),
                ));
                if interactive {
                    let classes: Vec<String> = doc.classes.iter().map(|c| c.name.clone()).collect();
                    let current = prompts.as_ref().and_then(|p| classes.iter().position(|c| c == p));
                    let (store, id3, kind) = (self.store.clone(), id.clone(), node.kind.clone());
                    let names = classes.clone();
                    col = col.child(row(
                        "Prompts",
                        MenuSelect::new("inspector-model-prompts", classes.iter().map(|c| SharedString::from(c.clone())).collect(), current, move |i, _, cx| {
                            let mut k = kind.clone();
                            if let NodeKind::Model { prompts, .. } = &mut k {
                                *prompts = Some(names[i].clone());
                            }
                            Self::exec(&store, Command::SetScript { id: id3.clone(), kind: k }, cx)
                        }),
                    ));
                }
                col.child(div().text_xs().text_color(color::text_muted()).child(if interactive {
                    "One object per box or point annotation of the prompt class on the plane shown. Reads the image itself · gives labels."
                } else {
                    "Every nucleus of the region. Reads the image itself · gives labels. Answers are kept, so moving back over a region does not run it again."
                }))
                .into_any_element()
            }
            NodeKind::Script { path, inputs, output, params, enabled } => {
                let p = std::path::PathBuf::from(path);
                let lang = crate::data::script::Language::of(&p).map_or("?", |l| l.title());
                let (store, id2, on) = (self.store.clone(), id.clone(), *enabled);
                let kind = node.kind.clone();
                let (open_p, st_open, st_reload, id_reload) = (p.clone(), self.store.clone(), self.store.clone(), id.clone());
                let mut nums = v_flex().gap(px(theme::XS));
                for (i, q) in params.iter().enumerate() {
                    let (store, id, kind) = (self.store.clone(), id.clone(), kind.clone());
                    let (store_set, id_set, kind_set) = (self.store.clone(), id.clone(), kind.clone());
                    // A step of 1 % of the value (0.01 near zero); drag or click to type
                    let step = (q.value.abs() * 0.01).max(0.01);
                    let display = crate::ui::param_editor::trim_number(q.value);
                    nums = nums.child(row(
                        q.name.clone(),
                        scrub_number(
                            &self.scrub,
                            format!("script-param-{i}"),
                            display.clone(),
                            display,
                            "Drag horizontally to scrub, or click to type an exact value",
                            move |d, _, cx| {
                                let mut k = kind.clone();
                                if let NodeKind::Script { params, .. } = &mut k
                                    && let Some(p) = params.get_mut(i)
                                {
                                    p.value += d as f64 * step;
                                }
                                Self::exec(&store, Command::SetScript { id: id.clone(), kind: k }, cx);
                            },
                            move |v, _, cx| {
                                let mut k = kind_set.clone();
                                if let NodeKind::Script { params, .. } = &mut k
                                    && let Some(p) = params.get_mut(i)
                                {
                                    p.value = v;
                                }
                                Self::exec(&store_set, Command::SetScript { id: id_set.clone(), kind: k }, cx);
                            },
                            cx,
                        ),
                    ));
                }
                let declared = inputs.iter().map(|q| format!("{}: {}", q.name, q.port.name())).collect::<Vec<_>>().join(", ");
                let kind_on = kind.clone();
                let path_code = path.clone();
                let st_code = self.store.clone();
                v_flex()
                    .gap(px(theme::MD))
                    .child(Checkbox::new("inspector-script-on").checked(on).label("On (off: passes its first input through)").on_click(move |v: &bool, _, cx| {
                        let mut k = kind_on.clone();
                        if let NodeKind::Script { enabled, .. } = &mut k {
                            *enabled = *v;
                        }
                        Self::exec(&store, Command::SetScript { id: id2.clone(), kind: k }, cx)
                    }))
                    .when(!params.is_empty(), |d| d.child(Section::new("Parameters").child(nums)))
                    .when(params.is_empty(), |d| {
                        d.child(div().text_xs().text_color(color::text_muted()).child("No parameters declared yet. Add `@lucem.param(\"sigma\", 3.0)` or `# lucem: param sigma = 3.0` in the Code tab."))
                    })
                    .child(row("Language", div().text_sm().child(lang)))
                    .child(row("File", div().text_xs().font_family("monospace").truncate().child(path.clone())))
                    .child(
                        h_flex()
                            .gap(px(theme::SM))
                            .flex_wrap()
                            .child(
                                Button::new("script-open-code")
                                    .outline()
                                    .small()
                                    .icon(Icon::new(IconName::Code))
                                    .label("Open in Code tab")
                                    .on_click(move |_, _, cx| {
                                        let p = path_code.clone();
                                        st_code.update(cx, |s, cx| {
                                            s.set_workspace(crate::navigation::route::Workspace::Code, cx);
                                            cx.emit(AppEvent::OpenScriptInCode(p));
                                        });
                                    }),
                            )
                            .child(Button::new("script-reload").small().ghost().icon(Icon::new(IconName::RefreshCw)).label("Reload").tooltip("Read the header again and run").on_click(move |_, _, cx| st_reload.update(cx, |s, cx| s.reload_script(&id_reload, cx))))
                            .when(cfg!(not(target_family = "wasm")), |h| {
                                h.child(Button::new("script-open").small().ghost().icon(Icon::new(IconName::SquarePen)).label("External editor").on_click(move |_, _, cx| {
                                    if let Err(e) = crate::data::script::open_in_editor(&open_p) {
                                        st_open.update(cx, |s, cx| s.set_status(format!("Could not open the editor: {e}"), cx));
                                    }
                                }))
                            }),
                    )
                    .child(div().text_xs().text_color(color::text_muted()).child(format!("Takes {} · gives {}. Declared with `@lucem.param` or `# lucem:`.", if declared.is_empty() { "nothing".to_string() } else { declared }, output.name())))
                    .into_any_element()
            }
        };
        let status = {
            let mut sec = Section::new("Result");
            if let Some(a) = &analysis {
                if let Some(e) = a.errors.get(&id) {
                    sec = sec.child(div().text_xs().text_color(color::danger()).child(e.clone()));
                }
                if let Some(t) = a.thresholds.get(&id) {
                    sec = sec.child(div().text_xs().text_color(color::text_muted()).child(format!("Otsu on the whole image: {t:.4} (of 0–1)")));
                }
                if let Some(t) = a.tables.get(&id) {
                    sec = sec.child(div().text_xs().text_color(color::text_muted()).child(format!("{} objects on the whole image (level {}) · see Analyze", t.rows.len(), a.level)));
                }
            }
            if let Some(p) = preview {
                sec = sec.child(div().w_full().h(px(200.)).rounded(px(6.)).overflow_hidden().bg(color::canvas()).child(gpui_kit::img(p).size_full().object_fit(gpui_kit::ObjectFit::Contain)));
            }
            if let Some(r) = layer {
                let line = match &r.error {
                    Some(e) => e.clone(),
                    None => format!("In view: level {} · {:.0} ms", r.level, r.ms),
                };
                sec = sec.child(div().text_xs().text_color(if r.error.is_some() { color::danger() } else { color::text_muted() }).child(line));
            }
            Some(sec)
        };
        v_flex().gap(px(theme::XL)).child(header).child(body).children(ports).children(status).into_any_element()
    }
}

impl Render for Inspector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body = self.body(window, cx);
        v_flex()
            .id("pipeline-inspector")
            .size_full()
            .child(div().id("inspector-body").flex_1().min_h_0().overflow_y_scroll().p(px(theme::MD)).child(body))
    }
}

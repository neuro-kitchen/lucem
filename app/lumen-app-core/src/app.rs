//! The root view: one title bar (menus, the three workspaces, the project), the workspace shown,
//! one status bar. Without an image the body is the home view.

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _, TitleBar};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, ExternalPaths, FocusHandle, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, PathPromptOptions, Render,
    SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::data::store::{AppEvent, Arrangement, Store, ViewKind, ViewRequest};
use crate::domain::command::Command;
use crate::navigation::actions::*;
use crate::navigation::route::Workspace;
use crate::domain::tool::Tool;
use crate::screens::analyze::AnalyzeView;
use crate::screens::explore::ExploreView;
use crate::screens::palette::{Command as PaletteCommand, Palette, PaletteEvent};
use crate::screens::pipeline::PipelineView;
use crate::screens::pipeline::catalog::{operations, Template};
use crate::ui::theme::{self, color};
use gpui_kit::assets::IconName;

/// The bottom bar: the last message, the focused view's position and values, its level and zoom
/// (draggable), Fit and 1:1, the cache and the device. Its own entity, so the pointer moving over
/// the image redraws only this bar.
pub struct StatusBar {
    store: Entity<Store>,
    explore: Entity<ExploreView>,
    /// The canvas followed, and the subscription that follows it.
    watched: Option<(gpui_kit::EntityId, Subscription)>,
    _subs: Vec<Subscription>,
}

impl StatusBar {
    fn new(store: Entity<Store>, explore: Entity<ExploreView>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify()), cx.observe(&explore, |_, _, cx| cx.notify())];
        Self { store, explore, watched: None, _subs: subs }
    }
}

impl Render for StatusBar {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rounded = frame_rounded(window);
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
        let busy = s.ml.busy.as_ref().map(|(what, f)| format!("{what}{}", f.map_or(String::new(), |f| format!(" {:.0} %", f * 100.0))));
        h_flex()
            .flex_none()
            .h(px(theme::STATUS_BAR_H))
            .px(px(theme::MD))
            .gap(px(theme::LG))
            .items_center()
            .bg(color::chrome())
            .when(rounded, |d| d.rounded_b(px(WINDOW_RADIUS - 1.0)))
            .border_t_1()
            .border_color(color::border())
            .text_xs()
            .child(div().flex_none().max_w(px(460.)).truncate().child(s.status.clone()))
            .when(explore, |d| {
                d.child(div().flex_1().min_w_0().truncate().font_family("monospace").text_color(color::text_muted()).child([st.position.clone(), st.values.clone()].into_iter().flatten().collect::<Vec<_>>().join("   ")))
            })
            .when(!explore, |d| d.child(div().flex_1()))
            .children(busy.map(|b| div().text_color(color::accent_hover()).child(b)))
            .when(s.computing, |d| d.child(div().text_color(color::warning()).child(crate::data::model_node::progress().unwrap_or_else(|| "computing results…".to_string()))))
            .when(explore, |d| {
                d.children(st.level.clone().map(|l| div().font_family("monospace").text_color(color::text_muted()).child(l)))
                    .child(div().font_family("monospace").child(crate::ui::widgets::scrub_label("status-zoom", format!("{:.1} %", st.zoom * 100.0), move |d, _, cx| c1.update(cx, |c, cx| c.zoom_at(if d > 0.0 { 1.05 } else { 1.0 / 1.05 }, None, cx)))))
                    .child(Button::new("status-fit").ghost().xsmall().label("Fit").tooltip("Fit the image (F · Ctrl+0)").on_click(move |_, _, cx| c2.update(cx, |c, cx| c.fit(cx))))
                    .child(Button::new("status-100").ghost().xsmall().label("1:1").tooltip("Actual pixels (Ctrl+1)").on_click(move |_, _, cx| c3.update(cx, |c, cx| c.actual_size(cx))))
                    .when(st.cached_mb > 0, |d| d.child(div().text_color(color::text_muted()).child(format!("{} MB cached", st.cached_mb))))
            })
            .child(div().text_color(color::text_muted()).child(device))
    }
}

pub struct LumenApp {
    store: Entity<Store>,
    pub explore: Entity<ExploreView>,
    pub pipeline: Entity<PipelineView>,
    analyze: Entity<AnalyzeView>,
    status: Entity<StatusBar>,
    palette: Option<(Entity<Palette>, Subscription)>,
    show_help: bool,
    settings: Option<Entity<crate::screens::settings::SettingsView>>,
    focus: FocusHandle,
    _subs: Vec<Subscription>,
}

impl LumenApp {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        theme::configure(&store.read(cx).prefs, window);
        theme::apply(window, cx);
        let pipeline = cx.new(|cx| PipelineView::new(store.clone(), window, cx));
        // Explore can dock the same Pipeline under its views
        let explore = cx.new(|cx| ExploreView::new(store.clone(), pipeline.clone(), window, cx));
        let analyze = cx.new(|cx| AnalyzeView::new(store.clone(), window, cx));
        // Model plugins are looked for once, in the background
        store.update(cx, |s, cx| s.scan_plugins(cx));
        let status = cx.new(|cx| StatusBar::new(store.clone(), explore.clone(), cx));
        let subs = vec![
            cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify()),
            // A theme that follows the system follows it while running
            cx.observe_window_appearance(window, |this, window, cx| {
                let prefs = this.store.read(cx).prefs.clone();
                if prefs.theme.is_none() {
                    theme::configure(&prefs, window);
                    theme::apply(window, cx);
                }
            }),
        ];
        let focus = cx.focus_handle();
        window.focus(&focus, cx);
        Self { store, explore, pipeline, analyze, status, palette: None, show_help: false, settings: None, focus, _subs: subs }
    }

    /// Opens Settings (or closes it).
    fn toggle_settings(&mut self, cx: &mut Context<Self>) {
        if self.settings.take().is_none() {
            let me = cx.entity().downgrade();
            let store = self.store.clone();
            self.settings = Some(cx.new(|_| {
                crate::screens::settings::SettingsView::new(store, move |_, cx| {
                    let _ = me.update(cx, |a, cx| {
                        a.settings = None;
                        cx.notify();
                    });
                })
            }));
        }
        cx.notify();
    }

    pub fn open_path(&mut self, path: std::path::PathBuf, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.open(path, cx));
    }

    fn prompt(store: Entity<Store>, directories: bool, files: bool, prompt: &'static str, save_folder: bool, cx: &mut App) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files, directories, multiple: false, prompt: Some(prompt.into()) });
        cx.spawn(async move |cx| match rx.await {
            Ok(Ok(Some(paths))) => {
                if let Some(path) = paths.into_iter().next() {
                    store.update(cx, |s, cx| if save_folder { s.save_in(path, cx) } else { s.open(path, cx) });
                }
            }
            Ok(Err(e)) => store.update(cx, |s, cx| s.set_status(format!("File dialog failed: {e}"), cx)),
            _ => {}
        })
        .detach();
    }

    fn open_image(store: &Entity<Store>, cx: &mut App) {
        // Directories too: an .ome.zarr container is a folder
        Self::prompt(store.clone(), true, true, "Open image or OME-Zarr", false, cx);
    }

    fn open_project(store: &Entity<Store>, cx: &mut App) {
        Self::prompt(store.clone(), true, true, "Open project (folder or project.lumen.json)", false, cx);
    }

    fn save_project(store: &Entity<Store>, cx: &mut App) {
        Self::prompt(store.clone(), true, false, "Save the project in this folder", true, cx);
    }

    /// Applies `op` to what Explore shows: before its first result, or as a new chain from the
    /// first visible channel.
    fn apply_op(store: &Entity<Store>, op: lumen_pipeline::Op, cx: &mut App) {
        store.update(cx, |s, cx| {
            let Some(img) = s.doc.active().cloned() else {
                s.set_status("Open an image first", cx);
                return;
            };
            let view = s.doc.graph.views().find(|v| s.doc.graph.images_of(&v.id).contains(&img.id)).map(|v| v.id.clone());
            let channel = img.channels.iter().position(|c| c.visible).unwrap_or(0);
            let title = op.title();
            if s.execute(Command::ApplyOp { op, view, image: img.id.clone(), channel }, cx).is_ok() {
                s.set_status(format!("{title} added to the pipeline"), cx);
            }
        });
    }

    /// Every command the search offers (`ops_only`: the operations alone, by family).
    fn commands(&self, ops_only: bool, cx: &mut Context<Self>) -> Vec<PaletteCommand> {
        use std::rc::Rc;
        let mut out = Vec::new();
        let store = self.store.clone();
        let mut add = |group: &'static str, icon: IconName, name: String, hint: String, run: crate::screens::palette::Run| out.push(PaletteCommand { group, name, hint, icon, run });
        if !ops_only {
            for w in Workspace::ALL {
                let st = store.clone();
                add("Go to", IconName::AppWindow, format!("{} workspace", w.title()), "Switch the workspace".into(), Rc::new(move |_, cx| st.update(cx, |s, cx| s.set_workspace(w, cx))));
            }
            let st = store.clone();
            add("File", IconName::ImagePlus, "Open image or OME-Zarr…".into(), "Ctrl+O · add an image to the project".into(), Rc::new(move |_, cx| Self::open_image(&st, cx)));
            let st = store.clone();
            add("File", IconName::FolderOpen, "Open project…".into(), "Ctrl+Shift+O".into(), Rc::new(move |_, cx| Self::open_project(&st, cx)));
            let st = store.clone();
            add("File", IconName::Save, "Save project in folder…".into(), "Ctrl+Shift+S".into(), Rc::new(move |_, cx| Self::save_project(&st, cx)));
            let me = cx.entity().downgrade();
            add("File", IconName::Settings, "Settings…".into(), "Ctrl+,".into(), Rc::new(move |_, cx| {
                let _ = me.update(cx, |a, cx| a.toggle_settings(cx));
            }));
            let st = store.clone();
            add("Edit", IconName::Undo2, "Undo".into(), "Ctrl+Z".into(), Rc::new(move |_, cx| st.update(cx, |s, cx| s.undo(false, cx))));
            let st = store.clone();
            add("Edit", IconName::Redo2, "Redo".into(), "Ctrl+Shift+Z".into(), Rc::new(move |_, cx| st.update(cx, |s, cx| s.undo(true, cx))));
            for t in Tool::BAR.iter().flatten().copied() {
                let st = store.clone();
                add("Tools", t.icon(), t.name().to_string(), format!("Key {}", t.key().to_uppercase()), Rc::new(move |_, cx| st.update(cx, |s, cx| {
                    s.set_tool(t, cx);
                    s.set_workspace(Workspace::Explore, cx);
                })));
            }
            let st = store.clone();
            add("View", IconName::Ruler, "Rulers on / off".into(), "Ctrl+R".into(), Rc::new(move |_, cx| st.update(cx, |s, cx| s.update_tool(|t| t.rulers = !t.rulers, cx))));
            let st = store.clone();
            add("View", IconName::ChartColumn, "Results over the image on / off".into(), "Show or hide the pipeline's results".into(), Rc::new(move |_, cx| st.update(cx, |s, cx| s.update_tool(|t| t.show_results = !t.show_results, cx))));
            let (a, b, c) = (store.clone(), store.clone(), store.clone());
            add("Analyze", IconName::Sparkles, "Find cells".into(), "In the selected regions".into(), Rc::new(move |_, cx| a.update(cx, |s, cx| s.hist_detect_cells(cx))));
            add("Analyze", IconName::ScanSearch, "Find tissue".into(), "Regions of class Tissue".into(), Rc::new(move |_, cx| b.update(cx, |s, cx| s.hist_detect_tissue(cx))));
            add("Classify", IconName::BrainCircuit, "Train from the classed shapes".into(), "Random Trees on cells and tiles".into(), Rc::new(move |_, cx| c.update(cx, |s, cx| s.hist_train_classifier(cx))));
        }
        // Operations under their family, as Process ▸ shows them
        for c in operations() {
            for it in c.items {
                let Template::Op(op) = it.template else { continue };
                let st = store.clone();
                add(c.name, c.icon, it.label.to_string(), format!("{} · a step on the shown channel", it.hint), Rc::new(move |_, cx| Self::apply_op(&st, op.clone(), cx)));
            }
        }
        out
    }

    fn toggle_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.open_search(false, window, cx);
    }

    /// The search (Ctrl+K: everything; Process ▸ Find an operation: operations only); closes it
    /// when open.
    pub fn open_search(&mut self, ops_only: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.palette.take().is_some() {
            window.focus(&self.focus, cx);
            cx.notify();
            return;
        }
        let commands = self.commands(ops_only, cx);
        let (title, placeholder) = if ops_only { ("Find an operation", "Blur, threshold, watershed…") } else { ("Search", "An operation, a tool or an action…") };
        let palette = cx.new(|cx| Palette::new(commands, title, placeholder, window, cx));
        let sub = cx.subscribe_in(&palette, window, |this, _, e: &PaletteEvent, window, cx| {
            this.palette = None;
            window.focus(&this.focus, cx);
            cx.notify();
            // The chosen command runs once the search is gone (it may open Settings, take focus)
            if let PaletteEvent::Run(run) = e {
                let run = run.clone();
                // On the window, with no entity borrowed (the command may update the app itself)
                window.defer(cx, move |window, cx| run(window, cx));
            }
        });
        self.palette = Some((palette, sub));
        cx.notify();
    }

    // ------------------------------------------------------------------------
    // Title bar
    // ------------------------------------------------------------------------

    /// The menu bar by purpose (each menu a verb): File, Edit, View, Image, Shapes, Process,
    /// Analyze, Classify, Automate, Help; items carry the toolbar's icons. Models appear under
    /// the purpose they serve, next to the built-in way, only when they fit the image.
    /// Returns the Lumen menu (File's items, opened from the app's name), Help, and the menu row
    /// (Edit … Automate) shown under the title bar.
    fn menus(&self, cx: &mut Context<Self>) -> (AnyElement, AnyElement, AnyElement) {
        let store = self.store.clone();
        let (can_undo, can_redo) = (self.store.read(cx).can_undo(), self.store.read(cx).can_redo());
        let recent = self.store.read(cx).prefs.recent.clone();
        let has_selection = !self.store.read(cx).selection.is_empty();
        let item = |label: &str, icon: IconName| PopupMenuItem::new(label.to_string()).icon(Icon::new(icon));
        let file = {
            let store = store.clone();
            let me = cx.entity().downgrade();
            Button::new("menu-file").ghost().small().child(div().font_weight(FontWeight::SEMIBOLD).child("Lumen")).tooltip("Files, projects, Settings").dropdown_menu(move |menu: PopupMenu, _, _| {
                let (a, b, c, d, e, f, g) = (store.clone(), store.clone(), store.clone(), store.clone(), store.clone(), store.clone(), store.clone());
                let mut m = menu
                    .item(item("New project  Ctrl+N", IconName::FilePlus).on_click(move |_, _, cx| d.update(cx, |s, cx| s.new_project(cx))))
                    .item(item("Open image or OME-Zarr…  Ctrl+O", IconName::ImagePlus).on_click(move |_, _, cx| Self::open_image(&a, cx)))
                    .item(item("Open project…  Ctrl+Shift+O", IconName::FolderOpen).on_click(move |_, _, cx| Self::open_project(&b, cx)))
                    .item(item("Save project in folder…  Ctrl+Shift+S", IconName::Save).on_click(move |_, _, cx| Self::save_project(&c, cx)))
                    .separator()
                    .label("Shapes")
                    .item(item("Import GeoJSON…", IconName::FileInput).on_click(move |_, _, cx| {
                        let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some("GeoJSON file".into()) });
                        let st = e.clone();
                        cx.spawn(async move |cx| {
                            if let Ok(Ok(Some(paths))) = rx.await
                                && let Some(p) = paths.into_iter().next() {
                                    st.update(cx, |s, cx| s.import_geojson(p, cx));
                                }
                        })
                        .detach();
                    }))
                    .item(item("Export as GeoJSON", IconName::FileOutput).on_click(move |_, _, cx| f.update(cx, |s, cx| s.export_geojson(cx))))
                    .item(item("Export cell measurements", IconName::Sheet).on_click(move |_, _, cx| g.update(cx, |s, cx| s.hist_export(cx))));
                if !recent.is_empty() {
                    m = m.separator().label("Recent");
                    for p in recent.iter().take(8) {
                        let (st, p) = (store.clone(), p.clone());
                        m = m.item(PopupMenuItem::new(crate::data::store::display_name(&p)).on_click(move |_, _, cx| st.update(cx, |s, cx| s.open(p.clone(), cx))));
                    }
                }
                m.separator()
                    .item(item("Settings…  Ctrl+,", IconName::Settings).on_click({
                        let me = me.clone();
                        move |_, _, cx| {
                            let _ = me.update(cx, |a, cx| a.toggle_settings(cx));
                        }
                    }))
                    .item(item("Quit  Ctrl+Q", IconName::Power).on_click(|_, _, cx| cx.quit()))
            })
        };
        let edit = {
            let store = store.clone();
            Button::new("menu-edit").ghost().small().label("Edit").dropdown_menu(move |menu: PopupMenu, _, _| {
                let (a, b, c) = (store.clone(), store.clone(), store.clone());
                menu.item(item("Undo  Ctrl+Z", IconName::Undo2).disabled(!can_undo).on_click(move |_, _, cx| a.update(cx, |s, cx| s.undo(false, cx))))
                    .item(item("Redo  Ctrl+Shift+Z", IconName::Redo2).disabled(!can_redo).on_click(move |_, _, cx| b.update(cx, |s, cx| s.undo(true, cx))))
                    .separator()
                    .item(item("Delete selected  Del", IconName::Trash).disabled(!has_selection).on_click(move |_, _, cx| c.update(cx, |s, cx| s.delete_selection(cx))))
            })
        };
        // Process: change pixels (image → image), one submenu per family
        let ops = {
            let store = store.clone();
            let me_find = cx.entity().downgrade();
            Button::new("menu-ops").ghost().small().label("Process").dropdown_menu(move |menu: PopupMenu, window, cx| {
                let me = me_find.clone();
                let mut m = menu
                    .item(PopupMenuItem::new("Find an operation…  Ctrl+K").icon(Icon::new(IconName::Search)).on_click(move |_, window, cx| {
                        let _ = me.update(cx, |a, cx| a.open_search(true, window, cx));
                    }))
                    .separator()
                    .label("A step on the shown channel");
                for c in operations() {
                    let (store, icon) = (store.clone(), c.icon);
                    let items = c.items;
                    m = m.submenu_with_icon(Some(Icon::new(icon)), c.name, window, cx, move |mut sub, _, _| {
                        for it in &items {
                            let Template::Op(op) = it.template.clone() else { continue };
                            let st = store.clone();
                            sub = sub.item(PopupMenuItem::new(format!("{}   {}", it.label, it.hint)).icon(Icon::new(icon)).on_click(move |_, _, cx| Self::apply_op(&st, op.clone(), cx)));
                        }
                        sub
                    });
                }
                m
            })
        };
        let view = {
            let store = store.clone();
            let me_view = cx.entity().downgrade();
            let tool = self.store.read(cx).tool.clone();
            let (link, pipeline_open) = (self.store.read(cx).prefs.link_views, self.store.read(cx).prefs.pipeline_dock);
            Button::new("menu-view").ghost().small().label("View").dropdown_menu(move |menu: PopupMenu, window, cx| {
                let request = |r: ViewRequest| {
                    let st = store.clone();
                    move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| st.update(cx, |s, cx| s.request_view(r, cx))
                };
                let toggle = |f: fn(&mut crate::domain::tool::ToolState)| {
                    let st = store.clone();
                    move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| st.update(cx, |s, cx| s.update_tool(f, cx))
                };
                let mut m = menu.label("Show over the image")
                    .item(item("Shapes  A", IconName::Shapes).checked(tool.show_annotations).on_click(toggle(|t| t.show_annotations = !t.show_annotations)))
                    .item(item("Fill regions  Shift+F", IconName::PaintBucket).checked(tool.fill_annotations).on_click(toggle(|t| t.fill_annotations = !t.fill_annotations)))
                    .item(item("Names  Shift+N", IconName::Type).checked(tool.show_names).on_click(toggle(|t| t.show_names = !t.show_names)))
                    .item(item("Results", IconName::ChartColumn).checked(tool.show_results).on_click(toggle(|t| t.show_results = !t.show_results)))
                    .item(item("Crosshair (3D cursor)", IconName::Crosshair).checked(tool.crosshair).on_click(toggle(|t| t.crosshair = !t.crosshair)))
                    .item(item("Rulers  Ctrl+R", IconName::Ruler).checked(tool.rulers).on_click(toggle(|t| t.rulers = !t.rulers)))
                    .separator();
                {
                    let store = store.clone();
                    m = m.submenu_with_icon(Some(Icon::new(IconName::LayoutGrid)), "This view shows", window, cx, move |mut sub, _, _| {
                        for kind in ViewKind::ALL {
                            let st = store.clone();
                            sub = sub.item(PopupMenuItem::new(format!("{}: {}", kind.title(), kind.hint())).on_click(move |_, _, cx| st.update(cx, |s, cx| s.request_view(ViewRequest::Show(kind), cx))));
                        }
                        sub
                    });
                }
                {
                    let store = store.clone();
                    m = m.submenu_with_icon(Some(Icon::new(IconName::SquarePlus)), "Views", window, cx, move |mut sub, _, _| {
                        for kind in ViewKind::ALL {
                            let st = store.clone();
                            sub = sub.item(PopupMenuItem::new(format!("Add a {} view", kind.title())).on_click(move |_, _, cx| st.update(cx, |s, cx| s.request_view(ViewRequest::Add(kind), cx))));
                        }
                        let st = store.clone();
                        sub = sub.item(PopupMenuItem::new("Close this view").on_click(move |_, _, cx| st.update(cx, |s, cx| s.request_view(ViewRequest::ClosePane, cx))));
                        for how in Arrangement::ALL {
                            let st = store.clone();
                            sub = sub.item(PopupMenuItem::new(format!("Arrange: {}", how.title())).on_click(move |_, _, cx| st.update(cx, |s, cx| s.request_view(ViewRequest::Arrange(how), cx))));
                        }
                        sub
                    });
                }
                m.item(item("Link views", IconName::Link).checked(link).on_click(request(ViewRequest::ToggleLink)))
                    .item(item("Pipeline under the views", IconName::Workflow).checked(pipeline_open).on_click(request(ViewRequest::TogglePipeline)))
                    .separator()
                    .item(item("Layers  Ctrl+[", IconName::Layers2).on_click(request(ViewRequest::ToggleLeft)))
                    .item(item("Side panel  Ctrl+]", IconName::PanelRight).on_click(request(ViewRequest::ToggleRight)))
                    .item(item("Reset panel layout", IconName::RotateCcw).on_click(request(ViewRequest::ResetLayout)))
                    .separator()
                    .item({
                        let me = me_view.clone();
                        item("Light / dark", IconName::SunMoon).on_click(move |_, window, cx| {
                            let _ = me.update(cx, |a, cx| a.toggle_theme(window, cx));
                        })
                    })
            })
        };
        let help = {
            let me = cx.entity().downgrade();
            Button::new("menu-help").ghost().small().label("Help").dropdown_menu(move |menu: PopupMenu, _, _| {
                let me = me.clone();
                menu.item(item("Keyboard & mouse  F1", IconName::Keyboard).on_click(move |_, _, cx| {
                    let _ = me.update(cx, |a, cx| {
                        a.show_help = true;
                        cx.notify();
                    });
                }))
            })
        };
        let domain = self.domain_menus(cx);
        (file.into_any_element(), help.into_any_element(), h_flex().gap(px(2.)).child(edit).child(view).children(domain.0).child(ops).children(domain.1).into_any_element())
    }

    /// Image and Shapes (before Process), then Analyze, Classify and Automate (after it).
    fn domain_menus(&self, cx: &mut Context<Self>) -> (Vec<AnyElement>, Vec<AnyElement>) {
        use crate::data::script::Language;
        let store = self.store.clone();
        let s = self.store.read(cx);
        let channels: Vec<(usize, String, bool)> = s.doc.active().map(|i| i.channels.iter().enumerate().map(|(k, c)| (k, c.label.clone(), c.visible)).collect()).unwrap_or_default();
        let images: Vec<(String, String, bool)> = s.doc.images.iter().map(|i| (i.id.clone(), i.name.clone(), s.doc.active_image.as_deref() == Some(i.id.as_str()))).collect();
        let tool = s.tool.clone();
        let image_id = s.doc.active_image.clone();
        let masks = s.selected_masks().len();
        let has_selection = !s.selection.is_empty();
        let brightfield = image_id.as_deref().map(|i| s.hist_is_brightfield(i));
        // Models by purpose, those fitting the image
        let detectors: Vec<(String, String, bool)> = brightfield.map(|b| s.ml.nucleus_models_for(b).into_iter().map(|(_, m)| (m.id, m.name, m.installed)).collect()).unwrap_or_default();
        let classifiers: Vec<(String, String, bool)> = s.ml.plugins.iter().flat_map(|p| p.models.iter()).filter(|m| m.task == lumen_ml::Task::Classification).map(|m| (m.id.clone(), m.name.clone(), m.installed)).collect();
        let segmenter = s.ml.segmenter(s.prefs.ml_model.as_deref()).is_some();
        let detector = s.hist.detector.clone();
        let item = |label: &str, icon: IconName| PopupMenuItem::new(label.to_string()).icon(Icon::new(icon));
        let image = {
            let store = store.clone();
            Button::new("menu-image").ghost().small().label("Image").dropdown_menu(move |menu: PopupMenu, _, _| {
                let mut m = menu.label("Channels shown");
                for (k, name, on) in &channels {
                    let (st, k, img) = (store.clone(), *k, image_id.clone());
                    m = m.item(PopupMenuItem::new(name.clone()).checked(*on).on_click(move |_, _, cx| {
                        let Some(image) = img.clone() else { return };
                        st.update(cx, |s, cx| {
                            if let Some(mut display) = s.doc.image(&image).and_then(|i| i.channels.get(k).cloned()) {
                                display.visible = !display.visible;
                                let _ = s.execute(Command::SetChannel { image, channel: k, display }, cx);
                            }
                        })
                    }));
                }
                m = m.separator().label("Images of the project");
                for (id, name, on) in &images {
                    let (st, id) = (store.clone(), id.clone());
                    m = m.item(PopupMenuItem::new(name.clone()).icon(Icon::new(IconName::Image)).checked(*on).on_click(move |_, _, cx| st.update(cx, |s, cx| {
                        let _ = s.execute(Command::SelectImage { id: id.clone() }, cx);
                    })));
                }
                let st = store.clone();
                m = m.separator().item(item("Open image or OME-Zarr…  Ctrl+O", IconName::ImagePlus).on_click(move |_, _, cx| Self::open_image(&st, cx)));
                let (st, img) = (store.clone(), image_id.clone());
                m.item(item("Remove from the project (the file stays)", IconName::Trash).disabled(img.is_none()).on_click(move |_, _, cx| {
                    if let Some(id) = img.clone() {
                        st.update(cx, |s, cx| s.remove_image(id, cx));
                    }
                }))
            })
            .into_any_element()
        };
        // Shapes: draw and manage regions, points, lines and detected objects
        let shapes = {
            let store = store.clone();
            Button::new("menu-shapes").ghost().small().label("Shapes").dropdown_menu(move |menu: PopupMenu, window, cx| {
                let tool_item = |t: Tool| {
                    let st = store.clone();
                    PopupMenuItem::new(format!("{}  {}", t.name(), t.key().to_uppercase())).icon(Icon::new(t.icon())).checked(tool.tool == t).on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_tool(t, cx)))
                };
                let mut m = menu.label("Select").item(tool_item(Tool::Select)).item(tool_item(Tool::Marquee)).separator().label("Draw");
                {
                    // The selected regions as one (Affinity's Add, Subtract, Intersect, Xor)
                    let store = store.clone();
                    m = m.submenu_with_icon(Some(Icon::new(IconName::SquaresUnite)), "Combine the selected shapes", window, cx, move |mut sub, _, _| {
                        use crate::domain::boolean::ShapeOp;
                        for op in ShapeOp::ALL {
                            let st = store.clone();
                            let icon = match op {
                                ShapeOp::Add => IconName::SquaresUnite,
                                ShapeOp::Subtract => IconName::SquaresSubtract,
                                ShapeOp::Intersect => IconName::SquaresIntersect,
                                ShapeOp::Xor => IconName::SquaresExclude,
                            };
                            sub = sub.item(PopupMenuItem::new(op.hint()).icon(Icon::new(icon)).on_click(move |_, _, cx| st.update(cx, |s, cx| s.combine_shapes(op, cx))));
                        }
                        sub
                    });
                }
                for t in [Tool::Rect, Tool::Ellipse, Tool::Polygon, Tool::Lasso, Tool::Point, Tool::Brush, Tool::Eraser, Tool::Wand, Tool::Threshold] {
                    m = m.item(tool_item(t));
                }
                if segmenter {
                    m = m.item(tool_item(Tool::Segment));
                }
                {
                    let store = store.clone();
                    m = m.separator().submenu_with_icon(Some(Icon::new(IconName::Combine)), "Combine the selected", window, cx, move |sub, _, _| {
                        let combine = |label: &'static str, need: usize, icon: IconName, op: crate::data::store::MaskOperation| {
                            let st = store.clone();
                            PopupMenuItem::new(label).icon(Icon::new(icon)).disabled(masks < need).on_click(move |_, _, cx| st.update(cx, |s, cx| s.apply_mask_operation(op.clone(), cx)))
                        };
                        let r = 2usize;
                        sub.item(combine("Union", 2, IconName::SquaresUnite, crate::data::store::MaskOperation::Union))
                            .item(combine("Intersect", 2, IconName::SquaresIntersect, crate::data::store::MaskOperation::Intersect))
                            .item(combine("Subtract", 2, IconName::SquaresSubtract, crate::data::store::MaskOperation::Subtract))
                            .item(combine("Invert", 1, IconName::Contrast, crate::data::store::MaskOperation::Invert))
                            .item(combine("Grow 2 px", 1, IconName::Expand, crate::data::store::MaskOperation::Op(lumen_pipeline::Op::Dilate { radius: r })))
                            .item(combine("Shrink 2 px", 1, IconName::Shrink, crate::data::store::MaskOperation::Op(lumen_pipeline::Op::Erode { radius: r })))
                    });
                }
                let del = store.clone();
                m.separator()
                    .label("Measure")
                    .item(tool_item(Tool::Ruler))
                    .item(tool_item(Tool::Probe))
                    .separator()
                    .item(item("Delete selected  Del", IconName::Trash).disabled(!has_selection).on_click(move |_, _, cx| del.update(cx, |s, cx| s.delete_selection(cx))))
            })
            .into_any_element()
        };
        // Analyze: find and measure (in the selected regions)
        let analyze_menu = {
            let store = store.clone();
            let detectors = detectors.clone();
            let detector = detector.clone();
            Button::new("menu-analyze").ghost().small().label("Analyze").dropdown_menu(move |menu: PopupMenu, window, cx| {
                let (a, b, c, d, e, f, g) = (store.clone(), store.clone(), store.clone(), store.clone(), store.clone(), store.clone(), store.clone());
                let mut m = menu
                    .label("Find (in the selected regions)")
                    .item(item("Find tissue", IconName::ScanSearch).on_click(move |_, _, cx| a.update(cx, |s, cx| s.hist_detect_tissue(cx))))
                    .item(item("Find cells", IconName::Sparkles).disabled(brightfield.is_none()).on_click(move |_, _, cx| b.update(cx, |s, cx| s.hist_detect_cells(cx))));
                {
                    let (store, detectors, detector) = (store.clone(), detectors.clone(), detector.clone());
                    m = m.submenu_with_icon(Some(Icon::new(IconName::Microscope)), "Find cells with", window, cx, move |mut sub, _, _| {
                        let st = store.clone();
                        sub = sub.item(PopupMenuItem::new("Threshold (built in)").checked(detector.is_none()).on_click(move |_, _, cx| st.update(cx, |s, cx| s.hist_set_detector(None, cx))));
                        for (id, name, installed) in &detectors {
                            let (st, id) = (store.clone(), id.clone());
                            let label = if *installed { name.clone() } else { format!("{name} (install in Settings → Models)") };
                            sub = sub.item(PopupMenuItem::new(label).icon(Icon::new(IconName::BrainCircuit)).checked(detector.as_deref() == Some(id.as_str())).disabled(!installed).on_click(move |_, _, cx| st.update(cx, |s, cx| s.hist_set_detector(Some(id.clone()), cx))));
                        }
                        sub
                    });
                }
                m.item(item("Superpixels", IconName::Grid3x3).on_click(move |_, _, cx| c.update(cx, |s, cx| s.hist_superpixels(cx))))
                    .separator()
                    .label("Density")
                    .item(item("Density map", IconName::Flame).on_click(move |_, _, cx| d.update(cx, |s, cx| s.hist_density(cx))))
                    .item(item("Mark 3 hotspots", IconName::MapPin).on_click(move |_, _, cx| e.update(cx, |s, cx| s.hist_hotspots(3, cx))))
                    .separator()
                    .label("Measure")
                    .item(item("Measurements table  Ctrl+3", IconName::Table2).on_click(move |_, _, cx| f.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx))))
                    .item(item("Export cell measurements", IconName::Sheet).on_click(move |_, _, cx| g.update(cx, |s, cx| s.hist_export(cx))))
            })
            .into_any_element()
        };
        // Classify: objects and pixels, built-in first, then models for the purpose
        let classify = {
            let store = store.clone();
            Button::new("menu-classify").ghost().small().label("Classify").dropdown_menu(move |menu: PopupMenu, _, _| {
                let (a, b, c, d) = (store.clone(), store.clone(), store.clone(), store.clone());
                let mut m = menu
                    .label("Cells and tiles")
                    .item(item("Train from the classed shapes (Random Trees)", IconName::BrainCircuit).on_click(move |_, _, cx| a.update(cx, |s, cx| s.hist_train_classifier(cx))))
                    .item(item("Intensity classes (1+ / 2+ / 3+)", IconName::Gauge).on_click(move |_, _, cx| b.update(cx, |s, cx| s.hist_apply_intensity(cx))));
                if !classifiers.is_empty() {
                    m = m.label("Models");
                    for (_, name, installed) in &classifiers {
                        m = m.item(PopupMenuItem::new(if *installed { name.clone() } else { format!("{name} (install in Settings → Models)") }).icon(Icon::new(IconName::BrainCircuit)).disabled(true));
                    }
                }
                m.separator()
                    .label("Pixels")
                    .item(item("Train and classify pixels", IconName::Palette).on_click(move |_, _, cx| c.update(cx, |s, cx| s.hist_train_pixels(cx))))
                    .item(item("Class map → shapes", IconName::Shapes).on_click(move |_, _, cx| d.update(cx, |s, cx| s.hist_class_regions(cx))))
            })
            .into_any_element()
        };
        let analyze = self.analyze.clone();
        let automate = {
            let store = store.clone();
            let analyze = analyze.clone();
            Button::new("menu-automate").ghost().small().label("Automate").dropdown_menu(move |menu: PopupMenu, _, _| {
                let (a, b, c, d, e) = (store.clone(), store.clone(), store.clone(), store.clone(), store.clone());
                let an = analyze.clone();
                menu.label("Pipeline")
                    .item(item("Pipeline workspace  Ctrl+2", IconName::Workflow).on_click(move |_, _, cx| a.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx))))
                    .item(item("Pipeline as a panel of Explore", IconName::PanelBottom).on_click({
                        let st = b.clone();
                        move |_, _, cx| st.update(cx, |s, cx| s.request_view(ViewRequest::TogglePipeline, cx))
                    }))
                    .separator()
                    .label("Scripts (Python, Rust)")
                    .item(item("New Python script node…", IconName::SquareTerminal).on_click(move |_, _, cx| c.update(cx, |s, cx| s.add_script_node(Language::Python, cx))))
                    .item(item("New Rust script node…", IconName::SquareTerminal).on_click(move |_, _, cx| d.update(cx, |s, cx| s.add_script_node(Language::Rust, cx))))
                    .item(item("Code on the measurement table", IconName::Code).on_click(move |_, _, cx| {
                        e.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx));
                        an.update(cx, |a, cx| a.open_code(cx));
                    }))
            })
            .into_any_element()
        };
        (vec![image, shapes], vec![analyze_menu, classify, automate])
    }

    /// Dark ↔ light (as dsp-kitchen's title bar button); Settings keeps System, the accent and more.
    fn toggle_theme(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.update_prefs(|p| p.theme = Some(!theme::is_dark()), cx));
        let prefs = self.store.read(cx).prefs.clone();
        theme::configure(&prefs, window);
        theme::apply(window, cx);
    }

    fn workspaces(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let current = self.store.read(cx).workspace;
        h_flex().gap(px(2.)).p(px(2.)).rounded(px(6.)).bg(color::control().opacity(0.6)).children(Workspace::ALL.into_iter().map(|w| {
            let store = self.store.clone();
            Button::new(SharedString::from(format!("workspace-{}", w.title().to_lowercase())))
                .small()
                .ghost()
                .icon(w.icon())
                .label(w.title())
                .tooltip(w.hint())
                .selected(w == current)
                .on_click(move |_, _, cx| store.update(cx, |s, cx| s.set_workspace(w, cx)))
        }))
    }

    fn title_bar(&self, rounded: bool, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let s = self.store.read(cx);
        let saved = s.project_file.is_some();
        let name = s.doc.name.clone();
        let (lumen, help, explore_row) = self.menus(cx);
        // Each workspace has its own menus in the row under the bar
        let row = match self.store.read(cx).workspace {
            Workspace::Pipeline => self.pipeline.update(cx, |p, cx| p.menus(cx)),
            Workspace::Analyze => self.analyze.update(cx, |a, cx| a.menus(cx)),
            Workspace::Explore => explore_row,
        };
        // Affinity's layout: the app's menu (Lumen: files, Settings) and Help at the top left,
        // the workspaces next to them; the menus of what to do sit in a row under the bar
        let bar = TitleBar::new().when(rounded, |t| t.rounded_t(px(WINDOW_RADIUS - 1.0))).child(
            h_flex()
                .w_full()
                .h(px(theme::TITLE_BAR_H))
                .gap(px(theme::MD))
                .pr(px(theme::SM))
                .items_center()
                .child(h_flex().gap(px(2.)).child(lumen).child(help))
                .child(self.workspaces(cx))
                .child(div().flex_1())
                .child(div().text_sm().text_color(color::text_muted()).truncate().child(if saved { name } else { format!("{name} · not saved") }))
                .child(
                    Button::new("title-right")
                        .ghost()
                        .small()
                        .icon(IconName::PanelRight)
                        .tooltip("Side panel (Ctrl+]; Ctrl+[ for Layers)")
                        .on_click(cx.listener(|this, _, _, cx| this.store.update(cx, |s, cx| s.request_view(ViewRequest::ToggleRight, cx)))),
                )
                .child(
                    Button::new("title-theme")
                        .ghost()
                        .small()
                        .icon(if theme::is_dark() { IconName::Sun } else { IconName::Moon })
                        .tooltip(if theme::is_dark() { "Light theme" } else { "Dark theme" })
                        .on_click(cx.listener(|this, _, window, cx| this.toggle_theme(window, cx))),
                ),
        );
        v_flex()
            .flex_none()
            .w_full()
            .child(bar)
            .child(h_flex().flex_none().w_full().h(px(30.)).px(px(theme::XS)).items_center().border_b_1().border_color(color::border()).bg(color::panel()).child(row))
    }

    // ------------------------------------------------------------------------
    // Body
    // ------------------------------------------------------------------------

    fn body(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let s = self.store.read(cx);
        match s.workspace {
            Workspace::Explore if s.doc.images.is_empty() => {
                let (a, b) = (self.store.clone(), self.store.clone());
                crate::screens::home::home(&self.store, move |_, cx| Self::open_image(&a, cx), move |_, cx| Self::open_project(&b, cx), cx).into_any_element()
            }
            Workspace::Explore => self.explore.clone().into_any_element(),
            Workspace::Pipeline => self.pipeline.clone().into_any_element(),
            Workspace::Analyze => self.analyze.clone().into_any_element(),
        }
    }

    fn help(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        div()
            .id("help-overlay")
            .absolute()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui_kit::hsla(0., 0., 0., 0.55))
            .on_click(cx.listener(|this, _, _, cx| {
                this.show_help = false;
                cx.notify();
            }))
            .child(
                v_flex()
                    .id("help")
                    .w(px(600.))
                    .p(px(theme::LG))
                    .gap(px(theme::MD))
                    .rounded(px(8.))
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
                    .child(h_flex().justify_end().child(Button::new("help-close").outline().small().label("Close").on_click(cx.listener(|this, _, _, cx| {
                        this.show_help = false;
                        cx.notify();
                    })))),
            )
    }
}

/// The shadow around the window when its corners are rounded (px).
const WINDOW_SHADOW: f32 = 10.;
/// Rounded corners of the window when it floats.
const WINDOW_RADIUS: f32 = 10.;

/// The edge a press at `p` (window px) resizes, within the shadow band of a `size` window.
fn resize_edge(p: gpui_kit::Point<gpui_kit::Pixels>, size: gpui_kit::Size<gpui_kit::Pixels>) -> Option<gpui_kit::ResizeEdge> {
    use gpui_kit::ResizeEdge as E;
    let band = WINDOW_SHADOW + 4.0;
    let (x, y, w, h) = (p.x.as_f32(), p.y.as_f32(), size.width.as_f32(), size.height.as_f32());
    let (l, r, t, b) = (x < band, x > w - band, y < band, y > h - band);
    Some(match (l, r, t, b) {
        (true, _, true, _) => E::TopLeft,
        (_, true, true, _) => E::TopRight,
        (true, _, _, true) => E::BottomLeft,
        (_, true, _, true) => E::BottomRight,
        (true, ..) => E::Left,
        (_, true, ..) => E::Right,
        (_, _, true, _) => E::Top,
        (_, _, _, true) => E::Bottom,
        _ => return None,
    })
}

/// Whether the window draws its own rounded frame: client-side decorations on a floating window.
fn frame_rounded(window: &Window) -> bool {
    match window.window_decorations() {
        gpui_kit::Decorations::Client { tiling } => !(tiling.top || tiling.left || tiling.right || tiling.bottom) && !window.is_maximized() && !window.is_fullscreen(),
        gpui_kit::Decorations::Server => false,
    }
}

impl Render for LumenApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Linux draws the window frame itself (client-side decorations): rounded corners and a
        // shadow while the window floats, square edges when it is maximised, full screen or tiled
        // GPUI clips children to rectangles, not rounded corners: the title bar and the status bar
        // round their own corners, or their square backgrounds show past the frame's curve
        let csd = matches!(window.window_decorations(), gpui_kit::Decorations::Client { .. });
        let rounded = frame_rounded(window);
        if csd {
            window.set_client_inset(px(if rounded { WINDOW_SHADOW } else { 0.0 }));
        }
        let app = v_flex()
            .id("lumen-app")
            .track_focus(&self.focus)
            .size_full()
            .relative()
            .bg(color::panel())
            .text_color(color::text())
            .on_action(cx.listener(|this, _: &OpenImage, _, cx| Self::open_image(&this.store, cx)))
            .on_action(cx.listener(|this, _: &OpenProject, _, cx| Self::open_project(&this.store, cx)))
            .on_action(cx.listener(|this, _: &SaveProjectAs, _, cx| Self::save_project(&this.store, cx)))
            .on_action(cx.listener(|this, _: &NewProject, _, cx| this.store.update(cx, |s, cx| s.new_project(cx))))
            .on_action(cx.listener(|this, _: &Undo, _, cx| this.store.update(cx, |s, cx| s.undo(false, cx))))
            .on_action(cx.listener(|this, _: &Redo, _, cx| this.store.update(cx, |s, cx| s.undo(true, cx))))
            .on_action(cx.listener(|this, _: &CommandPalette, w, cx| this.toggle_palette(w, cx)))
            .on_action(cx.listener(|this, _: &FindOperation, w, cx| this.open_search(true, w, cx)))
            .on_action(cx.listener(|this, _: &OpenSettings, _, cx| this.toggle_settings(cx)))
            .on_action(cx.listener(|this, _: &ToggleLeftPanel, _, cx| this.store.update(cx, |s, cx| s.request_view(ViewRequest::ToggleLeft, cx))))
            .on_action(cx.listener(|this, _: &ToggleRightPanel, _, cx| this.store.update(cx, |s, cx| s.request_view(ViewRequest::ToggleRight, cx))))
            .on_key_down(cx.listener(|this, e: &gpui_kit::KeyDownEvent, _, cx| {
                if e.keystroke.key == "escape" && this.settings.is_some() {
                    this.settings = None;
                    cx.notify();
                    cx.stop_propagation();
                }
            }))
            .on_action(cx.listener(|this, _: &ShowHelp, _, cx| {
                this.show_help = !this.show_help;
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &ToggleSidebar, _, cx| this.store.update(cx, |s, cx| s.request_view(ViewRequest::ToggleSides, cx))))
            .on_action(cx.listener(|this, _: &ToggleRulers, _, cx| this.store.update(cx, |s, cx| s.update_tool(|t| t.rulers = !t.rulers, cx))))
            .on_action(cx.listener(|this, _: &GoExplore, _, cx| this.store.update(cx, |s, cx| s.set_workspace(Workspace::Explore, cx))))
            .on_action(cx.listener(|this, _: &GoPipeline, _, cx| this.store.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx))))
            .on_action(cx.listener(|this, _: &GoAnalyze, _, cx| this.store.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx))))
            .on_action(|_: &Quit, _, cx: &mut App| cx.quit())
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                if let Some(p) = paths.paths().first().cloned() {
                    this.open_path(p, cx);
                }
            }))
            .child(self.title_bar(rounded, cx))
            .child(div().flex_1().min_h_0().child(self.body(cx)))
            .child(self.status.clone())
            .children(self.palette.as_ref().map(|(p, _)| p.clone()))
            .when(self.show_help, |d| d.child(self.help(cx)))
            .children(self.settings.clone())
            .when(rounded, |d| d.rounded(px(WINDOW_RADIUS)).overflow_hidden().border_1().border_color(color::border()));
        div()
            .id("window-frame")
            .size_full()
            .when(rounded, |d| {
                d.p(px(WINDOW_SHADOW)).on_mouse_down(gpui_kit::MouseButton::Left, |e, window, _| {
                    if let Some(edge) = resize_edge(e.position, window.viewport_size()) {
                        window.start_window_resize(edge);
                    }
                })
            })
            .child(div().size_full().when(rounded, |d| d.rounded(px(WINDOW_RADIUS)).shadow_lg()).child(app))
    }
}

//! The root view: one title bar (menus, the three workspaces, the project), the workspace shown,
//! one status bar. Without an image the body is the home view.

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, Selectable as _, Sizable as _, TitleBar};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, ExternalPaths, FocusHandle, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, PathPromptOptions, Render,
    SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::data::store::{AppEvent, Store};
use crate::domain::command::Command;
use crate::navigation::actions::*;
use crate::navigation::route::Workspace;
use crate::screens::explore::ExploreView;
use crate::screens::pipeline::catalog::{operations, Template};
use crate::ui::theme::{self, color};
use crate::ui::widgets::EmptyState;

pub struct LumenApp {
    store: Entity<Store>,
    pub explore: Entity<ExploreView>,
    show_help: bool,
    focus: FocusHandle,
    _subs: Vec<Subscription>,
}

impl LumenApp {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        theme::apply(window, cx);
        let explore = cx.new(|cx| ExploreView::new(store.clone(), window, cx));
        let subs = vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify())];
        let focus = cx.focus_handle();
        window.focus(&focus, cx);
        Self { store, explore, show_help: false, focus, _subs: subs }
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

    // ------------------------------------------------------------------------
    // Title bar
    // ------------------------------------------------------------------------

    fn menus(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let store = self.store.clone();
        let (can_undo, can_redo) = (self.store.read(cx).can_undo(), self.store.read(cx).can_redo());
        let recent = self.store.read(cx).prefs.recent.clone();
        let file = {
            let store = store.clone();
            Button::new("menu-file").ghost().small().label("File").dropdown_menu(move |menu: PopupMenu, _, _| {
                let (a, b, c, d) = (store.clone(), store.clone(), store.clone(), store.clone());
                let mut m = menu
                    .item(PopupMenuItem::new("Open image or OME-Zarr…  Ctrl+O").on_click(move |_, _, cx| Self::open_image(&a, cx)))
                    .item(PopupMenuItem::new("Open project…  Ctrl+Shift+O").on_click(move |_, _, cx| Self::open_project(&b, cx)))
                    .item(PopupMenuItem::new("Save project in folder…  Ctrl+Shift+S").on_click(move |_, _, cx| Self::save_project(&c, cx)))
                    .item(PopupMenuItem::new("New project  Ctrl+N").on_click(move |_, _, cx| d.update(cx, |s, cx| s.new_project(cx))));
                if !recent.is_empty() {
                    m = m.separator().label("Recent");
                    for p in recent.iter().take(8) {
                        let (st, p) = (store.clone(), p.clone());
                        m = m.item(PopupMenuItem::new(crate::data::store::display_name(&p)).on_click(move |_, _, cx| st.update(cx, |s, cx| s.open(p.clone(), cx))));
                    }
                }
                m.separator().item(PopupMenuItem::new("Quit  Ctrl+Q").on_click(|_, _, cx| cx.quit()))
            })
        };
        let edit = {
            let store = store.clone();
            Button::new("menu-edit").ghost().small().label("Edit").dropdown_menu(move |menu: PopupMenu, _, _| {
                let (a, b, c) = (store.clone(), store.clone(), store.clone());
                menu.item(PopupMenuItem::new("Undo  Ctrl+Z").disabled(!can_undo).on_click(move |_, _, cx| a.update(cx, |s, cx| s.undo(false, cx))))
                    .item(PopupMenuItem::new("Redo  Ctrl+Shift+Z").disabled(!can_redo).on_click(move |_, _, cx| b.update(cx, |s, cx| s.undo(true, cx))))
                    .separator()
                    .item(PopupMenuItem::new("Delete selected annotations  Del").on_click(move |_, _, cx| c.update(cx, |s, cx| s.delete_selection(cx))))
            })
        };
        // Photoshop style: one menu of operations, by family; each applies to what Explore shows
        let ops = {
            let store = store.clone();
            Button::new("menu-ops").ghost().small().label("Operations").dropdown_menu(move |menu: PopupMenu, _, _| {
                let mut m = menu.scrollable(true).max_h(px(560.));
                for c in operations() {
                    m = m.label(c.name);
                    for it in c.items {
                        let Template::Op(op) = it.template else { continue };
                        let st = store.clone();
                        m = m.item(PopupMenuItem::new(format!("{}   {}", it.label, it.hint)).on_click(move |_, _, cx| Self::apply_op(&st, op.clone(), cx)));
                    }
                }
                m
            })
        };
        let view = {
            let store = store.clone();
            Button::new("menu-view").ghost().small().label("View").dropdown_menu(move |menu: PopupMenu, _, _| {
                let (a, b, c) = (store.clone(), store.clone(), store.clone());
                menu.item(PopupMenuItem::new("Rulers  Ctrl+R").on_click(move |_, _, cx| a.update(cx, |s, cx| s.update_tool(|t| t.rulers = !t.rulers, cx))))
                    .item(PopupMenuItem::new("Results over the image").on_click(move |_, _, cx| b.update(cx, |s, cx| s.update_tool(|t| t.show_results = !t.show_results, cx))))
                    .item(PopupMenuItem::new("Side panel  Tab").on_click(move |_, _, cx| c.update(cx, |s, cx| {
                        let open = !s.prefs.sidebar_open;
                        s.set_sidebar(open, cx)
                    })))
            })
        };
        let help = {
            let me = cx.entity().downgrade();
            Button::new("menu-help").ghost().small().label("Help").dropdown_menu(move |menu: PopupMenu, _, _| {
                let me = me.clone();
                menu.item(PopupMenuItem::new("Keyboard & mouse  F1").on_click(move |_, _, cx| {
                    let _ = me.update(cx, |a, cx| {
                        a.show_help = true;
                        cx.notify();
                    });
                }))
            })
        };
        h_flex().gap(px(2.)).child(file).child(edit).child(ops).child(view).child(help)
    }

    fn workspaces(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let current = self.store.read(cx).workspace;
        h_flex().gap(px(2.)).children(Workspace::ALL.into_iter().map(|w| {
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

    fn title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let s = self.store.read(cx);
        let saved = s.project_file.is_some();
        let name = s.doc.name.clone();
        TitleBar::new().child(
            h_flex()
                .w_full()
                .h(px(theme::TITLE_BAR_H))
                .gap(px(theme::MD))
                .pr(px(theme::SM))
                .items_center()
                .child(div().pl(px(theme::XS)).text_sm().font_weight(FontWeight::SEMIBOLD).child("Lumen"))
                .child(self.menus(cx))
                .child(div().h(px(18.)).w(px(1.)).bg(color::border()))
                .child(self.workspaces(cx))
                .child(div().flex_1())
                .child(div().text_sm().text_color(color::text_muted()).child(if saved { name } else { format!("{name} · not saved") })),
        )
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
            Workspace::Pipeline => {
                let n = s.doc.graph.nodes.len();
                div().size_full().child(EmptyState::new(IconName::Workflow, "Pipeline canvas: next step", format!("The graph has {n} node(s). Its node editor (from dsp-kitchen, with typed ports) is being built; apply operations from the Operations menu meanwhile, their results show in Explore."))).into_any_element()
            }
            Workspace::Analyze => {
                let n = s.doc.annotations.len();
                div().size_full().child(EmptyState::new(IconName::Table2, "Analyze: next step", format!("{n} annotation(s). Measurement tables (annotations and objects from Measure nodes) come with the next build."))).into_any_element()
            }
        }
    }

    fn status_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let s = self.store.read(cx);
        let device = s.device.as_ref().map_or("GPU starts with the first result".to_string(), |d| d.label.clone());
        h_flex()
            .flex_none()
            .h(px(theme::STATUS_BAR_H))
            .px(px(theme::MD))
            .gap(px(theme::LG))
            .items_center()
            .bg(color::chrome())
            .border_t_1()
            .border_color(color::border())
            .text_xs()
            .child(div().flex_1().min_w_0().truncate().child(s.status.clone()))
            .child(div().text_color(color::text_muted()).child(device))
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

impl Render for LumenApp {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
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
            .on_action(cx.listener(|this, _: &CommandPalette, _, cx| this.store.update(cx, |s, cx| s.set_status("Search (Ctrl+K) comes with the next build; use the Operations menu", cx))))
            .on_action(cx.listener(|this, _: &ShowHelp, _, cx| {
                this.show_help = !this.show_help;
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &ToggleSidebar, _, cx| this.store.update(cx, |s, cx| {
                let open = !s.prefs.sidebar_open;
                s.set_sidebar(open, cx)
            })))
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
            .child(self.title_bar(cx))
            .child(div().flex_1().min_h_0().child(self.body(cx)))
            .child(self.status_bar(cx))
            .when(self.show_help, |d| d.child(self.help(cx)))
    }
}

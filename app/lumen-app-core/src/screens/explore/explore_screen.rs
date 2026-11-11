//! The Explore workspace (Affinity's layout on gpui-kit's dock, as dsp-kitchen's Explore): the
//! options bar on top, the tool column at the left edge, and one dock for everything else:
//! - the centre holds the views (each a tab, dragged into splits, or arranged at once);
//! - the left dock holds Layers, the right dock the studio (Display; Annotations, Results,
//!   Selection, Notes; Navigator, History); each side shows or hides, resizes, and every panel
//!   can be dragged to another group or side;
//! - the Pipeline is a panel too: shown under the focused view, and dragged anywhere.
//!
//! The focused view is the one the tools, the options bar, the Navigator and the menus act on;
//! views of one image can move and zoom together.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::base::dock::{DockArea, DockLayout, DockPlacement, InsertTarget, Panel as BasePanel, PanelEvent, PanelId, PanelInfo, PanelState};
use gpui_kit::base::Placement;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dock::{panel_handle, Panel};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, EntityId, EventEmitter, FocusHandle, Focusable, InteractiveElement as _, IntoElement, KeyDownEvent, MouseButton,
    ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, WeakEntity, Window,
};

use crate::data::store::{AppEvent, Arrangement, Store, ViewKind, ViewRequest};
use crate::screens::pipeline::PipelineView;
use crate::ui::theme::{self, color};
use crate::ui::widgets::EmptyState;

use crate::ui::container::{self, AddViewDialog, DialogEvent, ViewOption};

use super::canvas::ImageCanvas;
use super::panels::display::DisplayPanel;
use super::panels::navigator::Navigator;
use super::panels::notes::NotesPanel;
use super::panels::{annotations, histology, history, layers, metadata, results, selection, Tab};
use super::toolbar::{options_bar, overlay_toggles, tool_bar};
use super::view_panel::{PipelinePanel, ViewPanel};

/// Widths of the side docks and the Pipeline's height when first shown (px).
const RIGHT_W: f32 = 300.;
const PIPELINE_H: f32 = 320.;


pub struct ExploreView {
    store: Entity<Store>,
    /// The focused view's plane: what the tools, the options bar and the Navigator act on.
    pub canvas: Entity<ImageCanvas>,
    display: Entity<DisplayPanel>,
    notes: Entity<NotesPanel>,
    navigator: Entity<Navigator>,
    dock: Entity<DockArea>,
    pub(crate) panels: Vec<Entity<ViewPanel>>,
    studio: Vec<Entity<StudioPanel>>,
    /// How many views there are (the views read it to refuse closing the last one).
    count: Rc<Cell<usize>>,
    active: Option<EntityId>,
    pipeline: Entity<PipelinePanel>,
    pipeline_shown: bool,
    /// "Add view or container" is open; its keyboard focus and highlighted row.
    dialog: Option<(Entity<AddViewDialog>, Subscription)>,
    /// The project layout restored last (the Store's `layout_rev`), and the views saved last.
    layout_seen: u64,
    layout_saved: Option<Vec<String>>,
    /// The side panel shown beside the right rail (`None`: hidden), the last one shown, and its
    /// width (dragged).
    side: Option<Tab>,
    last_side: Tab,
    side_w: f32,
    _subs: Vec<Subscription>,
}

/// The right rail's panels, top to bottom (dsp-kitchen's rail: one icon per panel), and where a
/// divider goes (after the first three: what is used most).
const RAIL: [Tab; 6] = [Tab::Layers, Tab::Histology, Tab::Metadata, Tab::Selection, Tab::Notes, Tab::History];
const RAIL_SPLIT: usize = 3;
/// The Navigator's height at the bottom of the Layers panel (open).
const NAVIGATOR_H: f32 = 170.;
const RAIL_W: f32 = 36.;

/// Dragging the side panel's inner edge.
#[derive(Clone)]
struct SideResize;

impl Render for SideResize {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
    }
}

impl ExploreView {
    pub fn new(store: Entity<Store>, pipeline: Entity<PipelineView>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let me = cx.weak_entity();
        let count = Rc::new(Cell::new(1));
        let dock = container::new_dock("explore", window, cx);
        let first = cx.new(|cx| ViewPanel::new(store.clone(), me.clone(), ViewKind::Plane, count.clone(), true, cx));
        let canvas = first.read(cx).plane.clone();
        let display = cx.new(|cx| DisplayPanel::new(store.clone(), cx));
        let notes = cx.new(|cx| NotesPanel::new(store.clone(), canvas.clone(), window, cx));
        let navigator = cx.new(|cx| {
            let mut n = Navigator::new(store.clone(), cx);
            n.canvas = Some(canvas.downgrade());
            n
        });
        let pipeline = cx.new(|cx| PipelinePanel::new(pipeline, me.clone(), cx));
        let studio_panel = |tab: Tab, cx: &mut Context<Self>| cx.new(|cx| StudioPanel::new(tab, store.clone(), me.clone(), cx));
        let studio: Vec<Entity<StudioPanel>> = RAIL.iter().copied().map(|t| studio_panel(t, cx)).collect();
        let prefs = store.read(cx).prefs.clone();
        dock.update(cx, |d, cx| {
            d.set_center(DockLayout::tabs().panel_view(panel_handle(first.clone()), cx), window, cx);
        });
        let subs = vec![
            cx.subscribe_in(&store, window, |this: &mut Self, _, e: &AppEvent, window, cx| {
            if let AppEvent::View(req) = e {
                this.on_request(*req, window, cx);
            }
            if let AppEvent::Reveal(p) = e {
                let p = *p;
                this.canvas.update(cx, |c, cx| c.reveal(p, cx));
            }
            if !matches!(e, AppEvent::Status) {
                cx.notify();
            }
        }),
        ];
        let active = Some(first.entity_id());
        let mut this = Self {
            store,
            canvas,
            display,
            notes,
            navigator,
            dock,
            panels: vec![first],
            studio,
            count,
            active,
            pipeline,
            pipeline_shown: false,
            dialog: None,
            layout_seen: 0,
            layout_saved: None,
            side: prefs.right_open.then(|| prefs.right_tab.as_deref().and_then(Tab::from_key).filter(|t| RAIL.contains(t)).unwrap_or(Tab::Layers)),
            last_side: prefs.right_tab.as_deref().and_then(Tab::from_key).filter(|t| RAIL.contains(t)).unwrap_or(Tab::Layers),
            side_w: RIGHT_W,
            _subs: subs,
        };
        if prefs.pipeline_dock {
            this.set_pipeline(true, window, cx);
        }
        this
    }

    fn on_request(&mut self, req: ViewRequest, window: &mut Window, cx: &mut Context<Self>) {
        match req {
            ViewRequest::Show(kind) => self.show(kind, cx),
            ViewRequest::Add(kind) => self.add_view(kind, window, cx),
            ViewRequest::ClosePane => {
                if let Some(p) = self.active_panel() {
                    self.close_view(&p, window, cx);
                }
            }
            ViewRequest::Arrange(how) => self.arrange(how, window, cx),
            ViewRequest::TogglePipeline => {
                let on = !self.pipeline_shown;
                self.set_pipeline(on, window, cx);
            }
            ViewRequest::ToggleLink => self.store.update(cx, |s, cx| s.update_prefs(|p| p.link_views = !p.link_views, cx)),
            ViewRequest::ToggleLeft => self.show_side(Tab::Layers, cx),
            ViewRequest::ToggleRight | ViewRequest::ToggleSides => {
                let t = self.last_side;
                self.show_side(t, cx)
            }
            ViewRequest::ResetLayout => {
                self.side_w = RIGHT_W;
                self.side = None;
                self.show_side(Tab::Layers, cx);
                self.store.update(cx, |s, cx| s.update_prefs(|p| p.dock_sides = None, cx));
            }
        }
    }

    /// The focused view (the first when none is).
    fn active_panel(&self) -> Option<Entity<ViewPanel>> {
        self.panels.iter().find(|p| Some(p.entity_id()) == self.active).or(self.panels.first()).cloned()
    }

    /// Makes `panel` the focused view: the tools act on its plane, it asks for the results, the
    /// Navigator and the panels follow it.
    pub fn set_active(&mut self, panel: &Entity<ViewPanel>, cx: &mut Context<Self>) {
        let id = panel.entity_id();
        if self.active == Some(id) && self.canvas == panel.read(cx).plane {
            return;
        }
        self.active = Some(id);
        self.canvas = panel.read(cx).plane.clone();
        let planes: Vec<(bool, Entity<ImageCanvas>)> = self.panels.iter().map(|p| (p.entity_id() == id, p.read(cx).plane.clone())).collect();
        for (source, c) in planes {
            c.update(cx, |c, _| c.results_source = source);
        }
        self.canvas.update(cx, |c, cx| c.request_results(cx));
        let weak = self.canvas.downgrade();
        self.navigator.update(cx, |n, cx| {
            n.canvas = Some(weak);
            cx.notify();
        });
        for p in self.studio.clone() {
            p.update(cx, |_, cx| cx.notify());
        }
        cx.notify();
    }

    /// A view's camera moved: the Navigator follows, and linked views of the same image too.
    pub fn camera_moved(&mut self, from: &Entity<ViewPanel>, zoom: f64, center: (f64, f64), cx: &mut Context<Self>) {
        self.navigator.update(cx, |_, cx| cx.notify());
        if !self.store.read(cx).prefs.link_views {
            return;
        }
        let s = self.store.read(cx);
        let image = from.read(cx).plane.read(cx).image_id(s);
        let others: Vec<Entity<ImageCanvas>> = self.panels.iter().filter(|p| p.entity_id() != from.entity_id()).map(|p| p.read(cx).plane.clone()).filter(|c| c.read(cx).image_id(s) == image).collect();
        for c in others {
            c.update(cx, |c, cx| c.set_camera(zoom, center, cx));
        }
    }

    /// The focused view shows `kind`.
    fn show(&mut self, kind: ViewKind, cx: &mut Context<Self>) {
        if !ViewPanel::allowed(self.store.read(cx), kind) {
            self.store.update(cx, |s, cx| s.set_status("This image has no depth: the 2D and 3D views need a z axis", cx));
            return;
        }
        if let Some(p) = self.active_panel() {
            p.update(cx, |p, cx| p.set_kind(kind, cx));
        }
        cx.notify();
    }

    /// A new view of `kind` in its own group to the right of the focused one.
    pub fn add_view(&mut self, kind: ViewKind, window: &mut Window, cx: &mut Context<Self>) {
        let kind = if ViewPanel::allowed(self.store.read(cx), kind) { kind } else { ViewKind::Plane };
        let (store, me, count) = (self.store.clone(), cx.weak_entity(), self.count.clone());
        let panel = cx.new(|cx| ViewPanel::new(store, me, kind, count, false, cx));
        let focused = self.active_panel().map(|p| p.entity_id());
        self.panels.push(panel.clone());
        self.count.set(self.panels.len());
        container::add_beside(&self.dock, panel.clone(), focused.map(|f| (f, Placement::Right)), window, cx);
        self.set_active(&panel, cx);
    }

    /// Closes a view (the last one stays).
    pub fn close_view(&mut self, panel: &Entity<ViewPanel>, window: &mut Window, cx: &mut Context<Self>) {
        if self.panels.len() < 2 {
            self.store.update(cx, |s, cx| s.set_status("The last view stays open", cx));
            return;
        }
        self.dock.update(cx, |d, cx| d.remove_panel(panel.clone(), window, cx));
        self.removed(panel.entity_id(), cx);
    }

    /// A view left the dock (closed from its tab or its menu).
    pub fn removed(&mut self, id: EntityId, cx: &mut Context<Self>) {
        self.panels.retain(|p| p.entity_id() != id);
        self.count.set(self.panels.len());
        if self.active == Some(id) {
            self.active = None;
            if let Some(first) = self.panels.first().cloned() {
                self.set_active(&first, cx);
            }
        }
        cx.notify();
    }

    /// Lays the views out as `how`, in the order they were added (the Pipeline, when shown,
    /// stays under them).
    pub fn arrange(&mut self, how: Arrangement, window: &mut Window, cx: &mut Context<Self>) {
        let panels = self.panels.clone();
        if panels.is_empty() {
            return;
        }
        let views = container::arranged(&panels, how, cx);
        let layout = if self.pipeline_shown {
            DockLayout::v_split().child(views, None).child(DockLayout::tabs().panel_view(panel_handle(self.pipeline.clone()), cx), Some(px(PIPELINE_H)))
        } else {
            views
        };
        self.dock.update(cx, |d, cx| d.set_center(layout, window, cx));
        cx.notify();
    }

    /// Shows the Pipeline (under the focused view, as a panel the user can drag anywhere) or
    /// takes it away.
    fn set_pipeline(&mut self, on: bool, window: &mut Window, cx: &mut Context<Self>) {
        if on == self.pipeline_shown {
            return;
        }
        self.pipeline_shown = on;
        let pipeline = self.pipeline.clone();
        if on {
            let focused = self.active_panel().map(|p| p.entity_id());
            let id = PanelId::from(pipeline.entity_id());
            self.dock.update(cx, |d, cx| {
                let target = focused.and_then(|f| d.layout(DockPlacement::Center)?.find_panel_node(PanelId::from(f)));
                d.add_panel_view(panel_handle(pipeline), DockPlacement::Center, None, window, cx);
                if let Some(node) = target {
                    d.move_panel(id, InsertTarget::Split { node, placement: Placement::Bottom, size: Some(px(PIPELINE_H)) }, window, cx);
                }
            });
        } else {
            self.dock.update(cx, |d, cx| d.remove_panel(pipeline, window, cx));
        }
        self.store.update(cx, |s, cx| s.update_prefs(|p| p.pipeline_dock = on, cx));
        cx.notify();
    }

    /// The Pipeline's tab was closed.
    pub fn pipeline_closed(&mut self, cx: &mut Context<Self>) {
        if self.pipeline_shown {
            self.pipeline_shown = false;
            self.store.update(cx, |s, cx| s.update_prefs(|p| p.pipeline_dock = false, cx));
            cx.notify();
        }
    }

    /// Shows `tab` beside the right rail, or hides it when it is the one shown (remembered).
    pub fn show_side(&mut self, tab: Tab, cx: &mut Context<Self>) {
        self.side = if self.side == Some(tab) { None } else { Some(tab) };
        self.last_side = tab;
        let open = self.side.is_some();
        self.store.update(cx, |s, cx| {
            s.update_prefs(
                |p| {
                    p.right_tab = Some(tab.key().to_string());
                    p.right_open = open;
                },
                cx,
            )
        });
        cx.notify();
    }

    /// What a studio panel shows.
    fn tab_body(&self, tab: Tab, cx: &App) -> AnyElement {
        match tab {
            Tab::Layers => {
                // Editor of the selected layer, the tree, then the Navigator (folds)
                let open = !self.store.read(cx).layers_collapsed.contains("navigator");
                let st = self.store.clone();
                v_flex()
                    .size_full()
                    .child(div().id("layers-editor-scroll").flex_none().max_h(gpui_kit::relative(0.55)).overflow_y_scroll().border_b_1().border_color(color::border()).child(self.display.clone()))
                    .child(div().flex_1().min_h_0().child(layers::panel(&self.store, &self.canvas, cx)))
                    .child(
                        v_flex()
                            .flex_none()
                            .border_t_1()
                            .border_color(color::border())
                            .child(
                                h_flex()
                                    .id("navigator-fold")
                                    .h(px(28.))
                                    .px(px(theme::SM))
                                    .gap(px(theme::XS))
                                    .cursor_pointer()
                                    .hover(|d| d.bg(color::control()))
                                    .child(Icon::new(if open { IconName::ChevronDown } else { IconName::ChevronRight }).small())
                                    .child(Icon::new(IconName::Map).small().text_color(color::text_muted()))
                                    .child(div().text_sm().child("Navigator"))
                                    .on_click(move |_, _, cx| st.update(cx, |s, cx| s.toggle_layer_open("navigator", cx))),
                            )
                            .when(open, |d| d.child(div().h(px(NAVIGATOR_H)).child(self.navigator.clone()))),
                    )
                    .into_any_element()
            }
            Tab::Display => self.display.clone().into_any_element(),
            Tab::Annotations => annotations::panel(&self.store, &self.canvas, cx),
            Tab::Results => results::panel(&self.store, cx),
            Tab::Selection => selection::panel(&self.store, cx),
            Tab::Notes => self.notes.clone().into_any_element(),
            Tab::Navigator => self.navigator.clone().into_any_element(),
            Tab::History => history::panel(&self.store, cx),
            Tab::Histology => histology::panel(&self.store, cx),
            Tab::Metadata => metadata::panel(&self.store, cx),
        }
    }

    // ------------------------------------------------------------------------
    // "Add view or container" (the shared container's dialog)
    // ------------------------------------------------------------------------

    /// Explore's views for the dialog and the tabs (the 2D and 3D views need depth).
    pub fn view_options(store: &Store) -> Vec<ViewOption> {
        ViewKind::ALL
            .into_iter()
            .map(|k| {
                let hint = match k {
                    ViewKind::Plane => "One plane, full canvas; can show another image (its tab menu)",
                    ViewKind::Ortho => "The XY, XZ and YZ planes through the cursor",
                    ViewKind::Volume => "The volume rendered, with the annotations",
                    ViewKind::Channels => "Each channel on its own and all together, moving together",
                };
                let o = ViewOption::new(k.title(), k.icon(), hint);
                if ViewPanel::allowed(store, k) { o } else { o.unavailable("Needs an image with depth (a z axis)") }
            })
            .collect()
    }

    pub fn open_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let options = Self::view_options(self.store.read(cx));
        let can_arrange = !self.panels.is_empty();
        let dialog = cx.new(|cx| AddViewDialog::new(options, can_arrange, window, cx));
        let sub = cx.subscribe_in(&dialog, window, |this, _, e: &DialogEvent, window, cx| {
            this.dialog = None;
            match *e {
                DialogEvent::Arrange(how) => this.arrange(how, window, cx),
                DialogEvent::Add(i) => this.add_view(ViewKind::ALL[i], window, cx),
                DialogEvent::Close => {}
            }
            cx.notify();
        });
        self.dialog = Some((dialog, sub));
        cx.notify();
    }

    // ------------------------------------------------------------------------
    // Drawing
    // ------------------------------------------------------------------------

}

impl ExploreView {
    /// The views a project saved: the first view shows the first kind, one view is added for
    /// each of the others; views beyond them close.
    fn restore_layout(&mut self, kinds: Vec<String>, window: &mut Window, cx: &mut Context<Self>) {
        let kinds: Vec<ViewKind> = kinds.iter().filter_map(|t| ViewKind::ALL.iter().copied().find(|k| k.title() == t)).collect();
        let Some(first) = kinds.first().copied() else { return };
        for extra in self.panels.iter().skip(1).cloned().collect::<Vec<_>>() {
            self.dock.update(cx, |d, cx| d.remove_panel(extra.clone(), window, cx));
            self.removed(extra.entity_id(), cx);
        }
        if let Some(p) = self.panels.first().cloned() {
            p.update(cx, |p, cx| p.set_kind(first, cx));
        }
        // Its kind kept even while the image is still opening (adding by hand falls back to a
        // Plane when the image has no depth; a restored view says so itself when it has none)
        for kind in kinds.into_iter().skip(1) {
            self.add_view(kind, window, cx);
            if let Some(p) = self.panels.last().cloned() {
                p.update(cx, |p, cx| p.set_kind(kind, cx));
            }
        }
    }
}

impl Render for ExploreView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // A project brought its own views; otherwise the views, when changed, are saved
        let (rev, saved) = (self.store.read(cx).layout_rev, self.store.read(cx).layout.explore.views.clone());
        if rev != self.layout_seen {
            self.layout_seen = rev;
            self.layout_saved = Some(saved.clone());
            self.restore_layout(saved, window, cx);
        }
        let now: Vec<String> = self.panels.iter().map(|p| p.read(cx).kind.title().to_string()).collect();
        if self.layout_saved.as_ref() != Some(&now) {
            self.layout_saved = Some(now.clone());
            self.store.update(cx, |s, _| s.set_layout(|l| l.explore.views = now));
        }
        let s = self.store.read(cx);
        if s.doc.images.is_empty() && s.opening.is_empty() {
            return v_flex()
                .size_full()
                .child(EmptyState::new(IconName::Microscope, "No image yet", "Open an image or an OME-Zarr container from the File menu (Ctrl+O), or drop one on the window."))
                .into_any_element();
        }
        let toggles = overlay_toggles(&self.store, &self.canvas, cx).into_any_element();
        // The right rail (dsp-kitchen's): an icon per panel; the selected one shows beside it
        let me = cx.entity().downgrade();
        let mut rail = v_flex().flex_none().w(px(RAIL_W)).h_full().py(px(theme::XS)).gap(px(2.)).items_center().rounded(px(theme::RADIUS_LG)).bg(color::panel());
        for (i, tab) in RAIL.iter().copied().enumerate() {
            if i == RAIL_SPLIT {
                rail = rail.child(div().w(px(18.)).h(px(1.)).my(px(theme::XS)).bg(color::border()));
            }
            let me = me.clone();
            let on = self.side == Some(tab);
            let tip = match tab {
                Tab::Layers => "Layers (Ctrl+[)".to_string(),
                t => t.title().to_string(),
            };
            rail = rail.child(Button::new(SharedString::from(format!("rail-{}", tab.key()))).ghost().icon(Icon::new(tab.icon())).tooltip(tip).selected(on).on_click(move |_, _, cx| {
                let _ = me.update(cx, |e, cx| e.show_side(tab, cx));
            }));
        }
        let side_panel = self.side.and_then(|t| self.studio.iter().find(|p| p.read(cx).tab == t).cloned()).map(|p| {
            let title = p.read(cx).tab.title();
            v_flex()
                .relative()
                .flex_none()
                .w(px(self.side_w))
                .h_full()
                .rounded(px(theme::RADIUS_LG))
                .overflow_hidden()
                .bg(color::panel())
                .child(h_flex().flex_none().h(px(30.)).px(px(theme::MD)).items_center().border_b_1().border_color(color::border()).child(div().text_xs().font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(color::text_muted()).child(title.to_uppercase())))
                .child(div().flex_1().min_h_0().child(p))
                // Its inner edge resizes it
                .child(
                    div()
                        .id("side-resize")
                        .absolute()
                        .left(px(-3.))
                        .top_0()
                        .h_full()
                        .w(px(6.))
                        .cursor(gpui_kit::CursorStyle::ResizeLeftRight)
                        .on_drag(SideResize, |_, _, _, cx| cx.new(|_| SideResize)),
                )
        });
        v_flex()
            .size_full()
            .relative()
            .child(options_bar(&self.store, &self.canvas, toggles, cx))
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    // Panes are cards on the chrome, PANE_GAP apart (kitchen-ui)
                    .p(px(theme::PANE_GAP))
                    .gap(px(theme::PANE_GAP))
                    .bg(color::chrome())
                    .on_drag_move(cx.listener(|this, e: &gpui_kit::DragMoveEvent<SideResize>, _, cx| {
                        let w = f32::from(e.bounds.right() - e.event.position.x) - RAIL_W - 2.0 * theme::PANE_GAP;
                        this.side_w = w.clamp(220., 720.);
                        cx.notify();
                    }))
                    .child(tool_bar(&self.store, &self.canvas, cx))
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .rounded(px(theme::RADIUS_LG))
                            .overflow_hidden()
                            .bg(color::panel())
                            .child(div().relative().flex_1().min_h_0().w_full().child(self.dock.clone()).children(self.dialog.as_ref().map(|(d, _)| d.clone()))),
                    )
                    .children(side_panel)
                    .child(rail),
            )
            .into_any_element()
    }
}

// ----------------------------------------------------------------------------
// A panel of the studio, in a side dock
// ----------------------------------------------------------------------------

/// One studio tab (Layers, Display, …) as a dock panel: it can be dragged to another group or
/// side, and its dock resizes and hides.
pub struct StudioPanel {
    tab: Tab,
    explore: WeakEntity<ExploreView>,
    focus: FocusHandle,
    _subs: Vec<Subscription>,
}

impl StudioPanel {
    fn new(tab: Tab, store: Entity<Store>, explore: WeakEntity<ExploreView>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if !matches!(e, AppEvent::Status) {
                cx.notify();
            }
        })];
        Self { tab, explore, focus: cx.focus_handle(), _subs: subs }
    }
}

impl Render for StudioPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body = self.explore.upgrade().map(|e| e.read(cx).tab_body(self.tab, cx));
        // Panels taller than their group scroll
        // Delete in the panels without text fields removes what is selected there (as in the canvas)
        let deletes = matches!(self.tab, Tab::Layers | Tab::Results | Tab::Selection);
        let focus = self.focus.clone();
        let ex = self.explore.clone();
        div()
            .id("studio-body")
            .size_full()
            .overflow_y_scroll()
            .track_focus(&self.focus)
            .bg(color::panel())
            .when(deletes, |d| {
                d.on_mouse_down(MouseButton::Left, move |_, window, cx| window.focus(&focus, cx)).on_key_down(move |e: &KeyDownEvent, _, cx| {
                    if matches!(e.keystroke.key.as_str(), "delete" | "backspace") {
                        if let Some(ex) = ex.upgrade() {
                            let store = ex.read(cx).store.clone();
                            store.update(cx, |s, cx| s.delete_selection(cx));
                        }
                        cx.stop_propagation();
                    }
                })
            })
            .children(body)
    }
}

impl EventEmitter<PanelEvent> for StudioPanel {}

impl Focusable for StudioPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for StudioPanel {
    fn panel_name(&self) -> &'static str {
        "lumen-app.studio"
    }

    /// Which panel it is, so a saved layout puts it back where it was.
    fn dump(&self, _: &App) -> PanelState {
        let mut s = PanelState::new(self.panel_name());
        s.info = PanelInfo::panel(serde_json::json!({ "tab": self.tab.key() }));
        s
    }

    /// The studio's panels stay (a side hides with the bar's buttons).
    fn closable(&self, _: &App) -> bool {
        false
    }

    fn zoomable(&self, _: &App) -> bool {
        false
    }
}

impl Panel for StudioPanel {
    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        // Names only: icons truncate four tabs in a narrow dock
        self.tab.title()
    }

    /// No zoom control: with nothing else in it, the tab bar shows no "⋯".
    fn zoom_control(&self, _: &App) -> Option<gpui_kit::component::dock::PanelControl> {
        None
    }
}

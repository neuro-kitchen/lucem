//! The Explore workspace, laid out as dsp-kitchen's (VS Code / Zed): the options bar on top, an
//! icon rail at each side edge showing one panel at a time on that side (left: Display,
//! Annotations, Results; right: Selection, Notes), the image canvas in the middle with the tools
//! floating over it. A click on the shown tab's icon hides its side; Tab hides both.

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, Styled as _, Subscription, Window};

use crate::data::store::{AppEvent, Store, ViewKind, ViewRequest};
use crate::ui::theme::{self, color};
use crate::ui::widgets::EmptyState;

use super::canvas::ImageCanvas;
use super::panels::display::DisplayPanel;
use super::panels::notes::NotesPanel;
use super::panels::{annotations, layers, results, selection, Tab};
use super::toolbar::{options_bar, tool_bar};
use super::views::{OrthoView, VolumeView};

const LEFT_W: f32 = 296.;
const RIGHT_W: f32 = 296.;
const RAIL_W: f32 = 40.;
/// Views side by side at most.
const MAX_PANES: usize = 4;

/// One view of the image: what it shows, and its screens, made when first shown.
struct Pane {
    kind: ViewKind,
    plane: Entity<ImageCanvas>,
    ortho: Option<Entity<OrthoView>>,
    volume: Option<Entity<VolumeView>>,
}

pub struct ExploreView {
    store: Entity<Store>,
    pub canvas: Entity<ImageCanvas>,
    display: Entity<DisplayPanel>,
    notes: Entity<NotesPanel>,
    panes: Vec<Pane>,
    active_pane: usize,
    _subs: Vec<Subscription>,
}

impl ExploreView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let canvas = cx.new(|cx| ImageCanvas::new(store.clone(), cx));
        let display = cx.new(|cx| DisplayPanel::new(store.clone(), cx));
        let notes = cx.new(|cx| NotesPanel::new(store.clone(), canvas.clone(), window, cx));
        let subs = vec![cx.subscribe(&store, |this: &mut Self, _, e: &AppEvent, cx| {
            match e {
                AppEvent::View(ViewRequest::Show(kind)) => this.show(*kind, cx),
                AppEvent::View(ViewRequest::AddPane) => this.add_pane(cx),
                AppEvent::View(ViewRequest::ClosePane) => this.close_pane(cx),
                _ => {}
            }
            if !matches!(e, AppEvent::Status) {
                cx.notify();
            }
        })];
        let panes = vec![Pane { kind: ViewKind::Plane, plane: canvas.clone(), ortho: None, volume: None }];
        Self { store, canvas, display, notes, panes, active_pane: 0, _subs: subs }
    }

    /// Whether the data allows `kind`: the 2D and 3D views need depth.
    fn allowed(&self, kind: ViewKind, cx: &Context<Self>) -> bool {
        kind == ViewKind::Plane || self.store.read(cx).active_pyramid().is_some_and(|p| p.is_volume())
    }

    /// Switches the active view to `kind`.
    fn show(&mut self, kind: ViewKind, cx: &mut Context<Self>) {
        if !self.allowed(kind, cx) {
            self.store.update(cx, |s, cx| s.set_status("This image has no depth: the 2D and 3D views need a z axis", cx));
            return;
        }
        let store = self.store.clone();
        let p = &mut self.panes[self.active_pane];
        p.kind = kind;
        match kind {
            ViewKind::Ortho if p.ortho.is_none() => p.ortho = Some(cx.new(|cx| OrthoView::new(store, cx))),
            ViewKind::Volume if p.volume.is_none() => p.volume = Some(cx.new(|cx| VolumeView::new(store, cx))),
            _ => {}
        }
        cx.notify();
    }

    /// Adds a view beside the others (a plane, until switched).
    fn add_pane(&mut self, cx: &mut Context<Self>) {
        if self.panes.len() >= MAX_PANES {
            return;
        }
        let store = self.store.clone();
        let plane = cx.new(|cx| ImageCanvas::new(store, cx));
        self.panes.push(Pane { kind: ViewKind::Plane, plane, ortho: None, volume: None });
        self.active_pane = self.panes.len() - 1;
        cx.notify();
    }

    /// Closes the active view (the last one stays).
    fn close_pane(&mut self, cx: &mut Context<Self>) {
        if self.panes.len() > 1 {
            self.panes.remove(self.active_pane);
            self.active_pane = self.active_pane.min(self.panes.len() - 1);
            cx.notify();
        }
    }

    /// The view switcher above the panes.
    fn view_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let current = self.panes[self.active_pane].kind;
        let me = cx.entity().downgrade();
        let toggle = |id: &'static str, icon: IconName, tip: &'static str, kind: ViewKind, allowed: bool| {
            let me = me.clone();
            Button::new(id).ghost().small().icon(Icon::new(icon)).tooltip(if allowed { tip } else { "Needs an image with depth (a z axis)" }).selected(current == kind).disabled(!allowed).on_click(move |_, _, cx| {
                let _ = me.update(cx, |e, cx| e.show(kind, cx));
            })
        };
        let (ortho, volume) = (self.allowed(ViewKind::Ortho, cx), self.allowed(ViewKind::Volume, cx));
        let (add, close) = (me.clone(), me.clone());
        h_flex()
            .flex_none()
            .h(px(30.))
            .px(px(theme::SM))
            .gap(px(2.))
            .border_b_1()
            .border_color(color::border())
            .bg(color::chrome())
            .child(toggle("view-plane", IconName::Square, "Plane: one plane, full canvas", ViewKind::Plane, true))
            .child(toggle("view-ortho", IconName::LayoutGrid, "2D: the XY, XZ and YZ planes together", ViewKind::Ortho, ortho))
            .child(toggle("view-volume", IconName::Box, "3D: the volume rendered", ViewKind::Volume, volume))
            .child(div().flex_1())
            .child(Button::new("view-add").ghost().small().icon(Icon::new(IconName::Columns2)).tooltip("Add a view beside this one").disabled(self.panes.len() >= MAX_PANES).on_click(move |_, _, cx| {
                let _ = add.update(cx, |e, cx| e.add_pane(cx));
            }))
            .child(Button::new("view-close").ghost().small().icon(Icon::new(IconName::X)).tooltip("Close this view").disabled(self.panes.len() < 2).on_click(move |_, _, cx| {
                let _ = close.update(cx, |e, cx| e.close_pane(cx));
            }))
    }

    /// The views side by side; a click makes one the active view.
    fn pane_row(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let several = self.panes.len() > 1;
        let tools = self.panes[0].kind == ViewKind::Plane;
        let mut row = h_flex().flex_1().min_w_0().h_full();
        for (i, p) in self.panes.iter().enumerate() {
            let body: AnyElement = match p.kind {
                ViewKind::Plane => p.plane.clone().into_any_element(),
                ViewKind::Ortho => p.ortho.clone().map_or_else(|| div().into_any_element(), IntoElement::into_any_element),
                ViewKind::Volume => p.volume.clone().map_or_else(|| div().into_any_element(), IntoElement::into_any_element),
            };
            let me = cx.entity().downgrade();
            let active = several && i == self.active_pane;
            row = row.child(
                div()
                    .id(("view-pane", i))
                    .relative()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .border_color(if active { color::accent() } else { color::border() })
                    .when(several, |d| d.border_1())
                    .on_mouse_down(gpui_kit::MouseButton::Left, move |_, _, cx| {
                        let _ = me.update(cx, |e, cx| {
                            if e.active_pane != i {
                                e.active_pane = i;
                                cx.notify();
                            }
                        });
                    })
                    .child(body)
                    .when(i == 0 && tools, |d| d.child(tool_bar(&self.store, &self.canvas, cx))),
            );
        }
        row
    }

    fn tabs(&self, cx: &Context<Self>) -> (Option<Tab>, Option<Tab>) {
        let p = &self.store.read(cx).prefs;
        let open = p.sidebar_open;
        (p.left_tab.as_deref().and_then(Tab::from_key).filter(|_| open), p.right_tab.as_deref().and_then(Tab::from_key).filter(|_| open))
    }

    /// Shows `tab` on its side, or hides that side when it is the one shown.
    fn pick(&self, tab: Tab, cx: &mut Context<Self>) {
        let (left, right) = self.tabs(cx);
        let toggle = |cur: Option<Tab>| if cur == Some(tab) { None } else { Some(tab.key().to_string()) };
        let (l, r) = if Tab::LEFT.contains(&tab) { (toggle(left), right.map(|t| t.key().to_string())) } else { (left.map(|t| t.key().to_string()), toggle(right)) };
        self.store.update(cx, |s, cx| {
            s.prefs.sidebar_open = true;
            s.set_side_tabs(l, r, cx)
        });
    }

    fn rail(&self, tabs: &[Tab], shown: Option<Tab>, left: bool, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let me = cx.entity().downgrade();
        let bar = v_flex().flex_none().w(px(RAIL_W)).h_full().py(px(theme::SM)).gap(px(theme::XS)).items_center().bg(color::chrome()).border_color(color::border());
        let bar = if left { bar.border_r_1() } else { bar.border_l_1() };
        bar.children(tabs.iter().copied().map(|t| {
            let me = me.clone();
            Button::new(SharedString::from(format!("rail-{}", t.key()))).ghost().icon(Icon::new(t.icon())).tooltip(t.title()).selected(shown == Some(t)).on_click(move |_, _, cx| {
                let _ = me.update(cx, |e, cx| e.pick(t, cx));
            })
        }))
    }

    fn panel(&self, tab: Tab, width: f32, left: bool, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let body: AnyElement = match tab {
            Tab::Layers => layers::panel(&self.store, cx),
            Tab::Display => self.display.clone().into_any_element(),
            Tab::Annotations => annotations::panel(&self.store, &self.canvas, cx),
            Tab::Results => results::panel(&self.store, cx),
            Tab::Selection => selection::panel(&self.store, cx),
            Tab::Notes => self.notes.clone().into_any_element(),
        };
        let me = cx.entity().downgrade();
        let hide = Button::new(SharedString::from(format!("hide-{}", tab.key())))
            .ghost()
            .icon(Icon::new(if left { IconName::PanelLeftClose } else { IconName::PanelRightClose }))
            .tooltip(format!("Hide {}", tab.title()))
            .on_click(move |_, _, cx| {
                let _ = me.update(cx, |e, cx| e.pick(tab, cx));
            });
        let panel = v_flex()
            .flex_none()
            .w(px(width))
            .h_full()
            .bg(color::panel())
            .border_color(color::border())
            .child(
                h_flex()
                    .flex_none()
                    .h(px(32.))
                    .px(px(theme::SM))
                    .border_b_1()
                    .border_color(color::border())
                    .child(div().pl(px(theme::XS)).text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child(tab.title().to_uppercase()))
                    .child(div().flex_1())
                    .child(hide),
            )
            .child(div().flex_1().min_h_0().child(body));
        if left { panel.border_r_1() } else { panel.border_l_1() }
    }
}

impl Render for ExploreView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let s = self.store.read(cx);
        if s.doc.images.is_empty() && s.opening.is_empty() {
            return v_flex()
                .size_full()
                .child(EmptyState::new(IconName::Microscope, "No image yet", "Open an image or an OME-Zarr container from the File menu (Ctrl+O), or drop one on the window."))
                .into_any_element();
        }
        let (left, right) = self.tabs(cx);
        v_flex()
            .size_full()
            .child(options_bar(&self.store, &self.canvas, cx))
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .child(self.rail(&Tab::LEFT, left, true, cx))
                    .children(left.map(|t| self.panel(t, LEFT_W, true, cx)))
                    .child(v_flex().flex_1().min_w_0().h_full().child(self.view_bar(cx)).child(div().flex_1().min_h_0().w_full().flex().child(self.pane_row(cx))))
                    .children(right.map(|t| self.panel(t, RIGHT_W, false, cx)))
                    .child(self.rail(&Tab::RIGHT, right, false, cx)),
            )
            .into_any_element()
    }
}

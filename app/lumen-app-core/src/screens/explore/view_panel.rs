//! What Explore's dock holds (as dsp-kitchen's time views): each [`ViewPanel`] is one view of an
//! image (a plane, the three planes, the volume, or the channels side by side) in a tab the user
//! drags into splits; [`PipelinePanel`] is the Pipeline docked under the views (Blender: the same
//! editor in its own workspace or as an area of this one).

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::base::dock::{Panel as BasePanel, PanelEvent};
use gpui_kit::component::dock::{Panel, PanelControl};
use gpui_kit::component::menu::{PopupMenu, PopupMenuItem};
use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable, InteractiveElement as _, IntoElement, MouseButton, ParentElement as _, Render, SharedString,
    Styled as _, Subscription, WeakEntity, Window,
};

use crate::data::store::{AppEvent, Store, ViewKind};
use crate::screens::pipeline::PipelineView;
use crate::ui::theme::{self, color};

use super::canvas::{CanvasEvent, ImageCanvas};
use super::explore_screen::ExploreView;
use super::views::{OrthoView, VolumeView};

/// Tile reader threads of a view that is not the first.
const SIDE_VIEW_THREADS: usize = 2;

/// One view in the dock.
pub struct ViewPanel {
    store: Entity<Store>,
    explore: WeakEntity<ExploreView>,
    pub kind: ViewKind,
    /// The plane (shown for [`ViewKind::Plane`]; also the canvas the tools act on).
    pub plane: Entity<ImageCanvas>,
    ortho: Option<Entity<OrthoView>>,
    volume: Option<Entity<VolumeView>>,
    channels: Option<Entity<ChannelsView>>,
    /// How many views Explore has (the last one cannot be closed).
    count: Rc<Cell<usize>>,
    focus: FocusHandle,
    _subs: Vec<Subscription>,
}

impl ViewPanel {
    pub fn new(store: Entity<Store>, explore: WeakEntity<ExploreView>, kind: ViewKind, count: Rc<Cell<usize>>, first: bool, cx: &mut Context<Self>) -> Self {
        let readers = store.read(cx).prefs.tile_threads;
        let plane = cx.new(|cx| {
            let mut c = ImageCanvas::new(store.clone(), cx);
            if !first {
                c.threads = SIDE_VIEW_THREADS;
            } else if readers > 0 {
                c.threads = readers;
            }
            // The app's bottom bar shows the focused view's position, level and zoom
            c.footer = false;
            c
        });
        let me = cx.entity().downgrade();
        let ex = explore.clone();
        let subs = vec![
            cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify()),
            // Linked views follow this one's camera
            cx.subscribe(&plane, move |_, _, e: &CanvasEvent, cx| {
                let (CanvasEvent::Camera { zoom, center }, me, ex) = (*e, me.clone(), ex.clone());
                cx.defer(move |cx| {
                    if let (Some(ex), Some(me)) = (ex.upgrade(), me.upgrade()) {
                        ex.update(cx, |e, cx| e.camera_moved(&me, zoom, center, cx));
                    }
                });
            }),
        ];
        let mut this = Self { store, explore, kind: ViewKind::Plane, plane, ortho: None, volume: None, channels: None, count, focus: cx.focus_handle(), _subs: subs };
        this.set_kind(kind, cx);
        this
    }

    /// Whether the shown image allows `kind` (the 2D and 3D views need depth).
    pub fn allowed(store: &Store, kind: ViewKind) -> bool {
        !kind.needs_depth() || store.active_pyramid().is_some_and(|p| p.is_volume())
    }

    /// Shows `kind`, making its screen the first time.
    pub fn set_kind(&mut self, kind: ViewKind, cx: &mut Context<Self>) {
        let store = self.store.clone();
        match kind {
            ViewKind::Ortho if self.ortho.is_none() => self.ortho = Some(cx.new(|cx| OrthoView::new(store, cx))),
            ViewKind::Volume if self.volume.is_none() => self.volume = Some(cx.new(|cx| VolumeView::new(store, cx))),
            ViewKind::Channels if self.channels.is_none() => self.channels = Some(cx.new(|cx| ChannelsView::new(store, cx))),
            _ => {}
        }
        self.kind = kind;
        cx.notify();
    }

    /// The tab's title: the kind and the image.
    fn title_text(&self, cx: &App) -> String {
        let s = self.store.read(cx);
        let id = if self.kind == ViewKind::Plane { self.plane.read(cx).image_id(s) } else { s.doc.active_image.clone() };
        let name = id.and_then(|i| s.doc.image(&i).map(|i| i.name.clone()));
        let pinned = self.kind == ViewKind::Plane && self.plane.read(cx).pinned.is_some();
        match name {
            Some(n) => format!("{} · {n}{}", self.kind.title(), if pinned { " (pinned)" } else { "" }),
            None => self.kind.title().to_string(),
        }
    }

    /// Tells Explore this view is the one the tools and menus act on.
    fn focus_me(&self, cx: &mut Context<Self>) {
        let (ex, me) = (self.explore.clone(), cx.entity().downgrade());
        cx.defer(move |cx| {
            if let (Some(ex), Some(me)) = (ex.upgrade(), me.upgrade()) {
                ex.update(cx, |e, cx| e.set_active(&me, cx));
            }
        });
    }
}

impl Render for ViewPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let s = self.store.read(cx);
        let body: AnyElement = if !Self::allowed(s, self.kind) {
            v_flex().size_full().items_center().justify_center().text_sm().text_color(color::text_muted()).child(format!("{}: this image has no depth (no z axis)", self.kind.title())).into_any_element()
        } else {
            match self.kind {
                ViewKind::Plane => self.plane.clone().into_any_element(),
                ViewKind::Ortho => self.ortho.clone().map_or_else(|| div().into_any_element(), IntoElement::into_any_element),
                ViewKind::Volume => self.volume.clone().map_or_else(|| div().into_any_element(), IntoElement::into_any_element),
                ViewKind::Channels => self.channels.clone().map_or_else(|| div().into_any_element(), IntoElement::into_any_element),
            }
        };
        div()
            .id("view-panel")
            .track_focus(&self.focus)
            .size_full()
            .bg(color::canvas())
            .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| this.focus_me(cx)))
            .child(body)
    }
}

impl EventEmitter<PanelEvent> for ViewPanel {}

impl Focusable for ViewPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for ViewPanel {
    fn panel_name(&self) -> &'static str {
        "lumen-app.view"
    }

    fn closable(&self, _: &App) -> bool {
        self.count.get() > 1
    }

    fn set_active(&mut self, active: bool, _: &mut Window, cx: &mut Context<Self>) {
        if active {
            self.focus_me(cx);
        }
    }

    fn on_removed(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let (ex, id) = (self.explore.clone(), cx.entity_id());
        cx.defer(move |cx| {
            if let Some(ex) = ex.upgrade() {
                ex.update(cx, |e, cx| e.removed(id, cx));
            }
        });
    }
}

impl Panel for ViewPanel {
    fn title(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let text = SharedString::from(self.title_text(cx));
        div().max_w(px(280.)).truncate().child(text)
    }

    fn zoom_control(&self, _: &App) -> Option<PanelControl> {
        Some(PanelControl::Toolbar)
    }

    /// In the view's tab: what it shows (Plane, 2D, 3D, Channels) and "Add view…".
    fn toolbar_buttons(&mut self, _: &mut Window, cx: &mut Context<Self>) -> Option<Vec<Button>> {
        let s = self.store.read(cx);
        let me = cx.entity().downgrade();
        let mut out: Vec<Button> = ViewKind::ALL
            .into_iter()
            .map(|kind| {
                let (me, allowed) = (me.clone(), Self::allowed(s, kind));
                Button::new(SharedString::from(format!("view-kind-{}", kind.title())))
                    .ghost()
                    .xsmall()
                    .icon(Icon::new(kind.icon()))
                    .tooltip(if allowed { format!("{}: {}", kind.title(), kind.hint()) } else { format!("{}: needs an image with depth (a z axis)", kind.title()) })
                    .selected(self.kind == kind)
                    .disabled(!allowed)
                    .on_click(move |_, _, cx| {
                        let _ = me.update(cx, |p, cx| {
                            p.set_kind(kind, cx);
                            p.focus_me(cx);
                        });
                    })
            })
            .collect();
        let ex = self.explore.clone();
        out.push(Button::new("view-add").ghost().xsmall().icon(Icon::new(IconName::Plus)).tooltip("Add a view, or arrange the views").on_click(move |_, window, cx| {
            let _ = ex.update(cx, |e, cx| e.open_dialog(window, cx));
        }));
        Some(out)
    }

    /// Switch what it shows, pin an image, add or close a view.
    fn dropdown_menu(&mut self, menu: PopupMenu, _: &mut Window, cx: &mut Context<Self>) -> PopupMenu {
        let s = self.store.read(cx);
        let me = cx.entity().downgrade();
        let mut m = menu.label("Show");
        for kind in ViewKind::ALL {
            let me = me.clone();
            m = m.item(PopupMenuItem::new(kind.title()).checked(self.kind == kind).disabled(!Self::allowed(s, kind)).on_click(move |_, _, cx| {
                let _ = me.update(cx, |p, cx| p.set_kind(kind, cx));
            }));
        }
        // A plane can keep showing one image while another is selected (two images side by side)
        m = m.separator().label("Image");
        let pinned = self.plane.read(cx).pinned.clone();
        let canvas = self.plane.clone();
        {
            let canvas = canvas.clone();
            m = m.item(PopupMenuItem::new("The selected image").checked(pinned.is_none()).on_click(move |_, _, cx| {
                canvas.update(cx, |c, cx| {
                    c.pinned = None;
                    cx.notify();
                })
            }));
        }
        for img in &s.doc.images {
            let (canvas, id) = (canvas.clone(), img.id.clone());
            m = m.item(PopupMenuItem::new(img.name.clone()).checked(pinned.as_deref() == Some(img.id.as_str())).on_click(move |_, _, cx| {
                canvas.update(cx, |c, cx| {
                    c.pinned = Some(id.clone());
                    cx.notify();
                })
            }));
        }
        let ex = self.explore.clone();
        let close = self.explore.clone();
        let me2 = cx.entity().downgrade();
        m.separator()
            .item(PopupMenuItem::new("Add a view beside").on_click(move |_, window, cx| {
                let _ = ex.update(cx, |e, cx| e.add_view(ViewKind::Plane, window, cx));
            }))
            .item(PopupMenuItem::new("Close view").on_click(move |_, window, cx| {
                if let Some(me) = me2.upgrade() {
                    let _ = close.update(cx, |e, cx| e.close_view(&me, window, cx));
                }
            }))
    }
}

// ----------------------------------------------------------------------------
// Channels: each channel on its own, and all together
// ----------------------------------------------------------------------------

/// A grid of planes: one per channel, then the channels together; they move and zoom together.
pub struct ChannelsView {
    store: Entity<Store>,
    /// The image and channel count the cells were made for.
    made_for: Option<(String, usize)>,
    cells: Vec<(Option<usize>, Entity<ImageCanvas>)>,
    _subs: Vec<Subscription>,
    cell_subs: Vec<Subscription>,
}

impl ChannelsView {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, _: &AppEvent, cx| cx.notify())];
        Self { store, made_for: None, cells: Vec::new(), _subs: subs, cell_subs: Vec::new() }
    }

    /// One cell per channel and one for all of them, made again when the image changes.
    fn sync(&mut self, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let want = s.doc.active().map(|i| (i.id.clone(), i.channels.len()));
        if want == self.made_for {
            return;
        }
        self.made_for = want.clone();
        self.cells.clear();
        self.cell_subs.clear();
        let Some((_, n)) = want else { return };
        let store = self.store.clone();
        for ch in (0..n).map(Some).chain(std::iter::once(None)) {
            let canvas = cx.new(|cx| {
                let mut c = ImageCanvas::new(store.clone(), cx);
                c.channel = ch;
                c.compact = true;
                c.results_source = false;
                c.threads = SIDE_VIEW_THREADS;
                c
            });
            self.cells.push((ch, canvas));
        }
        // Every cell follows the one that moved
        let all: Vec<Entity<ImageCanvas>> = self.cells.iter().map(|(_, c)| c.clone()).collect();
        for c in &all {
            let others = all.clone();
            self.cell_subs.push(cx.subscribe(c, move |_, from, e: &CanvasEvent, cx| {
                let CanvasEvent::Camera { zoom, center } = *e;
                let others: Vec<Entity<ImageCanvas>> = others.iter().filter(|o| o.entity_id() != from.entity_id()).cloned().collect();
                cx.defer(move |cx| {
                    for o in others {
                        o.update(cx, |c, cx| c.set_camera(zoom, center, cx));
                    }
                });
            }));
        }
    }
}

impl Render for ChannelsView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.sync(cx);
        let s = self.store.read(cx);
        let Some(img) = s.doc.active().cloned() else { return v_flex().size_full().into_any_element() };
        let n = self.cells.len().max(1);
        let cols = (n as f64).sqrt().ceil().max(1.0) as usize;
        let mut grid = v_flex().size_full();
        for row in self.cells.chunks(cols) {
            let mut r = div().flex().flex_row().flex_1().min_h_0();
            for (ch, canvas) in row {
                let (name, color) = match ch {
                    Some(c) => img.channels.get(*c).map_or((format!("Channel {c}"), [255, 255, 255]), |d| (d.label.clone(), d.color)),
                    None => ("All channels".to_string(), [255, 255, 255]),
                };
                r = r.child(
                    div()
                        .relative()
                        .flex_1()
                        .min_w_0()
                        .h_full()
                        .border_1()
                        .border_color(color::border())
                        .child(canvas.clone())
                        .child(
                            h_flex()
                                .absolute()
                                .top(px(theme::SM))
                                .left(px(theme::SM))
                                .gap(px(theme::XS))
                                .px(px(theme::SM))
                                .py(px(2.))
                                .rounded(px(4.))
                                .bg(gpui_kit::black().opacity(0.55))
                                .child(div().size(px(8.)).rounded(px(2.)).bg(theme::rgb8(color)))
                                .child(div().text_xs().text_color(gpui_kit::white()).child(name)),
                        ),
                );
            }
            // Empty cells keep the columns even
            for _ in row.len()..cols {
                r = r.child(div().flex_1().min_w_0().h_full());
            }
            grid = grid.child(r);
        }
        grid.into_any_element()
    }
}

// ----------------------------------------------------------------------------
// The Pipeline, docked under the views
// ----------------------------------------------------------------------------

/// The Pipeline workspace's editor as a panel of Explore's dock, dragged anywhere like a view
/// (never on screen in both places: one workspace is shown at a time).
pub struct PipelinePanel {
    pipeline: Entity<PipelineView>,
    explore: WeakEntity<ExploreView>,
    focus: FocusHandle,
}

impl PipelinePanel {
    pub fn new(pipeline: Entity<PipelineView>, explore: WeakEntity<ExploreView>, cx: &mut Context<Self>) -> Self {
        Self { pipeline, explore, focus: cx.focus_handle() }
    }
}

impl Render for PipelinePanel {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().track_focus(&self.focus).child(self.pipeline.clone())
    }
}

impl EventEmitter<PanelEvent> for PipelinePanel {}

impl Focusable for PipelinePanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for PipelinePanel {
    fn panel_name(&self) -> &'static str {
        "lumen-app.pipeline"
    }

    /// Its tab closed: Explore's Pipeline button follows.
    fn on_removed(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let ex = self.explore.clone();
        cx.defer(move |cx| {
            if let Some(ex) = ex.upgrade() {
                ex.update(cx, |e, cx| e.pipeline_closed(cx));
            }
        });
    }
}

impl Panel for PipelinePanel {
    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        "Pipeline"
    }
}

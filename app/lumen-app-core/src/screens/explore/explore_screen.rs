//! The Explore workspace, laid out as dsp-kitchen's (VS Code / Zed): the options bar on top, an
//! icon rail at each side edge showing one panel at a time on that side (left: Display,
//! Annotations, Results; right: Selection, Notes), the image canvas in the middle with the tools
//! floating over it. A click on the shown tab's icon hides its side; Tab hides both.

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _};
use gpui_kit::{div, px, AnyElement, AppContext as _, Context, Entity, FontWeight, IntoElement, ParentElement as _, Render, SharedString, Styled as _, Subscription, Window};

use crate::data::store::{AppEvent, Store};
use crate::ui::theme::{self, color};
use crate::ui::widgets::EmptyState;

use super::canvas::ImageCanvas;
use super::panels::display::DisplayPanel;
use super::panels::notes::NotesPanel;
use super::panels::{annotations, results, selection, Tab};
use super::toolbar::{options_bar, tool_bar};

const LEFT_W: f32 = 296.;
const RIGHT_W: f32 = 296.;
const RAIL_W: f32 = 40.;

pub struct ExploreView {
    store: Entity<Store>,
    pub canvas: Entity<ImageCanvas>,
    display: Entity<DisplayPanel>,
    notes: Entity<NotesPanel>,
    _subs: Vec<Subscription>,
}

impl ExploreView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let canvas = cx.new(|cx| ImageCanvas::new(store.clone(), cx));
        let display = cx.new(|cx| DisplayPanel::new(store.clone(), cx));
        let notes = cx.new(|cx| NotesPanel::new(store.clone(), canvas.clone(), window, cx));
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if !matches!(e, AppEvent::Status) {
                cx.notify();
            }
        })];
        Self { store, canvas, display, notes, _subs: subs }
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
                    .child(div().relative().flex_1().min_w_0().h_full().child(self.canvas.clone()).child(tool_bar(&self.store, &self.canvas, cx)))
                    .children(right.map(|t| self.panel(t, RIGHT_W, false, cx)))
                    .child(self.rail(&Tab::RIGHT, right, false, cx)),
            )
            .into_any_element()
    }
}

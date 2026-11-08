//! Small pieces used by several views (from dsp-kitchen): a dropdown of fixed options, a titled
//! section, a labelled row, a centred empty state, an icon button, a number stepper, a colour
//! swatch, a panel header. Plain data in, no store.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, App, ElementId, FontWeight, InteractiveElement as _, IntoElement, ParentElement, RenderOnce, SharedString, StatefulInteractiveElement as _, Styled as _, Window};

use super::theme::{self, color};

type OnPick = Rc<dyn Fn(usize, &mut Window, &mut App)>;

/// A button showing the current option; pressing it lists the options.
#[derive(IntoElement)]
pub struct MenuSelect {
    id: ElementId,
    current: SharedString,
    options: Vec<SharedString>,
    selected: Option<usize>,
    on_pick: OnPick,
    full: bool,
}

impl MenuSelect {
    pub fn new(id: impl Into<ElementId>, options: Vec<SharedString>, selected: Option<usize>, on_pick: impl Fn(usize, &mut Window, &mut App) + 'static) -> Self {
        let current = selected.and_then(|i| options.get(i).cloned()).unwrap_or_else(|| "Choose…".into());
        Self { id: id.into(), current, options, selected, on_pick: Rc::new(on_pick), full: false }
    }

    /// Spans its row.
    pub fn full_width(mut self) -> Self {
        self.full = true;
        self
    }
}

impl RenderOnce for MenuSelect {
    fn render(self, _: &mut Window, _: &mut App) -> impl IntoElement {
        let Self { id, current, options, selected, on_pick, full } = self;
        let face = h_flex().w_full().min_w_0().gap_1().child(div().flex_1().min_w_0().truncate().text_left().child(current)).child(Icon::new(IconName::ChevronDown).xsmall());
        Button::new(id).outline().small().min_w_0().max_w_full().child(face).cursor_pointer().when(full, |b| b.w_full()).dropdown_menu(move |menu, _, _| {
            let mut menu = menu.scrollable(true).max_h(px(360.));
            for (i, label) in options.iter().enumerate() {
                let on_pick = on_pick.clone();
                menu = menu.item(PopupMenuItem::new(label.clone()).checked(selected == Some(i)).on_click(move |_, window, cx| on_pick(i, window, cx)));
            }
            menu
        })
    }
}

/// A titled block of a panel: a small caps title, then its rows.
#[derive(IntoElement)]
pub struct Section {
    title: SharedString,
    extra: Option<AnyElement>,
    children: Vec<AnyElement>,
}

impl Section {
    pub fn new(title: impl Into<SharedString>) -> Self {
        Self { title: title.into(), extra: None, children: Vec::new() }
    }

    /// Something at the right of the title (a count, a button).
    pub fn extra(mut self, e: impl IntoElement) -> Self {
        self.extra = Some(e.into_any_element());
        self
    }
}

impl ParentElement for Section {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.children.extend(elements);
    }
}

impl RenderOnce for Section {
    fn render(self, _: &mut Window, _: &mut App) -> impl IntoElement {
        v_flex()
            .w_full()
            .gap(px(theme::SM))
            .child(
                h_flex()
                    .h(px(20.))
                    .gap(px(theme::SM))
                    .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child(self.title.to_uppercase()))
                    .child(div().flex_1())
                    .children(self.extra),
            )
            .child(v_flex().w_full().gap(px(theme::SM)).children(self.children))
    }
}

/// `label    control` on one line, never wider than its panel.
pub fn row(label: impl Into<SharedString>, control: impl IntoElement) -> impl IntoElement {
    h_flex()
        .w_full()
        .min_w_0()
        .gap(px(theme::SM))
        .child(div().flex_none().w(px(96.)).truncate().text_sm().text_color(color::text_muted()).child(label.into()))
        .child(h_flex().flex_1().min_w_0().justify_end().overflow_hidden().child(control))
}

/// Centred icon, title and a line of explanation, with actions under it.
#[derive(IntoElement)]
pub struct EmptyState {
    icon: IconName,
    title: SharedString,
    description: SharedString,
    actions: Vec<AnyElement>,
}

impl EmptyState {
    pub fn new(icon: IconName, title: impl Into<SharedString>, description: impl Into<SharedString>) -> Self {
        Self { icon, title: title.into(), description: description.into(), actions: Vec::new() }
    }
}

impl ParentElement for EmptyState {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.actions.extend(elements);
    }
}

impl RenderOnce for EmptyState {
    fn render(self, _: &mut Window, _: &mut App) -> impl IntoElement {
        v_flex()
            .size_full()
            .items_center()
            .justify_center()
            .gap(px(theme::MD))
            .child(Icon::new(self.icon).size(px(32.)).text_color(color::text_muted()))
            .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child(self.title))
            .child(div().max_w(px(460.)).text_sm().text_center().text_color(color::text_muted()).child(self.description))
            .child(h_flex().pt(px(theme::SM)).gap(px(theme::SM)).children(self.actions))
    }
}

/// A small ghost icon button with a tooltip.
pub fn icon_button(id: impl Into<ElementId>, icon: impl Into<Icon>, tooltip: impl Into<SharedString>) -> Button {
    Button::new(id).ghost().small().icon(icon).tooltip(tooltip.into())
}

/// `− value +`: a number changed in steps.
pub fn stepper(id: &str, value: impl Into<SharedString>, on_step: impl Fn(f32, &mut Window, &mut App) + 'static) -> impl IntoElement {
    let on_step = Rc::new(on_step);
    let (a, b) = (on_step.clone(), on_step);
    h_flex()
        .gap(px(2.))
        .child(Button::new(SharedString::from(format!("{id}-minus"))).ghost().xsmall().icon(IconName::Minus).on_click(move |_, w, cx| a(-1.0, w, cx)))
        .child(div().min_w(px(52.)).text_center().text_sm().font_family("monospace").child(value.into()))
        .child(Button::new(SharedString::from(format!("{id}-plus"))).ghost().xsmall().icon(IconName::Plus).on_click(move |_, w, cx| b(1.0, w, cx)))
}

/// A slider over positions `0..count` (a position along an axis of an image). Dragging anywhere
/// on the track sets the position.
pub fn axis_slider(id: String, value: usize, count: usize, on_change: impl Fn(usize, &mut Window, &mut App) + 'static) -> impl IntoElement {
    use std::cell::Cell;
    let bounds = Rc::new(Cell::new(gpui_kit::Bounds::<gpui_kit::Pixels>::default()));
    let on_change = Rc::new(on_change);
    let frac = if count > 1 { value as f32 / (count - 1) as f32 } else { 0.0 };
    let at = move |x: gpui_kit::Pixels, b: gpui_kit::Bounds<gpui_kit::Pixels>| -> usize {
        let f = ((x - b.origin.x).as_f32() / b.size.width.as_f32().max(1.0)).clamp(0.0, 1.0);
        ((f * (count.saturating_sub(1)) as f32).round() as usize).min(count.saturating_sub(1))
    };
    let (keep, down, moved) = (bounds.clone(), bounds.clone(), bounds);
    let (on_down, on_moved) = (on_change.clone(), on_change);
    div()
        .id(SharedString::from(id))
        .relative()
        .w_full()
        .h(px(18.))
        .cursor_pointer()
        .child(gpui_kit::canvas(
            move |b, _, _| keep.set(b),
            move |b, _, window, _| {
                let y = b.origin.y + px(b.size.height.as_f32() / 2.0 - 1.5);
                let track = gpui_kit::Bounds { origin: gpui_kit::point(b.origin.x, y), size: gpui_kit::size(b.size.width, px(3.)) };
                window.paint_quad(gpui_kit::fill(track, color::control()));
                let filled = gpui_kit::Bounds { origin: track.origin, size: gpui_kit::size(px(b.size.width.as_f32() * frac), px(3.)) };
                window.paint_quad(gpui_kit::fill(filled, color::accent()));
                let knob = gpui_kit::Bounds { origin: gpui_kit::point(b.origin.x + px(b.size.width.as_f32() * frac - 5.0), b.origin.y + px(b.size.height.as_f32() / 2.0 - 5.0)), size: gpui_kit::size(px(10.), px(10.)) };
                window.paint_quad(gpui_kit::fill(knob, color::accent_hover()));
            },
        ).absolute().size_full())
        .on_mouse_down(gpui_kit::MouseButton::Left, move |e, w, cx| {
            cx.stop_propagation();
            on_down(at(e.position.x, down.get()), w, cx);
        })
        .on_mouse_move(move |e, w, cx| {
            if e.pressed_button == Some(gpui_kit::MouseButton::Left) {
                on_moved(at(e.position.x, moved.get()), w, cx);
            }
        })
}

/// A colour chip.
pub fn swatch(c: [u8; 3], size: f32) -> impl IntoElement {
    div().flex_none().size(px(size)).rounded(px(3.)).border_1().border_color(color::border()).bg(theme::rgb8(c))
}

/// A thin vertical divider for bars.
pub fn divider_v() -> impl IntoElement {
    div().flex_none().w(px(1.)).h(px(18.)).mx(px(theme::XS)).bg(color::border())
}

/// A clickable list row: icon, label, trailing text; highlighted when selected.
pub fn list_row(id: impl Into<ElementId>, icon: Option<IconName>, label: impl Into<SharedString>, trailing: impl Into<SharedString>, selected: bool, cx: &App) -> gpui_kit::Stateful<gpui_kit::Div> {
    let t = cx.theme();
    h_flex()
        .id(id)
        .w_full()
        .h(px(28.))
        .px(px(theme::SM))
        .gap(px(theme::SM))
        .rounded(px(4.))
        .cursor_pointer()
        .when(selected, |d| d.bg(t.list_active))
        .when(!selected, |d| d.hover(|d| d.bg(color::control())))
        .children(icon.map(|i| Icon::new(i).small().text_color(color::text_muted())))
        .child(div().flex_1().min_w_0().truncate().text_sm().child(label.into()))
        .child(div().flex_none().text_xs().text_color(color::text_muted()).child(trailing.into()))
}

/// `1234567` → `1 234 567` (thin spaces), for counts.
pub fn group_digits(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push('\u{2009}');
        }
        out.push(ch);
    }
    out
}

/// A length in µm (or px when the size is unknown), readable.
pub fn format_length(px_len: f64, pixel_um: Option<f64>) -> String {
    match pixel_um {
        Some(um) => {
            let v = px_len * um;
            if v >= 1000.0 { format!("{:.2} mm", v / 1000.0) } else if v >= 10.0 { format!("{v:.1} µm") } else { format!("{v:.2} µm") }
        }
        None => format!("{px_len:.1} px"),
    }
}

/// An area in µm² (or px²).
pub fn format_area(px_area: f64, pixel_um: Option<f64>) -> String {
    match pixel_um {
        Some(um) => {
            let v = px_area * um * um;
            if v >= 1.0e6 { format!("{:.3} mm²", v / 1.0e6) } else { format!("{v:.1} µm²") }
        }
        None => format!("{px_area:.0} px²"),
    }
}

/// A clickable chip label (for menus without a menu).
pub fn chip(id: impl Into<ElementId>, label: impl Into<SharedString>, on: bool, on_click: impl Fn(&mut Window, &mut App) + 'static) -> impl IntoElement {
    div()
        .id(id)
        .px(px(theme::SM))
        .py(px(2.))
        .rounded(px(4.))
        .text_xs()
        .cursor_pointer()
        .border_1()
        .border_color(if on { color::accent() } else { color::border() })
        .when(on, |d| d.bg(color::accent().opacity(0.25)))
        .child(label.into())
        .on_click(move |_, w, cx| on_click(w, cx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(group_digits(1234567), "1\u{2009}234\u{2009}567");
        assert_eq!(format_length(100.0, Some(0.5)), "50.0 µm");
        assert_eq!(format_length(3000.0, Some(0.5)), "1.50 mm");
        assert_eq!(format_area(100.0, None), "100 px²");
    }
}

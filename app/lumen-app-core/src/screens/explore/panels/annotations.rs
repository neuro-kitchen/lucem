//! The Annotations tab: the classes (colour, count, visibility; a click makes one the drawing
//! class) and the annotations of the shown image (a click selects one and centres it).

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, App, Entity, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _};

use crate::data::store::Store;
use crate::domain::annotation::{ClassDoc, Shape};
use crate::domain::command::Command;
use crate::screens::explore::canvas::ImageCanvas;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{format_area, format_length, group_digits, list_row, swatch, Section};

/// Rows listed at most.
const LISTED: usize = 300;

pub fn panel(store: &Entity<Store>, canvas: &Entity<ImageCanvas>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let Some(img) = s.doc.active_image.clone() else { return div().into_any_element() };
    let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
    let mut classes = v_flex().w_full().gap(px(2.));
    for c in &s.doc.classes {
        let n = s.doc.annotations_of(&img).filter(|a| a.class == c.name).count();
        let active = s.tool.class == c.name;
        let (st1, st2, cls1, cls2) = (store.clone(), store.clone(), c.clone(), c.name.clone());
        classes = classes.child(
            h_flex()
                .id(SharedString::from(format!("class-{}", c.name)))
                .h(px(28.))
                .px(px(theme::XS))
                .gap(px(theme::SM))
                .rounded(px(4.))
                .cursor_pointer()
                .when(active, |d| d.bg(color::accent().opacity(0.25)))
                .when(!active, |d| d.hover(|d| d.bg(color::control())))
                .child(Button::new(SharedString::from(format!("class-eye-{}", c.name))).ghost().xsmall().icon(Icon::new(if c.visible { IconName::Eye } else { IconName::EyeOff })).on_click(move |_, _, cx| {
                    cx.stop_propagation();
                    let class = ClassDoc { visible: !cls1.visible, ..cls1.clone() };
                    st1.update(cx, |s, cx| {
                        let _ = s.execute(Command::SetClass { class }, cx);
                    });
                }))
                .child(swatch(c.color, 12.))
                .child(div().flex_1().text_sm().child(c.name.clone()))
                .child(div().text_xs().text_color(color::text_muted()).child(group_digits(n)))
                .on_click(move |_, _, cx| {
                    let name = cls2.clone();
                    st2.update(cx, |s, cx| s.update_tool(|t| t.class = name, cx));
                }),
        );
    }
    let list: Vec<_> = s.doc.annotations_of(&img).collect();
    let mut rows = v_flex().w_full().gap(px(1.));
    for a in list.iter().rev().take(LISTED) {
        let measure = match (&a.shape, a.shape.area()) {
            (_, Some(area)) => format_area(area, um),
            (Shape::Line { .. }, _) | (Shape::Stroke { .. }, _) => a.shape.length().map_or(String::new(), |l| format_length(l, um)),
            _ => String::new(),
        };
        let selected = s.selection.contains(&a.id);
        let (store, canvas, id) = (store.clone(), canvas.clone(), a.id.clone());
        let (bx0, by0, bx1, by1) = a.shape.bounds();
        let centre = [(bx0 + bx1) / 2.0, (by0 + by1) / 2.0];
        rows = rows.child(list_row(SharedString::from(format!("ann-{}", a.id)), None, format!("{} · {}", a.shape.kind(), a.class), measure, selected, cx).on_click(move |_, _, cx| {
            let id = id.clone();
            store.update(cx, |s, cx| s.select([id], cx));
            canvas.update(cx, |c, cx| c.reveal(centre, cx));
        }));
    }
    let more = (list.len() > LISTED).then(|| div().text_xs().text_color(color::text_muted()).child(format!("{} more (see Analyze)", list.len() - LISTED)));
    v_flex()
        .id("annotations-panel")
        .size_full()
        .overflow_y_scroll()
        .p(px(theme::MD))
        .gap(px(theme::XL))
        .child(Section::new("Classes").child(classes))
        .child(
            Section::new(format!("On this image · {}", group_digits(list.len())))
                .when(list.is_empty(), |d| d.child(div().text_xs().text_color(color::text_muted()).child("Draw with the tools on the canvas (M O P L N B E W T R).")))
                .child(rows)
                .children(more),
        )
        .into_any_element()
}

//! The Selection tab: what is selected on the image — its class, measurements, and actions.

use gpui_kit::assets::IconName;
use gpui_kit::component::button::Button;
use gpui_kit::component::{v_flex, Icon, Sizable as _};
use gpui_kit::{div, px, AnyElement, App, Entity, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _};

use crate::data::store::Store;
use crate::domain::annotation::Annotation;
use crate::domain::command::Command;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{format_area, format_length, row, MenuSelect, Section};

pub fn panel(store: &Entity<Store>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
    let chosen: Vec<Annotation> = s.doc.annotations.iter().filter(|a| s.selection.contains(&a.id)).cloned().collect();
    let body: AnyElement = match chosen.as_slice() {
        [] => Section::new("Nothing selected")
            .child(div().text_xs().text_color(color::text_muted()).child("Select an annotation with V (Select) or in the Annotations tab: its class and measurements show here."))
            .into_any_element(),
        [a] => {
            let names: Vec<SharedString> = s.doc.classes.iter().map(|c| SharedString::from(c.name.clone())).collect();
            let raw: Vec<String> = s.doc.classes.iter().map(|c| c.name.clone()).collect();
            let current = raw.iter().position(|c| *c == a.class);
            let (st, a2) = (store.clone(), a.clone());
            let mut sec = Section::new(format!("{} · {}", a.shape.kind(), a.id)).child(row(
                "Class",
                MenuSelect::new("selection-class", names, current, move |i, _, cx| {
                    let annotation = Annotation { class: raw[i].clone(), ..a2.clone() };
                    st.update(cx, |s, cx| {
                        let _ = s.execute(Command::SetAnnotation { annotation }, cx);
                    });
                }),
            ));
            if let Some(area) = a.shape.area() {
                sec = sec.child(row("Area", div().text_sm().child(format_area(area, um))));
            }
            if let Some(len) = a.shape.length() {
                sec = sec.child(row(if a.shape.is_region() { "Perimeter" } else { "Length" }, div().text_sm().child(format_length(len, um))));
            }
            let (x0, y0, x1, y1) = a.shape.bounds();
            sec = sec.child(row("Centre", div().text_sm().font_family("monospace").child(match um {
                Some(u) => format!("{:.1}, {:.1} µm", (x0 + x1) / 2.0 * u, (y0 + y1) / 2.0 * u),
                None => format!("{:.0}, {:.0} px", (x0 + x1) / 2.0, (y0 + y1) / 2.0),
            })));
            sec.into_any_element()
        }
        many => {
            let area: f64 = many.iter().filter_map(|a| a.shape.area()).sum();
            Section::new(format!("{} selected", many.len())).child(row("Total area", div().text_sm().child(format_area(area, um)))).into_any_element()
        }
    };
    let st = store.clone();
    let delete = (!chosen.is_empty()).then(|| Button::new("selection-delete").outline().small().icon(Icon::new(IconName::Trash)).label("Delete").on_click(move |_, _, cx| st.update(cx, |s, cx| s.delete_selection(cx))));
    v_flex().id("selection-panel").size_full().overflow_y_scroll().p(px(theme::MD)).gap(px(theme::LG)).child(body).children(delete).into_any_element()
}

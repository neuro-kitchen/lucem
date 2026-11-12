//! The Metadata panel: what the shown image is (moved out of the Layers tree, which lists only
//! what is drawn): its file, size, pixel type, physical scale, pyramid levels and channels;
//! removing it from the project.

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Sizable as _};
use gpui_kit::{div, px, AnyElement, App, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _};

use crate::data::store::Store;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{group_digits, sig3, swatch};

/// `label   value`, the value allowed to wrap (paths are long).
fn line(label: &'static str, value: impl Into<SharedString>) -> impl IntoElement {
    h_flex().w_full().gap(px(theme::SM)).items_start().child(div().flex_none().w(px(96.)).text_sm().text_color(color::text_muted()).child(label)).child(div().flex_1().min_w_0().text_sm().child(value.into()))
}

fn heading(text: &'static str) -> impl IntoElement {
    div().pt(px(theme::MD)).text_base().font_weight(FontWeight::SEMIBOLD).child(text)
}

pub fn panel(store: &Entity<Store>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let Some(img) = s.doc.active().cloned() else { return div().p(px(theme::MD)).text_sm().text_color(color::text_muted()).child("No image").into_any_element() };
    let Some(p) = s.pyramids.get(&img.id).cloned() else { return div().p(px(theme::MD)).text_sm().text_color(color::text_muted()).child("Opening…").into_any_element() };
    let mut col = v_flex().id("metadata-panel").size_full().overflow_y_scroll().p(px(theme::MD)).gap(px(theme::SM));
    col = col
        .child(div().text_base().font_weight(FontWeight::SEMIBOLD).child(img.name.clone()))
        .child(line("File", crate::domain::paths::short_path(&img.path)))
        .child(line("Size", format!("{} × {} px{}", group_digits(p.width()), group_digits(p.height()), if p.is_volume() { format!(" × {} planes", p.depth()) } else { String::new() })))
        .child(line("Pixel type", p.dtype.clone()))
        .child(line("Pixel size", p.pixel_size_um.map_or("Unknown".to_string(), |u| format!("{} µm", sig3(u)))))
        .children(p.z_spacing_um.map(|z| line("Plane spacing", format!("{} µm", sig3(z)))))
        .children(p.pixel_size_um.map(|u| line("Area", crate::ui::widgets::format_area(p.width() as f64 * p.height() as f64, Some(u)))));
    col = col.child(heading("Channels"));
    for c in &p.channels {
        col = col.child(
            h_flex()
                .gap(px(theme::SM))
                .child(swatch(c.color, 12.))
                .child(div().flex_1().min_w_0().text_sm().child(c.label.clone()))
                .child(div().flex_none().text_xs().font_family("monospace").text_color(color::text_muted()).child(format!("{} – {}", sig3(c.range.0 as f64), sig3(c.range.1 as f64)))),
        );
    }
    col = col.child(heading("Levels"));
    for (k, l) in p.levels.iter().enumerate() {
        col = col.child(
            h_flex()
                .gap(px(theme::SM))
                .child(div().flex_none().w(px(28.)).text_xs().text_color(color::text_muted()).child(format!("{k}")))
                .child(div().flex_1().min_w_0().text_sm().child(format!("{} × {}{}", group_digits(l.width), group_digits(l.height), if l.depth > 1 { format!(" × {}", l.depth) } else { String::new() })))
                .child(div().flex_none().text_xs().font_family("monospace").text_color(color::text_muted()).child(format!("1:{}", sig3(l.downsample)))),
        );
    }
    let (st, id) = (store.clone(), img.id.clone());
    col = col.child(div().pt(px(theme::LG)).child(Button::new("metadata-remove").ghost().small().icon(Icon::new(IconName::Trash)).label("Remove from the project (the file stays)").on_click(move |_, _, cx| st.update(cx, |s, cx| s.remove_image(id.clone(), cx)))));
    col.into_any_element()
}

//! The Layers tab (Figma's left panel, Photoshop's Layers): everything on the canvas in one list,
//! channels, results and annotation classes, each with its visibility, a thumbnail or colour and a
//! name. Click selects a layer, Shift or Ctrl adds to the selection; masks that are selected can be
//! combined, which adds a new layer and leaves the originals as they were.

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, img, px, AnyElement, App, ClickEvent, Entity, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _};
use lumen_pipeline::{Op, PortType};

use crate::data::store::{MaskOperation, Store};
use crate::domain::annotation::ClassDoc;
use crate::domain::command::Command;
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::ui::theme::{self, color};
use crate::ui::widgets::{group_digits, stepper, swatch, Section};

const THUMB: f32 = 28.;

fn select(store: &Entity<Store>, key: &str, e: &ClickEvent, cx: &mut App) {
    let additive = e.modifiers().shift || e.modifiers().control || e.modifiers().platform;
    let key = key.to_string();
    store.update(cx, |s, cx| s.select_layer(key, additive, cx));
}

/// One row: visibility, thumbnail, name, what it is.
#[allow(clippy::too_many_arguments)]
fn layer_row(id: String, selected: bool, visible: bool, thumb: AnyElement, name: String, kind: String, on_eye: impl Fn(&mut App) + 'static, on_click: impl Fn(&ClickEvent, &mut App) + 'static) -> impl IntoElement {
    h_flex()
        .id(SharedString::from(format!("layer-{id}")))
        .h(px(36.))
        .px(px(theme::XS))
        .gap(px(theme::SM))
        .rounded(px(4.))
        .cursor_pointer()
        .when(selected, |d| d.bg(color::accent().opacity(0.3)))
        .when(!selected, |d| d.hover(|d| d.bg(color::control())))
        .child(Button::new(SharedString::from(format!("layer-eye-{id}"))).ghost().xsmall().icon(Icon::new(if visible { IconName::Eye } else { IconName::EyeOff })).on_click(move |_, _, cx| {
            cx.stop_propagation();
            on_eye(cx);
        }))
        .child(thumb)
        .child(div().flex_1().min_w_0().overflow_hidden().text_sm().when(!visible, |d| d.text_color(color::text_muted())).child(name))
        .child(div().flex_none().text_xs().text_color(color::text_muted()).child(kind))
        .on_click(move |e, _, cx| on_click(e, cx))
}

fn chip(c: [u8; 3]) -> AnyElement {
    div().flex_none().size(px(THUMB)).flex().items_center().justify_center().child(swatch(c, 14.)).into_any_element()
}

pub fn panel(store: &Entity<Store>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let Some(image) = s.doc.active_image.clone() else { return div().into_any_element() };
    let mut list = v_flex().w_full().gap(px(1.));

    // Results first: on top of the image, as they are drawn
    let views: Vec<_> = s.doc.graph.views().filter(|v| s.doc.graph.images_of(&v.id).contains(&image)).cloned().collect();
    for v in views.iter().rev() {
        let NodeKind::View { name, style } = &v.kind else { continue };
        let port = s.doc.graph.input(&v.id, 0).and_then(|i| s.doc.graph.node(i)).and_then(|n| n.kind.output());
        let kind = port.map_or("—", PortType::name).to_string();
        let key = format!("view:{}", v.id);
        let preview = s.analysis.as_ref().and_then(|a| a.previews.get(&v.id).cloned());
        let thumb = match preview {
            Some(p) => div().flex_none().size(px(THUMB)).rounded(px(3.)).overflow_hidden().border_1().border_color(color::border()).child(img(p).size_full()).into_any_element(),
            None => chip(style.color),
        };
        let (st, id, nm, sty) = (store.clone(), v.id.clone(), name.clone(), style.clone());
        let (st2, key2) = (store.clone(), key.clone());
        list = list.child(layer_row(
            key.clone(),
            s.layers_selected.contains(&key),
            style.visible,
            thumb,
            name.clone(),
            kind,
            move |cx| {
                let style = ViewStyle { visible: !sty.visible, ..sty.clone() };
                let (id, name) = (id.clone(), nm.clone());
                st.update(cx, |s, cx| {
                    let _ = s.execute(Command::SetView { id, name, style }, cx);
                });
            },
            move |e, cx| select(&st2, &key2, e, cx),
        ));
    }

    // Annotation classes
    for c in &s.doc.classes {
        let n = s.doc.annotations_of(&image).filter(|a| a.class == c.name).count();
        if n == 0 {
            continue;
        }
        let key = format!("class:{}", c.name);
        let (st, cls) = (store.clone(), c.clone());
        let (st2, key2) = (store.clone(), key.clone());
        list = list.child(layer_row(
            key.clone(),
            s.layers_selected.contains(&key),
            c.visible,
            chip(c.color),
            c.name.clone(),
            format!("{} annot.", group_digits(n)),
            move |cx| {
                let class = ClassDoc { visible: !cls.visible, ..cls.clone() };
                st.update(cx, |s, cx| {
                    let _ = s.execute(Command::SetClass { class }, cx);
                });
            },
            move |e, cx| select(&st2, &key2, e, cx),
        ));
    }

    // Channels at the bottom: the image itself
    if let Some(active) = s.doc.active() {
        for (i, c) in active.channels.iter().enumerate().rev() {
            let key = format!("channel:{i}");
            let (st, doc, img_id) = (store.clone(), c.clone(), image.clone());
            let (st2, key2) = (store.clone(), key.clone());
            list = list.child(layer_row(
                key.clone(),
                s.layers_selected.contains(&key),
                c.visible,
                chip(c.color),
                c.label.clone(),
                "channel".into(),
                move |cx| {
                    let mut display = doc.clone();
                    display.visible = !display.visible;
                    let image = img_id.clone();
                    st.update(cx, |s, cx| {
                        let _ = s.execute(Command::SetChannel { image, channel: i, display }, cx);
                    });
                },
                move |e, cx| select(&st2, &key2, e, cx),
            ));
        }
    }

    // What the selection allows
    let masks = s.selected_masks();
    let action = |id: &'static str, icon: IconName, tip: &'static str, enabled: bool, op: MaskOperation| {
        let st = store.clone();
        Button::new(id).ghost().small().icon(Icon::new(icon)).tooltip(tip).disabled(!enabled).on_click(move |_, _, cx| st.update(cx, |s, cx| s.apply_mask_operation(op.clone(), cx)))
    };
    let (two, one) = (masks.len() >= 2, !masks.is_empty());
    let ops = h_flex()
        .flex_wrap()
        .gap(px(2.))
        .child(action("mask-union", IconName::Plus, "Union: on where any selected mask is on", two, MaskOperation::Union))
        .child(action("mask-intersect", IconName::Combine, "Intersect: on where all selected masks are on", two, MaskOperation::Intersect))
        .child(action("mask-subtract", IconName::Minus, "Subtract: the first selected mask, minus the others", two, MaskOperation::Subtract))
        .child(action("mask-invert", IconName::Contrast, "Invert the first selected mask", one, MaskOperation::Invert))
        .child(action("mask-grow", IconName::Maximize2, "Grow the first selected mask", one, MaskOperation::Op(Op::Dilate { radius: s.mask_radius })))
        .child(action("mask-shrink", IconName::Minimize2, "Shrink the first selected mask", one, MaskOperation::Op(Op::Erode { radius: s.mask_radius })));
    let st = store.clone();
    let radius = stepper("mask-radius", format!("{} px", s.mask_radius), move |d, _, cx| {
        st.update(cx, |s, cx| {
            s.mask_radius = ((s.mask_radius as i64 + d as i64).clamp(1, 64)) as usize;
            cx.notify();
        })
    });
    let hint = if one { format!("{} mask(s) selected", masks.len()) } else { "Select masks (Shift adds) to combine them. Annotation classes count as masks.".to_string() };

    v_flex()
        .id("layers-panel")
        .size_full()
        .overflow_y_scroll()
        .p(px(theme::MD))
        .gap(px(theme::XL))
        .child(Section::new("Layers").child(list))
        .child(Section::new("Mask operations").child(v_flex().gap(px(theme::SM)).child(ops).child(h_flex().gap(px(theme::SM)).child(div().text_xs().text_color(color::text_muted()).child("Grow / shrink by")).child(radius)).child(div().text_xs().text_color(color::text_muted()).child(hint))))
        .into_any_element()
}

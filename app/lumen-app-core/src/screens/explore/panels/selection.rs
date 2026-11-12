//! The Selection tab: what is selected on the image and its numbers.
//!
//! - **One shape**: its class (to change), area, perimeter, centre; a region also its cells (by
//!   class), with those of the regions inside it.
//! - **Several shapes**: how many, total and mean area, the cells they hold.
//! - **One detected object** (a cell, a tile): its class, intensity class, region, and every
//!   measurement.
//! - **Several objects**: how many, their classes, and the mean of every measurement.

use std::collections::BTreeMap;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Sizable as _};
use gpui_kit::{div, px, AnyElement, App, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _};

use crate::data::store::Store;
use crate::domain::annotation::Annotation;
use crate::domain::command::Command;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{format_area, format_length, group_digits, row, sig3, MenuSelect};

fn heading(text: impl Into<SharedString>) -> impl IntoElement {
    div().text_base().font_weight(FontWeight::SEMIBOLD).child(text.into())
}

/// `name   value`: the name wraps, the value never does.
fn value_row(name: impl Into<SharedString>, value: impl Into<SharedString>) -> impl IntoElement {
    h_flex().w_full().gap(px(theme::SM)).items_start().child(div().flex_1().min_w_0().text_sm().text_color(color::text_muted()).child(name.into())).child(div().flex_none().text_sm().font_family("monospace").child(value.into()))
}

fn number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e9 { group_digits(v.abs() as usize) } else { sig3(v) }
}

/// The selection's numbers (also shown in the Analyze workspace).
pub fn summary(store: &Entity<Store>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
    let img = s.doc.active_image.clone().unwrap_or_default();
    let mut col = v_flex().gap(px(theme::SM));
    // Detected objects first (they are what was clicked last when any)
    let objects: Vec<(String, &lumen_hist::HistObject)> = s.sel.objects.iter().filter_map(|k| s.object(k).map(|o| (k.clone(), o))).collect();
    if !objects.is_empty() {
        let hier = s.hierarchy(&img);
        let cells = s.hist.cells.get(&img);
        let region_of = |key: &str| -> Option<String> {
            let id: u32 = key.get(1..)?.parse().ok()?;
            let i = cells?.iter().position(|o| o.id == id)?;
            hier.object_parent.get(i).cloned().flatten()
        };
        match objects.as_slice() {
            [(key, o)] => {
                col = col
                    .child(heading(format!("{} {key}", if key.starts_with('c') { "Cell" } else { "Tile" })))
                    .child(value_row("Class", o.class.clone().unwrap_or_else(|| "Unclassified".into())))
                    .children(o.intensity.map(|i| value_row("Intensity", i.name())))
                    .child(value_row("Region", region_of(key).unwrap_or_else(|| "the image".into())))
                    .child(value_row("Centre", match um {
                        Some(u) => format!("{:.1}, {:.1} µm", o.centroid[0] * u, o.centroid[1] * u),
                        None => format!("{:.0}, {:.0} px", o.centroid[0], o.centroid[1]),
                    }));
                for (k, v) in &o.measurements {
                    col = col.child(value_row(k.replace("µm^2", "µm²"), number(*v)));
                }
            }
            many => {
                let mut classes: BTreeMap<String, usize> = BTreeMap::new();
                let mut sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();
                for (_, o) in many {
                    *classes.entry(o.class.clone().unwrap_or_else(|| "Unclassified".into())).or_default() += 1;
                    for (k, v) in &o.measurements {
                        let e = sums.entry(k.clone()).or_default();
                        e.0 += v;
                        e.1 += 1;
                    }
                }
                col = col.child(heading(format!("{} objects selected", group_digits(many.len()))));
                for (c, n) in &classes {
                    col = col.child(value_row(c.clone(), group_digits(*n)));
                }
                col = col.child(div().pt(px(theme::SM)).text_xs().text_color(color::text_muted()).child("Mean of each measurement"));
                for (k, (sum, n)) in &sums {
                    col = col.child(value_row(k.replace("µm^2", "µm²"), number(sum / *n as f64)));
                }
            }
        }
        let st = store.clone();
        col = col.child(div().pt(px(theme::SM)).child(Button::new("selection-clear-objects").ghost().small().icon(Icon::new(IconName::X)).label("Clear the selection").on_click(move |_, _, cx| {
            st.update(cx, |s, cx| {
                s.sel.objects.clear();
                cx.notify();
            })
        })));
        return col.into_any_element();
    }
    let chosen: Vec<Annotation> = s.doc.annotations.iter().filter(|a| s.sel.shapes.contains(&a.id)).cloned().collect();
    let hier = s.hierarchy(&img);
    let cells = s.hist.cells.get(&img).cloned().unwrap_or_default();
    // The cells a region holds (with the regions inside it), by class
    let cells_by_class = |id: &str| -> BTreeMap<String, usize> {
        let sub = hier.subtree(id);
        let mut out = BTreeMap::new();
        for (i, p) in hier.object_parent.iter().enumerate() {
            if p.as_ref().is_some_and(|p| sub.contains(p)) {
                *out.entry(cells[i].class.clone().unwrap_or_else(|| "Unclassified".into())).or_insert(0) += 1;
            }
        }
        out
    };
    match chosen.as_slice() {
        [] => col = col.child(heading("Nothing selected")).child(div().text_xs().text_color(color::text_muted()).child("Click a shape or a cell (Select, V), or a row in Layers or Analyze. Shift adds.")),
        [a] => {
            let names: Vec<SharedString> = s.doc.classes.iter().map(|c| SharedString::from(c.name.clone())).collect();
            let raw: Vec<String> = s.doc.classes.iter().map(|c| c.name.clone()).collect();
            let current = raw.iter().position(|c| *c == a.class);
            let (st, a2) = (store.clone(), a.clone());
            col = col.child(heading(format!("{} {}", a.shape.kind(), a.id))).child(row(
                "Class",
                MenuSelect::new("selection-class", names, current, move |i, _, cx| {
                    let annotation = Annotation { class: raw[i].clone(), ..a2.clone() };
                    st.update(cx, |s, cx| {
                        let _ = s.execute(Command::SetAnnotation { annotation }, cx);
                    });
                })
            ));
            if let Some(area) = a.shape.area() {
                col = col.child(value_row("Area", format_area(area, um)));
            }
            if let Some(len) = a.shape.length() {
                col = col.child(value_row(if a.shape.is_region() { "Perimeter" } else { "Length" }, format_length(len, um)));
            }
            let (x0, y0, x1, y1) = a.shape.bounds();
            col = col.child(value_row("Centre", match um {
                Some(u) => format!("{:.1}, {:.1} µm", (x0 + x1) / 2.0 * u, (y0 + y1) / 2.0 * u),
                None => format!("{:.0}, {:.0} px", (x0 + x1) / 2.0, (y0 + y1) / 2.0),
            }));
            if a.shape.is_region() {
                col = col.child(value_row("Inside", hier.region_parent.get(&a.id).cloned().flatten().unwrap_or_else(|| "the image".into())));
                let by = cells_by_class(&a.id);
                let total: usize = by.values().sum();
                if total > 0 {
                    col = col.child(value_row("Cells", group_digits(total)));
                    if let Some(area) = a.shape.area().zip(um).map(|(a, u)| a * u * u / 1e6).filter(|a| *a > 0.0) {
                        col = col.child(value_row("Cells per mm²", sig3(total as f64 / area)));
                    }
                    for (c, n) in &by {
                        col = col.child(value_row(format!("· {c}"), format!("{} ({:.1} %)", group_digits(*n), 100.0 * *n as f64 / total as f64)));
                    }
                }
            }
        }
        many => {
            let areas: Vec<f64> = many.iter().filter_map(|a| a.shape.area()).collect();
            let total: f64 = areas.iter().sum();
            let n_cells: usize = many.iter().filter(|a| a.shape.is_region()).map(|a| hier.in_subtree.get(&a.id).copied().unwrap_or(0)).sum();
            col = col.child(heading(format!("{} shapes selected", many.len()))).child(value_row("Total area", format_area(total, um)));
            if !areas.is_empty() {
                col = col.child(value_row("Mean area", format_area(total / areas.len() as f64, um)));
            }
            if n_cells > 0 {
                col = col.child(value_row("Cells", group_digits(n_cells)));
            }
        }
    }
    if !chosen.is_empty() {
        let st = store.clone();
        col = col.child(div().pt(px(theme::SM)).child(Button::new("selection-delete").ghost().small().icon(Icon::new(IconName::Trash)).label("Delete").on_click(move |_, _, cx| st.update(cx, |s, cx| s.delete_selection(cx)))));
    }
    col.into_any_element()
}

pub fn panel(store: &Entity<Store>, cx: &App) -> AnyElement {
    v_flex().id("selection-panel").size_full().overflow_y_scroll().p(px(theme::MD)).child(summary(store, cx)).into_any_element()
}

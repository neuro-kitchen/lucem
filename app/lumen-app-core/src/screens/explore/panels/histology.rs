//! The Histology tab: the `lumen-hist` workflows on the shown image, top to bottom in the order
//! of an analysis — stains, tissue, cells, classes, intensity, hotspots and export.

use std::collections::BTreeSet;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::{div, px, AnyElement, App, Entity, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _};

use crate::data::hist::StainChoice;
use crate::data::store::Store;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{row, stepper, MenuSelect, Section};

fn note(text: impl Into<SharedString>) -> impl IntoElement {
    div().text_xs().text_color(color::text_muted()).child(text.into())
}

fn trim(v: f64) -> String {
    if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else if v.abs() >= 1.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// A button that runs a store action.
fn action(store: &Entity<Store>, id: &'static str, icon: IconName, label: &'static str, enabled: bool, f: fn(&mut Store, &mut gpui_kit::Context<Store>)) -> Button {
    let st = store.clone();
    Button::new(id).outline().small().icon(Icon::new(icon)).label(label).disabled(!enabled).on_click(move |_, _, cx| st.update(cx, f))
}

/// A stepped number of the detection settings.
fn setting(store: &Entity<Store>, id: &str, value: f64, step: f64, f: fn(&mut lumen_hist::cells::CellDetection, f64)) -> impl IntoElement {
    let st = store.clone();
    stepper(id, trim(value), move |d, _, cx| {
        st.update(cx, |s, cx| {
            let v = (value + d as f64 * step).max(0.0);
            f(&mut s.hist.detection, v);
            cx.notify();
        })
    })
}

pub fn panel(store: &Entity<Store>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let Some(img) = s.doc.active_image.clone() else { return div().p(px(theme::MD)).child(note("No image")).into_any_element() };
    let p = s.pyramids.get(&img).cloned();
    let busy = s.hist.busy.clone();
    let idle = busy.is_none();
    let stains = s.hist_stains(&img);
    let fluorescence = matches!(stains, StainChoice::Fluorescence(_));
    let cells = s.hist.cells.get(&img).cloned().unwrap_or_default();
    let selected_regions = s.doc.annotations.iter().filter(|a| a.image == img && s.selection.contains(&a.id) && a.shape.is_region()).count();
    let mut col = v_flex().id("histology-panel").size_full().overflow_y_scroll().p(px(theme::MD)).gap(px(theme::LG));
    if p.as_ref().is_some_and(|p| p.pixel_size_um.is_none()) {
        col = col.child(div().text_xs().text_color(color::warning()).child("This image has no pixel size (µm): histology measures in µm and needs it."));
    }
    if let Some(b) = &busy {
        col = col.child(h_flex().gap(px(theme::SM)).child(Icon::new(IconName::LoaderCircle).small()).child(div().text_sm().child(b.clone())));
    }

    // Stains
    let channels: Vec<String> = p.as_ref().map(|p| p.channels.iter().map(|c| c.label.clone()).collect()).unwrap_or_default();
    let mut choices: Vec<(SharedString, StainChoice)> = vec![("H-DAB".into(), StainChoice::HDab), ("H&E".into(), StainChoice::HE), ("H-E-DAB".into(), StainChoice::HEDab)];
    if let StainChoice::Estimated(v) = &stains {
        choices.push((stains.label().into(), StainChoice::Estimated(v.clone())));
    }
    choices.extend(channels.iter().enumerate().map(|(i, n)| (SharedString::from(format!("Fluorescence: nuclei on {n}")), StainChoice::Fluorescence(i))));
    let current = choices.iter().position(|(_, c)| *c == stains);
    let st = store.clone();
    let picks: Vec<StainChoice> = choices.iter().map(|(_, c)| c.clone()).collect();
    let stains_pick = MenuSelect::new("hist-stains", choices.iter().map(|(l, _)| l.clone()).collect(), current, move |i, _, cx| {
        let c = picks[i].clone();
        st.update(cx, |s, cx| s.hist_set_stains(c, cx));
    });
    col = col.child(
        Section::new("Stains")
            .child(row("Image", stains_pick))
            .child(h_flex().gap(px(theme::SM)).child(action(store, "hist-fit-stains", IconName::Pipette, "Fit to selected region", idle && !fluorescence && selected_regions > 0, Store::hist_estimate_stains)))
            .child(note(if fluorescence { "Nuclei are found on the chosen channel; every channel is measured." } else { "Brightfield is separated into stains (optical density); nuclei are found on the first. Fit the stains to a region of this slide for its real colours." })),
    );

    // Tissue
    col = col.child(Section::new("Tissue").child(h_flex().child(action(store, "hist-tissue", IconName::ScanSearch, "Find tissue", idle && p.is_some(), Store::hist_detect_tissue))).child(note("Regions of class Tissue (undoable); select them to find cells in them.")));

    // Cells
    let d = s.hist.detection.clone();
    let st = store.clone();
    let auto = s.hist.auto_threshold;
    let threshold_row: AnyElement = if fluorescence {
        let st2 = store.clone();
        h_flex()
            .gap(px(theme::XS))
            .child(Button::new("hist-auto-threshold").ghost().xsmall().label("Auto").selected(auto).on_click(move |_, _, cx| st2.update(cx, |s, cx| {
                s.hist.auto_threshold = !s.hist.auto_threshold;
                cx.notify();
            })))
            .when_some((!auto).then_some(()), |r, _| r.child(setting(store, "hist-threshold", d.threshold, (d.threshold * 0.1).max(1.0), |d, v| d.threshold = v)))
            .into_any_element()
    } else {
        setting(store, "hist-threshold", d.threshold, 0.02, |d, v| d.threshold = v).into_any_element()
    };
    let count = cells.len();
    let show = s.hist.show;
    let eye = Button::new("hist-show").ghost().xsmall().icon(Icon::new(if show { IconName::Eye } else { IconName::EyeOff })).tooltip("Show the cells on the image").on_click(move |_, _, cx| st.update(cx, |s, cx| {
        s.hist.show = !s.hist.show;
        s.hist.rev += 1;
        cx.notify();
    }));
    col = col.child(
        Section::new("Cells")
            .extra(h_flex().gap(px(theme::XS)).child(div().text_xs().text_color(color::text_muted()).child(format!("{count} cells"))).child(eye))
            .child(row("Threshold", threshold_row))
            .child(row("Smoothing σ (µm)", setting(store, "hist-sigma", d.sigma_um, 0.25, |d, v| d.sigma_um = v)))
            .child(row("Background radius (µm)", setting(store, "hist-bg", d.background_radius_um, 1.0, |d, v| d.background_radius_um = v)))
            .child(row("Cell expansion (µm)", setting(store, "hist-expand", d.cell_expansion_um, 1.0, |d, v| d.cell_expansion_um = v)))
            .child(row("Nucleus area min (µm²)", setting(store, "hist-min", d.min_area_um2, 5.0, |d, v| d.min_area_um2 = v)))
            .child(row("Nucleus area max (µm²)", setting(store, "hist-max", d.max_area_um2, 25.0, |d, v| d.max_area_um2 = v)))
            .child(
                h_flex()
                    .gap(px(theme::SM))
                    .child(action(store, "hist-cells", IconName::Sparkles, "Find cells", idle && selected_regions > 0, Store::hist_detect_cells))
                    .child(action(store, "hist-clear", IconName::Trash, "Clear", idle && count > 0, Store::hist_clear)),
            )
            .child(note(if selected_regions == 0 { "Select the region annotations to search (Shift adds)." } else { "Cells are found tile by tile on the device, inside the selected regions." })),
    );

    // Classes
    let trained = s.hist.classifier.is_some();
    col = col.child(
        Section::new("Classify cells")
            .child(h_flex().child(action(store, "hist-train", IconName::BrainCircuit, if trained { "Train again" } else { "Train from annotations" }, idle && count > 0, Store::hist_train_classifier)))
            .child(note("Draw regions or points over cells and give them classes (Tumor, Stroma, …); Random Trees learn from the cells they cover and classify all.")),
    );

    // Intensity
    let names: Vec<String> = cells.iter().flat_map(|c| c.measurements.keys().filter(|k| k.ends_with(" mean") && !k.starts_with("Smoothed")).cloned()).collect::<BTreeSet<_>>().into_iter().collect();
    let t = s.hist.intensity.clone();
    let st = store.clone();
    let picks = names.clone();
    let measurement = MenuSelect::new("hist-measurement", names.iter().map(|n| SharedString::from(n.clone())).collect(), names.iter().position(|n| *n == t.measurement), move |i, _, cx| {
        let m = picks[i].clone();
        st.update(cx, |s, cx| {
            s.hist.intensity.measurement = m;
            cx.notify();
        })
    })
    .full_width();
    let mut sec = Section::new("Intensity").child(row("Measurement", measurement));
    for k in 0..3 {
        let v = t.thresholds.get(k).copied().unwrap_or(0.0);
        let st = store.clone();
        let step = if v.abs() > 5.0 { (v.abs() * 0.05).max(1.0) } else { 0.05 };
        sec = sec.child(row(
            ["Threshold 1+", "Threshold 2+", "Threshold 3+"][k],
            stepper(&format!("hist-t{k}"), trim(v), move |d, _, cx| {
                st.update(cx, |s, cx| {
                    let th = &mut s.hist.intensity.thresholds;
                    th.resize(3, 0.0);
                    th[k] = (th[k] + d as f64 * step).max(0.0);
                    cx.notify();
                })
            }),
        ));
    }
    let summary = lumen_hist::intensity::summarize(cells.iter());
    sec = sec.child(h_flex().child(action(store, "hist-intensity", IconName::Gauge, "Apply", idle && count > 0, Store::hist_apply_intensity)));
    if summary.total > 0 {
        sec = sec.child(div().text_sm().child(format!("Positive {:.1} % · H-score {:.0} · 1+ {} · 2+ {} · 3+ {}", summary.positive_percent, summary.h_score, summary.one_plus, summary.two_plus, summary.three_plus)));
    }
    col = col.child(sec);

    // Density and export
    let st = store.clone();
    col = col.child(
        Section::new("Density and export")
            .child(
                h_flex()
                    .gap(px(theme::SM))
                    .child(Button::new("hist-hotspots").outline().small().icon(Icon::new(IconName::Flame)).label("Mark 3 hotspots").disabled(!(idle && count > 0)).on_click(move |_, _, cx| st.update(cx, |s, cx| s.hist_hotspots(3, cx))))
                    .child(action(store, "hist-export", IconName::Download, "Export", count > 0, Store::hist_export)),
            )
            .child(note("Export writes the cells' measurements and a summary per region annotation (counts, positive %, H-score, densities) next to the project.")),
    );
    col.into_any_element()
}

trait WhenSome: Sized {
    fn when_some<T>(self, v: Option<T>, f: impl FnOnce(Self, T) -> Self) -> Self {
        match v {
            Some(v) => f(self, v),
            None => self,
        }
    }
}

impl WhenSome for gpui_kit::Div {}

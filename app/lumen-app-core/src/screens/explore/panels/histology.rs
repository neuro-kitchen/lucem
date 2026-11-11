//! The Histology panel: the slide, its recipes, and the readout.
//!
//! - **Slide**: what every analysis depends on, set once: the stains (or the fluorescence nuclear
//!   channel), the pixel size, the tissue.
//! - **Recipes**: a named protocol each (count cells, positive % and H-score, cell classes,
//!   tissue classes, superpixel classes, density); one open at a time, with its settings and one
//!   primary action; progress with Cancel while it runs; its outcome under the button, in plain
//!   words.
//! - **Readout**: the numbers of the selected region: area, cells by class, positive %, H-score,
//!   density; Export.
//!
//! The processing of each channel is in Layers; the same steps are in the Analyze and Classify
//! menus.

use std::collections::BTreeSet;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, App, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _};

use crate::data::hist::{Recipe, StainChoice};
use crate::data::store::Store;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{group_digits, row, sig3, stepper, MenuSelect};

fn trim(v: f64) -> String {
    if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else if v.abs() >= 1.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// A section heading: sentence case, semibold, larger than the rows (`type-hierarchy`,
/// `font-weight-contrast`).
fn heading(text: &'static str) -> impl IntoElement {
    div().text_base().font_weight(FontWeight::SEMIBOLD).child(text)
}

fn note(text: impl Into<SharedString>) -> impl IntoElement {
    div().text_xs().text_color(color::text_muted()).child(text.into())
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

/// The one primary action of the open recipe (filled), disabled with its reason said beside.
fn run(store: &Entity<Store>, id: &'static str, label: &'static str, enabled: bool, f: fn(&mut Store, &mut gpui_kit::Context<Store>)) -> Button {
    let st = store.clone();
    Button::new(id).primary().small().icon(Icon::new(IconName::Play)).label(label).disabled(!enabled).on_click(move |_, _, cx| st.update(cx, f))
}

/// A secondary action (outline).
fn secondary(store: &Entity<Store>, id: &'static str, icon: IconName, label: &'static str, enabled: bool, f: fn(&mut Store, &mut gpui_kit::Context<Store>)) -> Button {
    let st = store.clone();
    Button::new(id).outline().small().icon(Icon::new(icon)).label(label).disabled(!enabled).on_click(move |_, _, cx| st.update(cx, f))
}

pub fn panel(store: &Entity<Store>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let Some(img) = s.doc.active_image.clone() else { return div().p(px(theme::MD)).child(note("No image")).into_any_element() };
    let p = s.pyramids.get(&img).cloned();
    let um = p.as_ref().and_then(|p| p.pixel_size_um);
    let busy = s.hist.busy.clone();
    let idle = busy.is_none();
    let stains = s.hist_stains(&img);
    let brightfield = !matches!(stains, StainChoice::Fluorescence(_));
    let cells = s.hist.cells.get(&img).cloned().unwrap_or_default();
    let count = cells.len();
    let tiles = s.hist.tiles.get(&img).map_or(0, Vec::len);
    let selected: Vec<&crate::domain::annotation::Annotation> = s.doc.annotations.iter().filter(|a| a.image == img && s.selection.contains(&a.id) && a.shape.is_region()).collect();
    let regions = selected.len();

    // ---- Slide --------------------------------------------------------------
    let channels: Vec<String> = p.as_ref().map(|p| p.channels.iter().map(|c| c.label.clone()).collect()).unwrap_or_default();
    let mut choices: Vec<(SharedString, StainChoice)> = vec![("H-DAB".into(), StainChoice::HDab), ("H&E".into(), StainChoice::HE), ("H-E-DAB".into(), StainChoice::HEDab)];
    if let StainChoice::Estimated(v) = &stains {
        choices.push((stains.label().into(), StainChoice::Estimated(v.clone())));
    }
    choices.extend(channels.iter().enumerate().map(|(i, n)| (SharedString::from(format!("Fluorescence · {n}")), StainChoice::Fluorescence(i))));
    let current = choices.iter().position(|(_, c)| *c == stains);
    let st = store.clone();
    let picks: Vec<StainChoice> = choices.iter().map(|(_, c)| c.clone()).collect();
    let stains_pick = MenuSelect::new("hist-stains", choices.iter().map(|(l, _)| l.clone()).collect(), current, move |i, _, cx| {
        let c = picks[i].clone();
        st.update(cx, |s, cx| s.hist_set_stains(c, cx));
    });
    let tissue: Vec<&crate::domain::annotation::Annotation> = s.doc.annotations_of(&img).filter(|a| a.class == "Tissue" && a.shape.is_region()).collect();
    let tissue_text = if tissue.is_empty() {
        "Not found yet".to_string()
    } else {
        let px2: f64 = tissue.iter().filter_map(|a| a.shape.area()).sum();
        format!("{} region(s) · {}", tissue.len(), crate::ui::widgets::format_area(px2, um))
    };
    let mut slide = v_flex()
        .gap(px(theme::SM))
        .child(heading("Slide"))
        .child(row("Stains", stains_pick))
        .child(row("Pixel size", div().text_sm().when(um.is_none(), |d| d.text_color(color::warning())).child(um.map_or("Unknown: measurements need it".to_string(), |u| format!("{} µm", sig3(u))))))
        .child(row("Tissue", h_flex().gap(px(theme::SM)).child(div().text_sm().child(tissue_text)).child(secondary(store, "hist-tissue", IconName::ScanSearch, "Find", idle && p.is_some(), Store::hist_detect_tissue))));
    if brightfield {
        slide = slide.child(h_flex().child(secondary(store, "hist-fit-stains", IconName::Pipette, "Fit stains to the selected region", idle && regions > 0, Store::hist_estimate_stains)));
    }

    // ---- Recipes ------------------------------------------------------------
    let open = s.hist.recipe;
    let mut recipes = v_flex().gap(px(theme::XS)).child(heading("Recipes"));
    for r in Recipe::ALL {
        let is_open = r == open;
        // What it has given so far, on its header
        let status = match r {
            Recipe::Count if count > 0 => format!("{} cells", group_digits(count)),
            Recipe::Positivity => {
                let sm = lumen_hist::intensity::summarize(cells.iter());
                if sm.total > 0 { format!("{:.1} % positive", sm.positive_percent) } else { String::new() }
            }
            Recipe::CellClasses if s.hist.classifier.is_some() => "trained".into(),
            Recipe::TissueClasses if s.hist.class_maps.contains_key(&img) => "classified".into(),
            Recipe::Superpixels if tiles > 0 => format!("{} tiles", group_digits(tiles)),
            Recipe::Density if s.hist.density_maps.contains_key(&img) => "mapped".into(),
            _ => String::new(),
        };
        let st = store.clone();
        let header = h_flex()
            .id(SharedString::from(format!("recipe-{r:?}")))
            .h(px(32.))
            .px(px(theme::SM))
            .gap(px(theme::SM))
            .rounded(px(6.))
            .cursor_pointer()
            .when(is_open, |d| d.bg(color::raised()))
            .when(!is_open, |d| d.hover(|d| d.bg(color::control())))
            .child(Icon::new(r.icon()).small().text_color(if is_open { color::accent_hover() } else { color::text_muted() }))
            .child(div().flex_1().min_w_0().text_sm().when(is_open, |d| d.font_weight(FontWeight::SEMIBOLD)).child(r.title()))
            .child(div().flex_none().text_xs().text_color(color::text_muted()).child(status))
            .child(Icon::new(if is_open { IconName::ChevronDown } else { IconName::ChevronRight }).small().text_color(color::text_muted()))
            .on_click(move |_, _, cx| {
                st.update(cx, |s, cx| {
                    s.hist.recipe = r;
                    s.hist.outcome = None;
                    cx.notify();
                })
            });
        recipes = recipes.child(header);
        if is_open {
            recipes = recipes.child(recipe_body(store, s, r, &img, &cells, regions, brightfield, idle, cx));
        }
    }

    // ---- Readout --------------------------------------------------------------
    let mut readout = v_flex().gap(px(theme::SM)).child(heading("Readout"));
    match (selected.first(), um) {
        (Some(a), Some(um)) if count > 0 => {
            let m = lumen_hist::export::region_summary(&a.shape.outline(), &cells, um);
            let name = if a.note.is_empty() { format!("{} {}", a.shape.kind(), a.id) } else { a.note.lines().next().unwrap_or_default().to_string() };
            readout = readout.child(note(format!("{name} ({})", a.class)));
            for (k, v) in &m {
                let value = if k.contains("µm^2") && *v >= 1.0e4 { format!("{} mm²", sig3(v / 1.0e6)) } else if v.fract() == 0.0 { group_digits(*v as usize) } else { sig3(*v) };
                let label = k.replace("µm^2", "µm²").replace("Num Detections", "Cells").replace("mm^2", "mm²");
                readout = readout.child(h_flex().gap(px(theme::SM)).child(div().flex_1().min_w_0().text_sm().text_color(color::text_muted()).child(label)).child(div().text_sm().font_family("monospace").child(value)));
            }
        }
        _ => readout = readout.child(note(if count == 0 { "Run a recipe; the selected region's numbers show here." } else { "Select a region to read its numbers." })),
    }
    readout = readout.child(h_flex().child(secondary(store, "hist-export", IconName::Download, "Export all", count > 0, Store::hist_export)));

    let mut col = v_flex().id("histology-panel").size_full().overflow_y_scroll().p(px(theme::MD)).gap(px(32.));
    if um.is_none() && p.is_some() {
        col = col.child(div().text_xs().text_color(color::warning()).child("This image has no pixel size: histology measures in µm and needs it."));
    }
    col.child(slide).child(recipes).child(readout).into_any_element()
}

/// The open recipe: its purpose, its settings, its primary action, progress or outcome.
#[allow(clippy::too_many_arguments)]
fn recipe_body(store: &Entity<Store>, s: &Store, r: Recipe, img: &str, cells: &[lumen_hist::HistObject], regions: usize, brightfield: bool, idle: bool, cx: &App) -> AnyElement {
    let count = cells.len();
    let d = s.hist.detection.clone();
    let mut body = v_flex().gap(px(theme::SM)).px(px(theme::SM)).pb(px(theme::MD)).child(note(r.purpose()));
    // What blocks the primary action, said beside it (`wcag-forms` 3.3.3)
    let (enabled, why): (bool, Option<&str>) = match r {
        Recipe::Count | Recipe::Superpixels if regions == 0 => (false, Some("Select one or more regions (Shift adds)")),
        Recipe::Positivity | Recipe::CellClasses | Recipe::Density if count == 0 => (false, Some("Count cells first")),
        _ => (true, None),
    };
    match r {
        Recipe::Count => {
            // Only the detectors that fit the image (`default-effect`)
            let models = s.ml.nucleus_models_for(brightfield);
            let mut labels: Vec<SharedString> = vec!["Threshold (built in)".into()];
            labels.extend(models.iter().map(|(_, m)| SharedString::from(if m.installed { m.name.clone() } else { format!("{} (not installed)", m.name) })));
            let cur = s.hist.detector.as_ref().map_or(Some(0), |id| models.iter().position(|(_, m)| &m.id == id).map(|i| i + 1));
            let ids: Vec<String> = models.iter().map(|(_, m)| m.id.clone()).collect();
            let st = store.clone();
            body = body.child(row(
                "Detector",
                MenuSelect::new("hist-detector", labels, cur, move |i, _, cx| {
                    let d = (i > 0).then(|| ids[i - 1].clone());
                    st.update(cx, |s, cx| s.hist_set_detector(d, cx))
                }),
            ));
            if s.hist.detector.is_some() && cur.is_none() {
                body = body.child(div().text_xs().text_color(color::warning()).child("The chosen model does not fit this image; choose another."));
            }
            if s.hist.detector.is_none() {
                let fluorescence = !brightfield;
                let auto = s.hist.auto_threshold;
                let threshold: AnyElement = if fluorescence {
                    let st = store.clone();
                    h_flex()
                        .gap(px(theme::XS))
                        .child(Button::new("hist-auto-threshold").ghost().xsmall().label("Auto").selected(auto).on_click(move |_, _, cx| st.update(cx, |s, cx| {
                            s.hist.auto_threshold = !s.hist.auto_threshold;
                            cx.notify();
                        })))
                        .when(!auto, |r| r.child(setting(store, "hist-threshold", d.threshold, (d.threshold * 0.1).max(1.0), |d, v| d.threshold = v)))
                        .into_any_element()
                } else {
                    setting(store, "hist-threshold", d.threshold, 0.02, |d, v| d.threshold = v).into_any_element()
                };
                let (st1, st2) = (store.clone(), store.clone());
                body = body
                    .child(row("Threshold", threshold))
                    .child(row(
                        "Split touching",
                        h_flex()
                            .gap(px(2.))
                            .p(px(2.))
                            .rounded(px(6.))
                            .bg(color::control())
                            .child(Button::new("hist-split-gaps").ghost().xsmall().label("At gaps").selected(d.split_by_intensity).tooltip("Boundaries along the dark gaps (dense tissue)").on_click(move |_, _, cx| st1.update(cx, |s, cx| {
                                s.hist.detection.split_by_intensity = true;
                                s.hist.detection.split_by_shape = true;
                                cx.notify();
                            })))
                            .child(Button::new("hist-split-shape").ghost().xsmall().label("By shape").selected(!d.split_by_intensity && d.split_by_shape).tooltip("Round pieces cut at their necks (sparse nuclei)").on_click(move |_, _, cx| st2.update(cx, |s, cx| {
                                s.hist.detection.split_by_intensity = false;
                                s.hist.detection.split_by_shape = true;
                                cx.notify();
                            }))),
                    ))
                    .child(row("Smoothing (µm)", setting(store, "hist-sigma", d.sigma_um, 0.25, |d, v| d.sigma_um = v)))
                    .child(row("Background (µm)", setting(store, "hist-bg", d.background_radius_um, 1.0, |d, v| d.background_radius_um = v)));
            }
            body = body
                .child(row("Min nucleus (µm²)", setting(store, "hist-min", d.min_area_um2, 5.0, |d, v| d.min_area_um2 = v)))
                .child(row("Max nucleus (µm²)", setting(store, "hist-max", d.max_area_um2, 25.0, |d, v| d.max_area_um2 = v)))
                .child(row("Cell growth (µm)", setting(store, "hist-expand", d.cell_expansion_um, 1.0, |d, v| d.cell_expansion_um = v)));
        }
        Recipe::Positivity => {
            let names: Vec<String> = cells.iter().flat_map(|c| c.measurements.keys().filter(|k| k.ends_with(" mean") && !k.starts_with("Smoothed")).cloned()).collect::<BTreeSet<_>>().into_iter().collect();
            let t = s.hist.intensity.clone();
            let st = store.clone();
            let picks = names.clone();
            body = body.child(row(
                "Measurement",
                MenuSelect::new("hist-measurement", names.iter().map(|n| SharedString::from(n.clone())).collect(), names.iter().position(|n| *n == t.measurement), move |i, _, cx| {
                    let m = picks[i].clone();
                    st.update(cx, |s, cx| {
                        s.hist.intensity.measurement = m;
                        cx.notify();
                    })
                })
                .full_width(),
            ));
            for k in 0..3 {
                let v = t.thresholds.get(k).copied().unwrap_or(0.0);
                let st = store.clone();
                let step = if v.abs() > 5.0 { (v.abs() * 0.05).max(1.0) } else { 0.05 };
                body = body.child(row(
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
            let sm = lumen_hist::intensity::summarize(cells.iter());
            if sm.total > 0 {
                body = body.child(div().text_sm().child(format!("Positive {:.1} % · H-score {:.0} · 1+ {} · 2+ {} · 3+ {}", sm.positive_percent, sm.h_score, sm.one_plus, sm.two_plus, sm.three_plus)));
            }
        }
        Recipe::CellClasses => {
            body = body.child(note("Draw regions or points over cells, give them classes (Tumor, Stroma, …) in Layers, then train: every cell gets a class."));
        }
        Recipe::TissueClasses => {
            let st = store.clone();
            body = body.child(row("Resolution (µm/px)", stepper("hist-px-um", trim(s.hist.pixel_um), move |d, _, cx| st.update(cx, |s, cx| {
                let v = s.hist.pixel_um;
                s.hist.pixel_um = if d > 0.0 { v * 2.0 } else { v / 2.0 }.clamp(0.25, 64.0);
                cx.notify();
            }))));
            if let Some(m) = s.hist.class_maps.get(img) {
                for a in &m.areas {
                    body = body.child(row(SharedString::from(a.class.clone()), div().text_sm().child(format!("{} mm² · {:.1} %", sig3(a.area_um2 / 1e6), a.percent))));
                }
            } else {
                body = body.child(note("Paint or draw two or more classes over the image; the selected regions (else the painted area) are classed."));
            }
        }
        Recipe::Superpixels => {
            let sp = s.hist.superpixel.clone();
            let (st, st2) = (store.clone(), store.clone());
            body = body
                .child(row("Size (µm)", stepper("hist-sp-size", trim(sp.spacing_um), move |d, _, cx| st.update(cx, |s, cx| {
                    s.hist.superpixel.spacing_um = (s.hist.superpixel.spacing_um + d as f64 * 5.0).max(5.0);
                    cx.notify();
                }))))
                .child(row("Compactness", stepper("hist-sp-compact", trim(sp.compactness), move |d, _, cx| st2.update(cx, |s, cx| {
                    s.hist.superpixel.compactness = (s.hist.superpixel.compactness + d as f64 * 0.05).clamp(0.01, 10.0);
                    cx.notify();
                }))));
        }
        Recipe::Density => {
            let dens = s.hist.density.clone();
            let kinds = [("Count within radius", lumen_hist::density::DensityValue::Count), ("Cells per mm²", lumen_hist::density::DensityValue::PerMm2), ("% positive", lumen_hist::density::DensityValue::PercentPositive)];
            let st = store.clone();
            let value_pick = MenuSelect::new("hist-density-value", kinds.iter().map(|(l, _)| SharedString::from(*l)).collect(), kinds.iter().position(|(_, v)| *v == dens.value), move |i, _, cx| {
                st.update(cx, |s, cx| {
                    s.hist.density.value = kinds[i].1;
                    cx.notify();
                })
            });
            let classes: Vec<String> = cells.iter().filter_map(|c| c.class.clone()).collect::<BTreeSet<_>>().into_iter().collect();
            let mut class_labels: Vec<SharedString> = vec!["All cells".into()];
            class_labels.extend(classes.iter().map(|c| SharedString::from(c.clone())));
            let current = dens.class.as_ref().and_then(|c| classes.iter().position(|k| k == c)).map_or(0, |i| i + 1);
            let st = store.clone();
            let class_pick = MenuSelect::new("hist-density-class", class_labels, Some(current), move |i, _, cx| {
                let c = (i > 0).then(|| classes[i - 1].clone());
                st.update(cx, |s, cx| {
                    s.hist.density.class = c;
                    cx.notify();
                })
            });
            let st = store.clone();
            body = body.child(row("Value", value_pick)).child(row("Cells", class_pick)).child(row("Radius (µm)", stepper("hist-density-r", trim(dens.radius_um), move |d, _, cx| st.update(cx, |s, cx| {
                s.hist.density.radius_um = (s.hist.density.radius_um + d as f64 * 10.0).max(10.0);
                cx.notify();
            }))));
        }
    }
    // The primary action, then the secondary ones
    let primary = match r {
        Recipe::Count => run(store, "recipe-run-count", "Find cells", idle && enabled, Store::hist_detect_cells),
        Recipe::Positivity => run(store, "recipe-run-positivity", "Score cells", idle && enabled, Store::hist_apply_intensity),
        Recipe::CellClasses => run(store, "recipe-run-classes", "Train and classify", idle && enabled, Store::hist_train_classifier),
        Recipe::TissueClasses => run(store, "recipe-run-tissue", "Train and classify", idle && enabled, Store::hist_train_pixels),
        Recipe::Superpixels => run(store, "recipe-run-superpixels", "Make superpixels", idle && enabled, Store::hist_superpixels),
        Recipe::Density => run(store, "recipe-run-density", "Density map", idle && enabled, Store::hist_density),
    };
    let mut actions = h_flex().gap(px(theme::SM)).flex_wrap().child(primary);
    actions = match r {
        Recipe::Count => actions.child(Button::new("recipe-clear").ghost().small().icon(Icon::new(IconName::Trash)).label("Clear").disabled(!(idle && count > 0)).on_click({
            let st = store.clone();
            move |_, _, cx| st.update(cx, Store::hist_clear)
        })),
        Recipe::TissueClasses => actions.child(secondary(store, "recipe-tissue-shapes", IconName::Shapes, "Class map → shapes", idle && s.hist.class_maps.contains_key(img), Store::hist_class_regions)),
        Recipe::Superpixels => actions.child(secondary(store, "recipe-sp-train", IconName::BrainCircuit, "Train on tiles", idle && s.hist.tiles.get(img).is_some_and(|t| !t.is_empty()), Store::hist_train_classifier)),
        Recipe::Density => {
            let st = store.clone();
            actions.child(Button::new("recipe-hotspots").outline().small().icon(Icon::new(IconName::MapPin)).label("Mark 3 hotspots").disabled(!(idle && count > 0)).on_click(move |_, _, cx| st.update(cx, |s, cx| s.hist_hotspots(3, cx))))
        }
        _ => actions,
    };
    body = body.child(actions);
    if let Some(why) = why {
        body = body.child(note(why));
    }
    // Running: what, how far, Cancel; done: the outcome, in plain words
    let _ = cx;
    if let Some(b) = &s.hist.busy {
        let f = s.hist.progress.unwrap_or(0.0).clamp(0.0, 1.0);
        let st = store.clone();
        body = body.child(
            v_flex()
                .gap(px(theme::XS))
                .child(h_flex().gap(px(theme::SM)).child(Icon::new(IconName::LoaderCircle).small()).child(div().flex_1().min_w_0().text_sm().child(b.clone())).child(Button::new("recipe-cancel").ghost().xsmall().label("Cancel").on_click(move |_, _, cx| st.update(cx, |s, cx| s.hist_cancel(cx)))))
                .child(div().w_full().h(px(4.)).rounded_full().bg(color::control()).child(div().h_full().w(gpui_kit::relative(f)).rounded_full().bg(color::accent()))),
        );
    } else if let Some((ok, msg)) = &s.hist.outcome {
        body = body.child(
            h_flex()
                .gap(px(theme::SM))
                .items_start()
                .child(Icon::new(if *ok { IconName::CircleCheck } else { IconName::TriangleAlert }).small().text_color(if *ok { color::success() } else { color::warning() }))
                .child(div().flex_1().min_w_0().text_sm().child(msg.clone())),
        );
    }
    body.into_any_element()
}

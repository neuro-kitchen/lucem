//! Histology in the app: the `lumen-hist` workflows run on the shown image, their cells kept by
//! image (many more objects than annotations: they live here, drawn from a raster, and are
//! exported rather than edited one by one).
//!
//! - Stains: the separation used (H-DAB, H&E, H-E-DAB, fitted to a selected region, or a
//!   fluorescence nuclear channel).
//! - Tissue: regions become `Tissue` annotations (undoable).
//! - Cells: detected inside the selected region annotations, tile by tile on the device.
//! - Classifiers: Random Trees trained from the image's classed annotations (regions and points),
//!   and intensity classes from one measurement.
//! - Hotspots: the densest places become point annotations; Export: a table and a summary.

use std::collections::HashMap;
use std::sync::Arc;

use gpui_kit::{AppContext as _, Context, RenderImage};
use lumen_hist::cells::CellDetection;
use lumen_hist::classify::{train_random_trees, Example, FeatureChoice, ObjectClassifier};
use lumen_hist::intensity::{self, IntensityThresholds};
use lumen_hist::slide::{auto_threshold, detect_cells_in_region, SlideRegion, Staining, Tiling};
use lumen_hist::stains::{estimate_background, estimate_macenko, StainVectors};
use lumen_hist::{HistObject, IntensityClass};

use crate::data::store::Store;
use crate::domain::annotation::{Annotation, ClassDoc, Shape};
use crate::domain::command::Command;

/// How nuclei show in the image.
#[derive(Debug, Clone, PartialEq)]
pub enum StainChoice {
    HDab,
    HE,
    HEDab,
    /// Fitted to a region of this image (Macenko).
    Estimated(StainVectors),
    /// Fluorescence: nuclei on this channel.
    Fluorescence(usize),
}

impl StainChoice {
    pub fn label(&self) -> String {
        match self {
            StainChoice::HDab => "H-DAB".into(),
            StainChoice::HE => "H&E".into(),
            StainChoice::HEDab => "H-E-DAB".into(),
            StainChoice::Estimated(s) => format!("{} (fitted)", s.stains.iter().take(2).map(|v| v.name.as_str()).collect::<Vec<_>>().join("-")),
            StainChoice::Fluorescence(c) => format!("Fluorescence, nuclei on channel {}", c + 1),
        }
    }

    fn staining(&self) -> Staining {
        match self {
            StainChoice::HDab => Staining::Brightfield { stains: StainVectors::h_dab() },
            StainChoice::HE => Staining::Brightfield { stains: StainVectors::h_e() },
            StainChoice::HEDab => Staining::Brightfield { stains: StainVectors::h_e_dab() },
            StainChoice::Estimated(s) => Staining::Brightfield { stains: s.clone() },
            StainChoice::Fluorescence(c) => Staining::Fluorescence { nuclear: *c },
        }
    }
}

pub struct HistState {
    /// Cells by image.
    pub cells: HashMap<String, Vec<HistObject>>,
    /// The stains by image (a guess from the image until chosen).
    pub stains: HashMap<String, StainChoice>,
    pub detection: CellDetection,
    /// Fluorescence: the threshold from the image (Otsu's) rather than `detection.threshold`.
    pub auto_threshold: bool,
    pub intensity: IntensityThresholds,
    pub classifier: Option<ObjectClassifier>,
    pub busy: Option<String>,
    /// Bumped when cells change (the canvas redraws them).
    pub rev: u64,
    pub show: bool,
}

impl Default for HistState {
    fn default() -> Self {
        Self {
            cells: HashMap::new(),
            stains: HashMap::new(),
            detection: CellDetection::default(),
            auto_threshold: true,
            intensity: IntensityThresholds { measurement: "Nucleus: DAB OD mean".into(), thresholds: vec![0.2, 0.4, 0.6], filter: None },
            classifier: None,
            busy: None,
            rev: 0,
            show: true,
        }
    }
}

/// The colour of an intensity class.
pub fn intensity_color(c: IntensityClass) -> [u8; 3] {
    match c {
        IntensityClass::Negative => [80, 120, 235],
        IntensityClass::OnePlus => [250, 210, 60],
        IntensityClass::TwoPlus => [245, 140, 40],
        IntensityClass::Positive | IntensityClass::ThreePlus => [225, 50, 50],
    }
}

/// A colour per class name, stable (classes the project does not have).
fn hashed_color(name: &str) -> [u8; 3] {
    let h = name.bytes().fold(2_166_136_261u32, |h, b| (h ^ b as u32).wrapping_mul(16_777_619));
    lumen_view::label_color(h | 1)
}

/// The cells of `rect` (level-0 `x, y, w, h`) as a raster at `ds` level-0 px per pixel: nuclei
/// filled in their colour, cell outlines drawn. `None` when no cell is there.
pub fn cells_raster(cells: &[HistObject], color: &dyn Fn(&HistObject) -> [u8; 3], rect: [f64; 4], ds: f64, alpha: f32) -> Option<Arc<RenderImage>> {
    let (w, h) = (((rect[2] / ds).ceil() as usize).max(1), ((rect[3] / ds).ceil() as usize).max(1));
    let origin = [rect[0], rect[1]];
    let inside: Vec<(usize, &HistObject)> = cells
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            let (x0, y0, x1, y1) = lumen_hist::polygon::bounds(&c.roi);
            x1 >= rect[0] && y1 >= rect[1] && x0 <= rect[0] + rect[2] && y0 <= rect[1] + rect[3]
        })
        .collect();
    if inside.is_empty() {
        return None;
    }
    let (mut nuc, mut cell) = (vec![0u32; w * h], vec![0u32; w * h]);
    for (i, c) in &inside {
        lumen_hist::polygon::fill(&c.roi, &mut cell, w, h, origin, ds, *i as u32 + 1);
        if let Some(n) = &c.nucleus {
            lumen_hist::polygon::fill(n, &mut nuc, w, h, origin, ds, *i as u32 + 1);
        }
    }
    let a_fill = (alpha.clamp(0.0, 1.0) * 150.0) as u8;
    let a_line = (alpha.clamp(0.0, 1.0) * 230.0) as u8;
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (n, c) = (nuc[i], cell[i]);
            let edge = c != 0 && [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)].iter().any(|&(nx, ny)| nx >= w || ny >= h || cell[ny * w + nx] != c);
            let (id, a) = if edge { (c, a_line) } else if n != 0 { (n, a_fill) } else { continue };
            let rgb = color(&cells[id as usize - 1]);
            out[i * 4..i * 4 + 4].copy_from_slice(&[rgb[2], rgb[1], rgb[0], a]);
        }
    }
    Some(lumen_view::bgra_image(w, h, out))
}

impl Store {
    /// The stains used for `image` (chosen, else guessed: RGB is H-DAB, else fluorescence with
    /// nuclei on the channel named like a nuclear stain, else the first).
    pub fn hist_stains(&self, image: &str) -> StainChoice {
        if let Some(c) = self.hist.stains.get(image) {
            return c.clone();
        }
        let Some(p) = self.pyramids.get(image) else { return StainChoice::HDab };
        if p.channels.len() == 3 && p.channels.iter().all(|c| c.range.1 <= 255.0) {
            return StainChoice::HDab;
        }
        let nuclear = p.channels.iter().position(|c| ["dapi", "hoechst", "nuc", "dna"].iter().any(|k| c.label.to_lowercase().contains(k))).unwrap_or(0);
        StainChoice::Fluorescence(nuclear)
    }

    pub fn hist_set_stains(&mut self, choice: StainChoice, cx: &mut Context<Self>) {
        if let Some(img) = self.doc.active_image.clone() {
            self.hist.stains.insert(img, choice);
            cx.notify();
        }
    }

    /// The selected region annotations of the shown image (their outlines, level-0 px, and plane).
    fn hist_selected_regions(&self) -> Vec<(Vec<[f64; 2]>, usize)> {
        let Some(img) = self.doc.active_image.as_deref() else { return Vec::new() };
        self.doc.annotations.iter().filter(|a| a.image == img && self.selection.contains(&a.id) && a.shape.is_region()).map(|a| (a.shape.outline(), a.z.unwrap_or(0))).collect()
    }

    fn hist_job(&mut self, what: &str, cx: &mut Context<Self>) -> Option<(String, Arc<lumen_io::Pyramid>, Arc<super::device::Device>)> {
        if self.hist.busy.is_some() {
            self.set_status("Histology is still working on the last step", cx);
            return None;
        }
        let image = self.doc.active_image.clone()?;
        let Some(p) = self.pyramids.get(&image).cloned() else {
            self.set_status("The image is still opening", cx);
            return None;
        };
        if p.pixel_size_um.is_none() {
            self.set_status("Histology needs the image's pixel size (µm): it has none", cx);
            return None;
        }
        let Some(device) = self.device.clone() else {
            self.ensure_device(cx);
            self.set_status("Starting the device; try again when it is up", cx);
            return None;
        };
        self.hist.busy = Some(what.to_string());
        cx.notify();
        Some((image, p, device))
    }

    /// Fits the stains to the (first) selected region.
    pub fn hist_estimate_stains(&mut self, cx: &mut Context<Self>) {
        let Some((outline, z)) = self.hist_selected_regions().into_iter().next() else {
            return self.set_status("Select a region with both stains (and some background) to fit the stains to", cx);
        };
        let Some(image) = self.doc.active_image.clone() else { return };
        let Some(p) = self.pyramids.get(&image).cloned() else { return };
        if p.channels.len() < 3 {
            return self.set_status("Stains are fitted to RGB (brightfield) images", cx);
        }
        let names = match self.hist_stains(&image) {
            StainChoice::HE => ("Hematoxylin".to_string(), "Eosin".to_string()),
            _ => ("Hematoxylin".to_string(), "DAB".to_string()),
        };
        self.hist.busy = Some("Fitting the stains".into());
        let task = cx.background_spawn(async move {
            let (x0, y0, x1, y1) = lumen_hist::polygon::bounds(&outline);
            let level = (0..p.levels.len()).find(|&l| (x1 - x0).max(y1 - y0) / p.levels[l].downsample <= 2048.0).unwrap_or(p.coarsest());
            let l = &p.levels[level];
            let (x, y) = ((x0.max(0.0) / l.downsample) as usize, (y0.max(0.0) / l.downsample) as usize);
            let (w, h) = (((x1 - x0) / l.downsample).ceil().max(1.0) as usize, ((y1 - y0) / l.downsample).ceil().max(1.0) as usize);
            let planes: Vec<Vec<f32>> = (0..3).map(|c| p.read_region_at(level, c, z, x, y, w, h).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
            let scale: Vec<f32> = (0..3).map(|c| 255.0 / p.channels[c].range.1.max(1e-6)).collect();
            let rgb: Vec<u8> = (0..w * h).flat_map(|i| (0..3).map(move |c| (i, c))).map(|(i, c)| (planes[c][i] * scale[c]).round().clamp(0.0, 255.0) as u8).collect();
            let bg = estimate_background(&rgb);
            estimate_macenko(&rgb, bg, (&names.0, &names.1), 0.15, 1.0).map_err(|e| e.to_string())
        });
        cx.spawn(async move |this, cx| {
            let r = task.await;
            let _ = this.update(cx, |s, cx| {
                s.hist.busy = None;
                match r {
                    Ok(v) => {
                        let msg = format!("Stains fitted: {}", v.stains.iter().take(2).map(|st| format!("{} ({:.2}, {:.2}, {:.2})", st.name, st.od[0], st.od[1], st.od[2])).collect::<Vec<_>>().join(", "));
                        s.hist.stains.insert(image, StainChoice::Estimated(v));
                        s.set_status(msg, cx);
                    }
                    Err(e) => s.set_status(format!("Fitting the stains: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Finds the tissue at a coarse level; its regions become `Tissue` annotations.
    pub fn hist_detect_tissue(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let Some(p) = self.pyramids.get(&image).cloned() else { return };
        let Some(um) = p.pixel_size_um else { return self.set_status("Tissue detection needs the image's pixel size (µm)", cx) };
        let brightfield = !matches!(self.hist_stains(&image), StainChoice::Fluorescence(_));
        self.hist.busy = Some("Finding the tissue".into());
        cx.notify();
        let task = cx.background_spawn(async move {
            let level = (0..p.levels.len()).find(|&l| p.levels[l].width.max(p.levels[l].height) <= 4096).unwrap_or(p.coarsest());
            let l = p.levels[level].clone();
            let channels = if brightfield { 3.min(p.channels.len()) } else { p.channels.len() };
            let planes: Vec<Vec<f32>> = (0..channels).map(|c| p.read_region_at(level, c, 0, 0, 0, l.width, l.height).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
            let scale: Vec<f32> = (0..channels).map(|c| if brightfield { 255.0 / p.channels[c].range.1.max(1e-6) } else { 1.0 }).collect();
            let img: Vec<f32> = (0..l.width * l.height).flat_map(|i| (0..channels).map(move |c| (i, c))).map(|(i, c)| planes[c][i] * scale[c]).collect();
            let grid = lumen_hist::raster::Grid::new(l.width, l.height, [0.0, 0.0], l.downsample);
            Ok::<_, String>(lumen_hist::tissue::detect(&img, channels, grid, um, &lumen_hist::tissue::TissueSettings { brightfield, ..Default::default() }))
        });
        cx.spawn(async move |this, cx| {
            let r = task.await;
            let _ = this.update(cx, |s, cx| {
                s.hist.busy = None;
                match r {
                    Ok(t) if t.regions.is_empty() => s.set_status("No tissue found", cx),
                    Ok(t) => {
                        let mut commands = Vec::new();
                        if s.doc.class("Tissue").is_none() {
                            commands.push(Command::SetClass { class: ClassDoc { name: "Tissue".into(), color: [120, 200, 120], visible: true, opacity: 1.0 } });
                        }
                        let annotations: Vec<Annotation> = t.regions.iter().map(|r| Annotation { id: String::new(), image: image.clone(), shape: Shape::Polygon { points: r.roi.clone() }, class: "Tissue".into(), note: String::new(), z: None }).collect();
                        let n = annotations.len();
                        commands.push(Command::AddAnnotations { annotations });
                        if s.execute(Command::Batch { commands }, cx).is_ok() {
                            let mm2: f64 = t.regions.iter().filter_map(|r| r.get("Area µm^2")).sum::<f64>() / 1e6;
                            s.set_status(format!("{n} tissue region(s), {mm2:.2} mm² (threshold {:.3})", t.threshold), cx);
                        }
                    }
                    Err(e) => s.set_status(format!("Tissue: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Detects cells inside the selected region annotations (tile by tile on the device); they
    /// replace the image's cells inside those regions.
    pub fn hist_detect_cells(&mut self, cx: &mut Context<Self>) {
        let regions = self.hist_selected_regions();
        if regions.is_empty() {
            return self.set_status("Select one or more region annotations (or draw one) to find cells in", cx);
        }
        let stains = self.doc.active_image.as_deref().map(|i| self.hist_stains(i)).unwrap_or(StainChoice::HDab);
        let mut settings = self.hist.detection.clone();
        let auto = self.hist.auto_threshold && matches!(stains, StainChoice::Fluorescence(_));
        let Some((image, p, device)) = self.hist_job("Finding cells", cx) else { return };
        let (tx, rx) = async_channel::unbounded::<(usize, usize)>();
        let task = cx.background_spawn(async move {
            let um = p.pixel_size_um.unwrap_or(1.0);
            let names: Vec<String> = p.channels.iter().map(|c| c.label.clone()).collect();
            let ranges: Vec<f32> = p.channels.iter().map(|c| c.range.1).collect();
            let staining = stains.staining();
            let mut all: Vec<HistObject> = Vec::new();
            let mut threshold_note = String::new();
            for (outline, z) in &regions {
                let (x0, y0, x1, y1) = lumen_hist::polygon::bounds(outline);
                if let (true, Staining::Fluorescence { nuclear }) = (auto, &staining) {
                    let cl = (0..p.levels.len()).find(|&k| (x1 - x0).max(y1 - y0) / p.levels[k].downsample <= 2048.0).unwrap_or(p.coarsest());
                    let c = &p.levels[cl];
                    let (cx0, cy0) = ((x0.max(0.0) / c.downsample) as usize, (y0.max(0.0) / c.downsample) as usize);
                    let (cw, ch) = ((((x1 - x0) / c.downsample).ceil() as usize).max(1), (((y1 - y0) / c.downsample).ceil() as usize).max(1));
                    let v = p.read_region_at(cl, *nuclear, *z, cx0, cy0, cw, ch).map_err(|e| e.to_string())?;
                    settings.threshold = auto_threshold(&v, settings.background_radius_um > 0.0).0;
                    threshold_note = format!(" (threshold {:.1})", settings.threshold);
                }
                let region = SlideRegion { x0: x0.max(0.0) as usize, y0: y0.max(0.0) as usize, width: (x1.min(p.width() as f64) - x0.max(0.0)).max(0.0).ceil() as usize, height: (y1.min(p.height() as f64) - y0.max(0.0)).max(0.0).ceil() as usize, downsample: 1.0, pixel_um: um, within: Some(outline.clone()) };
                let z = *z;
                let mut read = |x: usize, y: usize, w: usize, h: usize| -> lumen_hist::HistResult<Vec<Vec<f32>>> { (0..p.channels.len()).map(|c| p.read_region_at(0, c, z, x, y, w, h).map_err(|e| lumen_hist::HistError::Invalid(e.to_string()))).collect() };
                let mut found = detect_cells_in_region(&device.client, &mut read, &names, &ranges, &staining, &region, &settings, Tiling::default(), &mut |d, t| {
                    let _ = tx.try_send((d, t));
                })
                .map_err(|e| e.to_string())?;
                for o in &mut found {
                    o.id += all.len() as u32;
                }
                all.extend(found);
            }
            Ok::<_, String>((all, regions, threshold_note))
        });
        cx.spawn(async move |this, cx| {
            while let Ok((d, t)) = rx.recv().await {
                if this.update(cx, |s, cx| {
                    s.hist.busy = Some(format!("Finding cells: tile {d} of {t}"));
                    cx.notify();
                }).is_err() {
                    return;
                }
            }
        })
        .detach();
        cx.spawn(async move |this, cx| {
            let r = task.await;
            let _ = this.update(cx, |s, cx| {
                s.hist.busy = None;
                match r {
                    Ok((mut found, regions, note)) => {
                        if let Some(c) = &s.hist.classifier {
                            c.apply(&mut found);
                        }
                        let n = found.len();
                        let list = s.hist.cells.entry(image).or_default();
                        list.retain(|c| !regions.iter().any(|(o, _)| lumen_hist::polygon::contains(o, c.centroid)));
                        list.extend(found);
                        for (i, c) in list.iter_mut().enumerate() {
                            c.id = i as u32 + 1;
                        }
                        s.hist.rev += 1;
                        s.set_status(format!("{n} cell(s) found{note}"), cx);
                    }
                    Err(e) => s.set_status(format!("Cells: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Trains Random Trees on the shown image's cells from its classed annotations (regions and
    /// points; not `Tissue`, `Region`, `Measurement`) and classifies every cell.
    pub fn hist_train_classifier(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let examples: Vec<Example> = self
            .doc
            .annotations_of(&image)
            .filter(|a| !["Tissue", "Region", "Measurement"].contains(&a.class.as_str()))
            .filter_map(|a| match &a.shape {
                Shape::Point { x, y } => Some(Example::Point([*x, *y], a.class.clone())),
                s if s.is_region() => Some(Example::Region(s.outline(), a.class.clone())),
                _ => None,
            })
            .collect();
        let Some(cells) = self.hist.cells.get_mut(&image).filter(|c| !c.is_empty()) else {
            return self.set_status("Find cells first", cx);
        };
        match train_random_trees(cells, &examples, &FeatureChoice::All, false, None, &lumen_hist::forest::ForestSettings::default()) {
            Ok(c) => {
                c.apply(cells);
                let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
                for cell in cells.iter() {
                    *counts.entry(cell.class.clone().unwrap_or_else(|| "none".into())).or_default() += 1;
                }
                self.hist.classifier = Some(c);
                self.hist.rev += 1;
                self.set_status(format!("Classified: {}", counts.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", ")), cx);
            }
            Err(e) => self.set_status(format!("Classifier: {e} — draw regions or points of each class over cells, then train"), cx),
        }
        cx.notify();
    }

    /// Intensity classes of the shown image's cells.
    pub fn hist_apply_intensity(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let t = self.hist.intensity.clone();
        let Some(cells) = self.hist.cells.get_mut(&image).filter(|c| !c.is_empty()) else {
            return self.set_status("Find cells first", cx);
        };
        if t.measurement.is_empty() || !cells.iter().any(|c| c.get(&t.measurement).is_some()) {
            return self.set_status(format!("The cells have no measurement {:?}", t.measurement), cx);
        }
        intensity::classify(cells, &t);
        let s = intensity::summarize(cells.iter());
        self.hist.rev += 1;
        self.set_status(format!("Positive {:.1} % of {} (1+ {}, 2+ {}, 3+ {}; H-score {:.0})", s.positive_percent, s.total, s.one_plus, s.two_plus, s.three_plus, s.h_score), cx);
        cx.notify();
    }

    /// The `count` densest places of the shown image's cells (within 50 µm) as point
    /// annotations of class `Hotspot`.
    pub fn hist_hotspots(&mut self, count: usize, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let Some(um) = self.pyramids.get(&image).and_then(|p| p.pixel_size_um) else { return };
        let Some(cells) = self.hist.cells.get(&image).filter(|c| !c.is_empty()) else {
            return self.set_status("Find cells first", cx);
        };
        let b = cells.iter().map(|c| c.centroid).fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, p| (b.0.min(p[0]), b.1.min(p[1]), b.2.max(p[0]), b.3.max(p[1])));
        let s = lumen_hist::density::DensitySettings { pixel_um: 10.0, radius_um: 50.0, ..Default::default() };
        let map = lumen_hist::density::density_map(cells, (b.0, b.1, b.2 + 1.0, b.3 + 1.0), um, &s);
        let spots = lumen_hist::density::hotspots(&map, count, 200.0, 1.0, um);
        let mut commands = Vec::new();
        if self.doc.class("Hotspot").is_none() {
            commands.push(Command::SetClass { class: ClassDoc { name: "Hotspot".into(), color: [255, 80, 200], visible: true, opacity: 1.0 } });
        }
        let annotations: Vec<Annotation> = spots.iter().map(|(p, v)| Annotation { id: String::new(), image: image.clone(), shape: Shape::Point { x: p[0], y: p[1] }, class: "Hotspot".into(), note: format!("{v:.0} cells within 50 µm"), z: None }).collect();
        let n = annotations.len();
        commands.push(Command::AddAnnotations { annotations });
        if self.execute(Command::Batch { commands }, cx).is_ok() {
            self.set_status(format!("{n} hotspot(s) marked"), cx);
        }
    }

    /// Writes the shown image's cell measurements (TSV) and a summary per region annotation (TSV)
    /// next to the project.
    pub fn hist_export(&mut self, cx: &mut Context<Self>) {
        let Some(img) = self.doc.active().cloned() else { return };
        let Some(um) = self.pyramids.get(&img.id).and_then(|p| p.pixel_size_um) else { return };
        let Some(cells) = self.hist.cells.get(&img.id).filter(|c| !c.is_empty()) else {
            return self.set_status("Find cells first", cx);
        };
        let folder = self.folder().or_else(|| std::env::current_dir().ok()).unwrap_or_default();
        let stem = img.name.replace([' ', '/'], "_");
        let table = lumen_hist::export::measurements_table(cells, &img.name, um);
        let cells_path = folder.join(format!("{stem}.cells.tsv"));
        let mut summary = lumen_hist::export::Table { header: vec!["Image".into(), "Annotation".into(), "Class".into()], rows: Vec::new() };
        let regions: Vec<&Annotation> = self.doc.annotations_of(&img.id).filter(|a| a.shape.is_region()).collect();
        let sums: Vec<lumen_hist::Measurements> = regions.iter().map(|a| lumen_hist::export::region_summary(&a.shape.outline(), cells, um)).collect();
        let keys: std::collections::BTreeSet<&String> = sums.iter().flat_map(|m| m.keys()).collect();
        summary.header.extend(keys.iter().map(|k| k.to_string()));
        for (a, m) in regions.iter().zip(&sums) {
            let mut row = vec![img.name.clone(), a.id.clone(), a.class.clone()];
            row.extend(keys.iter().map(|k| m.get(*k).map_or(String::new(), |v| format!("{v}"))));
            summary.rows.push(row);
        }
        let summary_path = folder.join(format!("{stem}.regions.tsv"));
        let r = std::fs::write(&cells_path, table.to_delimited('\t')).and_then(|_| std::fs::write(&summary_path, summary.to_delimited('\t')));
        let msg = match r {
            Ok(()) => format!("{} cells → {}; {} region(s) → {}", cells.len(), cells_path.display(), regions.len(), summary_path.display()),
            Err(e) => format!("Could not write the measurements: {e}"),
        };
        self.set_status(msg, cx);
    }

    pub fn hist_clear(&mut self, cx: &mut Context<Self>) {
        if let Some(img) = self.doc.active_image.clone() {
            self.hist.cells.remove(&img);
            self.hist.rev += 1;
            self.set_status("Cells cleared", cx);
            cx.notify();
        }
    }

    /// The colour a cell is drawn in: its intensity class, else its class's colour.
    pub fn hist_color(&self) -> impl Fn(&HistObject) -> [u8; 3] + use<> {
        let classes: HashMap<String, [u8; 3]> = self.doc.classes.iter().map(|c| (c.name.clone(), c.color)).collect();
        move |c: &HistObject| match (c.intensity, &c.class) {
            (Some(i), _) => intensity_color(i),
            (None, Some(name)) => classes.get(name).copied().unwrap_or_else(|| hashed_color(name)),
            (None, None) => [90, 200, 255],
        }
    }
}

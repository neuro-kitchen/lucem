//! The one way a histology step runs. A step is a serializable [`Action`] (what the session
//! records); [`apply`] runs it on an image's [`Analysis`] (what the steps before it left) and
//! says what it did as an [`Outcome`] (data: shown by the app, returned to an agent, printed by
//! the command line). The app, `lucem session run` and an agent all go through here, so a step
//! does the same wherever it runs.
//!
//! Training a classifier is not a step: it makes a step's parameters (the classifier) from the
//! user's examples; applying it is the step.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};

use lucem_core::cubecl::prelude::Client;
use lucem_io::Pyramid;
use serde::Serialize;

use crate::area::ClassArea;
use crate::classify::ObjectClassifier;
use crate::density::{DensityMap, DensityValue};
use crate::error::{invalid, HistResult};
use crate::objects::Polygon;
use crate::session::{Action, Within};
use crate::slide::{auto_threshold, cells_in_region, SlideRegion, Staining, Tiling};
use crate::HistObject;

/// What the steps so far left on one image.
#[derive(Debug, Default, Clone)]
pub struct Analysis {
    /// Tissue regions (level-0 outlines).
    pub tissue: Vec<Polygon>,
    pub cells: Vec<HistObject>,
    pub tiles: Vec<HistObject>,
    /// The classifiers in use: new cells (tiles) are classed by them as they are found.
    pub classifier: Option<ObjectClassifier>,
    pub tile_classifier: Option<ObjectClassifier>,
    pub class_map: Option<ClassMap>,
    pub density: Option<Density>,
}

/// Every pixel's class over a region.
#[derive(Debug, Clone)]
pub struct ClassMap {
    /// Level-0 position of the map's first pixel, and level-0 pixels per map pixel.
    pub origin: [f64; 2],
    pub ds: f64,
    pub width: usize,
    pub height: usize,
    /// 0: none (outside the regions), k: `classes[k - 1]`.
    pub map: Vec<u8>,
    pub classes: Vec<String>,
    pub areas: Vec<ClassArea>,
}

/// A density map and how to read it.
#[derive(Debug, Clone)]
pub struct Density {
    pub map: DensityMap,
    pub max: f32,
    /// What a value means ("Tumor: cells per mm²").
    pub label: String,
}

/// What a step did, as data.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "step", rename_all = "snake_case")]
pub enum Outcome {
    Tissue { regions: usize, area_mm2: f64, threshold: f64 },
    Cells { found: usize, total: usize, threshold: Option<f64> },
    Classified { tiles: bool, counts: BTreeMap<String, usize> },
    Intensity { total: usize, positive_percent: f64, one_plus: usize, two_plus: usize, three_plus: usize, h_score: f64 },
    PixelClasses { areas: Vec<ClassArea> },
    Superpixels { found: usize, total: usize },
    Density { max: f32, label: String },
    /// Writing is the front end's (the output store); nothing to compute.
    Export,
}

impl Outcome {
    /// One line for people (the status bar, the command line).
    pub fn summary(&self) -> String {
        match self {
            Outcome::Tissue { regions: 0, .. } => "No tissue found".into(),
            Outcome::Tissue { regions, area_mm2, threshold } => format!("{regions} tissue region(s), {area_mm2:.2} mm² (threshold {threshold:.3})"),
            Outcome::Cells { found, threshold, .. } => format!("{found} cell(s) found{}", threshold.map(|t| format!(" (threshold {t:.1})")).unwrap_or_default()),
            Outcome::Classified { tiles, counts } => format!("Classified {}: {}", if *tiles { "tiles" } else { "cells" }, counts.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", ")),
            Outcome::Intensity { total, positive_percent, one_plus, two_plus, three_plus, h_score } => {
                format!("Positive {positive_percent:.1} % of {total} (1+ {one_plus}, 2+ {two_plus}, 3+ {three_plus}; H-score {h_score:.0})")
            }
            Outcome::PixelClasses { areas } => format!("Pixel classes: {}", areas.iter().map(|a| format!("{} {:.3} mm² ({:.1} %)", a.class, a.area_um2 / 1e6, a.percent)).collect::<Vec<_>>().join(", ")),
            Outcome::Superpixels { found, .. } => format!("{found} superpixel(s)"),
            Outcome::Density { max, label } => format!("Density map: {label}, up to {max:.1}"),
            Outcome::Export => "Exported".into(),
        }
    }
}

/// How a step reaches nucleus models (plugins): the app through its running servers, the
/// command line by starting the plugin programs.
pub trait Models {
    /// The input channels model `id` takes (it is found and started when needed).
    fn channels(&mut self, id: &str) -> HistResult<usize>;
    /// Sends `req` to the plugin running model `id`.
    fn request(&mut self, id: &str, req: &lucem_ml::Request) -> Result<serde_json::Value, String>;
}

/// No model plugins: a step that names a model fails, saying so.
pub struct NoModels;

impl Models for NoModels {
    fn channels(&mut self, id: &str) -> HistResult<usize> {
        Err(invalid(format!("the model {id} is not available here")))
    }
    fn request(&mut self, id: &str, _: &lucem_ml::Request) -> Result<serde_json::Value, String> {
        Err(format!("the model {id} is not available here"))
    }
}

/// The plugin programs found on this machine, each started once when a step needs it.
#[derive(Default)]
pub struct PluginModels {
    found: Option<Vec<lucem_ml::PluginInfo>>,
    running: HashMap<String, lucem_ml::PluginProcess>,
}

impl PluginModels {
    fn find(&mut self, id: &str) -> HistResult<(String, usize)> {
        let found = self.found.get_or_insert_with(|| lucem_ml::discover().into_iter().filter_map(Result::ok).collect());
        found.iter().find_map(|p| p.models.iter().find(|m| m.id == id).map(|m| (p.program.clone(), m.channels as usize))).ok_or_else(|| invalid(format!("no plugin offers the model {id}")))
    }
}

impl Models for PluginModels {
    fn channels(&mut self, id: &str) -> HistResult<usize> {
        Ok(self.find(id)?.1)
    }
    fn request(&mut self, id: &str, req: &lucem_ml::Request) -> Result<serde_json::Value, String> {
        let (program, _) = self.find(id).map_err(|e| e.to_string())?;
        if !self.running.contains_key(&program) {
            self.running.insert(program.clone(), lucem_ml::PluginProcess::start(&program)?);
        }
        let p = self.running.get_mut(&program).ok_or("the plugin stopped")?;
        p.request(req, &mut |_, _| {})
    }
}

/// What a step runs with.
pub struct Ctx<'a> {
    pub client: &'a Client,
    pub pyramid: &'a Pyramid,
    /// The plane (of a volume) it reads.
    pub z: usize,
    pub models: &'a mut dyn Models,
    /// What it is doing, and how far (0..1) when known.
    pub progress: &'a mut dyn FnMut(&str, Option<f32>),
    /// Set to stop the step after its current tile.
    pub cancel: &'a AtomicBool,
}

fn cancelled(cancel: &AtomicBool) -> HistResult<()> {
    if cancel.load(Ordering::Relaxed) { Err(invalid("cancelled")) } else { Ok(()) }
}

/// The outlines a step looks in.
fn outlines(within: &Within, a: &Analysis, whole: &Polygon) -> Vec<Polygon> {
    match within {
        Within::Image => vec![whole.clone()],
        Within::Tissue if !a.tissue.is_empty() => a.tissue.clone(),
        Within::Tissue => vec![whole.clone()],
        Within::Regions { outlines } => outlines.clone(),
    }
}

/// The finest level whose pixels are at most `target_um` (a little over is fine), coarser
/// still while the `w × h` level-0 rectangle would be over `max_side` pixels.
pub fn level_for(p: &Pyramid, um: f64, target_um: f64, w: f64, h: f64, max_side: f64) -> usize {
    let mut level = 0;
    for (k, l) in p.levels.iter().enumerate() {
        if l.downsample * um <= target_um * 1.05 {
            level = k;
        }
    }
    while level < p.coarsest() && w.max(h) / p.levels[level].downsample > max_side {
        level += 1;
    }
    level
}

/// Every channel of a level-0 rectangle read at `level` (plane `z`), scaled to 0..1 by its
/// range: names and values, the level region `[x, y, w, h]` and its downsample.
pub type Channels = (Vec<(String, Vec<f32>)>, [usize; 4], f64);

pub fn read_channels(p: &Pyramid, level: usize, z: usize, b: (f64, f64, f64, f64)) -> HistResult<Channels> {
    let l = &p.levels[level];
    let ds = l.downsample;
    let x = (b.0.max(0.0) / ds) as usize;
    let y = (b.1.max(0.0) / ds) as usize;
    let w = ((b.2.min(p.width() as f64) / ds).ceil() as usize).saturating_sub(x).max(1);
    let h = ((b.3.min(p.height() as f64) / ds).ceil() as usize).saturating_sub(y).max(1);
    let channels = p
        .channels
        .iter()
        .enumerate()
        .map(|(c, ch)| {
            let scale = 1.0 / ch.range.1.max(1e-6);
            p.read_region_at(level, c, z, x, y, w, h).map(|v| (ch.label.clone(), v.into_iter().map(|v| v * scale).collect())).map_err(|e| invalid(e.to_string()))
        })
        .collect::<HistResult<_>>()?;
    Ok((channels, [x, y, w, h], ds))
}

fn union_bounds(outlines: &[Polygon]) -> Option<(f64, f64, f64, f64)> {
    outlines.iter().map(|o| crate::polygon::bounds(o)).reduce(|a, b| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)))
}

fn renumber(objects: &mut [HistObject]) {
    for (i, o) in objects.iter_mut().enumerate() {
        o.id = i as u32 + 1;
    }
}

fn class_counts(objects: &[HistObject]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for o in objects {
        *counts.entry(o.class.clone().unwrap_or_else(|| "none".into())).or_default() += 1;
    }
    counts
}

/// Runs `action` on `a`.
pub fn apply(ctx: &mut Ctx, a: &mut Analysis, action: &Action) -> HistResult<Outcome> {
    let p = ctx.pyramid;
    let um = p.pixel_size_um.ok_or_else(|| invalid("the image has no pixel size (µm)"))?;
    let (iw, ih) = (p.width() as f64, p.height() as f64);
    let whole: Polygon = vec![[0.0, 0.0], [iw, 0.0], [iw, ih], [0.0, ih]];
    (ctx.progress)(action.title(), None);
    match action {
        Action::Tissue { settings } => {
            let level = (0..p.levels.len()).find(|&l| p.levels[l].width.max(p.levels[l].height) <= 4096).unwrap_or(p.coarsest());
            let l = p.levels[level].clone();
            let n_ch = if settings.brightfield { 3.min(p.channels.len()) } else { p.channels.len() };
            let planes: Vec<Vec<f32>> = (0..n_ch).map(|c| p.read_region_at(level, c, ctx.z, 0, 0, l.width, l.height).map_err(|e| invalid(e.to_string()))).collect::<HistResult<_>>()?;
            let scale: Vec<f32> = (0..n_ch).map(|c| if settings.brightfield { 255.0 / p.channels[c].range.1.max(1e-6) } else { 1.0 }).collect();
            let img: Vec<f32> = (0..l.width * l.height).flat_map(|i| (0..n_ch).map(move |c| (i, c))).map(|(i, c)| planes[c][i] * scale[c]).collect();
            let t = crate::tissue::detect(&img, n_ch, crate::raster::Grid::new(l.width, l.height, [0.0, 0.0], l.downsample), um, settings);
            let area_mm2 = t.regions.iter().filter_map(|r| r.get("Area µm^2")).sum::<f64>() / 1e6;
            a.tissue = t.regions.into_iter().map(|r| r.roi).collect();
            Ok(Outcome::Tissue { regions: a.tissue.len(), area_mm2, threshold: t.threshold })
        }
        Action::Cells { within, staining, detection, auto_threshold: auto, model } => {
            let regions = outlines(within, a, &whole);
            let names: Vec<String> = p.channels.iter().map(|c| c.label.clone()).collect();
            let ranges: Vec<f32> = p.channels.iter().map(|c| c.range.1).collect();
            let model_channels = match model {
                Some(id) => Some(ctx.models.channels(id)?),
                None => None,
            };
            // Models read at about their own scale (≤ 0.55 µm per pixel), the threshold at full
            let level = if model.is_some() { crate::models::model_level(&p.levels.iter().map(|l| l.downsample).collect::<Vec<_>>(), um) } else { 0 };
            let ds = p.levels[level].downsample;
            let (lw, lh) = (p.levels[level].width as f64, p.levels[level].height as f64);
            let (z, cancel) = (ctx.z, ctx.cancel);
            let mut found_all: Vec<HistObject> = Vec::new();
            let mut threshold = None;
            for outline in &regions {
                cancelled(cancel)?;
                let (x0, y0, x1, y1) = crate::polygon::bounds(outline);
                let mut settings = detection.clone();
                if let (true, Staining::Fluorescence { nuclear }, None) = (*auto, staining, model) {
                    let cl = (0..p.levels.len()).find(|&k| (x1 - x0).max(y1 - y0) / p.levels[k].downsample <= 2048.0).unwrap_or(p.coarsest());
                    let c = &p.levels[cl];
                    let (cx0, cy0) = ((x0.max(0.0) / c.downsample) as usize, (y0.max(0.0) / c.downsample) as usize);
                    let (cw, chh) = ((((x1 - x0) / c.downsample).ceil() as usize).max(1), (((y1 - y0) / c.downsample).ceil() as usize).max(1));
                    let v = p.read_region_at(cl, *nuclear, z, cx0, cy0, cw, chh).map_err(|e| invalid(e.to_string()))?;
                    settings.threshold = auto_threshold(&v, settings.background_radius_um > 0.0).0;
                    threshold = Some(settings.threshold);
                }
                let (rx0, ry0) = ((x0.max(0.0) / ds).floor(), (y0.max(0.0) / ds).floor());
                let region = SlideRegion { x0: rx0 as usize, y0: ry0 as usize, width: ((x1 / ds).min(lw) - rx0).max(0.0).ceil() as usize, height: ((y1 / ds).min(lh) - ry0).max(0.0).ceil() as usize, downsample: ds, pixel_um: um, within: Some(outline.clone()) };
                let mut read = |x: usize, y: usize, w: usize, h: usize| -> HistResult<Vec<Vec<f32>>> {
                    cancelled(cancel)?;
                    (0..p.channels.len()).map(|c| p.read_region_at(level, c, z, x, y, w, h).map_err(|e| invalid(e.to_string()))).collect()
                };
                let progress = &mut *ctx.progress;
                let mut tiles_done = |d: usize, t: usize| progress(&format!("Finding cells: tile {d} of {t}"), Some(d as f32 / t.max(1) as f32));
                let mut found = match (model, model_channels) {
                    (Some(id), Some(channels)) => {
                        let nuclear = match staining {
                            Staining::Fluorescence { nuclear } => *nuclear,
                            Staining::Brightfield { .. } => 0,
                        };
                        let models = &mut *ctx.models;
                        let mut send = |req: &lucem_ml::Request| models.request(id, req);
                        let mut labeller = |raw: &[Vec<f32>], w: usize, h: usize| crate::models::model_nuclei(&mut send, id, channels, nuclear, raw, &ranges, w, h, um * ds);
                        cells_in_region(ctx.client, &mut read, &names, &ranges, staining, &region, &settings, Tiling { tile: 2048, overlap: 64 }, Some(&mut labeller), &mut tiles_done)?
                    }
                    _ => cells_in_region(ctx.client, &mut read, &names, &ranges, staining, &region, &settings, Tiling::default(), None, &mut tiles_done)?,
                };
                found_all.append(&mut found);
            }
            if let Some(c) = &a.classifier {
                c.apply(&mut found_all);
            }
            let n = found_all.len();
            // The new cells replace those inside the regions looked in
            a.cells.retain(|c| !regions.iter().any(|o| crate::polygon::contains(o, c.centroid)));
            a.cells.extend(found_all);
            renumber(&mut a.cells);
            Ok(Outcome::Cells { found: n, total: a.cells.len(), threshold })
        }
        Action::ClassifyObjects { classifier, tiles } => {
            let list = if *tiles { &mut a.tiles } else { &mut a.cells };
            classifier.apply(list);
            let counts = class_counts(list);
            if *tiles {
                a.tile_classifier = Some(classifier.clone());
            } else {
                a.classifier = Some(classifier.clone());
            }
            Ok(Outcome::Classified { tiles: *tiles, counts })
        }
        Action::Intensity { thresholds } => {
            if a.cells.is_empty() {
                return Err(invalid("find cells first"));
            }
            if thresholds.measurement.is_empty() || !a.cells.iter().any(|c| c.get(&thresholds.measurement).is_some()) {
                return Err(invalid(format!("the cells have no measurement {:?}", thresholds.measurement)));
            }
            crate::intensity::classify(&mut a.cells, thresholds);
            let s = crate::intensity::summarize(a.cells.iter());
            Ok(Outcome::Intensity { total: s.total, positive_percent: s.positive_percent, one_plus: s.one_plus, two_plus: s.two_plus, three_plus: s.three_plus, h_score: s.h_score })
        }
        Action::Superpixels { within, settings, pixel_um } => {
            let regions = outlines(within, a, &whole);
            let mut found_all: Vec<HistObject> = Vec::new();
            for (i, outline) in regions.iter().enumerate() {
                cancelled(ctx.cancel)?;
                (ctx.progress)(&format!("Superpixels: region {} of {}", i + 1, regions.len()), Some(i as f32 / regions.len().max(1) as f32));
                let b = crate::polygon::bounds(outline);
                let level = level_for(p, um, *pixel_um, b.2 - b.0, b.3 - b.1, 4096.0);
                let (channels, [x, y, w, h], ds) = read_channels(p, level, ctx.z, b)?;
                let refs: Vec<(String, &[f32])> = channels.iter().map(|(n, v)| (n.clone(), v.as_slice())).collect();
                let grid = crate::raster::Grid::new(w, h, [x as f64 * ds, y as f64 * ds], ds);
                let (_, objects) = crate::superpixel::superpixels(ctx.client, &refs, grid, um, settings)?;
                found_all.extend(objects.into_iter().filter(|o| crate::polygon::contains(outline, o.centroid)));
            }
            if let Some(c) = &a.tile_classifier {
                c.apply(&mut found_all);
            }
            let n = found_all.len();
            a.tiles.retain(|c| !regions.iter().any(|o| crate::polygon::contains(o, c.centroid)));
            a.tiles.extend(found_all);
            renumber(&mut a.tiles);
            Ok(Outcome::Superpixels { found: n, total: a.tiles.len() })
        }
        Action::PixelClasses { classifier, pixel_um, within } => {
            let regions = outlines(within, a, &whole);
            let b = union_bounds(&regions).ok_or_else(|| invalid("nowhere to class pixels"))?;
            let level = level_for(p, um, *pixel_um, b.2 - b.0, b.3 - b.1, 4096.0);
            let (channels, [x, y, w, h], ds) = read_channels(p, level, ctx.z, b)?;
            cancelled(ctx.cancel)?;
            let refs: Vec<(String, &[f32])> = channels.iter().map(|(n, v)| (n.clone(), v.as_slice())).collect();
            let mut map = crate::pixel::classify(ctx.client, classifier, &refs, w, h)?;
            let origin = [x as f64 * ds, y as f64 * ds];
            // Only inside the regions (the whole image needs no mask)
            let inside: Option<Vec<u8>> = (!matches!(within, Within::Image)).then(|| {
                let mut m = vec![0u32; w * h];
                for o in &regions {
                    crate::polygon::fill(o, &mut m, w, h, origin, ds, 1);
                }
                m.into_iter().map(|v| u8::from(v != 0)).collect()
            });
            if let Some(m) = &inside {
                for (c, &i) in map.iter_mut().zip(m) {
                    if i == 0 {
                        *c = 0;
                    }
                }
            }
            let px_um2 = (ds * um) * (ds * um);
            let areas = crate::area::class_areas(&map, &classifier.classes, inside.as_deref(), px_um2);
            a.class_map = Some(ClassMap { origin, ds, width: w, height: h, map, classes: classifier.classes.clone(), areas: areas.clone() });
            Ok(Outcome::PixelClasses { areas })
        }
        Action::Density { settings: s } => {
            if a.cells.is_empty() {
                return Err(invalid("find cells first"));
            }
            let b = a.cells.iter().map(|c| c.centroid).fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, q| (b.0.min(q[0]), b.1.min(q[1]), b.2.max(q[0]), b.3.max(q[1])));
            let pad = s.radius_um / um;
            let map = crate::density::density_map(&a.cells, (b.0 - pad, b.1 - pad, b.2 + pad + 1.0, b.3 + pad + 1.0), um, s);
            let max = map.values.iter().copied().fold(0.0f32, f32::max);
            if max <= 0.0 {
                return Err(invalid("no cell counted (check the class chosen)"));
            }
            let unit = match s.value {
                DensityValue::Count => format!("cells within {:.0} µm", s.radius_um),
                DensityValue::PerMm2 => "cells per mm²".into(),
                DensityValue::PercentPositive => format!("% positive within {:.0} µm", s.radius_um),
            };
            let label = format!("{}{unit}", s.class.as_ref().map(|c| format!("{c}: ")).unwrap_or_default());
            a.density = Some(Density { map, max, label: label.clone() });
            Ok(Outcome::Density { max, label })
        }
        Action::Export => Ok(Outcome::Export),
    }
}

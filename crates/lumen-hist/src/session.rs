//! The session: every step that changed the analysis, in order, with its parameters — what
//! was done, kept as JSON (`session.json` next to the project, and in each output's
//! provenance), so the same analysis runs again without the app: `lumen session run
//! session.json <images…>` on a server, image after image.
//!
//! Only steps that change data are recorded (finding tissue or cells, training, scoring,
//! classing pixels, exporting); views, panels and selections are not.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::cells::CellDetection;
use crate::classify::ObjectClassifier;
use crate::density::DensitySettings;
use crate::error::{invalid, HistResult};
use crate::intensity::IntensityThresholds;
use crate::objects::Polygon;
use crate::pixel::PixelClassifier;
use crate::slide::Staining;
use crate::superpixel::SuperpixelSettings;
use crate::tissue::TissueSettings;

/// The format this crate reads and writes (`lumen_session` field).
pub const SESSION_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub lumen_session: u32,
    #[serde(default)]
    pub steps: Vec<Step>,
}

impl Default for Session {
    fn default() -> Self {
        Self { lumen_session: SESSION_VERSION, steps: Vec::new() }
    }
}

/// One step: when (seconds since 1970), on which image (its name), and what.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub at: u64,
    pub image: String,
    #[serde(flatten)]
    pub action: Action,
}

/// Where a step looks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "in", rename_all = "snake_case")]
pub enum Within {
    /// The whole image.
    Image,
    /// The tissue found (a `tissue` step before it, or found then).
    Tissue,
    /// These regions (level-0 outlines); on another image, the tissue is used instead.
    Regions { outlines: Vec<Polygon> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    /// Regions of class Tissue.
    Tissue { settings: TissueSettings },
    /// Cells found and measured (`model`: a nucleus model plugin, else the threshold).
    Cells {
        within: Within,
        staining: Staining,
        detection: CellDetection,
        /// Fluorescence: the threshold from each image (Otsu's) rather than `detection`'s.
        #[serde(default)]
        auto_threshold: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    /// The cells (or tiles) classed by a trained classifier.
    ClassifyObjects {
        classifier: ObjectClassifier,
        #[serde(default)]
        tiles: bool,
    },
    /// Intensity classes (negative, 1+, 2+, 3+).
    Intensity { thresholds: IntensityThresholds },
    /// Every pixel classed.
    PixelClasses { classifier: PixelClassifier, pixel_um: f64 },
    /// Superpixel tiles.
    Superpixels { within: Within, settings: SuperpixelSettings, pixel_um: f64 },
    /// A density map.
    Density { settings: DensitySettings },
    /// Everything written to the output store.
    Export,
}

impl Action {
    pub fn title(&self) -> &'static str {
        match self {
            Action::Tissue { .. } => "Find tissue",
            Action::Cells { .. } => "Find cells",
            Action::ClassifyObjects { .. } => "Classify",
            Action::Intensity { .. } => "Intensity classes",
            Action::PixelClasses { .. } => "Class pixels",
            Action::Superpixels { .. } => "Superpixels",
            Action::Density { .. } => "Density map",
            Action::Export => "Export",
        }
    }
}

impl Session {
    pub fn load(path: &Path) -> HistResult<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        let s: Session = serde_json::from_str(&text).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        if s.lumen_session != SESSION_VERSION {
            return Err(invalid(format!("{}: session version {}, this lumen reads {SESSION_VERSION}", path.display(), s.lumen_session)));
        }
        Ok(s)
    }

    pub fn save(&self, path: &Path) -> HistResult<()> {
        let text = serde_json::to_string_pretty(self).map_err(|e| invalid(e.to_string()))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| invalid(format!("{}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, path).map_err(|e| invalid(format!("{}: {e}", path.display())))
    }

    /// Records a step on `image` now.
    pub fn push(&mut self, image: &str, action: Action) {
        let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        self.steps.push(Step { at, image: image.to_string(), action });
    }

    /// The steps to run on another image: the last of each kind, in their order (a step done
    /// again replaces the earlier one: the session keeps what was settled on).
    pub fn recipe(&self) -> Vec<&Action> {
        let mut out: Vec<&Action> = Vec::new();
        for s in &self.steps {
            let kind = std::mem::discriminant(&s.action);
            out.retain(|a| std::mem::discriminant(*a) != kind);
            out.push(&s.action);
        }
        out
    }
}

// ----------------------------------------------------------------------------
// Running a session on an image (headless)
// ----------------------------------------------------------------------------

/// What a run gives for one image.
#[derive(Debug, Default)]
pub struct RunResult {
    pub tissue: Vec<Polygon>,
    pub cells: Vec<crate::HistObject>,
    pub tiles: Vec<crate::HistObject>,
    /// Steps skipped, and why.
    pub notes: Vec<String>,
}

fn read_channels(p: &lumen_io::Pyramid, level: usize, b: (f64, f64, f64, f64)) -> HistResult<(Vec<(String, Vec<f32>)>, [usize; 4], f64)> {
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
            p.read_region_at(level, c, 0, x, y, w, h).map(|v| (ch.label.clone(), v.into_iter().map(|v| v * scale).collect())).map_err(|e| invalid(e.to_string()))
        })
        .collect::<HistResult<_>>()?;
    Ok((channels, [x, y, w, h], ds))
}

/// Runs `recipe` (see [`Session::recipe`]) on one image: tissue, cells (the threshold or the
/// nucleus model named, through its plugin), classes, intensity, superpixels. Pixel classes
/// and density maps are not run headless yet (said in `notes`).
pub fn run(client: &lumen_core::cubecl::prelude::Client, p: &lumen_io::Pyramid, recipe: &[&Action], progress: &mut dyn FnMut(&str)) -> HistResult<RunResult> {
    use crate::slide::{auto_threshold, cells_in_region, SlideRegion, Tiling};
    let um = p.pixel_size_um.ok_or_else(|| invalid("the image has no pixel size (µm)"))?;
    let (iw, ih) = (p.width() as f64, p.height() as f64);
    let whole: Polygon = vec![[0.0, 0.0], [iw, 0.0], [iw, ih], [0.0, ih]];
    let mut out = RunResult::default();
    let mut plugin: Option<(lumen_ml::PluginProcess, usize)> = None;
    for action in recipe {
        progress(action.title());
        match action {
            Action::Tissue { settings } => {
                let level = (0..p.levels.len()).find(|&l| p.levels[l].width.max(p.levels[l].height) <= 4096).unwrap_or(p.coarsest());
                let l = p.levels[level].clone();
                let n_ch = if settings.brightfield { 3.min(p.channels.len()) } else { p.channels.len() };
                let planes: Vec<Vec<f32>> = (0..n_ch).map(|c| p.read_region_at(level, c, 0, 0, 0, l.width, l.height).map_err(|e| invalid(e.to_string()))).collect::<HistResult<_>>()?;
                let scale: Vec<f32> = (0..n_ch).map(|c| if settings.brightfield { 255.0 / p.channels[c].range.1.max(1e-6) } else { 1.0 }).collect();
                let img: Vec<f32> = (0..l.width * l.height).flat_map(|i| (0..n_ch).map(move |c| (i, c))).map(|(i, c)| planes[c][i] * scale[c]).collect();
                let t = crate::tissue::detect(&img, n_ch, crate::raster::Grid::new(l.width, l.height, [0.0, 0.0], l.downsample), um, settings);
                out.tissue = t.regions.into_iter().map(|r| r.roi).collect();
            }
            Action::Cells { within, staining, detection, auto_threshold: auto, model } => {
                let outlines: Vec<Polygon> = match within {
                    Within::Image => vec![whole.clone()],
                    Within::Tissue | Within::Regions { .. } if !out.tissue.is_empty() => out.tissue.clone(),
                    _ => {
                        out.notes.push("cells: no tissue found first; the whole image was searched".into());
                        vec![whole.clone()]
                    }
                };
                let names: Vec<String> = p.channels.iter().map(|c| c.label.clone()).collect();
                let ranges: Vec<f32> = p.channels.iter().map(|c| c.range.1).collect();
                if let (Some(id), None) = (model, &plugin) {
                    let (program, info) = lumen_ml::discover().into_iter().filter_map(Result::ok).find_map(|pl| pl.models.iter().find(|m| &m.id == id).map(|m| (pl.program.clone(), m.clone()))).ok_or_else(|| invalid(format!("no plugin offers the model {id}")))?;
                    plugin = Some((lumen_ml::PluginProcess::start(&program).map_err(invalid)?, info.channels as usize));
                }
                let level = if model.is_some() { crate::models::model_level(&p.levels.iter().map(|l| l.downsample).collect::<Vec<_>>(), um) } else { 0 };
                let ds = p.levels[level].downsample;
                let (lw, lh) = (p.levels[level].width as f64, p.levels[level].height as f64);
                for outline in &outlines {
                    let (x0, y0, x1, y1) = crate::polygon::bounds(outline);
                    let mut settings = detection.clone();
                    if let (true, Staining::Fluorescence { nuclear }, None) = (*auto, staining, model) {
                        let cl = (0..p.levels.len()).find(|&k| (x1 - x0).max(y1 - y0) / p.levels[k].downsample <= 2048.0).unwrap_or(p.coarsest());
                        let c = &p.levels[cl];
                        let (cx0, cy0) = ((x0.max(0.0) / c.downsample) as usize, (y0.max(0.0) / c.downsample) as usize);
                        let (cw, chh) = ((((x1 - x0) / c.downsample).ceil() as usize).max(1), (((y1 - y0) / c.downsample).ceil() as usize).max(1));
                        let v = p.read_region_at(cl, *nuclear, 0, cx0, cy0, cw, chh).map_err(|e| invalid(e.to_string()))?;
                        settings.threshold = auto_threshold(&v, settings.background_radius_um > 0.0).0;
                    }
                    let (rx0, ry0) = ((x0.max(0.0) / ds).floor(), (y0.max(0.0) / ds).floor());
                    let region = SlideRegion { x0: rx0 as usize, y0: ry0 as usize, width: ((x1 / ds).min(lw) - rx0).max(0.0).ceil() as usize, height: ((y1 / ds).min(lh) - ry0).max(0.0).ceil() as usize, downsample: ds, pixel_um: um, within: Some(outline.clone()) };
                    let mut read = |x: usize, y: usize, w: usize, h: usize| -> HistResult<Vec<Vec<f32>>> { (0..p.channels.len()).map(|c| p.read_region_at(level, c, 0, x, y, w, h).map_err(|e| invalid(e.to_string()))).collect() };
                    let mut tiles_done = |d: usize, t: usize| progress(&format!("Find cells: tile {d} of {t}"));
                    let mut found = match (&mut plugin, model) {
                        (Some((proc, channels)), Some(id)) => {
                            let nuclear = match staining {
                                Staining::Fluorescence { nuclear } => *nuclear,
                                Staining::Brightfield { .. } => 0,
                            };
                            let channels = *channels;
                            let mut send = |req: &lumen_ml::Request| proc.request(req, &mut |_, _| {});
                            let mut labeller = |raw: &[Vec<f32>], w: usize, h: usize| crate::models::model_nuclei(&mut send, id, channels, nuclear, raw, &ranges, w, h, um * ds);
                            cells_in_region(client, &mut read, &names, &ranges, staining, &region, &settings, Tiling { tile: 2048, overlap: 64 }, Some(&mut labeller), &mut tiles_done)?
                        }
                        _ => cells_in_region(client, &mut read, &names, &ranges, staining, &region, &settings, Tiling::default(), None, &mut tiles_done)?,
                    };
                    for o in &mut found {
                        o.id += out.cells.len() as u32;
                    }
                    out.cells.extend(found);
                }
            }
            Action::ClassifyObjects { classifier, tiles } => {
                classifier.apply(if *tiles { &mut out.tiles } else { &mut out.cells });
            }
            Action::Intensity { thresholds } => {
                crate::intensity::classify(&mut out.cells, thresholds);
            }
            Action::Superpixels { settings, pixel_um, .. } => {
                let outlines = if out.tissue.is_empty() { vec![whole.clone()] } else { out.tissue.clone() };
                for outline in &outlines {
                    let b = crate::polygon::bounds(outline);
                    let mut level = 0;
                    for (k, l) in p.levels.iter().enumerate() {
                        if l.downsample * um <= pixel_um * 1.05 {
                            level = k;
                        }
                    }
                    while level < p.coarsest() && (b.2 - b.0).max(b.3 - b.1) / p.levels[level].downsample > 4096.0 {
                        level += 1;
                    }
                    let (channels, [x, y, w, h], ds) = read_channels(p, level, b)?;
                    let refs: Vec<(String, &[f32])> = channels.iter().map(|(n, v)| (n.clone(), v.as_slice())).collect();
                    let grid = crate::raster::Grid::new(w, h, [x as f64 * ds, y as f64 * ds], ds);
                    let (_, objects) = crate::superpixel::superpixels(client, &refs, grid, um, settings)?;
                    out.tiles.extend(objects.into_iter().filter(|o| crate::polygon::contains(outline, o.centroid)));
                }
            }
            Action::PixelClasses { .. } => out.notes.push("pixel classes: run in the app for now (not headless yet)".into()),
            Action::Density { .. } => out.notes.push("density map: run in the app for now (not headless yet)".into()),
            Action::Export => {}
        }
    }
    for (i, o) in out.cells.iter_mut().enumerate() {
        o.id = i as u32 + 1;
    }
    for (i, o) in out.tiles.iter_mut().enumerate() {
        o.id = i as u32 + 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_reads_back_and_keeps_the_last_of_each_step() {
        let mut s = Session::default();
        s.push("a", Action::Tissue { settings: TissueSettings::default() });
        s.push("a", Action::Cells { within: Within::Tissue, staining: Staining::Fluorescence { nuclear: 0 }, detection: CellDetection::default(), auto_threshold: true, model: None });
        let detection = CellDetection { threshold: 300.0, ..CellDetection::default() };
        s.push("a", Action::Cells { within: Within::Image, staining: Staining::Fluorescence { nuclear: 0 }, detection: detection.clone(), auto_threshold: false, model: Some("stardist-fluo".into()) });
        s.push("a", Action::Export);
        let path = std::env::temp_dir().join(format!("lumen-session-{}.json", std::process::id()));
        s.save(&path).unwrap();
        let back = Session::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back, s);
        let r = back.recipe();
        assert_eq!(r.len(), 3, "tissue, the last cells, export");
        assert!(matches!(r[1], Action::Cells { model: Some(_), detection: d, .. } if *d == detection));
    }
}

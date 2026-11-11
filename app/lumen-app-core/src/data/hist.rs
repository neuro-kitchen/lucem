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
//! - Pixel classifier: Random Trees on multiscale pixel features, trained from classed regions
//!   and brush strokes; the class map is a layer, its areas are measured, its regions can become
//!   annotations.
//! - Superpixels: SLIC tiles inside the selected regions, classified like cells.
//! - Density map: cells (of a class, or positive ones) within a radius, as a layer.
//!
//! Saved with the project in `<folder>/histology/`: `settings.json` (stains, settings,
//! classifiers) and `<image>.objects.json.gz` (cells and tiles).

use std::collections::HashMap;
use std::sync::Arc;

use std::path::{Path, PathBuf};

use gpui_kit::{AppContext as _, Context, RenderImage};
use lumen_hist::area::ClassArea;
use lumen_hist::cells::CellDetection;
use lumen_hist::density::DensitySettings;
use lumen_hist::pixel::PixelClassifier;
use lumen_hist::superpixel::SuperpixelSettings;
use serde::{Deserialize, Serialize};
use lumen_hist::classify::{train_random_trees, Example, FeatureChoice, ObjectClassifier};
use lumen_hist::intensity::{self, IntensityThresholds};
use lumen_hist::slide::{auto_threshold, detect_cells_in_region, SlideRegion, Staining, Tiling};
use lumen_hist::stains::{estimate_background, estimate_macenko, StainVectors};
use lumen_hist::{HistObject, IntensityClass};

use crate::data::store::Store;
use crate::domain::annotation::{Annotation, ClassDoc, Shape};
use crate::domain::command::Command;

/// How nuclei show in the image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// Superpixel tiles by image (classified like cells, by their own classifier).
    pub tiles: HashMap<String, Vec<HistObject>>,
    pub tile_classifier: Option<ObjectClassifier>,
    pub superpixel: SuperpixelSettings,
    /// The pixel classifier and the resolution it works at (µm per pixel).
    pub pixel_classifier: Option<PixelClassifier>,
    pub pixel_um: f64,
    /// Class maps by image.
    pub class_maps: HashMap<String, ClassMap>,
    pub density: DensitySettings,
    /// Density maps by image.
    pub density_maps: HashMap<String, DensityLayer>,
    /// The class and density maps drawn.
    pub show_maps: bool,
    /// The model plugin that finds nuclei (its model id), else the threshold of `detection`.
    pub detector: Option<String>,
    /// The recipe card open in the Histology panel.
    pub recipe: Recipe,
    /// How the last step went (`true`: done; `false`: a problem or what is missing), shown
    /// under the button that ran it.
    pub outcome: Option<(bool, String)>,
    /// How far a running step is (0–1), and its cancel flag (checked between tiles).
    pub progress: Option<f32>,
    pub cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Every step that changed the analysis, in order (`session.json` next to the project;
    /// `lumen session run` runs it again on other images).
    pub session: lumen_hist::session::Session,
}

/// The Histology panel's recipes: a named protocol each, run on the selected regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Recipe {
    #[default]
    Count,
    Positivity,
    CellClasses,
    TissueClasses,
    Superpixels,
    Density,
}

impl Recipe {
    pub const ALL: [Recipe; 6] = [Recipe::Count, Recipe::Positivity, Recipe::CellClasses, Recipe::TissueClasses, Recipe::Superpixels, Recipe::Density];

    pub fn title(self) -> &'static str {
        match self {
            Recipe::Count => "Count cells",
            Recipe::Positivity => "Positive % and H-score",
            Recipe::CellClasses => "Cell classes",
            Recipe::TissueClasses => "Tissue classes",
            Recipe::Superpixels => "Superpixel classes",
            Recipe::Density => "Density and hotspots",
        }
    }

    /// One line: what it gives.
    pub fn purpose(self) -> &'static str {
        match self {
            Recipe::Count => "Every cell in the selected regions, measured.",
            Recipe::Positivity => "Cells scored negative, 1+, 2+, 3+ by one stain.",
            Recipe::CellClasses => "Cells classed like the shapes you class over them.",
            Recipe::TissueClasses => "Every pixel classed from painted examples; areas measured.",
            Recipe::Superpixels => "Tiles that follow edges, classed without finding cells.",
            Recipe::Density => "Cells per area as a map, and the densest spots.",
        }
    }

    pub fn icon(self) -> gpui_kit::assets::IconName {
        use gpui_kit::assets::IconName;
        match self {
            Recipe::Count => IconName::Sparkles,
            Recipe::Positivity => IconName::Gauge,
            Recipe::CellClasses => IconName::BrainCircuit,
            Recipe::TissueClasses => IconName::Palette,
            Recipe::Superpixels => IconName::Grid3x3,
            Recipe::Density => IconName::Flame,
        }
    }
}

/// Whether an error says the GPU stopped responding (a lost or poisoned device).
pub fn is_device_lost(msg: &str) -> bool {
    let m = msg.to_lowercase();
    m.contains("poisoned") || m.contains("device is lost") || m.contains("device lost") || m.contains("devicelost")
}

/// What a lost device is called to a person.
pub const DEVICE_LOST: &str = "The GPU stopped responding (often a job too large for it). lumen restarts it on the next run: run again, or try a smaller region.";

/// A pixel classifier's output over a region of an image.
pub struct ClassMap {
    /// Level-0 `x, y, w, h` it covers; `ds` level-0 px per map pixel.
    pub rect: [f64; 4],
    pub width: usize,
    pub height: usize,
    pub ds: f64,
    /// `0`: none (outside the region), `k`: `classes[k − 1]`.
    pub map: Vec<u8>,
    pub classes: Vec<String>,
    pub areas: Vec<ClassArea>,
    pub image: Arc<RenderImage>,
}

pub struct DensityLayer {
    pub rect: [f64; 4],
    pub max: f32,
    pub label: String,
    pub image: Arc<RenderImage>,
}

/// What `settings.json` holds.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct HistDoc {
    stains: std::collections::BTreeMap<String, StainChoice>,
    detection: CellDetection,
    auto_threshold: bool,
    intensity: Option<IntensityThresholds>,
    classifier: Option<ObjectClassifier>,
    tile_classifier: Option<ObjectClassifier>,
    superpixel: SuperpixelSettings,
    pixel_classifier: Option<PixelClassifier>,
    pixel_um: f64,
    density: DensitySettings,
    detector: Option<String>,
}

/// What `<image>.objects.json.gz` holds.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct ObjectsDoc {
    cells: Vec<HistObject>,
    tiles: Vec<HistObject>,
}

/// Classes a pixel or object classifier never learns from.
const NOT_TRAINING: [&str; 5] = ["Tissue", "Region", "Measurement", "Hotspot", "Ignore*"];

fn histology_dir(folder: &Path) -> PathBuf {
    folder.join("histology")
}

fn objects_file(folder: &Path, image: &str) -> PathBuf {
    let safe: String = image.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    histology_dir(folder).join(format!("{safe}.objects.json.gz"))
}

fn write_gz(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let tmp = path.with_extension("tmp");
    let mut e = flate2::write::GzEncoder::new(std::fs::File::create(&tmp)?, flate2::Compression::fast());
    e.write_all(bytes)?;
    e.finish()?;
    std::fs::rename(tmp, path)
}

fn read_gz(path: &Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read as _;
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path)?).read_to_end(&mut out)?;
    Ok(out)
}

/// The histology of a project folder: settings and the objects of the images given.
pub(crate) fn load_folder(folder: &Path, images: &[String]) -> (HistState, Vec<String>) {
    let mut st = HistState::default();
    let mut errors = Vec::new();
    let dir = histology_dir(folder);
    let session_path = folder.join("session.json");
    if session_path.is_file() {
        match lumen_hist::session::Session::load(&session_path) {
            Ok(sess) => st.session = sess,
            Err(e) => errors.push(e.to_string()),
        }
    }
    if let Ok(text) = std::fs::read_to_string(dir.join("settings.json")) {
        match serde_json::from_str::<HistDoc>(&text) {
            Ok(d) => {
                st.stains = d.stains.into_iter().collect();
                st.detection = d.detection;
                st.auto_threshold = d.auto_threshold;
                if let Some(t) = d.intensity {
                    st.intensity = t;
                }
                st.classifier = d.classifier;
                st.tile_classifier = d.tile_classifier;
                st.superpixel = d.superpixel;
                st.pixel_classifier = d.pixel_classifier;
                if d.pixel_um > 0.0 {
                    st.pixel_um = d.pixel_um;
                }
                if d.density.pixel_um > 0.0 {
                    st.density = d.density;
                }
                st.detector = d.detector;
            }
            Err(e) => errors.push(format!("histology settings: {e}")),
        }
    }
    for img in images {
        let f = objects_file(folder, img);
        if !f.is_file() {
            continue;
        }
        match read_gz(&f).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<ObjectsDoc>(&b).map_err(|e| e.to_string())) {
            Ok(o) => {
                if !o.cells.is_empty() {
                    st.cells.insert(img.clone(), o.cells);
                }
                if !o.tiles.is_empty() {
                    st.tiles.insert(img.clone(), o.tiles);
                }
            }
            Err(e) => errors.push(format!("{}: {e}", f.display())),
        }
    }
    (st, errors)
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
            tiles: HashMap::new(),
            tile_classifier: None,
            superpixel: SuperpixelSettings::default(),
            pixel_classifier: None,
            pixel_um: 2.0,
            class_maps: HashMap::new(),
            density: DensitySettings::default(),
            density_maps: HashMap::new(),
            show_maps: true,
            detector: None,
            recipe: Recipe::Count,
            outcome: None,
            progress: None,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            session: Default::default(),
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
/// Selected objects (`selected`: their indices) are outlined in white.
pub fn cells_raster(cells: &[HistObject], color: &dyn Fn(&HistObject) -> [u8; 3], rect: [f64; 4], ds: f64, alpha: f32, selected: &std::collections::HashSet<usize>) -> Option<Arc<RenderImage>> {
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
        match &c.nucleus {
            Some(n) => lumen_hist::polygon::fill(n, &mut nuc, w, h, origin, ds, *i as u32 + 1),
            // A tile or detection: filled when it has a class (else its outline only)
            None if c.class.is_some() || c.intensity.is_some() => lumen_hist::polygon::fill(&c.roi, &mut nuc, w, h, origin, ds, *i as u32 + 1),
            None => {}
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
            let picked = selected.contains(&(id as usize - 1));
            let (rgb, a) = if picked && edge { ([255, 255, 255], 255) } else { (color(&cells[id as usize - 1]), a) };
            out[i * 4..i * 4 + 4].copy_from_slice(&[rgb[2], rgb[1], rgb[0], a]);
        }
    }
    Some(lumen_view::bgra_image(w, h, out))
}

impl Store {
    /// The outcome of a histology step: under its button in the panel, and on the status bar.
    /// A lost GPU is said in plain words and the device is dropped (it opens again on the next
    /// run, never in a loop).
    pub fn hist_report(&mut self, ok: bool, msg: impl Into<String>, cx: &mut Context<Self>) {
        let mut msg = msg.into();
        if !ok && is_device_lost(&msg) {
            self.drop_device(cx);
            msg = DEVICE_LOST.to_string();
        }
        self.hist.outcome = Some((ok, msg.clone()));
        self.hist.progress = None;
        self.set_status(msg, cx);
    }

    /// Records a step of the session (on the shown image) and saves it with the project.
    pub fn hist_record(&mut self, action: lumen_hist::session::Action, cx: &mut Context<Self>) {
        let name = self.doc.active().map(|i| i.name.clone()).unwrap_or_default();
        self.hist.session.push(&name, action);
        let Some(folder) = self.folder() else { return };
        let session = self.hist.session.clone();
        cx.background_spawn(async move {
            let _ = session.save(&folder.join("session.json"));
        })
        .detach();
    }

    /// Whether the image is brightfield (RGB stains) rather than fluorescence.
    pub fn hist_is_brightfield(&self, image: &str) -> bool {
        !matches!(self.hist_stains(image), StainChoice::Fluorescence(_))
    }

    /// The nucleus detector: a model id, or `None` for the threshold.
    pub fn hist_set_detector(&mut self, id: Option<String>, cx: &mut Context<Self>) {
        self.hist.detector = id;
        self.hist.outcome = None;
        cx.notify();
    }

    /// Stops the running step after its current tile.
    pub fn hist_cancel(&mut self, cx: &mut Context<Self>) {
        self.hist.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        cx.notify();
    }

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
            self.hist_report(false, "Histology is still working on the last step", cx);
            return None;
        }
        let image = self.doc.active_image.clone()?;
        let Some(p) = self.pyramids.get(&image).cloned() else {
            self.hist_report(false, "The image is still opening", cx);
            return None;
        };
        if p.pixel_size_um.is_none() {
            self.hist_report(false, "Histology needs the image's pixel size (µm): it has none", cx);
            return None;
        }
        let Some(device) = self.device.clone() else {
            self.ensure_device(cx);
            self.hist_report(false, "Starting the device; try again when it is up", cx);
            return None;
        };
        self.hist.busy = Some(what.to_string());
        self.hist.outcome = None;
        self.hist.progress = Some(0.0);
        self.hist.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        cx.notify();
        Some((image, p, device))
    }

    /// Fits the stains to the (first) selected region.
    pub fn hist_estimate_stains(&mut self, cx: &mut Context<Self>) {
        let Some((outline, z)) = self.hist_selected_regions().into_iter().next() else {
            return self.hist_report(false, "Select a region with both stains (and some background) to fit the stains to", cx);
        };
        let Some(image) = self.doc.active_image.clone() else { return };
        let Some(p) = self.pyramids.get(&image).cloned() else { return };
        if p.channels.len() < 3 {
            return self.hist_report(false, "Stains are fitted to RGB (brightfield) images", cx);
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
                        s.hist_report(true, msg, cx);
                    }
                    Err(e) => s.hist_report(false, format!("Fitting the stains: {e}"), cx),
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
        let Some(um) = p.pixel_size_um else { return self.hist_report(false, "Tissue detection needs the image's pixel size (µm)", cx) };
        let brightfield = !matches!(self.hist_stains(&image), StainChoice::Fluorescence(_));
        let record = lumen_hist::session::Action::Tissue { settings: lumen_hist::tissue::TissueSettings { brightfield, ..Default::default() } };
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
                    Ok(t) if t.regions.is_empty() => s.hist_report(true, "No tissue found", cx),
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
                            s.hist_record(record.clone(), cx);
                            s.hist_report(true, format!("{n} tissue region(s), {mm2:.2} mm² (threshold {:.3})", t.threshold), cx);
                        }
                    }
                    Err(e) => s.hist_report(false, format!("Tissue: {e}"), cx),
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
            return self.hist_report(false, "Select one or more region annotations (or draw one) to find cells in", cx);
        }
        let stains = self.doc.active_image.as_deref().map(|i| self.hist_stains(i)).unwrap_or(StainChoice::HDab);
        let mut settings = self.hist.detection.clone();
        // A model plugin finds the nuclei, when one is chosen
        let model = match self.hist.detector.clone() {
            None => None,
            Some(id) => match self.ml.nucleus_models().into_iter().find(|(_, m)| m.id == id) {
                None => return self.hist_report(false, format!("The nucleus model {id:?} is not found: Settings → Models (or choose Threshold)"), cx),
                Some((_, m)) if !m.installed => return self.hist_report(false, format!("{} is not installed: Settings → Models → Install", m.name), cx),
                Some((program, m)) => Some((self.ml.server(&program), program, m.id, m.channels as usize)),
            },
        };
        let auto = model.is_none() && self.hist.auto_threshold && matches!(stains, StainChoice::Fluorescence(_));
        let record = lumen_hist::session::Action::Cells {
            within: lumen_hist::session::Within::Regions { outlines: regions.iter().map(|(o, _)| o.clone()).collect() },
            staining: stains.staining(),
            detection: self.hist.detection.clone(),
            auto_threshold: auto,
            model: self.hist.detector.clone(),
        };
        let Some((image, p, device)) = self.hist_job(if model.is_some() { "Finding cells with the model" } else { "Finding cells" }, cx) else { return };
        let cancel = self.hist.cancel.clone();
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
                // Models read at about their own scale (≤ 0.55 µm per pixel), the threshold at full
                let level = if model.is_some() { lumen_hist::models::model_level(&p.levels.iter().map(|l| l.downsample).collect::<Vec<_>>(), um) } else { 0 };
                let ds = p.levels[level].downsample;
                let (lw, lh) = (p.levels[level].width as f64, p.levels[level].height as f64);
                let (rx0, ry0) = ((x0.max(0.0) / ds).floor(), (y0.max(0.0) / ds).floor());
                let region = SlideRegion { x0: rx0 as usize, y0: ry0 as usize, width: ((x1 / ds).min(lw) - rx0).max(0.0).ceil() as usize, height: ((y1 / ds).min(lh) - ry0).max(0.0).ceil() as usize, downsample: ds, pixel_um: um, within: Some(outline.clone()) };
                let z = *z;
                let mut read = |x: usize, y: usize, w: usize, h: usize| -> lumen_hist::HistResult<Vec<Vec<f32>>> {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        return Err(lumen_hist::HistError::Invalid("cancelled".into()));
                    }
                    (0..p.channels.len()).map(|c| p.read_region_at(level, c, z, x, y, w, h).map_err(|e| lumen_hist::HistError::Invalid(e.to_string()))).collect()
                };
                let mut progress = |d, t| {
                    let _ = tx.try_send((d, t));
                };
                let mut found = match &model {
                    None => detect_cells_in_region(&device.client, &mut read, &names, &ranges, &staining, &region, &settings, Tiling::default(), &mut progress),
                    Some((server, program, id, channels)) => {
                        let mut send = |req: &lumen_ml::Request| super::ml::request(server, program, req, &mut |_, _| {});
                        let nuclear = match &staining {
                            Staining::Fluorescence { nuclear } => *nuclear,
                            Staining::Brightfield { .. } => 0,
                        };
                        let mut labeller = |raw: &[Vec<f32>], w: usize, h: usize| lumen_hist::models::model_nuclei(&mut send, id, *channels, nuclear, raw, &ranges, w, h, um * ds);
                        lumen_hist::slide::cells_in_region(&device.client, &mut read, &names, &ranges, &staining, &region, &settings, Tiling { tile: 2048, overlap: 64 }, Some(&mut labeller), &mut progress)
                    }
                }
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
                    s.hist.progress = Some(d as f32 / t.max(1) as f32);
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
                        let image_saved = image.clone();
                        let list = s.hist.cells.entry(image).or_default();
                        list.retain(|c| !regions.iter().any(|(o, _)| lumen_hist::polygon::contains(o, c.centroid)));
                        list.extend(found);
                        for (i, c) in list.iter_mut().enumerate() {
                            c.id = i as u32 + 1;
                        }
                        s.hist.rev += 1;
                        s.hist_persist(vec![image_saved], cx);
                        s.hist_record(record.clone(), cx);
                        s.hist_report(true, format!("{n} cell(s) found{note}"), cx);
                    }
                    Err(e) if e.contains("cancelled") => s.hist_report(false, "Cancelled: the cells found before were kept", cx),
                    Err(e) => s.hist_report(false, format!("Finding cells failed: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// The shown image's classed annotations an object classifier learns from (regions and
    /// points; not `Tissue`, `Region`, `Measurement`, `Hotspot`).
    fn hist_examples(&self, image: &str) -> Vec<Example> {
        self.doc
            .annotations_of(image)
            .filter(|a| !NOT_TRAINING.contains(&a.class.as_str()))
            .filter_map(|a| match &a.shape {
                Shape::Point { x, y } => Some(Example::Point([*x, *y], a.class.clone())),
                s if s.is_region() => Some(Example::Region(s.outline(), a.class.clone())),
                _ => None,
            })
            .collect()
    }

    /// Trains Random Trees on the shown image's cells (and, separately, its superpixel tiles)
    /// from its classed annotations and classifies them all.
    pub fn hist_train_classifier(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let examples = self.hist_examples(&image);
        let mut done = Vec::new();
        let mut errors = Vec::new();
        let mut records = Vec::new();
        for tiles in [false, true] {
            let list = if tiles { self.hist.tiles.get_mut(&image) } else { self.hist.cells.get_mut(&image) };
            let Some(objects) = list.filter(|c| !c.is_empty()) else { continue };
            match train_random_trees(objects, &examples, &FeatureChoice::All, false, None, &lumen_hist::forest::ForestSettings::default()) {
                Ok(c) => {
                    c.apply(objects);
                    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
                    for o in objects.iter() {
                        *counts.entry(o.class.clone().unwrap_or_else(|| "none".into())).or_default() += 1;
                    }
                    done.push(format!("{}: {}", if tiles { "tiles" } else { "cells" }, counts.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", ")));
                    records.push(lumen_hist::session::Action::ClassifyObjects { classifier: c.clone(), tiles });
                    if tiles {
                        self.hist.tile_classifier = Some(c);
                    } else {
                        self.hist.classifier = Some(c);
                    }
                }
                Err(e) => errors.push(format!("{}: {e}", if tiles { "tiles" } else { "cells" })),
            }
        }
        if done.is_empty() && errors.is_empty() {
            return self.hist_report(false, "Find cells or superpixels first", cx);
        }
        self.hist.rev += 1;
        self.hist_persist(vec![image], cx);
        for r in records {
            self.hist_record(r, cx);
        }
        let msg = if errors.is_empty() {
            format!("Classified {}", done.join("; "))
        } else {
            format!("{} — draw regions or points of each class over the objects, then train", [done, errors].concat().join("; "))
        };
        self.hist_report(true, msg, cx);
        cx.notify();
    }

    /// Intensity classes of the shown image's cells.
    pub fn hist_apply_intensity(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let t = self.hist.intensity.clone();
        let Some(cells) = self.hist.cells.get_mut(&image).filter(|c| !c.is_empty()) else {
            return self.hist_report(false, "Find cells first", cx);
        };
        if t.measurement.is_empty() || !cells.iter().any(|c| c.get(&t.measurement).is_some()) {
            return self.hist_report(false, format!("The cells have no measurement {:?}", t.measurement), cx);
        }
        intensity::classify(cells, &t);
        let s = intensity::summarize(cells.iter());
        self.hist.rev += 1;
        self.hist_persist(vec![image], cx);
        self.hist_record(lumen_hist::session::Action::Intensity { thresholds: t.clone() }, cx);
        self.hist_report(true, format!("Positive {:.1} % of {} (1+ {}, 2+ {}, 3+ {}; H-score {:.0})", s.positive_percent, s.total, s.one_plus, s.two_plus, s.three_plus, s.h_score), cx);
        cx.notify();
    }

    /// The `count` densest places of the shown image's cells (within 50 µm) as point
    /// annotations of class `Hotspot`.
    pub fn hist_hotspots(&mut self, count: usize, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let Some(um) = self.pyramids.get(&image).and_then(|p| p.pixel_size_um) else { return };
        let Some(cells) = self.hist.cells.get(&image).filter(|c| !c.is_empty()) else {
            return self.hist_report(false, "Find cells first", cx);
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
            self.hist_report(true, format!("{n} hotspot(s) marked"), cx);
        }
    }

    /// Writes the shown image's analysis to its output store (in the background): inside its
    /// OME-Zarr when it can be written, else `<project>/analysis/<image>.lumen.zarr` —
    /// `labels/cells`, `labels/nuclei`, `lumen/cells.parquet`, `lumen/tiles.parquet`,
    /// `lumen/shapes.parquet` (GeoParquet), `lumen/provenance.json` (the session, the pipeline,
    /// the version).
    pub fn hist_export(&mut self, cx: &mut Context<Self>) {
        let Some(img) = self.doc.active().cloned() else { return };
        let Some(p) = self.pyramids.get(&img.id).cloned() else { return };
        let Some(um) = p.pixel_size_um else { return self.hist_report(false, "Export needs the image's pixel size (µm)", cx) };
        let cells = self.hist.cells.get(&img.id).cloned().unwrap_or_default();
        let tiles = self.hist.tiles.get(&img.id).cloned().unwrap_or_default();
        if cells.is_empty() && tiles.is_empty() {
            return self.hist_report(false, "Find cells or superpixels first: there is nothing to export", cx);
        }
        let folder = self.folder();
        let image_path = crate::domain::project::ProjectDoc::resolve(&img.path, folder.as_deref());
        let fallback = folder.clone().or_else(|| std::env::current_dir().ok()).unwrap_or_default().join("analysis");
        let hier = self.hierarchy(&img.id);
        let cell_regions = hier.object_parent.clone();
        let tile_h = crate::domain::hierarchy::Hierarchy::build(self.doc.annotations_of(&img.id), &tiles);
        let tile_regions = tile_h.object_parent.clone();
        let regions: Vec<(String, String, Vec<[f64; 2]>)> = self.doc.annotations_of(&img.id).filter(|a| a.shape.is_region()).map(|a| (a.id.clone(), a.class.clone(), a.shape.outline())).collect();
        self.hist_record(lumen_hist::session::Action::Export, cx);
        let provenance = serde_json::json!({
            "lumen": env!("CARGO_PKG_VERSION"),
            "image": image_path.display().to_string(),
            "run": "lumen app",
            "session": self.hist.session,
            "pipeline": self.doc.graph,
        });
        self.hist.busy = Some("Exporting".into());
        cx.notify();
        let name = img.name.clone();
        let task = cx.background_spawn(async move {
            use lumen_hist::output;
            let store = output::store_for(&image_path, &name, &fallback);
            if store != image_path {
                output::ensure_group(&store).map_err(|e| e.to_string())?;
            }
            if !cells.is_empty() {
                output::write_objects(&store, "cells", &cells, &cell_regions, um).map_err(|e| e.to_string())?;
                output::write_labels(&store, "cells", &cells, false, (p.width(), p.height()), Some(um)).map_err(|e| e.to_string())?;
                output::write_labels(&store, "nuclei", &cells, true, (p.width(), p.height()), Some(um)).map_err(|e| e.to_string())?;
            }
            if !tiles.is_empty() {
                output::write_objects(&store, "tiles", &tiles, &tile_regions, um).map_err(|e| e.to_string())?;
            }
            let shapes: Vec<output::RegionShape> = regions.iter().map(|(id, class, o)| output::RegionShape { id, class, outline: o }).collect();
            output::write_shapes(&store, &[cells.clone(), tiles.clone()].concat(), &shapes).map_err(|e| e.to_string())?;
            output::write_provenance(&store, &provenance).map_err(|e| e.to_string())?;
            Ok::<_, String>((store, cells.len(), tiles.len()))
        });
        cx.spawn(async move |this, cx| {
            let r = task.await;
            let _ = this.update(cx, |s, cx| {
                s.hist.busy = None;
                match r {
                    Ok((store, c, t)) => s.hist_report(true, format!("{c} cells, {t} tiles → {} (labels, Parquet tables, GeoParquet shapes, provenance)", store.display()), cx),
                    Err(e) => s.hist_report(false, format!("Export failed: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    pub fn hist_clear(&mut self, cx: &mut Context<Self>) {
        if let Some(img) = self.doc.active_image.clone() {
            self.hist.cells.remove(&img);
            self.hist.tiles.remove(&img);
            self.hist.rev += 1;
            self.hist_persist(vec![img], cx);
            self.hist_report(true, "Cells and tiles cleared", cx);
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

/// The finest level whose pixels are at most `target_um` (a little over is fine), coarser
/// still while the `w × h` level-0 rectangle would be over `max_side` pixels.
fn level_for(p: &lumen_io::Pyramid, um: f64, target_um: f64, w: f64, h: f64, max_side: f64) -> usize {
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

/// Every channel of a level-0 rectangle read at `level`, scaled to 0..1 by its range: names and
/// values, the level region `(x, y, w, h)` and its downsample.
type Channels = (Vec<(String, Vec<f32>)>, [usize; 4], f64);

fn read_channels(p: &lumen_io::Pyramid, level: usize, z: usize, b: (f64, f64, f64, f64)) -> Result<Channels, String> {
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
            p.read_region_at(level, c, z, x, y, w, h).map(|v| (ch.label.clone(), v.into_iter().map(|v| v * scale).collect())).map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    Ok((channels, [x, y, w, h], ds))
}

fn union_bounds(outlines: impl Iterator<Item = Vec<[f64; 2]>>) -> Option<(f64, f64, f64, f64)> {
    outlines.map(|o| lumen_hist::polygon::bounds(&o)).reduce(|a, b| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)))
}

impl Store {
    /// Writes the histology settings and the objects of `images` into the project's folder (in
    /// the background); nothing until the project is saved somewhere.
    pub(crate) fn hist_persist(&mut self, images: Vec<String>, cx: &mut Context<Self>) {
        let Some(folder) = self.folder() else { return };
        let doc = HistDoc {
            stains: self.hist.stains.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            detection: self.hist.detection.clone(),
            auto_threshold: self.hist.auto_threshold,
            intensity: Some(self.hist.intensity.clone()),
            classifier: self.hist.classifier.clone(),
            tile_classifier: self.hist.tile_classifier.clone(),
            superpixel: self.hist.superpixel.clone(),
            pixel_classifier: self.hist.pixel_classifier.clone(),
            pixel_um: self.hist.pixel_um,
            density: self.hist.density.clone(),
            detector: self.hist.detector.clone(),
        };
        let objects: Vec<(String, ObjectsDoc)> = images
            .into_iter()
            .map(|i| {
                let d = ObjectsDoc { cells: self.hist.cells.get(&i).cloned().unwrap_or_default(), tiles: self.hist.tiles.get(&i).cloned().unwrap_or_default() };
                (i, d)
            })
            .collect();
        let task = cx.background_spawn(async move {
            std::fs::create_dir_all(histology_dir(&folder)).map_err(|e| e.to_string())?;
            let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
            std::fs::write(histology_dir(&folder).join("settings.json"), text).map_err(|e| e.to_string())?;
            for (img, o) in objects {
                let f = objects_file(&folder, &img);
                if o.cells.is_empty() && o.tiles.is_empty() {
                    let _ = std::fs::remove_file(&f);
                    continue;
                }
                let bytes = serde_json::to_vec(&o).map_err(|e| e.to_string())?;
                write_gz(&f, &bytes).map_err(|e| format!("{}: {e}", f.display()))?;
            }
            Ok::<_, String>(())
        });
        cx.spawn(async move |this, cx| {
            if let Err(e) = task.await {
                let _ = this.update(cx, |s, cx| s.set_status(format!("Histology not saved: {e}"), cx));
            }
        })
        .detach();
    }

    /// Saves every image's histology (when the project is saved somewhere new).
    pub(crate) fn hist_persist_all(&mut self, cx: &mut Context<Self>) {
        let images: Vec<String> = self.doc.images.iter().map(|i| i.id.clone()).collect();
        self.hist_persist(images, cx);
    }

    /// Reads the project's histology (in the background) after it opens.
    pub(crate) fn hist_load(&mut self, cx: &mut Context<Self>) {
        self.hist = HistState::default();
        let Some(folder) = self.folder() else { return };
        let images: Vec<String> = self.doc.images.iter().map(|i| i.id.clone()).collect();
        let task = cx.background_spawn(async move { load_folder(&folder, &images) });
        cx.spawn(async move |this, cx| {
            let (st, errors) = task.await;
            let _ = this.update(cx, |s, cx| {
                let n: usize = st.cells.values().map(Vec::len).sum();
                s.hist = st;
                s.hist.rev += 1;
                if !errors.is_empty() {
                    s.set_status(format!("Histology not read: {}", errors.join("; ")), cx);
                } else if n > 0 {
                    s.set_status(format!("Opened project {} ({n} cells)", s.doc.name), cx);
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Trains the pixel classifier from the shown image's classed regions and brush strokes and
    /// classifies the selected regions (or where it was trained); the class map becomes a layer
    /// and its areas are measured.
    pub fn hist_train_pixels(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let training: Vec<Annotation> = self
            .doc
            .annotations_of(&image)
            .filter(|a| matches!(a.shape, Shape::Stroke { erase: true, .. }) || (!NOT_TRAINING.contains(&a.class.as_str()) && (a.shape.is_region() || matches!(a.shape, Shape::Stroke { erase: false, .. }))))
            .cloned()
            .collect();
        let classes: Vec<String> = training.iter().filter(|a| !matches!(a.shape, Shape::Stroke { erase: true, .. })).map(|a| a.class.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        if classes.len() < 2 {
            return self.hist_report(false, "Draw regions or brush strokes of at least two classes (Tumor, Stroma, …) to train a pixel classifier", cx);
        }
        let selected: Vec<Annotation> = self.doc.annotations_of(&image).filter(|a| self.selection.contains(&a.id) && a.shape.is_region() && !classes.contains(&a.class)).cloned().collect();
        let colors: Vec<[u8; 3]> = classes.iter().map(|c| self.doc.class(c).map(|c| c.color).unwrap_or_else(|| hashed_color(c))).collect();
        let target_um = self.hist.pixel_um;
        let rec_um = target_um;
        let z = training.iter().find_map(|a| a.z).unwrap_or(0);
        let Some((image, p, device)) = self.hist_job("Training the pixel classifier", cx) else { return };
        let task = cx.background_spawn(async move {
            let um = p.pixel_size_um.unwrap_or(1.0);
            let train_b = union_bounds(training.iter().map(|a| a.shape.outline())).ok_or("nothing to train on")?;
            let area_b = union_bounds(selected.iter().map(|a| a.shape.outline())).unwrap_or(train_b);
            let b = (train_b.0.min(area_b.0), train_b.1.min(area_b.1), train_b.2.max(area_b.2), train_b.3.max(area_b.3));
            let level = level_for(&p, um, target_um, b.2 - b.0, b.3 - b.1, 4096.0);
            let (channels, [x, y, w, h], ds) = read_channels(&p, level, z, b)?;
            let refs: Vec<(String, &[f32])> = channels.iter().map(|(n, v)| (n.clone(), v.as_slice())).collect();
            let mut labels = vec![0u8; w * h];
            for (k, class) in classes.iter().enumerate() {
                let list: Vec<&Annotation> = training.iter().filter(|a| &a.class == class || matches!(a.shape, Shape::Stroke { erase: true, .. })).collect();
                let m = crate::domain::annotation::rasterize(&list, Some(class), x, y, w, h, ds);
                for (l, v) in labels.iter_mut().zip(m) {
                    if v != 0 {
                        *l = k as u8 + 1;
                    }
                }
            }
            let t = lumen_hist::pixel::PixelTraining::default();
            let c = lumen_hist::pixel::train(&device.client, &refs, w, h, &labels, &classes, &t).map_err(|e| e.to_string())?;
            let mut map = lumen_hist::pixel::classify(&device.client, &c, &refs, w, h).map_err(|e| e.to_string())?;
            // Only inside the selected regions (when some are)
            let within: Option<Vec<u8>> = (!selected.is_empty()).then(|| {
                let refs: Vec<&Annotation> = selected.iter().collect();
                crate::domain::annotation::rasterize(&refs, None, x, y, w, h, ds).into_iter().map(|v| u8::from(v != 0)).collect()
            });
            if let Some(m) = &within {
                for (c, &inside) in map.iter_mut().zip(m) {
                    if inside == 0 {
                        *c = 0;
                    }
                }
            }
            let px_um2 = (ds * um) * (ds * um);
            let areas = lumen_hist::area::class_areas(&map, &classes, within.as_deref(), px_um2);
            let mut bgra = vec![0u8; w * h * 4];
            for (i, &k) in map.iter().enumerate() {
                if k != 0 {
                    let [r, g, b] = colors[k as usize - 1];
                    bgra[i * 4..i * 4 + 4].copy_from_slice(&[b, g, r, 110]);
                }
            }
            let rect = [x as f64 * ds, y as f64 * ds, w as f64 * ds, h as f64 * ds];
            let image = lumen_view::bgra_image(w, h, bgra);
            Ok::<_, String>((c, ClassMap { rect, width: w, height: h, ds, map, classes, areas, image }))
        });
        cx.spawn(async move |this, cx| {
            let r = task.await;
            let _ = this.update(cx, |s, cx| {
                s.hist.busy = None;
                match r {
                    Ok((c, m)) => {
                        s.hist_record(lumen_hist::session::Action::PixelClasses { classifier: c.clone(), pixel_um: rec_um }, cx);
                        let msg = format!("Pixel classes: {}", m.areas.iter().map(|a| format!("{} {:.3} mm² ({:.1} %)", a.class, a.area_um2 / 1e6, a.percent)).collect::<Vec<_>>().join(", "));
                        s.hist.pixel_classifier = Some(c);
                        s.hist.class_maps.insert(image.clone(), m);
                        s.hist.show_maps = true;
                        s.hist.rev += 1;
                        s.hist_persist(vec![image], cx);
                        s.hist_report(true, msg, cx);
                    }
                    Err(e) => s.hist_report(false, format!("Pixel classifier: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// The shown image's class map as region annotations of its classes (undoable).
    pub fn hist_class_regions(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let Some(m) = self.hist.class_maps.get(&image) else { return self.hist_report(false, "Train the pixel classifier first", cx) };
        let um = self.pyramids.get(&image).and_then(|p| p.pixel_size_um).unwrap_or(1.0);
        let px_um2 = (m.ds * um) * (m.ds * um);
        // Pieces under 100 µm² dropped, holes under 100 µm² filled
        let min_px = (100.0 / px_um2).max(4.0);
        let grid = lumen_hist::raster::Grid::new(m.width, m.height, [m.rect[0], m.rect[1]], m.ds);
        let mut annotations = Vec::new();
        for (k, class) in m.classes.iter().enumerate() {
            for poly in lumen_hist::pixel::class_regions(&m.map, grid, k as u8 + 1, min_px, min_px) {
                annotations.push(Annotation { id: String::new(), image: image.clone(), shape: Shape::Polygon { points: poly }, class: class.clone(), note: "pixel classifier".into(), z: None });
            }
        }
        if annotations.is_empty() {
            return self.hist_report(false, "The class map has no region large enough", cx);
        }
        let n = annotations.len();
        if self.execute(Command::AddAnnotations { annotations }, cx).is_ok() {
            self.hist_report(true, format!("{n} class region(s) added (note \"pixel classifier\"); Ctrl+Z removes them"), cx);
        }
    }

    pub fn hist_remove_maps(&mut self, cx: &mut Context<Self>) {
        if let Some(img) = self.doc.active_image.clone() {
            self.hist.class_maps.remove(&img);
            self.hist.density_maps.remove(&img);
            self.hist.rev += 1;
            cx.notify();
        }
    }

    /// SLIC superpixels inside the selected regions: tiles with measurements, classified by the
    /// tile classifier when there is one; they replace the tiles inside those regions.
    pub fn hist_superpixels(&mut self, cx: &mut Context<Self>) {
        let regions = self.hist_selected_regions();
        if regions.is_empty() {
            return self.hist_report(false, "Select one or more region annotations to cut into superpixels", cx);
        }
        let settings = self.hist.superpixel.clone();
        let target_um = self.hist.pixel_um;
        let record = lumen_hist::session::Action::Superpixels { within: lumen_hist::session::Within::Regions { outlines: regions.iter().map(|(o, _)| o.clone()).collect() }, settings: settings.clone(), pixel_um: target_um };
        let Some((image, p, device)) = self.hist_job("Making superpixels", cx) else { return };
        let task = cx.background_spawn(async move {
            let um = p.pixel_size_um.unwrap_or(1.0);
            let mut all: Vec<HistObject> = Vec::new();
            for (outline, z) in &regions {
                let b = lumen_hist::polygon::bounds(outline);
                let level = level_for(&p, um, target_um, b.2 - b.0, b.3 - b.1, 4096.0);
                let (channels, [x, y, w, h], ds) = read_channels(&p, level, *z, b)?;
                let refs: Vec<(String, &[f32])> = channels.iter().map(|(n, v)| (n.clone(), v.as_slice())).collect();
                let grid = lumen_hist::raster::Grid::new(w, h, [x as f64 * ds, y as f64 * ds], ds);
                let (_, objects) = lumen_hist::superpixel::superpixels(&device.client, &refs, grid, um, &settings).map_err(|e| e.to_string())?;
                all.extend(objects.into_iter().filter(|o| lumen_hist::polygon::contains(outline, o.centroid)));
            }
            Ok::<_, String>((all, regions))
        });
        cx.spawn(async move |this, cx| {
            let r = task.await;
            let _ = this.update(cx, |s, cx| {
                s.hist.busy = None;
                match r {
                    Ok((mut found, regions)) => {
                        if let Some(c) = &s.hist.tile_classifier {
                            c.apply(&mut found);
                        }
                        let n = found.len();
                        let list = s.hist.tiles.entry(image.clone()).or_default();
                        list.retain(|c| !regions.iter().any(|(o, _)| lumen_hist::polygon::contains(o, c.centroid)));
                        list.extend(found);
                        for (i, c) in list.iter_mut().enumerate() {
                            c.id = i as u32 + 1;
                        }
                        s.hist.rev += 1;
                        s.hist_persist(vec![image], cx);
                        s.hist_record(record.clone(), cx);
                        s.hist_report(true, format!("{n} superpixel(s); give regions or points classes and Train to classify them"), cx);
                    }
                    Err(e) => s.hist_report(false, format!("Superpixels: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// The density map of the shown image's cells as a layer (transparent where it is 0).
    pub fn hist_density(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let Some(um) = self.pyramids.get(&image).and_then(|p| p.pixel_size_um) else { return };
        let Some(cells) = self.hist.cells.get(&image).filter(|c| !c.is_empty()) else {
            return self.hist_report(false, "Find cells first", cx);
        };
        let s = self.hist.density.clone();
        let b = cells.iter().map(|c| c.centroid).fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, p| (b.0.min(p[0]), b.1.min(p[1]), b.2.max(p[0]), b.3.max(p[1])));
        let pad = s.radius_um / um;
        let bounds = (b.0 - pad, b.1 - pad, b.2 + pad + 1.0, b.3 + pad + 1.0);
        let map = lumen_hist::density::density_map(cells, bounds, um, &s);
        let max = map.values.iter().copied().fold(0.0f32, f32::max);
        if max <= 0.0 {
            return self.hist_report(false, "No cell counted (check the class chosen)", cx);
        }
        let mut bgra = vec![0u8; map.width * map.height * 4];
        for (i, &v) in map.values.iter().enumerate() {
            if v > 0.0 {
                let t = v / max;
                let [r, g, b] = lumen_view::ColormapLut::Inferno.sample_rgb(t);
                bgra[i * 4..i * 4 + 4].copy_from_slice(&[b, g, r, (90.0 + 120.0 * t) as u8]);
            }
        }
        let unit = match s.value {
            lumen_hist::density::DensityValue::Count => format!("cells within {:.0} µm", s.radius_um),
            lumen_hist::density::DensityValue::PerMm2 => "cells per mm²".into(),
            lumen_hist::density::DensityValue::PercentPositive => format!("% positive within {:.0} µm", s.radius_um),
        };
        let label = format!("{}{unit}", s.class.as_ref().map(|c| format!("{c}: ")).unwrap_or_default());
        let rect = [map.origin[0], map.origin[1], map.width as f64 * map.ds, map.height as f64 * map.ds];
        let layer = DensityLayer { rect, max, label: label.clone(), image: lumen_view::bgra_image(map.width, map.height, bgra) };
        self.hist.density_maps.insert(image, layer);
        self.hist.show_maps = true;
        self.hist.rev += 1;
        self.hist_record(lumen_hist::session::Action::Density { settings: s.clone() }, cx);
        self.hist_report(true, format!("Density map: {label}, up to {max:.1}"), cx);
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histology_is_read_back_from_the_project_folder() {
        let folder = std::env::temp_dir().join(format!("lumen-hist-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(histology_dir(&folder)).unwrap();
        let mut cell = HistObject::new(1, lumen_hist::ObjectKind::Cell, vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]);
        cell.class = Some("Tumor".into());
        cell.measurements.insert("Nucleus: Area µm^2".into(), 12.5);
        let doc = HistDoc { stains: [("img1".to_string(), StainChoice::Fluorescence(2))].into_iter().collect(), auto_threshold: false, pixel_um: 4.0, ..Default::default() };
        std::fs::write(histology_dir(&folder).join("settings.json"), serde_json::to_string(&doc).unwrap()).unwrap();
        let objects = ObjectsDoc { cells: vec![cell.clone()], tiles: Vec::new() };
        write_gz(&objects_file(&folder, "img1"), &serde_json::to_vec(&objects).unwrap()).unwrap();

        let (st, errors) = load_folder(&folder, &["img1".to_string(), "img2".to_string()]);
        std::fs::remove_dir_all(&folder).ok();
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(st.cells.get("img1"), Some(&vec![cell]));
        assert!(!st.cells.contains_key("img2"));
        assert_eq!(st.stains.get("img1"), Some(&StainChoice::Fluorescence(2)));
        assert!(!st.auto_threshold);
        assert_eq!(st.pixel_um, 4.0);
    }
}

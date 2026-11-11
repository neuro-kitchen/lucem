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
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "in", rename_all = "snake_case")]
pub enum Within {
    /// The whole image.
    #[default]
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
    /// Every pixel classed (inside `within`).
    PixelClasses {
        classifier: PixelClassifier,
        pixel_um: f64,
        #[serde(default)]
        within: Within,
    },
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
    /// What each step did (one line each).
    pub notes: Vec<String>,
}

/// Runs `recipe` (see [`Session::recipe`]) on one image through the engine (the same steps
/// the app runs). Regions drawn on the image the session was recorded on become this image's
/// tissue (`Within::Tissue`).
pub fn run(client: &lumen_core::cubecl::prelude::Client, p: &lumen_io::Pyramid, recipe: &[&Action], progress: &mut dyn FnMut(&str)) -> HistResult<RunResult> {
    use crate::engine::{apply, Analysis, Ctx, PluginModels};
    let mut a = Analysis::default();
    let mut models = PluginModels::default();
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let mut notes = Vec::new();
    let mut report = |what: &str, _: Option<f32>| progress(what);
    for action in recipe {
        let action = retarget(action);
        let mut ctx = Ctx { client, pyramid: p, z: 0, models: &mut models, progress: &mut report, cancel: &cancel };
        let outcome = apply(&mut ctx, &mut a, &action)?;
        notes.push(outcome.summary());
    }
    Ok(RunResult { tissue: a.tissue, cells: a.cells, tiles: a.tiles, notes })
}

/// A recorded step for another image: its drawn regions become that image's tissue.
fn retarget(action: &Action) -> Action {
    let to_tissue = |w: &Within| if matches!(w, Within::Regions { .. }) { Within::Tissue } else { w.clone() };
    match action {
        Action::Cells { within, staining, detection, auto_threshold, model } => Action::Cells { within: to_tissue(within), staining: staining.clone(), detection: detection.clone(), auto_threshold: *auto_threshold, model: model.clone() },
        Action::Superpixels { within, settings, pixel_um } => Action::Superpixels { within: to_tissue(within), settings: settings.clone(), pixel_um: *pixel_um },
        Action::PixelClasses { classifier, pixel_um, within } => Action::PixelClasses { classifier: classifier.clone(), pixel_um: *pixel_um, within: to_tissue(within) },
        other => other.clone(),
    }
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

//! The tools an agent calls (MCP), over lucem without a window. The agent works as a person does
//! in the app: it opens images, runs histology steps on one, reads what they found, adjusts, and
//! the steps it settles on are a session (`session.json`) it can save and run over a folder,
//! headless. Every step it runs is recorded in that session, as the app records a person's.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use lucem_hist::session::{Action, Session, Within};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::schemars::{self, JsonSchema};
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::work;


/// The agent's working state: the session being built, and the runtime.
struct State {
    session: Session,
    /// The session's file, once saved or loaded.
    file: Option<PathBuf>,
    /// The images described, by the name steps use (their file name).
    images: Vec<(String, PathBuf)>,
}

#[derive(Clone)]
pub struct LucemTools {
    state: Arc<Mutex<State>>,
    runtime: Option<String>,
}

impl LucemTools {
    /// Tools on the runtime named `runtime` (else `LUCEM_RUNTIME`'s, else the first compiled in).
    pub fn new(runtime: Option<String>) -> Self {
        Self { state: Arc::new(Mutex::new(State { session: Session::default(), file: None, images: Vec::new() })), runtime }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The path of an image named by its path or by the name `describe_image` gave it.
    fn image_path(&self, image: &str) -> PathBuf {
        self.state().images.iter().find(|(n, _)| n == image).map_or_else(|| PathBuf::from(image), |(_, p)| p.clone())
    }
}

fn ok(v: impl Into<String>) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(v.into())])
}

fn ok_json(v: &Value) -> CallToolResult {
    ok(serde_json::to_string_pretty(v).unwrap_or_default())
}

fn fail(e: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(e.into())])
}

/// Runs blocking work (GPU, files) off the protocol's thread.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tokio::task::spawn_blocking(f).await.map_err(|e| e.to_string())?
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ImageArg {
    /// Path of the image: an OME-Zarr folder, a TIFF/OME-TIFF or another file lucem reads.
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StepArg {
    /// The image: its path, or the name describe_image gave it.
    pub image: String,
    /// The step as JSON, e.g. {"action": "tissue", "settings": {...}}; see step_reference.
    pub step: Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PathArg {
    /// A file path (session.json).
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunSessionArg {
    /// Images to run the session's steps on.
    pub images: Vec<String>,
    /// Folder for the results of images that are not OME-Zarr (TIFF, …). An .ome.zarr image keeps its
    /// results inside itself (`written` in the answer names where), whatever this says. Default:
    /// beside each image.
    #[serde(default)]
    pub out_dir: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PipelineArg {
    /// The pipeline, as `lucem run` reads it: `sources` maps a name to a channel index, `nodes` maps
    /// a name to `{"op": {"op": "<operation>", ...its parameters}, "inputs": [source or node names]}`,
    /// `target` names the node whose result is written. Example: `{"sources": {"src": 0}, "nodes":
    /// {"blur": {"op": {"op": "gaussian_blur", "sigma": 2.0}, "inputs": ["src"]}}, "target": "blur"}`.
    /// The operations and their parameters: list_operations.
    pub spec: lucem_pipeline::PipelineSpec,
    /// The input image.
    pub image: String,
    /// The output OME-Zarr folder.
    pub output: String,
    /// Block size in pixels (default: chosen from the image).
    #[serde(default)]
    pub block: Option<usize>,
}

#[tool_router]
impl LucemTools {
    #[tool(description = "Every image-processing operation lucem has (filters, intensity, thresholds, morphology, segmentation, measurements…): its family, input and output types, parameters with defaults, and what it does. Use them in run_pipeline.", annotations(read_only_hint = true))]
    async fn list_operations(&self) -> CallToolResult {
        ok_json(&work::operations())
    }

    #[tool(description = "Describes an image (size, channels, pixel size in µm, pyramid levels) and remembers it by its file name for the other tools.", annotations(read_only_hint = true))]
    async fn describe_image(&self, Parameters(a): Parameters<ImageArg>) -> CallToolResult {
        let path = PathBuf::from(&a.path);
        let p2 = path.clone();
        match blocking(move || work::describe(&p2)).await {
            Ok(v) => {
                let name = path.file_name().map_or_else(|| a.path.clone(), |n| n.to_string_lossy().into_owned());
                let mut st = self.state();
                st.images.retain(|(n, _)| *n != name);
                st.images.push((name, path));
                ok_json(&v)
            }
            Err(e) => fail(e),
        }
    }

    #[tool(description = "The histology steps run_step takes, each as an example JSON with its default settings and what it does.", annotations(read_only_hint = true))]
    async fn step_reference(&self) -> CallToolResult {
        ok_json(&step_reference())
    }

    #[tool(description = "Runs a histology step on one image and records it in the session. The session's steps (the last of each kind) run in order on the image, so a later step sees the earlier ones' results; running a kind again replaces it. Returns what each step found (tissue regions, cells, tiles, notes). An `export` step writes the results to the image's output store.")]
    async fn run_step(&self, Parameters(a): Parameters<StepArg>) -> CallToolResult {
        let action: Action = match serde_json::from_value(a.step.clone()) {
            Ok(x) => x,
            Err(e) => return fail(format!("not a step: {e}. See step_reference for their JSON.")),
        };
        let path = self.image_path(&a.image);
        let name = path.file_name().map_or_else(|| a.image.clone(), |n| n.to_string_lossy().into_owned());
        let export = matches!(action, Action::Export);
        let session = {
            let mut st = self.state();
            st.session.push(&name, action);
            st.session.clone()
        };
        let runtime = self.runtime.clone();
        let result = blocking(move || {
            let target = work::target(runtime.as_deref())?;
            let recipe = session.recipe();
            let (p, r) = work::analyse(&target, &path, &recipe, &mut |_| {})?;
            let store = if export { Some(work::write_results(&path, &p, &r, &session, "lucem mcp", None)?) } else { None };
            Ok(json!({
                "image": name,
                "steps_run": recipe.iter().map(|a| a.title()).collect::<Vec<_>>(),
                "tissue_regions": r.tissue.len(),
                "cells": r.cells.len(),
                "tiles": r.tiles.len(),
                "notes": r.notes,
                "written": store.map(|s| s.file_name().map(|n| n.to_string_lossy().into_owned())),
            }))
        })
        .await;
        match result {
            Ok(v) => ok_json(&v),
            Err(e) => {
                // A step that failed is not kept
                self.state().session.steps.pop();
                fail(e)
            }
        }
    }

    #[tool(description = "The session so far: every step run, on which image, with its settings, and the recipe (the last step of each kind, in order) that run_session would run.", annotations(read_only_hint = true))]
    async fn get_session(&self) -> CallToolResult {
        let st = self.state();
        ok_json(&json!({
            "file": st.file.as_ref().and_then(|f| f.file_name()).map(|n| n.to_string_lossy().into_owned()),
            "recipe": st.session.recipe().iter().map(|a| a.title()).collect::<Vec<_>>(),
            "session": st.session,
        }))
    }

    #[tool(description = "Writes the session to a session.json file (the app and `lucem session run` read it).")]
    async fn save_session(&self, Parameters(a): Parameters<PathArg>) -> CallToolResult {
        let path = PathBuf::from(&a.path);
        let mut st = self.state();
        match st.session.save(&path) {
            Ok(()) => {
                st.file = Some(path);
                ok(format!("saved {} step(s)", st.session.steps.len()))
            }
            Err(e) => fail(e.to_string()),
        }
    }

    #[tool(description = "Reads a session.json (from the app or saved earlier) as the session to continue from.")]
    async fn load_session(&self, Parameters(a): Parameters<PathArg>) -> CallToolResult {
        let path = PathBuf::from(&a.path);
        match Session::load(&path) {
            Ok(s) => {
                let mut st = self.state();
                let n = s.steps.len();
                st.session = s;
                st.file = Some(path);
                ok(format!("loaded {n} step(s); recipe: {}", st.session.recipe().iter().map(|a| a.title()).collect::<Vec<_>>().join(" → ")))
            }
            Err(e) => fail(e.to_string()),
        }
    }

    #[tool(description = "Runs the session's recipe on each image and writes each one's results (labels, Parquet tables, GeoParquet shapes, provenance): inside the image when it is OME-Zarr, else in out_dir (default beside it); `written` says where. As `lucem session run`.")]
    async fn run_session(&self, Parameters(a): Parameters<RunSessionArg>) -> CallToolResult {
        let session = self.state().session.clone();
        if session.recipe().is_empty() {
            return fail("the session has no step yet: run_step first, or load_session");
        }
        let runtime = self.runtime.clone();
        let out = a.out_dir.map(PathBuf::from);
        let images = a.images;
        let result = blocking(move || {
            let target = work::target(runtime.as_deref())?;
            let mut rows = Vec::new();
            for image in &images {
                let path = PathBuf::from(image);
                let t0 = std::time::Instant::now();
                rows.push(match work::run_session_on(&target, &path, &session, "lucem mcp", out.as_deref(), &mut |_| {}) {
                    Ok((store, cells, tiles, notes)) => json!({"image": image, "cells": cells, "tiles": tiles, "seconds": t0.elapsed().as_secs_f64(), "written": store.file_name().map(|n| n.to_string_lossy().into_owned()), "notes": notes}),
                    Err(e) => json!({"image": image, "error": e}),
                });
            }
            Ok(Value::Array(rows))
        })
        .await;
        match result {
            Ok(v) => ok_json(&v),
            Err(e) => fail(e),
        }
    }

    #[tool(description = "Runs a pipeline of operations (list_operations) over a whole image, block by block, and writes the result as OME-Zarr. As `lucem run`. Smallest spec: {\"sources\": {\"src\": 0}, \"nodes\": {\"blur\": {\"op\": {\"op\": \"gaussian_blur\", \"sigma\": 2.0}, \"inputs\": [\"src\"]}}, \"target\": \"blur\"}.")]
    async fn run_pipeline(&self, Parameters(a): Parameters<PipelineArg>) -> CallToolResult {
        let spec = a.spec;
        let runtime = self.runtime.clone();
        let (image, output) = (self.image_path(&a.image), PathBuf::from(&a.output));
        let result = blocking(move || {
            let target = work::target(runtime.as_deref())?;
            work::run_pipeline(&target, &spec, &image, &output, a.block, None)
        })
        .await;
        match result {
            Ok(d) => ok_json(&serde_json::to_value(&d).unwrap_or_default()),
            Err(e) => fail(e),
        }
    }
}

#[tool_handler(
    name = "lucem",
    instructions = "lucem: GPU image analysis for histology and microscopy, without its window. \
Start with describe_image on an image, then step_reference for the histology steps and their settings. \
run_step runs a step on one image (it is recorded in the session, and the session's steps run in order: \
tissue, then cells, then the rest); read its result and adjust the settings by running the step again. \
get_session shows the steps settled on; save_session writes session.json; run_session runs it over many \
images and writes each one's results (labels, Parquet tables, GeoParquet shapes, provenance). \
For image processing (filters, thresholds, morphology, segmentation, measurements), list_operations gives \
every operation and run_pipeline runs a pipeline of them over a whole image."
)]
impl ServerHandler for LucemTools {}

/// Each step an agent can run, with its default settings.
fn step_reference() -> Value {
    use lucem_hist::cells::CellDetection;
    use lucem_hist::density::DensitySettings;
    use lucem_hist::slide::Staining;
    use lucem_hist::superpixel::SuperpixelSettings;
    use lucem_hist::tissue::TissueSettings;
    let example = |a: Action| serde_json::to_value(a).unwrap_or(Value::Null);
    json!([
        {"does": "Finds the tissue (regions of class Tissue). Run it first on a slide.",
         "step": example(Action::Tissue { settings: TissueSettings::default() })},
        {"does": "Finds and measures cells inside the tissue (or the whole image: within image). Fluorescence: nuclei on channel `nuclear` (auto_threshold takes Otsu's per image); brightfield: {\"staining\": \"brightfield\", \"stains\": …} with the image's stain vectors. `model`: a nucleus model plugin (e.g. \"stardist\") instead of the threshold.",
         "step": example(Action::Cells { within: Within::default(), staining: Staining::Fluorescence { nuclear: 0 }, detection: CellDetection::default(), auto_threshold: true, model: None })},
        {"does": "Intensity classes (negative, 1+, 2+, 3+) of the cells by one measurement: one threshold, or three increasing ones; optionally only cells of one class.",
         "step": example(Action::Intensity { thresholds: lucem_hist::intensity::IntensityThresholds { measurement: "<a measurement name from the cells' table>".into(), thresholds: vec![0.2, 0.4, 0.6], filter: None } })},
        {"does": "Superpixel tiles over the tissue, measured (for tissue classes).",
         "step": example(Action::Superpixels { within: Within::default(), settings: SuperpixelSettings::default(), pixel_um: 2.0 })},
        {"does": "A density map of the cells (cells per mm², and hotspots).",
         "step": example(Action::Density { settings: DensitySettings::default() })},
        {"does": "Writes the results to the image's output store: labels (cells, nuclei), Parquet tables, GeoParquet shapes, provenance.",
         "step": example(Action::Export)},
        {"does": "Object and pixel classifiers (classify_objects, pixel_classes) are trained in the app on annotated examples; a session from the app carries them (load_session)."}
    ])
}

/// Serves the tools over stdin/stdout until the client goes away.
pub async fn serve_stdio(runtime: Option<String>) -> Result<(), String> {
    use rmcp::ServiceExt as _;
    let service = LucemTools::new(runtime).serve(rmcp::transport::stdio()).await.map_err(|e| e.to_string())?;
    service.waiting().await.map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example in run_pipeline's description reads, and the schema an agent sees describes the
    /// real format (sources, nodes, target), not another.
    #[test]
    fn the_pipeline_example_reads_and_the_schema_is_the_real_one() {
        let arg: PipelineArg = serde_json::from_value(json!({
            "spec": {"sources": {"src": 0}, "nodes": {"blur": {"op": {"op": "gaussian_blur", "sigma": 2.0}, "inputs": ["src"]}}, "target": "blur"},
            "image": "a.ome.zarr",
            "output": "b.ome.zarr"
        }))
        .expect("the documented example reads");
        assert_eq!(arg.spec.target, "blur");
        assert_eq!(arg.spec.sources.get("src"), Some(&0));
        let schema = serde_json::to_string(&schemars::schema_for!(PipelineArg)).unwrap();
        for field in ["sources", "nodes", "target", "inputs"] {
            assert!(schema.contains(field), "the schema names {field}");
        }
        assert!(!schema.contains("\"edges\""), "no edges field");
    }
}

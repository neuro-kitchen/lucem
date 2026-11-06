//! The shared state every screen reads: the project document (changed only through commands,
//! with undo and redo, saved after each change when the project has a folder), the images opened
//! for tile reads, the tool settings, the annotation selection, and the results of the graph over
//! what is on screen. Slow work (opening an image, starting the GPU, computing results) runs off
//! the UI thread and comes back as events.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use gpui_kit::{AppContext as _, Context, EventEmitter};
use lumen_io::{generate_phantom, PhantomKind, Pyramid};

use crate::domain::command::{apply, Changed, Command, CommandError};
use crate::domain::graph::NodeKind;
use crate::domain::annotation::{Annotation, Shape};
use crate::domain::geometry;
use crate::domain::project::{ChannelDoc, ProjectDoc, PROJECT_FILE};
use crate::domain::tool::{Tool, ToolState};
use crate::navigation::route::Workspace;

use super::device::Device;
use super::prefs::Prefs;
use super::results::{analysis_key, analyze, compute, Analysis, ResultRequest, Results};

/// Undo steps kept.
const UNDO_DEPTH: usize = 200;

/// What changed, for the screens that follow it.
#[derive(Debug, Clone, PartialEq)]
pub enum AppEvent {
    Document(Changed),
    /// An image finished opening (its id).
    ImageReady(String),
    Workspace,
    Status,
    Results,
    Tool,
    Selection,
}

pub struct Store {
    pub doc: ProjectDoc,
    /// The project file (`<folder>/project.lumen.json`); `None` until saved somewhere.
    pub project_file: Option<PathBuf>,
    history: Vec<ProjectDoc>,
    future: Vec<ProjectDoc>,
    pub prefs: Prefs,
    prefs_path: Option<PathBuf>,
    pub workspace: Workspace,
    pub status: String,
    /// Names of images being opened.
    pub opening: Vec<String>,
    /// Images open for tile reads, by image id.
    pub pyramids: HashMap<String, Arc<Pyramid>>,
    opening_ids: HashSet<String>,
    pub tool: ToolState,
    /// Selected annotations (ids).
    pub selection: BTreeSet<String>,
    pub results: Results,
    /// The whole-image pass of the shown image (previews, Otsu thresholds, tables).
    pub analysis: Option<Arc<Analysis>>,
    /// Per image, per channel: the histogram of the coarsest level.
    pub histograms: HashMap<String, Vec<ChannelHistogram>>,
    /// The document before a live edit (a drag on a histogram or colour wheel), to undo it in
    /// one step.
    live_base: Option<ProjectDoc>,
    /// Results are being computed.
    pub computing: bool,
    last_request: Option<ResultRequest>,
    results_gen: Arc<AtomicU64>,
    pub device: Option<Arc<Device>>,
    device_pending: bool,
    /// The selected node in the Pipeline (shared so Explore can show it).
    pub selected_node: Option<String>,
}

impl EventEmitter<AppEvent> for Store {}

impl Store {
    pub fn new(prefs: Prefs, prefs_path: Option<PathBuf>) -> Self {
        let tool = ToolState { rulers: prefs.rulers, ..ToolState::default() };
        Self {
            doc: ProjectDoc::default(),
            project_file: None,
            history: Vec::new(),
            future: Vec::new(),
            prefs,
            prefs_path,
            workspace: Workspace::Explore,
            status: "Ready".into(),
            opening: Vec::new(),
            pyramids: HashMap::new(),
            opening_ids: HashSet::new(),
            tool,
            selection: BTreeSet::new(),
            results: Results::new(),
            analysis: None,
            histograms: HashMap::new(),
            live_base: None,
            computing: false,
            last_request: None,
            results_gen: Arc::new(AtomicU64::new(0)),
            device: None,
            device_pending: false,
            selected_node: None,
        }
    }

    // ------------------------------------------------------------------------
    // The document
    // ------------------------------------------------------------------------

    /// Runs a command on the document, saves it and says what changed.
    pub fn execute(&mut self, cmd: Command, cx: &mut Context<Self>) -> Result<Changed, CommandError> {
        let before = self.doc.clone();
        let result = apply(&mut self.doc, cmd);
        match &result {
            Ok(change) => {
                if before != self.doc {
                    self.history.push(before);
                    if self.history.len() > UNDO_DEPTH {
                        self.history.remove(0);
                    }
                    self.future.clear();
                    self.save();
                }
                self.after_change(change.clone(), cx);
            }
            Err(e) => self.set_status(e.to_string(), cx),
        }
        result
    }

    /// Back to the document before the last change (`redo`: forward again).
    pub fn undo(&mut self, redo: bool, cx: &mut Context<Self>) {
        let next = if redo { self.future.pop() } else { self.history.pop() };
        let Some(doc) = next else {
            self.set_status(if redo { "Nothing to redo" } else { "Nothing to undo" }, cx);
            return;
        };
        let before = std::mem::replace(&mut self.doc, doc);
        if redo {
            self.history.push(before);
        } else {
            self.future.push(before);
        }
        self.save();
        self.selection.retain(|id| self.doc.annotations.iter().any(|a| &a.id == id));
        self.ensure_pyramids(cx);
        self.after_change(Changed::Graph, cx);
        cx.emit(AppEvent::Document(Changed::Images));
        cx.emit(AppEvent::Document(Changed::Annotations));
        self.set_status(if redo { "Redone" } else { "Undone" }, cx);
    }

    pub fn can_undo(&self) -> bool {
        !self.history.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }

    fn after_change(&mut self, change: Changed, cx: &mut Context<Self>) {
        match &change {
            Changed::Images => {
                self.ensure_pyramids(cx);
                self.results.retain(|_, r| self.doc.image(&r.image).is_some());
            }
            Changed::Graph => {
                self.results.retain(|v, _| self.doc.graph.contains(v));
                self.recompute(cx);
            }
            Changed::Layout => {
                // A view's style changes how its layer is drawn
                self.recompute(cx);
            }
            Changed::Annotations => {
                self.selection.retain(|id| self.doc.annotations.iter().any(|a| &a.id == id));
                let reads_annotations = self.doc.graph.nodes.iter().any(|n| matches!(n.kind, NodeKind::Annotations { .. }));
                if reads_annotations {
                    self.recompute(cx);
                }
            }
            _ => {}
        }
        cx.emit(AppEvent::Document(change));
        cx.notify();
    }

    /// Writes the project file when the project has a folder.
    fn save(&mut self) {
        if let Some(path) = &self.project_file {
            if let Err(e) = self.doc.save(path) {
                self.status = format!("Project not saved: {e}");
            }
        }
    }

    /// The project's folder.
    pub fn folder(&self) -> Option<PathBuf> {
        self.project_file.as_ref().and_then(|p| p.parent().map(Path::to_path_buf))
    }

    pub fn set_status(&mut self, s: impl Into<String>, cx: &mut Context<Self>) {
        self.status = s.into();
        cx.emit(AppEvent::Status);
        cx.notify();
    }

    pub fn set_workspace(&mut self, w: Workspace, cx: &mut Context<Self>) {
        if self.workspace != w {
            self.workspace = w;
            cx.emit(AppEvent::Workspace);
            cx.notify();
        }
    }

    fn save_prefs(&self) {
        self.prefs.save(self.prefs_path.as_deref());
    }

    // ------------------------------------------------------------------------
    // Projects
    // ------------------------------------------------------------------------

    /// A new, empty project (not saved anywhere yet).
    pub fn new_project(&mut self, cx: &mut Context<Self>) {
        self.history.push(std::mem::take(&mut self.doc));
        self.doc = ProjectDoc::default();
        self.future.clear();
        self.project_file = None;
        self.pyramids.clear();
        self.results.clear();
        self.selection.clear();
        self.after_change(Changed::Images, cx);
        self.set_status("New project", cx);
    }

    /// Saves the project as `<folder>/project.lumen.json` and keeps saving there.
    pub fn save_in(&mut self, folder: PathBuf, cx: &mut Context<Self>) {
        let file = folder.join(PROJECT_FILE);
        if self.doc.name == "Untitled" {
            if let Some(n) = folder.file_name() {
                self.doc.name = n.to_string_lossy().into_owned();
            }
        }
        // Image paths inside the folder become relative (the folder can move)
        let old = self.folder();
        for img in &mut self.doc.images {
            let abs = ProjectDoc::resolve(&img.path, old.as_deref());
            img.path = ProjectDoc::stored_path(&abs, Some(&folder));
        }
        match self.doc.save(&file) {
            Ok(()) => {
                self.project_file = Some(file.clone());
                self.prefs.push_recent(&file);
                self.save_prefs();
                self.set_status(format!("Project saved in {}", folder.display()), cx);
            }
            Err(e) => self.set_status(format!("Project not saved: {e}"), cx),
        }
        cx.emit(AppEvent::Document(Changed::Project));
    }

    fn open_project(&mut self, file: PathBuf, cx: &mut Context<Self>) {
        match ProjectDoc::load(&file) {
            Ok(doc) => {
                self.history.clear();
                self.future.clear();
                self.doc = doc;
                self.project_file = Some(file.clone());
                self.pyramids.clear();
                self.results.clear();
                self.selection.clear();
                self.prefs.push_recent(&file);
                self.save_prefs();
                self.after_change(Changed::Images, cx);
                self.set_workspace(Workspace::Explore, cx);
                self.set_status(format!("Opened project {}", self.doc.name), cx);
            }
            Err(e) => self.set_status(format!("Could not open the project: {e}"), cx),
        }
    }

    // ------------------------------------------------------------------------
    // Images
    // ------------------------------------------------------------------------

    /// Opens a project file or folder, an OME-Zarr container, an image file, or a sample
    /// (`sample:<name>`).
    pub fn open(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let s = path.to_string_lossy();
        if s.ends_with(".lumen.json") {
            return self.open_project(path, cx);
        }
        if path.is_dir() && path.join(PROJECT_FILE).is_file() {
            return self.open_project(path.join(PROJECT_FILE), cx);
        }
        self.open_image(path, cx);
    }

    fn open_image(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        // The same file again: show it
        if let Some(img) = self.doc.images.iter().find(|i| ProjectDoc::resolve(&i.path, self.folder().as_deref()) == path) {
            let id = img.id.clone();
            let _ = self.execute(Command::SelectImage { id }, cx);
            self.set_workspace(Workspace::Explore, cx);
            return;
        }
        let name = display_name(&path);
        self.opening.push(name.clone());
        self.set_status(format!("Opening {name}…"), cx);
        let task = cx.background_spawn({
            let path = path.clone();
            async move { open_pyramid(&path).map(|py| (default_channels(&py), py)) }
        });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |s, cx| {
                s.opening.retain(|n| n != &name);
                match result {
                    Ok((channels, py)) => {
                        if !path.to_string_lossy().starts_with("sample:") {
                            s.prefs.push_recent(&path);
                            s.save_prefs();
                        }
                        let stored = ProjectDoc::stored_path(&path, s.folder().as_deref());
                        let summary = format!("{} · {} × {} px · {} channel(s) · {} level(s)", py.name, py.width(), py.height(), py.channels.len(), py.levels.len());
                        let cmd = Command::AddImage { path: stored, name: py.name.clone(), channels };
                        if s.execute(cmd, cx).is_ok() {
                            let id = s.doc.images.last().map(|i| i.id.clone()).unwrap_or_default();
                            let py = Arc::new(py);
                            s.pyramids.insert(id.clone(), py.clone());
                            s.compute_histograms(&id, py, cx);
                            s.set_workspace(Workspace::Explore, cx);
                            s.set_status(format!("Opened {summary}"), cx);
                            cx.emit(AppEvent::ImageReady(id));
                        }
                    }
                    Err(e) => s.set_status(format!("Could not open {name}: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Opens the images of the document that are not open yet (after opening a project, undo).
    fn ensure_pyramids(&mut self, cx: &mut Context<Self>) {
        let folder = self.folder();
        let missing: Vec<(String, PathBuf)> = self
            .doc
            .images
            .iter()
            .filter(|i| !self.pyramids.contains_key(&i.id) && !self.opening_ids.contains(&i.id))
            .map(|i| (i.id.clone(), ProjectDoc::resolve(&i.path, folder.as_deref())))
            .collect();
        self.pyramids.retain(|id, _| self.doc.image(id).is_some());
        for (id, path) in missing {
            self.opening_ids.insert(id.clone());
            let task = cx.background_spawn({
                let path = path.clone();
                async move { open_pyramid(&path) }
            });
            cx.spawn(async move |this, cx| {
                let result = task.await;
                let _ = this.update(cx, |s, cx| {
                    s.opening_ids.remove(&id);
                    match result {
                        Ok(py) => {
                            // Channels never shown before get their defaults
                            if s.doc.image(&id).is_some_and(|i| i.channels.len() != py.channels.len()) {
                                let channels = default_channels(&py);
                                if let Some(img) = s.doc.image_mut(&id) {
                                    img.channels = channels;
                                }
                            }
                            let py = Arc::new(py);
                            s.pyramids.insert(id.clone(), py.clone());
                            s.compute_histograms(&id, py, cx);
                            cx.emit(AppEvent::ImageReady(id));
                        }
                        Err(e) => s.set_status(format!("Could not open {}: {e}", path.display()), cx),
                    }
                    cx.notify();
                });
            })
            .detach();
        }
    }

    /// Histograms of every channel of `id` (coarsest level), in the background.
    fn compute_histograms(&mut self, id: &str, py: Arc<Pyramid>, cx: &mut Context<Self>) {
        let id = id.to_string();
        let task = cx.background_spawn(async move {
            let level = py.coarsest();
            let l = py.levels[level].clone();
            (0..py.channels.len())
                .map(|c| {
                    let v = py.read_region(level, c, 0, 0, l.width, l.height).unwrap_or_default();
                    let lo = v.iter().copied().filter(|x| x.is_finite()).fold(f32::MAX, f32::min);
                    let hi = v.iter().copied().filter(|x| x.is_finite()).fold(f32::MIN, f32::max);
                    let (lo, hi) = if hi > lo { (lo, hi) } else { (0.0, 1.0) };
                    ChannelHistogram { bins: geometry::histogram(&v, lo, hi, HIST_BINS), lo, hi }
                })
                .collect::<Vec<_>>()
        });
        cx.spawn(async move |this, cx| {
            let h = task.await;
            let _ = this.update(cx, |s, cx| {
                s.histograms.insert(id, h);
                cx.notify();
            });
        })
        .detach();
    }

    pub fn active_pyramid(&self) -> Option<&Arc<Pyramid>> {
        self.doc.active_image.as_ref().and_then(|id| self.pyramids.get(id))
    }

    // ------------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------------

    /// What is on screen changed: compute the results there (debounced by the canvas).
    pub fn request_results(&mut self, req: ResultRequest, cx: &mut Context<Self>) {
        if self.last_request.as_ref() == Some(&req) && !self.results.is_empty() {
            return;
        }
        self.last_request = Some(req);
        self.recompute(cx);
    }

    /// Computes the results again for the last place on screen.
    pub fn recompute(&mut self, cx: &mut Context<Self>) {
        let Some(req) = self.last_request.clone() else { return };
        let has_nodes = self.doc.graph.nodes.iter().any(|n| self.doc.graph.images_of(&n.id).contains(&req.image));
        if !has_nodes {
            if !self.results.is_empty() || self.analysis.is_some() {
                self.results.clear();
                self.analysis = None;
                cx.emit(AppEvent::Results);
            }
            return;
        }
        let Some(device) = self.device.clone() else {
            self.ensure_device(cx);
            return;
        };
        let Some(pyramid) = self.pyramids.get(&req.image).cloned() else { return };
        let generation = self.results_gen.fetch_add(1, Ordering::AcqRel) + 1;
        let latest = self.results_gen.clone();
        let doc = self.doc.clone();
        let key = analysis_key(&doc, &req.image);
        // The whole-image pass again only when what it depends on changed
        let cached = self.analysis.clone().filter(|a| a.key == key);
        let show = self.tool.show_results;
        let image = req.image.clone();
        self.computing = true;
        cx.emit(AppEvent::Results);
        let task = cx.background_spawn(async move {
            if latest.load(Ordering::Acquire) != generation {
                return None;
            }
            let analysis = match cached {
                Some(a) => a,
                None => Arc::new(analyze(&device.client, &doc, &pyramid, &req.image)),
            };
            if latest.load(Ordering::Acquire) != generation {
                return Some((analysis, None));
            }
            let layers = show.then(|| compute(&device.client, &doc, &pyramid, &req, &analysis));
            Some((analysis, layers))
        });
        cx.spawn(async move |this, cx| {
            let out = task.await;
            let _ = this.update(cx, |s, cx| {
                if let Some((analysis, _)) = &out {
                    // A finished analysis is worth keeping even when a newer request waits
                    if s.analysis.as_ref().is_none_or(|a| a.key != analysis.key) {
                        s.analysis = Some(analysis.clone());
                    }
                }
                if s.results_gen.load(Ordering::Acquire) != generation {
                    return;
                }
                s.computing = false;
                if let Some((analysis, Some(list))) = out {
                    s.results.retain(|v, r| r.image != image && s.doc.graph.contains(v));
                    let errors: Vec<String> = list.iter().filter_map(|r| r.error.clone().map(|e| format!("{}: {e}", r.view))).chain(analysis.errors.iter().map(|(n, e)| format!("{n}: {e}"))).collect();
                    let ms: f64 = list.iter().map(|r| r.ms).sum();
                    for r in list {
                        s.results.insert(r.view.clone(), r);
                    }
                    s.status = match errors.first() {
                        Some(e) => e.clone(),
                        None => format!(
                            "Results in {ms:.0} ms (whole image at level {}: {:.0} ms) on {}",
                            analysis.level,
                            analysis.ms,
                            s.device.as_ref().map_or("—", |d| d.label.as_str())
                        ),
                    };
                    cx.emit(AppEvent::Status);
                }
                cx.emit(AppEvent::Results);
                cx.notify();
            });
        })
        .detach();
    }

    // ------------------------------------------------------------------------
    // Live edits (one undo step for a whole drag)
    // ------------------------------------------------------------------------

    /// Shows channel `channel` of `image` as `display` while a drag goes on (not saved, not in the
    /// undo history yet).
    pub fn set_channel_live(&mut self, image: &str, channel: usize, display: ChannelDoc, cx: &mut Context<Self>) {
        if self.live_base.is_none() {
            self.live_base = Some(self.doc.clone());
        }
        if let Some(slot) = self.doc.image_mut(image).and_then(|i| i.channels.get_mut(channel)) {
            *slot = display;
            cx.emit(AppEvent::Document(Changed::Display(image.to_string())));
            cx.notify();
        }
    }

    /// The drag ended: the change becomes one command (undoable, saved).
    pub fn commit_live(&mut self, cx: &mut Context<Self>) {
        let Some(base) = self.live_base.take() else { return };
        let edited = std::mem::replace(&mut self.doc, base);
        for img in &edited.images {
            let before = self.doc.image(&img.id).map(|i| i.channels.clone()).unwrap_or_default();
            for (c, ch) in img.channels.iter().enumerate() {
                if before.get(c) != Some(ch) {
                    let _ = self.execute(Command::SetChannel { image: img.id.clone(), channel: c, display: ch.clone() }, cx);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Threshold tool
    // ------------------------------------------------------------------------

    /// Annotations from the threshold on the whole image (at a level of ≤ 4096 px a side): one
    /// region per object above (or below) the threshold, of the drawing class.
    pub fn threshold_to_annotations(&mut self, cx: &mut Context<Self>) {
        let Some(img) = self.doc.active_image.clone() else { return };
        let Some(py) = self.pyramids.get(&img).cloned() else { return };
        let t = self.tool.clone();
        let class = t.class.clone();
        self.set_status("Thresholding the whole image…", cx);
        let task = cx.background_spawn(async move {
            let level = (0..py.levels.len()).find(|&l| py.levels[l].width.max(py.levels[l].height) <= 4096).unwrap_or(py.coarsest());
            let l = py.levels[level].clone();
            let ds = l.downsample;
            let mut v = py.read_region(level, t.threshold_channel, 0, 0, l.width, l.height).map_err(|e| e.to_string())?;
            geometry::blur(&mut v, l.width, l.height, t.threshold_sigma / ds);
            let mask: Vec<u8> = v.iter().map(|&x| u8::from((x >= t.threshold_value) == t.threshold_above)).collect();
            let min_px = (t.threshold_min_area / (ds * ds)).ceil().max(1.0) as usize;
            let polys = geometry::mask_to_polygons(&mask, l.width, l.height, min_px);
            let shapes: Vec<Shape> = polys
                .into_iter()
                .map(|p| geometry::simplify(&p, 0.75))
                .filter(|p| p.len() >= 3)
                .map(|p| Shape::Polygon { points: p.into_iter().map(|[x, y]| [x * ds, y * ds]).collect() })
                .collect();
            Ok::<_, String>((shapes, level))
        });
        cx.spawn(async move |this, cx| {
            let r = task.await;
            let _ = this.update(cx, |s, cx| match r {
                Ok((shapes, level)) => {
                    let n = shapes.len();
                    let annotations = shapes.into_iter().map(|shape| Annotation { id: String::new(), image: img.clone(), shape, class: class.clone(), note: String::new() }).collect();
                    if s.execute(Command::AddAnnotations { annotations }, cx).is_ok() {
                        s.set_status(format!("{n} region(s) of class {class} from the threshold (level {level})"), cx);
                    }
                }
                Err(e) => s.set_status(format!("Threshold failed: {e}"), cx),
            });
        })
        .detach();
    }

    /// The threshold as a pipeline chain: channel → blur → threshold → remove small → view.
    pub fn threshold_to_pipeline(&mut self, cx: &mut Context<Self>) {
        let Some(img) = self.doc.active().cloned() else { return };
        let t = self.tool.clone();
        let top = self.pyramids.get(&img.id).and_then(|p| p.channels.get(t.threshold_channel).map(|c| c.range.1)).unwrap_or(1.0).max(1.0);
        let value = (t.threshold_value / top) as f64;
        let ops = [
            lumen_pipeline::Op::GaussianBlur { sigma: t.threshold_sigma.max(0.5) },
            lumen_pipeline::Op::Threshold { value },
            lumen_pipeline::Op::RemoveSmall { min_size: t.threshold_min_area.round() as usize },
        ];
        let mut view: Option<String> = None;
        for op in ops {
            let before: Vec<String> = self.doc.graph.views().map(|v| v.id.clone()).collect();
            if self.execute(Command::ApplyOp { op, view: view.clone(), image: img.id.clone(), channel: t.threshold_channel }, cx).is_err() {
                return;
            }
            if view.is_none() {
                view = self.doc.graph.views().map(|v| v.id.clone()).find(|v| !before.contains(v));
            }
        }
        self.set_status("Threshold added to the pipeline (sizes in level-0 px)", cx);
    }
    /// Starts the compute device in the background; results follow once it is up.
    pub fn ensure_device(&mut self, cx: &mut Context<Self>) {
        if self.device.is_some() || self.device_pending {
            return;
        }
        self.device_pending = true;
        self.set_status("Starting the compute device…", cx);
        let task = cx.background_spawn(async move { Device::discover() });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |s, cx| {
                s.device_pending = false;
                match result {
                    Ok(d) => {
                        let label = d.label.clone();
                        s.device = Some(Arc::new(d));
                        s.set_status(format!("Compute device: {label}"), cx);
                        s.recompute(cx);
                    }
                    Err(e) => s.set_status(e, cx),
                }
            });
        })
        .detach();
    }

    // ------------------------------------------------------------------------
    // Tools and selection
    // ------------------------------------------------------------------------

    pub fn set_tool(&mut self, tool: Tool, cx: &mut Context<Self>) {
        if self.tool.tool != tool {
            self.tool.tool = tool;
            // The Threshold tool starts at the channel's Otsu threshold
            if tool == Tool::Threshold && self.tool.threshold_value == 0.0 {
                let c = self.tool.threshold_channel;
                if let Some(h) = self.doc.active_image.as_ref().and_then(|id| self.histograms.get(id)).and_then(|h| h.get(c)) {
                    self.tool.threshold_value = h.otsu();
                }
            }
            cx.emit(AppEvent::Tool);
            cx.notify();
        }
    }

    pub fn update_tool(&mut self, f: impl FnOnce(&mut ToolState), cx: &mut Context<Self>) {
        let before = self.tool.clone();
        f(&mut self.tool);
        if self.tool.rulers != before.rulers {
            self.prefs.rulers = self.tool.rulers;
            self.save_prefs();
        }
        if self.tool.show_results != before.show_results {
            if self.tool.show_results {
                self.recompute(cx);
            } else {
                self.results.clear();
            }
        }
        cx.emit(AppEvent::Tool);
        cx.notify();
    }

    /// Explore's side tabs (`None`: that side closed).
    pub fn set_side_tabs(&mut self, left: Option<String>, right: Option<String>, cx: &mut Context<Self>) {
        self.prefs.left_tab = left;
        self.prefs.right_tab = right;
        self.save_prefs();
        cx.notify();
    }

    pub fn set_sidebar(&mut self, open: bool, cx: &mut Context<Self>) {
        self.prefs.sidebar_open = open;
        self.save_prefs();
        cx.notify();
    }

    pub fn select(&mut self, ids: impl IntoIterator<Item = String>, cx: &mut Context<Self>) {
        self.selection = ids.into_iter().collect();
        cx.emit(AppEvent::Selection);
        cx.notify();
    }

    pub fn delete_selection(&mut self, cx: &mut Context<Self>) {
        if self.selection.is_empty() {
            return;
        }
        let ids: Vec<String> = self.selection.iter().cloned().collect();
        let n = ids.len();
        if self.execute(Command::RemoveAnnotations { ids }, cx).is_ok() {
            self.selection.clear();
            self.set_status(format!("Deleted {n} annotation(s)"), cx);
        }
    }
}

/// Bins of a channel histogram.
pub const HIST_BINS: usize = 256;

/// A channel's histogram over `[lo, hi]` (raw values of the coarsest level).
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelHistogram {
    pub bins: Vec<u32>,
    pub lo: f32,
    pub hi: f32,
}

impl ChannelHistogram {
    /// The Otsu threshold of the channel (raw value).
    pub fn otsu(&self) -> f32 {
        geometry::otsu(&self.bins, self.lo, self.hi)
    }
}

/// Opens a pyramid from a path or a `sample:<name>`.
pub fn open_pyramid(path: &Path) -> Result<Pyramid, String> {
    let s = path.to_string_lossy();
    if let Some(name) = s.strip_prefix("sample:") {
        let kind = PhantomKind::ALL.iter().copied().find(|k| sample_name(*k) == name).ok_or_else(|| format!("no sample {name:?}"))?;
        let mut img = generate_phantom(kind, 2048, 2048);
        img.name = kind.title().to_string();
        img.path = Some(path.to_path_buf());
        return Ok(Pyramid::from_loaded(img));
    }
    Pyramid::open(path).map_err(|e| e.to_string())
}

/// The id of a sample in `sample:<name>` paths.
pub fn sample_name(kind: PhantomKind) -> String {
    kind.title().to_lowercase().replace([' ', '&', '/'], "-").replace("--", "-")
}

/// How each channel starts: the file's colour, and its window, or the 0.1–99.8 % percentiles of
/// the coarsest level when the file gives the full range of the type.
pub fn default_channels(py: &Pyramid) -> Vec<ChannelDoc> {
    py.channels
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let (min, max) = if c.window == c.range && c.range.1 > 1.0 { py.auto_window(i, 0.001, 0.998).unwrap_or(c.window) } else { c.window };
            let (min, max) = if max > min { (min, max) } else { (min, min + 1.0) };
            ChannelDoc { label: c.label.clone(), visible: true, color: c.color, min, max, gamma: 1.0 }
        })
        .collect()
}

/// A file's name for messages.
pub fn display_name(path: &Path) -> String {
    let s = path.to_string_lossy();
    if let Some(n) = s.strip_prefix("sample:") {
        return n.to_string();
    }
    path.file_name().map_or_else(|| s.into_owned(), |n| n.to_string_lossy().into_owned())
}

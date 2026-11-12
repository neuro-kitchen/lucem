//! The shared state every screen reads: the project document (changed only through commands,
//! with undo and redo, saved after each change when the project has a folder), the images opened
//! for tile reads, the tool settings, the annotation selection, and the results of the graph over
//! what is on screen. Slow work (opening an image, starting the GPU, computing results) runs off
//! the UI thread and comes back as events.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use gpui_kit::{AppContext as _, Context, EventEmitter};
use lucem_io::{generate_phantom, PhantomKind, Pyramid};
use lucem_pipeline::{Op, PortType};

use crate::domain::command::{Changed, Command};
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::domain::annotation::{Annotation, Shape};
use crate::domain::geometry;
use crate::domain::project::{ChannelDoc, ProjectDoc, PROJECT_FILE};
use crate::domain::tool::{Tool, ToolState};
use crate::navigation::route::Workspace;

use super::device::Device;
use super::prefs::Prefs;
use super::results::{analysis_key, analyze, compute, Analysis, ResultRequest, Results};

/// Undo steps kept.
pub(crate) const UNDO_DEPTH: usize = 200;

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
    /// A view was asked for from the menu.
    View(ViewRequest),
    /// The 3D cursor of the shown image moved.
    Cursor,
    /// Bring a level-0 point of the shown image into the focused view.
    Reveal([f64; 2]),
    /// Select this Pipeline node and show the Pipeline (a new script node: its code opens in the
    /// Inspector's editor).
    RevealNode(String),
}

/// What a pane of Explore shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewKind {
    /// One plane, full canvas.
    Plane,
    /// The XY, XZ and YZ planes together.
    Ortho,
    /// The volume rendered.
    Volume,
    /// Each channel on its own, and the channels together, side by side (linked).
    Channels,
}

impl ViewKind {
    pub const ALL: [ViewKind; 4] = [ViewKind::Plane, ViewKind::Ortho, ViewKind::Volume, ViewKind::Channels];

    pub fn title(self) -> &'static str {
        match self {
            ViewKind::Plane => "Plane",
            ViewKind::Ortho => "2D",
            ViewKind::Volume => "3D",
            ViewKind::Channels => "Channels",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            ViewKind::Plane => "One plane, full canvas",
            ViewKind::Ortho => "The XY, XZ and YZ planes through the cursor",
            ViewKind::Volume => "The volume rendered",
            ViewKind::Channels => "Each channel on its own and all together, moving together",
        }
    }

    pub fn icon(self) -> gpui_kit::assets::IconName {
        use gpui_kit::assets::IconName;
        match self {
            ViewKind::Plane => IconName::Square,
            ViewKind::Ortho => IconName::LayoutGrid,
            ViewKind::Volume => IconName::Box,
            ViewKind::Channels => IconName::Columns3,
        }
    }

    /// Whether it needs an image with depth.
    pub fn needs_depth(self) -> bool {
        matches!(self, ViewKind::Ortho | ViewKind::Volume)
    }
}

/// How the views of Explore are laid out (dsp-kitchen's containers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arrangement {
    /// One tab group holding every view.
    Tabs,
    /// Side by side.
    Horizontal,
    /// Stacked.
    Vertical,
    /// Rows of up to ⌈√n⌉ views.
    Grid,
}

impl Arrangement {
    pub const ALL: [Arrangement; 4] = [Arrangement::Tabs, Arrangement::Horizontal, Arrangement::Vertical, Arrangement::Grid];

    pub fn title(self) -> &'static str {
        match self {
            Arrangement::Tabs => "Tabs",
            Arrangement::Horizontal => "Side by side",
            Arrangement::Vertical => "Stacked",
            Arrangement::Grid => "Grid",
        }
    }
}

/// A view menu command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewRequest {
    /// The focused view shows this.
    Show(ViewKind),
    /// A new view of this kind beside the focused one.
    Add(ViewKind),
    /// Closes the focused view (the last one stays).
    ClosePane,
    Arrange(Arrangement),
    /// The Pipeline docked under the views, shown or hidden.
    TogglePipeline,
    /// Views move and zoom together, or not.
    ToggleLink,
    /// Explore's side panel (beside the right rail): Layers, or the last panel shown, shown or
    /// hidden (ToggleSides as ToggleRight).
    ToggleLeft,
    ToggleRight,
    ToggleSides,
    /// The side docks back to how they come (and forget the saved arrangement).
    ResetLayout,
}

/// What the Layers panel can do with the selected masks.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskOperation {
    Union,
    Intersect,
    Subtract,
    Invert,
    /// A one-input operator on the first selected mask.
    Op(Op),
}

pub struct Store {
    pub doc: ProjectDoc,
    /// The project file (`<folder>/project.lucem.json`); `None` until saved somewhere.
    pub project_file: Option<PathBuf>,
    /// The changes made to `doc`, to undo and redo (see `super::document`).
    pub(crate) undo: super::document::History,
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
    /// The request the results on show answer (asked again: nothing to do).
    pub(crate) results_for: Option<ResultRequest>,
    /// Asked while computing: run once more when the computation ends (one at a time, so slow
    /// steps such as model nodes are never started over and over, nor thrown away).
    results_pending: bool,
    results_gen: Arc<AtomicU64>,
    pub device: Option<Arc<Device>>,
    device_pending: bool,
    /// Bumped when the device changes: what lives on the old one (3D volumes) is dropped.
    pub device_generation: u64,
    /// Machine-learning plugins and their jobs.
    pub ml: super::ml::MlState,
    /// Pixels a mask grows or shrinks by.
    pub mask_radius: usize,
    /// Rows of the Layers tree folded shut (view state).
    pub layers_collapsed: HashSet<String>,
    /// When each script node's file was last seen changed (the watcher runs while there are any).
    script_seen: HashMap<String, std::time::SystemTime>,
    script_watch: bool,
    /// The 3D cursor `[z, y, x]` (level-0 voxels) of each image: the plane every view shows and
    /// where the three planes of the 2D view cross. View state, not in the undo history.
    cursors: HashMap<String, [usize; 3]>,
    /// Bumped whenever annotations or classes change (views that rasterize them follow it).
    pub annotations_rev: u64,
    /// The views of each workspace (`layout.json`), and a counter bumped when a project brings
    /// its own (the workspaces restore their views when it moves).
    pub layout: super::layout::Layout,
    pub layout_rev: u64,
    /// Histology: cells by image, the stains, detection and classifier settings.
    pub hist: super::hist::HistState,
    /// What is selected: shapes, detected objects, Layers rows, the Pipeline node.
    pub sel: super::selection::Selection,
    /// The object hierarchy of the shown image, kept until its shapes or cells change (built
    /// once per change, never per frame).
    hierarchy: std::cell::RefCell<Option<(String, u64, u64, Arc<crate::domain::hierarchy::Hierarchy>)>>,
}

impl EventEmitter<AppEvent> for Store {}

impl Store {
    pub fn new(prefs: Prefs, prefs_path: Option<PathBuf>) -> Self {
        crate::data::tools::set(&prefs.tools);
        let tool = ToolState { rulers: prefs.rulers, ..ToolState::default() };
        Self {
            doc: ProjectDoc::default(),
            project_file: None,
            undo: Default::default(),
            prefs,
            prefs_path,
            workspace: Workspace::Explore,
            status: "Ready".into(),
            opening: Vec::new(),
            pyramids: HashMap::new(),
            opening_ids: HashSet::new(),
            tool,
            results: Results::new(),
            analysis: None,
            histograms: HashMap::new(),
            live_base: None,
            computing: false,
            last_request: None,
            results_for: None,
            results_pending: false,
            results_gen: Arc::new(AtomicU64::new(0)),
            device: None,
            device_pending: false,
            device_generation: 0,
            ml: Default::default(),
            mask_radius: 2,
            layers_collapsed: HashSet::new(),
            script_seen: HashMap::new(),
            script_watch: false,
            cursors: HashMap::new(),
            annotations_rev: 0,
            layout: Default::default(),
            layout_rev: 0,
            hist: Default::default(),
            hierarchy: Default::default(),
            sel: Default::default(),
        }
    }

    /// Changes the saved views (`layout.json`, written when the project has a folder). View state:
    /// no event, no redraw.
    pub fn set_layout(&mut self, f: impl FnOnce(&mut super::layout::Layout)) {
        let before = self.layout.clone();
        f(&mut self.layout);
        if self.layout != before
            && let Some(folder) = self.folder()
        {
            let _ = self.layout.save(&folder);
        }
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
        let before = std::mem::take(&mut self.doc);
        self.undo.record(before, "New project".into());
        self.sel.layers.clear();
        self.project_file = None;
        crate::data::tools::set_scripts_dir(None);
        self.hist = Default::default();
        self.pyramids.clear();
        self.results.clear();
        self.sel.shapes.clear();
        self.after_change(Changed::Images, cx);
        self.set_status("New project", cx);
    }

    /// Saves the project as `<folder>/project.lucem.json` and keeps saving there.
    pub fn save_in(&mut self, folder: PathBuf, cx: &mut Context<Self>) {
        let file = folder.join(PROJECT_FILE);
        if self.doc.name == "Untitled"
            && let Some(n) = folder.file_name() {
                self.doc.name = n.to_string_lossy().into_owned();
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
                crate::data::tools::set_scripts_dir(file.parent().map(|d| crate::data::script::scripts_dir(Some(d))));
                self.prefs.push_recent(&file);
                self.save_prefs();
                self.hist_persist_all(cx);
                let _ = self.layout.save(&folder);
                self.set_status(format!("Project saved in {}", crate::domain::paths::short_path(&folder)), cx);
            }
            Err(e) => self.set_status(format!("Project not saved: {e}"), cx),
        }
        cx.emit(AppEvent::Document(Changed::Project));
    }

    fn open_project(&mut self, file: PathBuf, cx: &mut Context<Self>) {
        match ProjectDoc::load(&file) {
            Ok(doc) => {
                self.undo = Default::default();
                self.sel.layers.clear();
                self.doc = doc;
                // A project saved under the old name is saved again as the new one
                let file = if file.file_name().is_some_and(|n| n == crate::domain::project::OLD_PROJECT_FILE) { file.with_file_name(PROJECT_FILE) } else { file };
                self.project_file = Some(file.clone());
                crate::data::tools::set_scripts_dir(file.parent().map(|d| crate::data::script::scripts_dir(Some(d))));
                self.pyramids.clear();
                self.results.clear();
                self.sel.shapes.clear();
                self.prefs.push_recent(&file);
                self.save_prefs();
                self.after_change(Changed::Images, cx);
                self.watch_scripts(cx);
                self.hist_load(cx);
                if let Some(folder) = file.parent() {
                    self.layout = super::layout::Layout::load(folder);
                    self.layout_rev += 1;
                }
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
        // `project.lumen.json`: a project saved under the old name (saved again as the new one)
        if s.ends_with(".lucem.json") || s.ends_with(".lumen.json") {
            return self.open_project(path, cx);
        }
        if path.is_dir() && path.join(PROJECT_FILE).is_file() {
            return self.open_project(path.join(PROJECT_FILE), cx);
        }
        if path.is_dir() && path.join(crate::domain::project::OLD_PROJECT_FILE).is_file() {
            return self.open_project(path.join(crate::domain::project::OLD_PROJECT_FILE), cx);
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
    pub(crate) fn ensure_pyramids(&mut self, cx: &mut Context<Self>) {
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
                        Err(e) => s.set_status(format!("Could not open {}: {e}", crate::domain::paths::short_path(&path)), cx),
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
                    // Every plane of a volume, not only the first
                    let v = py.read_block(level, c, [0, 0, 0], [l.depth, l.height, l.width]).unwrap_or_default();
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

    /// The 3D cursor of the shown image, `[z, y, x]` in level-0 voxels (the centre until moved).
    pub fn cursor(&self) -> Option<[usize; 3]> {
        self.cursor_of(self.doc.active_image.as_deref()?)
    }

    /// The 3D cursor of image `id`.
    pub fn cursor_of(&self, id: &str) -> Option<[usize; 3]> {
        let py = self.pyramids.get(id)?;
        let dims = [py.depth(), py.height(), py.width()];
        let at = self.cursors.get(id).copied().unwrap_or([dims[0] / 2, dims[1] / 2, dims[2] / 2]);
        Some(std::array::from_fn(|a| at[a].min(dims[a].saturating_sub(1))))
    }

    /// Moves the 3D cursor of the shown image (clamped to it).
    pub fn set_cursor(&mut self, at: [usize; 3], cx: &mut Context<Self>) {
        let Some(id) = self.doc.active_image.clone() else { return };
        self.set_cursor_of(&id, at, cx);
    }

    /// Moves the 3D cursor of image `id` (clamped to it).
    pub fn set_cursor_of(&mut self, id: &str, at: [usize; 3], cx: &mut Context<Self>) {
        let id = id.to_string();
        let Some(py) = self.pyramids.get(&id) else { return };
        let dims = [py.depth(), py.height(), py.width()];
        let at: [usize; 3] = std::array::from_fn(|a| at[a].min(dims[a].saturating_sub(1)));
        if self.cursor_of(&id) != Some(at) {
            self.cursors.insert(id, at);
            cx.emit(AppEvent::Cursor);
            cx.notify();
        }
    }

    /// Moves the 3D cursor along one axis (`0` z, `1` y, `2` x) by `delta` voxels.
    pub fn nudge_cursor(&mut self, axis: usize, delta: i64, cx: &mut Context<Self>) {
        let Some(mut at) = self.cursor() else { return };
        at[axis] = (at[axis] as i64 + delta).max(0) as usize;
        self.set_cursor(at, cx);
    }

    /// The plane annotations drawn now belong to: the cursor's for a volume, `None` for a 2D image.
    pub fn drawing_plane(&self) -> Option<usize> {
        self.active_pyramid().filter(|p| p.is_volume()).and(self.cursor()).map(|c| c[0])
    }

    // ------------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------------

    /// Switches the compute backend (`None`: automatic) while running: what lives on the old
    /// device is dropped and the results are computed again on the new one.
    pub fn set_runtime(&mut self, runtime: Option<String>, cx: &mut Context<Self>) {
        self.prefs.runtime = runtime;
        self.save_prefs();
        self.device = None;
        self.device_pending = false;
        self.device_generation += 1;
        self.results.clear();
        self.analysis = None;
        cx.emit(AppEvent::Results);
        self.ensure_device(cx);
    }

    /// Removes a processing step; a channel's last step takes its chain with it (its source and
    /// result view), so the channel is back to how it was. One undo step.
    pub fn remove_step(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let n = self.doc.image(&image).map_or(0, |i| i.channels.len());
        let last_of = (0..n).find_map(|c| {
            let (steps, view) = self.doc.graph.channel_chain(&image, c);
            (steps == [id.clone()]).then(|| {
                let source = self.doc.graph.nodes.iter().find(|g| matches!(&g.kind, NodeKind::Source { image: i, channel: ch } if *i == image && *ch == c)).map(|g| g.id.clone());
                view.into_iter().chain(source).collect::<Vec<String>>()
            })
        });
        let cmd = match last_of {
            Some(rest) => Command::Batch { commands: vec![Command::RemoveBridging { id: id.clone() }, Command::RemoveNodes { nodes: rest }] },
            None => Command::RemoveBridging { id },
        };
        if self.execute(cmd, cx).is_ok() {
            self.set_status("Step removed (Ctrl+Z brings it back)", cx);
        }
    }

    /// The object hierarchy of `image` (regions and cells), from the cache when nothing changed.
    pub fn hierarchy(&self, image: &str) -> Arc<crate::domain::hierarchy::Hierarchy> {
        if let Some((i, a, h, hier)) = self.hierarchy.borrow().as_ref()
            && i == image
            && *a == self.annotations_rev
            && *h == self.hist.rev
        {
            return hier.clone();
        }
        let empty = Vec::new();
        let cells = self.hist.cells.get(image).unwrap_or(&empty);
        let hier = Arc::new(crate::domain::hierarchy::Hierarchy::build(self.doc.annotations_of(image), cells));
        *self.hierarchy.borrow_mut() = Some((image.to_string(), self.annotations_rev, self.hist.rev, hier.clone()));
        hier
    }

    /// Channels shown processed (see `GraphDoc::processed_channels`).
    pub fn processed_channels(&self, image: &str) -> std::collections::HashMap<usize, String> {
        let n = self.doc.image(image).map_or(0, |i| i.channels.len());
        self.doc.graph.processed_channels(image, n)
    }

    /// The device stopped responding (lost or poisoned): every handle to it is dropped and the
    /// devices are shut down in the background, so the next run opens a fresh one (never
    /// reopened here: a job that loses it again must not loop).
    pub fn drop_device(&mut self, cx: &mut Context<Self>) {
        self.device = None;
        self.device_pending = false;
        self.device_generation += 1;
        self.results.clear();
        self.results_for = None;
        self.analysis = None;
        cx.emit(AppEvent::Results);
        cx.background_spawn(async { lucem_core::compute::shutdown_devices() }).detach();
    }

    /// A blur of a 512² image on the device, read back checked: the output must be finite, not
    /// all zero, and keep the image's mean (CubeCL #1461: zeros are not success). Says how long.
    pub fn test_device(&mut self, cx: &mut Context<Self>) {
        let Some(device) = self.device.clone() else {
            self.ensure_device(cx);
            return self.set_status("Starting the device; test again when it is up", cx);
        };
        self.set_status(format!("Testing {}…", device.label), cx);
        let task = cx.background_spawn(async move {
            use lucem_core::memory::DeviceArray;
            let n = 512usize;
            let host: Vec<f32> = (0..n * n).map(|i| ((i % n) as f32 / n as f32) * 0.8 + 0.1).collect();
            let mean_in = host.iter().sum::<f32>() / host.len() as f32;
            let t0 = std::time::Instant::now();
            let a = DeviceArray::from_host(&device.client, lucem_core::Shape::new([n, n]), host).map_err(|e| e.to_string())?;
            let out = Op::GaussianBlur { sigma: 2.0 }.run(&device.client, &[lucem_pipeline::Value::Image(a)])?;
            let back = out.to_host(&device.client).ok_or("no raster came back")?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let mean_out = back.iter().sum::<f32>() / back.len().max(1) as f32;
            if back.iter().any(|v| !v.is_finite()) {
                return Err("the device returned values that are not numbers".to_string());
            }
            if back.iter().all(|&v| v == 0.0) || (mean_out - mean_in).abs() > 0.05 {
                return Err(format!("the device returned a wrong result (mean {mean_out:.3}, expected {mean_in:.3})"));
            }
            Ok::<_, String>((device.label.clone(), ms))
        });
        cx.spawn(async move |this, cx| {
            let r = task.await;
            let _ = this.update(cx, |s, cx| match r {
                Ok((label, ms)) => s.set_status(format!("{label}: a blur of 512 × 512 px, checked, in {ms:.1} ms (first runs include compiling)"), cx),
                Err(e) => s.set_status(format!("Device test failed: {e}"), cx),
            });
        })
        .detach();
    }

    // ------------------------------------------------------------------------
    // Machine-learning plugins
    // ------------------------------------------------------------------------

    /// Looks for the plugin programs again (in the background) and reads their manifests.
    pub fn scan_plugins(&mut self, cx: &mut Context<Self>) {
        if self.ml.scanning {
            return;
        }
        self.ml.scanning = true;
        let task = cx.background_spawn(async move { lucem_ml::discover() });
        cx.spawn(async move |this, cx| {
            let found = task.await;
            let _ = this.update(cx, |s, cx| {
                s.ml.scanning = false;
                s.ml.scanned = true;
                s.ml.stop_servers();
                s.ml.set_plugins(found.iter().filter_map(|r| r.as_ref().ok().cloned()).collect());
                s.ml.problems = found.into_iter().filter_map(Result::err).collect();
                cx.emit(AppEvent::Status);
                cx.notify();
            });
        })
        .detach();
    }

    /// Runs a request on a plugin in the background, its progress in `ml.busy`; `done` gets the
    /// answer on the UI thread.
    fn ml_job(&mut self, program: String, label: String, req: lucem_ml::Request, done: impl FnOnce(&mut Self, Result<serde_json::Value, String>, &mut Context<Self>) + 'static, cx: &mut Context<Self>) {
        if self.ml.busy.is_some() {
            return self.set_status("A model is busy; wait for it to finish", cx);
        }
        self.ml.busy = Some((label, None));
        cx.notify();
        let server = self.ml.server(&program);
        let (tx, rx) = async_channel::unbounded::<(String, Option<f32>)>();
        let task = cx.background_spawn(async move {
            super::ml::request(&server, &program, &req, &mut |stage, f| {
                let _ = tx.try_send((stage.to_string(), f));
            })
        });
        cx.spawn(async move |this, cx| {
            while let Ok((stage, f)) = rx.recv().await {
                let _ = this.update(cx, |s, cx| {
                    s.ml.busy = Some((stage, f));
                    cx.notify();
                });
            }
            let r = task.await;
            let _ = this.update(cx, |s, cx| {
                s.ml.busy = None;
                done(s, r, cx);
                cx.notify();
            });
        })
        .detach();
    }

    /// Downloads and verifies a model's weights through its plugin.
    pub fn install_model(&mut self, program: String, model: String, cx: &mut Context<Self>) {
        let label = format!("Installing {model}");
        self.ml_job(
            program,
            label,
            lucem_ml::Request::Install { model: model.clone() },
            move |s, r, cx| {
                match r {
                    Ok(_) => s.set_status(
                        match model.as_str() {
                            lucem_ml::RUNTIME_ID => "ONNX Runtime installed and verified".to_string(),
                            lucem_ml::RUNTIME_CUDA_ID => "ONNX Runtime (GPU) installed and verified; it is used once cuDNN 9 is there".to_string(),
                            _ => format!("{model} installed and verified"),
                        },
                        cx,
                    ),
                    Err(e) => s.set_status(format!("{model}: {e}"), cx),
                }
                s.scan_plugins(cx);
            },
            cx,
        );
    }

    /// The model the Segment tool uses.
    pub fn set_ml_model(&mut self, id: String, cx: &mut Context<Self>) {
        self.update_prefs(|p| p.ml_model = Some(id), cx);
    }

    /// Segments the region of `job` with the chosen model; the regions become annotations.
    /// With `through`, the object is then followed through the planes above and below.
    pub fn segment(&mut self, job: super::ml::SegmentJob, through: Option<super::ml::ZRun>, cx: &mut Context<Self>) {
        let Some((program, model, installed)) = self.ml.segmenter(self.prefs.ml_model.as_deref()) else {
            if !self.ml.scanned {
                self.scan_plugins(cx);
            }
            return self.set_status("No segmentation model: install the SAM plugin (Settings → Models)", cx);
        };
        if !installed {
            return self.set_status(format!("{model} is not installed yet (Settings → Models → Install)"), cx);
        }
        if self.ml.busy.is_some() {
            return self.set_status("The model is still working on the last click", cx);
        }
        self.ml.busy = Some((format!("Segmenting with {model}"), None));
        cx.notify();
        let server = self.ml.server(&program);
        let task = cx.background_spawn(async move {
            let r = super::ml::segment(&server, &program, &model, &job, &mut |_, _| {});
            (r, job)
        });
        cx.spawn(async move |this, cx| {
            let (r, job) = task.await;
            let _ = this.update(cx, |s, cx| {
                s.ml.busy = None;
                match r {
                    Ok((shapes, score, res)) => {
                        let n = shapes.len();
                        let annotations = shapes.iter().cloned().map(|shape| Annotation { id: String::new(), image: job.image.clone(), shape, class: job.class.clone(), note: String::new(), z: job.z }).collect();
                        if s.execute(Command::AddAnnotations { annotations }, cx).is_ok() {
                            let enc = if res.encode_ms > 0.0 { format!("view encoded in {:.0} ms, ", res.encode_ms) } else { String::new() };
                            s.set_status(format!("{n} region(s) of {} (score {score:.2}; {enc}mask in {:.0} ms)", job.class, res.decode_ms), cx);
                            if let Some(run) = through {
                                s.carry_through_z(job, shapes, run, cx);
                            }
                        }
                    }
                    Err(e) => s.set_status(format!("Segment: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Follows a segmented object through the other planes (in the background); every plane it is
    /// found on adds its regions, all in one undo step.
    fn carry_through_z(&mut self, job: super::ml::SegmentJob, first: Vec<crate::domain::annotation::Shape>, run: super::ml::ZRun, cx: &mut Context<Self>) {
        let Some((program, model, true)) = self.ml.segmenter(self.prefs.ml_model.as_deref()) else { return };
        self.ml.busy = Some(("Following it through z".into(), None));
        cx.notify();
        let server = self.ml.server(&program);
        let (tx, rx) = async_channel::unbounded::<usize>();
        let task = cx.background_spawn(async move {
            let r = super::ml::carry_through_z(&server, &program, &model, &job, &first, &run, &mut |n| {
                let _ = tx.try_send(n);
            });
            (r, job)
        });
        cx.spawn(async move |this, cx| {
            while let Ok(n) = rx.recv().await {
                let alive = this.update(cx, |s, cx| {
                    s.ml.busy = Some((format!("Following it through z: {n} plane(s)"), None));
                    cx.notify();
                });
                if alive.is_err() {
                    return;
                }
            }
        })
        .detach();
        cx.spawn(async move |this, cx| {
            let (r, job) = task.await;
            let _ = this.update(cx, |s, cx| {
                s.ml.busy = None;
                match r {
                    Ok(planes) if planes.is_empty() => s.set_status("Through z: the object is not on the next planes (or the model is unsure)", cx),
                    Ok(planes) => {
                        let (lo, hi) = planes.iter().fold((usize::MAX, 0), |(lo, hi), (z, _)| (lo.min(*z), hi.max(*z)));
                        let n = planes.len();
                        let annotations: Vec<Annotation> = planes.into_iter().flat_map(|(z, shapes)| shapes.into_iter().map(move |shape| (z, shape))).map(|(z, shape)| Annotation { id: String::new(), image: job.image.clone(), shape, class: job.class.clone(), note: String::new(), z: Some(z) }).collect();
                        if s.execute(Command::AddAnnotations { annotations }, cx).is_ok() {
                            s.set_status(format!("Followed through {n} more plane(s), z {lo}–{hi}"), cx);
                        }
                    }
                    Err(e) => s.set_status(format!("Through z: {e}"), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    // ------------------------------------------------------------------------
    // Script nodes
    // ------------------------------------------------------------------------

    /// A new script node (Python or Rust) from the template, in the project's `scripts/` folder
    /// (or the user's), selected in the Pipeline so its code opens in the Inspector's editor.
    pub fn add_script_node(&mut self, lang: crate::data::script::Language, cx: &mut Context<Self>) {
        use crate::data::script;
        let dir = script::scripts_dir(self.folder().as_deref());
        let path = match script::new_script(&dir, lang) {
            Ok(p) => p,
            Err(e) => return self.set_status(format!("Could not write the script: {e}"), cx),
        };
        self.add_script_file(path, cx);
    }

    /// A script node running an existing `.py` or `.rs` file.
    pub fn add_script_file(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        use crate::data::script;
        if script::Language::of(&path).is_none() {
            return self.set_status(format!("{}: a script is a .py or .rs file", crate::domain::paths::short_path(&path)), cx);
        }
        let header = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| script::read_header(&t));
        let h = match header {
            Ok(h) => h,
            Err(e) => return self.set_status(format!("{}: {e}", crate::domain::paths::short_path(&path)), cx),
        };
        let kind = NodeKind::Script { path: path.to_string_lossy().into_owned(), inputs: h.inputs, output: h.output, params: h.params, enabled: true };
        let y = self.doc.graph.next_row();
        let before: std::collections::HashSet<String> = self.doc.graph.nodes.iter().map(|n| n.id.clone()).collect();
        if self.execute(Command::AddNode { kind, x: 0.0, y, id: None, after: self.sel.node.clone() }, cx).is_ok() {
            self.set_status(format!("Script {} added: edit it in the Inspector, Save and run", crate::domain::paths::short_path(&path)), cx);
            if let Some(id) = self.doc.graph.nodes.iter().find(|n| !before.contains(&n.id)).map(|n| n.id.clone()) {
                self.set_workspace(Workspace::Pipeline, cx);
                cx.emit(AppEvent::RevealNode(id));
            }
        }
    }

    /// A Model node running `model` (a plugin's); Segment Anything is prompted by the drawing
    /// class' boxes and points until another class is chosen.
    pub fn add_model_node(&mut self, model: String, interactive: bool, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else {
            return self.set_status("Open an image first: the model runs on it", cx);
        };
        let prompts = interactive.then(|| self.tool.class.clone());
        // One selected region annotation: the model runs inside it
        let picked: Vec<&crate::domain::annotation::Annotation> = self.doc.annotations_of(&image).filter(|a| self.sel.shapes.contains(&a.id) && a.shape.is_region()).collect();
        let roi = (picked.len() == 1).then(|| crate::domain::graph::Roi::Annotation { id: picked[0].id.clone() });
        let kind = NodeKind::Model { image, channels: Vec::new(), model: model.clone(), roi, prompts, enabled: true };
        let y = self.doc.graph.next_row();
        if self.execute(Command::AddNode { kind, x: 0.0, y, id: None, after: None }, cx).is_ok() {
            self.set_status(format!("Model {model} added on the shown image; connect it to a view (or a measure)"), cx);
        }
    }

    /// Reads a script node's header again (its ports and numbers; values of numbers kept).
    pub fn reload_script(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(NodeKind::Script { path, params, enabled, .. }) = self.doc.graph.node(id).map(|n| n.kind.clone()) else { return };
        match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| crate::data::script::read_header(&t)) {
            Ok(h) => {
                let params = h.params.into_iter().map(|p| params.iter().find(|q| q.name == p.name).cloned().unwrap_or(p)).collect();
                let kind = NodeKind::Script { path, inputs: h.inputs, output: h.output, params, enabled };
                if self.doc.graph.node(id).map(|n| &n.kind) != Some(&kind) {
                    let _ = self.execute(Command::SetScript { id: id.to_string(), kind }, cx);
                } else {
                    self.recompute(cx);
                }
            }
            Err(e) => self.set_status(format!("{path}: {e}"), cx),
        }
    }

    /// Watches the files of script nodes (once a second): a saved file is read again and runs.
    pub(crate) fn watch_scripts(&mut self, cx: &mut Context<Self>) {
        let any = self.doc.graph.nodes.iter().any(|n| matches!(n.kind, NodeKind::Script { .. }));
        if self.script_watch || !any {
            return;
        }
        self.script_watch = true;
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(std::time::Duration::from_secs(1)).await;
                let alive = this.update(cx, |s, cx| {
                    let scripts: Vec<(String, String)> = s.doc.graph.nodes.iter().filter_map(|n| if let NodeKind::Script { path, .. } = &n.kind { Some((n.id.clone(), path.clone())) } else { None }).collect();
                    if scripts.is_empty() {
                        s.script_watch = false;
                        return false;
                    }
                    for (id, path) in scripts {
                        let Ok(t) = std::fs::metadata(&path).and_then(|m| m.modified()) else { continue };
                        if s.script_seen.insert(id.clone(), t).is_some_and(|old| old != t) {
                            s.reload_script(&id, cx);
                        }
                    }
                    true
                });
                if !matches!(alive, Ok(true)) {
                    break;
                }
            }
        })
        .detach();
    }

    /// Folds a row of the Layers tree shut, or opens it.
    pub fn toggle_layer_open(&mut self, key: &str, cx: &mut Context<Self>) {
        if !self.layers_collapsed.remove(key) {
            self.layers_collapsed.insert(key.to_string());
        }
        cx.notify();
    }

    /// The selected layers that are masks, in selection order: mask results and annotation classes.
    pub fn selected_masks(&self) -> Vec<String> {
        self.sel.layers
            .iter()
            .filter(|k| match k.split_once(':') {
                Some(("class", _)) => true,
                Some(("view", id)) => self.doc.graph.input(id, 0).and_then(|i| self.doc.graph.node(i)).and_then(|n| n.kind.output()) == Some(PortType::Mask),
                _ => false,
            })
            .cloned()
            .collect()
    }

    /// Adds the result of `op` on the selected masks as a new layer (one undo step).
    pub fn apply_mask_operation(&mut self, op: MaskOperation, cx: &mut Context<Self>) {
        let masks = self.selected_masks();
        let Some(image) = self.doc.active_image.clone() else { return };
        let needed = if matches!(op, MaskOperation::Invert | MaskOperation::Op(_)) { 1 } else { 2 };
        if masks.len() < needed {
            return;
        }
        let mut commands = Vec::new();
        let mut graph = self.doc.graph.clone();
        let y = graph.next_row();
        let mut x = 0.0f32;
        // Node ids are reserved in a copy of the graph, so they are unique within the batch
        let fresh = |graph: &mut crate::domain::graph::GraphDoc, base: &str| {
            let id = graph.fresh_id(base);
            graph.nodes.push(crate::domain::graph::GraphNode { id: id.clone(), x: 0.0, y: 0.0, kind: NodeKind::View { name: String::new(), style: ViewStyle::default() } });
            id
        };
        let mut sources = Vec::new();
        let mut names = Vec::new();
        for key in if needed == 1 { &masks[..1] } else { &masks[..] } {
            match key.split_once(':') {
                Some(("view", id)) => {
                    let Some(from) = self.doc.graph.input(id, 0) else { return };
                    sources.push(from.to_string());
                    if let Some(NodeKind::View { name, .. }) = self.doc.graph.node(id).map(|n| &n.kind) {
                        names.push(name.clone());
                    }
                }
                Some(("class", class)) => {
                    let existing = self.doc.graph.nodes.iter().find(|n| matches!(&n.kind, NodeKind::Annotations { image: i, class: Some(c) } if *i == image && c == class));
                    let id = match existing {
                        Some(n) => n.id.clone(),
                        None => {
                            let id = fresh(&mut graph, "annotations");
                            commands.push(Command::AddNode { kind: NodeKind::Annotations { image: image.clone(), class: Some(class.to_string()) }, x, y, id: Some(id.clone()), after: None });
                            x += 240.0;
                            id
                        }
                    };
                    sources.push(id);
                    names.push(class.to_string());
                }
                _ => return,
            }
        }
        let binary = match &op {
            MaskOperation::Union => Some((Op::MaskUnion {}, "∪")),
            MaskOperation::Intersect => Some((Op::MaskIntersect {}, "∩")),
            MaskOperation::Subtract => Some((Op::MaskSubtract {}, "−")),
            _ => None,
        };
        let (last, title) = if let Some((bin, sign)) = binary {
            let mut current = sources[0].clone();
            for src in &sources[1..] {
                let id = fresh(&mut graph, "mask_op");
                commands.push(Command::AddNode { kind: NodeKind::Op { enabled: true, op: bin.clone() }, x, y, id: Some(id.clone()), after: None });
                commands.push(Command::Connect { from: current, to: id.clone(), port: 0 });
                commands.push(Command::Connect { from: src.clone(), to: id.clone(), port: 1 });
                current = id;
                x += 240.0;
            }
            (current, names.join(&format!(" {sign} ")))
        } else {
            let (one, label) = match &op {
                MaskOperation::Op(o) => (o.clone(), o.title().to_string()),
                _ => (Op::MaskInvert {}, "Invert".to_string()),
            };
            let id = fresh(&mut graph, "mask_op");
            commands.push(Command::AddNode { kind: NodeKind::Op { enabled: true, op: one }, x, y, id: Some(id.clone()), after: None });
            commands.push(Command::Connect { from: sources[0].clone(), to: id.clone(), port: 0 });
            x += 240.0;
            (id, format!("{label} {}", names.first().map_or("", String::as_str)))
        };
        let view = fresh(&mut graph, "view");
        commands.push(Command::AddNode { kind: NodeKind::View { name: title, style: ViewStyle::default() }, x, y, id: Some(view.clone()), after: None });
        commands.push(Command::Connect { from: last, to: view.clone(), port: 0 });
        if self.execute(Command::Batch { commands }, cx).is_ok() {
            self.sel.layers = vec![format!("view:{view}")];
            cx.emit(AppEvent::Selection);
        }
    }

    /// A view menu command, for Explore.
    pub fn request_view(&mut self, request: ViewRequest, cx: &mut Context<Self>) {
        cx.emit(AppEvent::View(request));
    }

    /// What is on screen changed: compute the results there (debounced by the canvas).
    pub fn request_results(&mut self, req: ResultRequest, cx: &mut Context<Self>) {
        if self.last_request.as_ref() == Some(&req) && (self.computing || self.results_for.as_ref() == Some(&req)) {
            return;
        }
        self.last_request = Some(req);
        self.recompute(cx);
    }

    /// Computes the results again for the last place on screen.
    pub fn recompute(&mut self, cx: &mut Context<Self>) {
        let Some(req) = self.last_request.clone() else { return };
        if self.computing {
            self.results_pending = true;
            return;
        }
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
        // The whole-image pass runs on the plane shown
        let key = analysis_key(&doc, &req.image) ^ (req.z as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // The whole-image pass again only when what it depends on changed
        let cached = self.analysis.clone().filter(|a| a.key == key);
        let show = self.tool.show_results;
        let image = req.image.clone();
        let asked = req.clone();
        self.computing = true;
        cx.emit(AppEvent::Results);
        let task = cx.background_spawn(async move {
            if latest.load(Ordering::Acquire) != generation {
                return None;
            }
            let analysis = match cached {
                Some(a) => a,
                None => Arc::new(analyze(&device.client, &doc, &pyramid, &req.image, req.z)),
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
                s.computing = false;
                // Shown even when something newer was asked meanwhile (each layer sits at its own
                // place); what was asked meanwhile runs next
                let current = s.results_gen.load(Ordering::Acquire) == generation;
                s.results_for = Some(asked);
                if s.results_pending || !current {
                    s.results_pending = false;
                    s.results_for = None;
                    cx.spawn(async move |this, cx| {
                        let _ = this.update(cx, |s, cx| s.recompute(cx));
                    })
                    .detach();
                }
                if let Some((analysis, Some(list))) = out {
                    s.results.retain(|v, r| r.image != image && s.doc.graph.contains(v));
                    // The views' errors first; the whole-image pass' only when no view was computed
                    // (a model node skips that pass on purpose, and says so in Pipeline)
                    let mut errors: Vec<String> = list.iter().filter_map(|r| r.error.clone().map(|e| format!("{}: {e}", r.view))).collect();
                    if list.is_empty() {
                        errors.extend(analysis.errors.iter().map(|(n, e)| format!("{n}: {e}")));
                    }
                    let ms: f64 = list.iter().map(|r| r.ms).sum();
                    if errors.iter().any(|e| crate::data::hist::is_device_lost(e)) {
                        s.drop_device(cx);
                        s.status = crate::data::hist::DEVICE_LOST.to_string();
                        cx.emit(AppEvent::Status);
                        cx.notify();
                        return;
                    }
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
        // A volume: the plane shown
        let plane = self.drawing_plane();
        self.set_status("Thresholding the whole image…", cx);
        let task = cx.background_spawn(async move {
            let level = (0..py.levels.len()).find(|&l| py.levels[l].width.max(py.levels[l].height) <= 4096).unwrap_or(py.coarsest());
            let l = py.levels[level].clone();
            let ds = l.downsample;
            let at = plane.map_or(0, |z| ((z as f64 / l.z_downsample) as usize).min(l.depth - 1));
            let mut v = py.read_region_at(level, t.threshold_channel, at, 0, 0, l.width, l.height).map_err(|e| e.to_string())?;
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
                    let annotations = shapes.into_iter().map(|shape| Annotation { id: String::new(), image: img.clone(), shape, class: class.clone(), note: String::new(), z: plane }).collect();
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
            lucem_pipeline::Op::GaussianBlur { sigma: t.threshold_sigma.max(0.5) },
            lucem_pipeline::Op::Threshold { value },
            lucem_pipeline::Op::RemoveSmall { min_size: t.threshold_min_area.round() as usize },
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
        let preferred = self.prefs.runtime.clone();
        let generation = self.device_generation;
        let task = cx.background_spawn(async move { Device::discover(preferred.as_deref()) });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |s, cx| {
                // A device asked for before a switch is not the one wanted now
                if s.device_generation != generation {
                    return;
                }
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

    /// Changes the preferences and saves them.
    pub fn update_prefs(&mut self, f: impl FnOnce(&mut Prefs), cx: &mut Context<Self>) {
        f(&mut self.prefs);
        crate::data::tools::set(&self.prefs.tools);
        self.save_prefs();
        cx.notify();
    }

    pub fn set_sidebar(&mut self, open: bool, cx: &mut Context<Self>) {
        self.prefs.sidebar_open = open;
        self.save_prefs();
        cx.notify();
    }

    /// Writes the shown image's annotations as GeoJSON next to the project (or in the working
    /// folder).
    pub fn export_geojson(&mut self, cx: &mut Context<Self>) {
        let Some(img) = self.doc.active().cloned() else { return };
        let list: Vec<&Annotation> = self.doc.annotations_of(&img.id).collect();
        let folder = self.folder().or_else(|| std::env::current_dir().ok()).unwrap_or_default();
        let path = folder.join(format!("{}.geojson", img.name.replace([' ', '/'], "_")));
        let msg = match std::fs::write(&path, super::geojson::export(&list)) {
            Ok(()) => format!("{} annotation(s) written to {}", list.len(), crate::domain::paths::short_path(&path)),
            Err(e) => format!("Could not write {}: {e}", crate::domain::paths::short_path(&path)),
        };
        self.set_status(msg, cx);
    }

    /// Adds the features of a GeoJSON file to the shown image (one undo step); classes it names
    /// that the project lacks are added.
    pub fn import_geojson(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let Some(img) = self.doc.active_image.clone() else { return self.set_status("Open an image first", cx) };
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => return self.set_status(format!("{}: {e}", crate::domain::paths::short_path(&path)), cx),
        };
        let annotations = match super::geojson::import(&text, &img, &self.tool.class) {
            Ok(a) => a,
            Err(e) => return self.set_status(format!("{}: {e}", crate::domain::paths::short_path(&path)), cx),
        };
        let n = annotations.len();
        let mut commands: Vec<Command> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for a in &annotations {
            if self.doc.class(&a.class).is_none() && seen.insert(a.class.clone()) {
                let hue = (seen.len() * 67 % 360) as f32 / 360.0;
                let c = gpui_kit::hsla(hue, 0.65, 0.55, 1.0).to_rgb();
                let class = crate::domain::annotation::ClassDoc { name: a.class.clone(), color: [(c.r * 255.0) as u8, (c.g * 255.0) as u8, (c.b * 255.0) as u8], visible: true, opacity: 1.0 };
                commands.push(Command::SetClass { class });
            }
        }
        commands.push(Command::AddAnnotations { annotations });
        if self.execute(Command::Batch { commands }, cx).is_ok() {
            self.set_status(format!("{n} annotation(s) imported from {}", crate::domain::paths::short_path(&path)), cx);
        }
    }

    /// Selects annotation `id`, shows its image and plane in Explore and brings it into view.
    pub fn reveal_annotation(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(a) = self.doc.annotations.iter().find(|a| a.id == id).cloned() else { return };
        if self.doc.active_image.as_deref() != Some(a.image.as_str()) {
            let _ = self.execute(Command::SelectImage { id: a.image.clone() }, cx);
        }
        self.select([a.id.clone()], cx);
        if let (Some(z), Some(mut at)) = (a.z, self.cursor_of(&a.image)) {
            at[0] = z;
            self.set_cursor_of(&a.image, at, cx);
        }
        let (x0, y0, x1, y1) = a.shape.bounds();
        self.set_workspace(Workspace::Explore, cx);
        cx.emit(AppEvent::Reveal([(x0 + x1) / 2.0, (y0 + y1) / 2.0]));
    }

    /// Deletes class `name`; its annotations move to `move_to`, else they go too. Undoable.
    pub fn delete_class(&mut self, name: String, move_to: Option<String>, cx: &mut Context<Self>) {
        let n = self.doc.annotations.iter().filter(|a| a.class == name).count();
        let what = match (&move_to, n) {
            (_, 0) => format!("Deleted class {name}"),
            (Some(to), n) => format!("Deleted class {name}; its {n} annotation(s) are now {to}"),
            (None, n) => format!("Deleted class {name} and its {n} annotation(s)"),
        };
        if self.execute(Command::RemoveClass { name, move_to }, cx).is_ok() {
            self.set_status(format!("{what} (Ctrl+Z brings it back)"), cx);
        }
    }

    /// Deletes every class without annotations in the project (not Measurement).
    pub fn delete_empty_classes(&mut self, cx: &mut Context<Self>) {
        let empty: Vec<String> = self.doc.classes.iter().filter(|c| c.name != crate::domain::command::MEASUREMENT && !self.doc.annotations.iter().any(|a| a.class == c.name)).map(|c| c.name.clone()).collect();
        let keep_one = self.doc.classes.iter().filter(|c| c.name != crate::domain::command::MEASUREMENT).count() == empty.len();
        let commands: Vec<Command> = empty.iter().skip(usize::from(keep_one)).map(|name| Command::RemoveClass { name: name.clone(), move_to: None }).collect();
        let n = commands.len();
        if n > 0 && self.execute(Command::Batch { commands }, cx).is_ok() {
            self.set_status(format!("Deleted {n} empty class(es)"), cx);
        }
    }

    /// Renames a class (its annotations follow).
    pub fn rename_class(&mut self, from: String, to: String, cx: &mut Context<Self>) {
        let _ = self.execute(Command::RenameClass { from, to }, cx);
    }

    /// Puts annotations in class `class`.
    pub fn move_to_class(&mut self, ids: Vec<String>, class: String, cx: &mut Context<Self>) {
        if !ids.is_empty() {
            let _ = self.execute(Command::SetAnnotationsClass { ids, class }, cx);
        }
    }

    /// Removes an image from the project (with its annotations and the nodes reading it); the
    /// file stays where it is. Undoable.
    pub fn remove_image(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(name) = self.doc.image(&id).map(|i| i.name.clone()) else { return };
        let n = self.doc.annotations.iter().filter(|a| a.image == id).count();
        if self.execute(Command::RemoveImage { id: id.clone() }, cx).is_ok() {
            self.set_status(format!("Removed {name} from the project{} — the file is untouched (Ctrl+Z brings it back)", if n > 0 { format!(" with its {n} annotation(s)") } else { String::new() }), cx);
        }
    }

    /// Removes a result (a view node of the pipeline).
    pub fn remove_result(&mut self, id: String, cx: &mut Context<Self>) {
        let _ = self.execute(Command::RemoveNodes { nodes: vec![id] }, cx);
    }

    /// Combines the selected region shapes of the shown image into new ones (Affinity's Add,
    /// Subtract, Intersect, Xor): Subtract takes the others from the first selected. The
    /// results take the first's class; the shapes combined are removed (one undo step).
    pub fn combine_shapes(&mut self, op: crate::domain::boolean::ShapeOp, cx: &mut Context<Self>) {
        let Some(image) = self.doc.active_image.clone() else { return };
        let picked: Vec<Annotation> = self.doc.annotations_of(&image).filter(|a| self.sel.shapes.contains(&a.id) && a.shape.is_region()).cloned().collect();
        if picked.len() < 2 {
            return self.set_status(format!("{} needs two or more selected regions", op.title()), cx);
        }
        let outlines: Vec<Vec<[f64; 2]>> = picked.iter().map(|a| a.shape.outline()).collect();
        let result = crate::domain::boolean::combine(&outlines, op);
        let first = picked[0].clone();
        let made: Vec<Annotation> = result.into_iter().map(|points| Annotation { id: String::new(), shape: Shape::Polygon { points }, ..first.clone() }).collect();
        let n = made.len();
        let ids: Vec<String> = picked.iter().map(|a| a.id.clone()).collect();
        let before: HashSet<String> = self.doc.annotations.iter().map(|a| a.id.clone()).collect();
        let mut commands = vec![Command::RemoveAnnotations { ids }];
        if n > 0 {
            commands.push(Command::AddAnnotations { annotations: made });
        }
        if self.execute(Command::Batch { commands }, cx).is_ok() {
            let new: Vec<String> = self.doc.annotations.iter().filter(|a| !before.contains(&a.id)).map(|a| a.id.clone()).collect();
            self.select(new, cx);
            self.set_status(if n == 0 { format!("{}: nothing left", op.title()) } else { format!("{}: {n} shape(s) (Ctrl+Z undoes)", op.title()) }, cx);
        }
    }

    pub fn delete_selection(&mut self, cx: &mut Context<Self>) {
        if self.sel.shapes.is_empty() {
            let views: Vec<String> = self.sel.layers.iter().filter_map(|k| k.strip_prefix("view:")).filter(|id| self.doc.graph.contains(id)).map(str::to_string).collect();
            let n = views.len();
            if n > 0 && self.execute(Command::RemoveNodes { nodes: views }, cx).is_ok() {
                self.sel.layers.retain(|k| !k.starts_with("view:"));
                self.set_status(format!("Removed {n} result layer(s) (Ctrl+Z brings them back)"), cx);
            } else if self.sel.layers.iter().any(|k| k.starts_with("class:")) {
                self.set_status("A class is deleted from its menu (right-click it in Layers)", cx);
            }
            return;
        }
        let ids: Vec<String> = self.sel.shapes.iter().cloned().collect();
        let n = ids.len();
        if self.execute(Command::RemoveAnnotations { ids }, cx).is_ok() {
            self.sel.shapes.clear();
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
    if s == "sample:volume" {
        let mut py = lucem_io::sample_volume(256, 128).map_err(|e| e.to_string())?;
        py.path = Some(path.to_path_buf());
        return Ok(py);
    }
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

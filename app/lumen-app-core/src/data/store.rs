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
use crate::domain::project::{ChannelDoc, ProjectDoc, PROJECT_FILE};
use crate::domain::tool::{Tool, ToolState};
use crate::navigation::route::Workspace;

use super::device::Device;
use super::prefs::Prefs;
use super::results::{compute, ResultRequest, Results};

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
                            s.pyramids.insert(id.clone(), Arc::new(py));
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
                            s.pyramids.insert(id.clone(), Arc::new(py));
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
        if !self.tool.show_results {
            return;
        }
        let has_views = self.doc.graph.views().any(|v| self.doc.graph.images_of(&v.id).contains(&req.image));
        if !has_views {
            if !self.results.is_empty() {
                self.results.clear();
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
        self.computing = true;
        cx.emit(AppEvent::Results);
        let task = cx.background_spawn(async move {
            // A newer request already waits: skip this one
            if latest.load(Ordering::Acquire) != generation {
                return None;
            }
            Some(compute(&device.client, &doc, &pyramid, &req))
        });
        cx.spawn(async move |this, cx| {
            let out = task.await;
            let _ = this.update(cx, |s, cx| {
                if s.results_gen.load(Ordering::Acquire) != generation {
                    return;
                }
                s.computing = false;
                if let Some(list) = out {
                    let errors: Vec<String> = list.iter().filter_map(|r| r.error.clone().map(|e| format!("{}: {e}", r.view))).collect();
                    let ms: f64 = list.iter().map(|r| r.ms).sum();
                    s.results.retain(|v, r| r.image != list.first().map_or(String::new(), |f| f.image.clone()) && s.doc.graph.contains(v));
                    for r in list {
                        s.results.insert(r.view.clone(), r);
                    }
                    if let Some(e) = errors.first() {
                        s.status = e.clone();
                        cx.emit(AppEvent::Status);
                    } else if ms > 0.0 {
                        s.status = format!("Results computed in {ms:.0} ms on {}", s.device.as_ref().map_or("—", |d| d.label.as_str()));
                        cx.emit(AppEvent::Status);
                    }
                }
                cx.emit(AppEvent::Results);
                cx.notify();
            });
        })
        .detach();
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

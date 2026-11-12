//! lumen's commands, declared once (kitchen-shell's [`Registry`]): each has an id, a title, an
//! icon, a line saying what it does, a group, a key, when it can run (and why not) and what it
//! does. The menus, the keys, the search (Ctrl+K) and an agent's tools all come from this list,
//! so none drifts from the others.
//!
//! Ids are dotted by area: `file.*`, `edit.*`, `view.*`, `image.*`, `tool.*`, `shapes.*`,
//! `process.*`, `hist.*`, `script.*`, `settings.*`, `remote.*`. Entries that follow what the
//! project holds (its images, channels, recent files, the models that fit) are menu providers
//! running a command with arguments (`file.open {path}`, `image.select {id}`…).
//! The Pipeline's and Analyze's own commands live with their views
//! ([`crate::screens::pipeline::PipelineView::commands`], [`crate::screens::analyze::AnalyzeView::commands`]).

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::{App, Entity, PathPromptOptions, SharedString, WeakEntity, Window};
use kitchen_shell::{CommandSpec, MenuItem, MenuSpec, Registry, Shell};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::data::store::{Arrangement, MaskOperation, Store, ViewKind, ViewRequest};
use crate::domain::boolean::ShapeOp;
use crate::domain::command::Command;
use crate::domain::tool::{Tool, ToolState};
use crate::navigation::route::Workspace;
use crate::screens::pipeline::catalog::{operations, Template};

/// What the commands act on.
#[derive(Clone)]
pub struct Ctx {
    pub store: Entity<Store>,
    /// The frame (set once it exists: commands only run after).
    pub shell: Rc<std::cell::OnceCell<WeakEntity<Shell>>>,
    pub pipeline: WeakEntity<crate::screens::pipeline::PipelineView>,
    pub analyze: WeakEntity<crate::screens::analyze::AnalyzeView>,
    /// Shows or hides the Keyboard & mouse sheet.
    pub toggle_help: Rc<dyn Fn(&mut Window, &mut App)>,
}

impl Ctx {
    /// Runs `f` on the frame.
    pub fn shell(&self, cx: &mut App, f: impl FnOnce(&mut Shell, &mut gpui_kit::Context<Shell>)) {
        if let Some(s) = self.shell.get().and_then(|w| w.upgrade()) {
            s.update(cx, f);
        }
    }
}

/// The command id of an operation of the catalog (`process.op.gaussian_blur`).
pub fn op_id(label: &str) -> String {
    let slug: String = label.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("process.op.{}", slug.split('_').filter(|s| !s.is_empty()).collect::<Vec<_>>().join("_"))
}

/// The command id of a tool (`tool.rectangle`).
pub fn tool_id(t: Tool) -> String {
    format!("tool.{}", t.name().to_lowercase().replace(|c: char| !c.is_ascii_alphanumeric(), "_"))
}

fn needs_image(store: &Entity<Store>) -> impl Fn(&App) -> Result<(), String> + 'static {
    let store = store.clone();
    move |cx| if store.read(cx).doc.active().is_some() { Ok(()) } else { Err("Open an image first".into()) }
}

// ----------------------------------------------------------------------------
// File dialogs
// ----------------------------------------------------------------------------

fn prompt(store: Entity<Store>, directories: bool, files: bool, prompt: &'static str, save_folder: bool, cx: &mut App) {
    let rx = cx.prompt_for_paths(PathPromptOptions { files, directories, multiple: false, prompt: Some(prompt.into()) });
    cx.spawn(async move |cx| match rx.await {
        Ok(Ok(Some(paths))) => {
            if let Some(path) = paths.into_iter().next() {
                store.update(cx, |s, cx| if save_folder { s.save_in(path, cx) } else { s.open(path, cx) });
            }
        }
        Ok(Err(e)) => store.update(cx, |s, cx| s.set_status(format!("File dialog failed: {e}"), cx)),
        _ => {}
    })
    .detach();
}

/// Asks for an image or OME-Zarr container (a folder) and opens it.
pub fn open_image(store: &Entity<Store>, cx: &mut App) {
    prompt(store.clone(), true, true, "Open image or OME-Zarr", false, cx);
}

/// Asks for a project (its folder or `project.lumen.json`) and opens it.
pub fn open_project(store: &Entity<Store>, cx: &mut App) {
    prompt(store.clone(), true, true, "Open project (folder or project.lumen.json)", false, cx);
}

fn save_project(store: &Entity<Store>, cx: &mut App) {
    prompt(store.clone(), true, false, "Save the project in this folder", true, cx);
}

/// Applies `op` to what Explore shows: before its first result, or as a new chain from the first
/// visible channel.
fn apply_op(store: &Entity<Store>, op: lumen_pipeline::Op, cx: &mut App) {
    store.update(cx, |s, cx| {
        let Some(img) = s.doc.active().cloned() else {
            s.set_status("Open an image first", cx);
            return;
        };
        let view = s.doc.graph.views().find(|v| s.doc.graph.images_of(&v.id).contains(&img.id)).map(|v| v.id.clone());
        let channel = img.channels.iter().position(|c| c.visible).unwrap_or(0);
        let title = op.title();
        if s.execute(Command::ApplyOp { op, view, image: img.id.clone(), channel }, cx).is_ok() {
            s.set_status(format!("{title} added to the pipeline"), cx);
        }
    });
}

// ----------------------------------------------------------------------------
// Arguments
// ----------------------------------------------------------------------------

/// A file to open.
#[derive(Deserialize, JsonSchema)]
struct PathArgs {
    /// An image, an OME-Zarr container or a project (its folder or `project.lumen.json`).
    path: String,
}

/// An image of the project.
#[derive(Deserialize, JsonSchema)]
struct ImageArgs {
    /// Its id.
    id: String,
}

/// A channel of the shown image.
#[derive(Deserialize, JsonSchema)]
struct ChannelArgs {
    /// Its index (0 is the first).
    index: usize,
}

/// A kind of view.
#[derive(Deserialize, JsonSchema)]
struct ViewArgs {
    /// "Plane", "Ortho", "3D" or "Channels" (as the View menu names them).
    kind: String,
}

/// An arrangement of the views.
#[derive(Deserialize, JsonSchema)]
struct ArrangeArgs {
    /// "Tabs", "Side by side", "Stacked" or "Grid" (as the View menu names them).
    how: String,
}

/// A nucleus detector.
#[derive(Deserialize, JsonSchema)]
struct DetectorArgs {
    /// A nucleus model plugin's id; none for the built-in threshold.
    #[serde(default)]
    model: Option<String>,
}

/// Where an operation goes.
#[derive(Deserialize, JsonSchema, Default)]
struct OpArgs {
    /// A node on the Pipeline's canvas instead of a step on the channel Explore shows.
    #[serde(default)]
    node: bool,
}

fn kind_named(name: &str) -> Result<ViewKind, String> {
    ViewKind::ALL.into_iter().find(|k| k.title().eq_ignore_ascii_case(name)).ok_or_else(|| format!("no view kind {name:?}"))
}

// ----------------------------------------------------------------------------
// The commands
// ----------------------------------------------------------------------------

/// Registers lumen's commands and menu providers (the Pipeline's and Analyze's register their own).
pub fn register(registry: &Registry, c: &Ctx) {
    let s = &c.store;
    let st = |f: fn(&mut Store, &mut gpui_kit::Context<Store>)| {
        let store = s.clone();
        move |_: &mut Window, cx: &mut App| store.update(cx, f)
    };
    let tool = |f: fn(&mut ToolState)| {
        let store = s.clone();
        move |_: &mut Window, cx: &mut App| store.update(cx, |s, cx| s.update_tool(f, cx))
    };
    let tool_on = |f: fn(&ToolState) -> bool| {
        let store = s.clone();
        move |cx: &App| f(&store.read(cx).tool)
    };
    let view = |r: ViewRequest| {
        let store = s.clone();
        move |_: &mut Window, cx: &mut App| store.update(cx, |s, cx| s.request_view(r, cx))
    };
    let (s1, s2, s3, s4, s5) = (s.clone(), s.clone(), s.clone(), s.clone(), s.clone());
    let mut list = vec![
        // File
        CommandSpec::new("file.new", "New project", IconName::FilePlus, st(|s, cx| s.new_project(cx))).group("File").hint("Close the project and start an empty one").key("secondary-n"),
        CommandSpec::new("file.open_image", "Open image or OME-Zarr…", IconName::ImagePlus, move |_, cx| open_image(&s1, cx)).group("File").hint("Add an image to the project").key("secondary-o"),
        CommandSpec::new("file.open_project", "Open project…", IconName::FolderOpen, move |_, cx| open_project(&s2, cx)).group("File").hint("A project folder or project.lumen.json").key("secondary-shift-o"),
        CommandSpec::new("file.save_in", "Save project in folder…", IconName::Save, move |_, cx| save_project(&s3, cx)).group("File").hint("Writes project.lumen.json, pipeline.json and layout.json in a folder").key("secondary-shift-s"),
        CommandSpec::with_args::<PathArgs>("file.open", "Open a file", IconName::FolderOpen, move |a, _, cx| {
            s4.update(cx, |s, cx| s.open(a.path.into(), cx));
            Ok(Value::Null)
        })
        .group("File")
        .hint("Open an image, an OME-Zarr container or a project by its path"),
        CommandSpec::new("file.import_geojson", "Import GeoJSON…", IconName::FileInput, move |_, cx| {
            let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some("GeoJSON file".into()) });
            let st = s5.clone();
            cx.spawn(async move |cx| {
                if let Ok(Ok(Some(paths))) = rx.await
                    && let Some(p) = paths.into_iter().next()
                {
                    st.update(cx, |s, cx| s.import_geojson(p, cx));
                }
            })
            .detach();
        })
        .group("File")
        .hint("Shapes from a GeoJSON file onto the shown image")
        .enabled(needs_image(s)),
        CommandSpec::new("file.export_geojson", "Export as GeoJSON", IconName::FileOutput, st(|s, cx| s.export_geojson(cx))).group("File").hint("The shown image's shapes as GeoJSON").enabled(needs_image(s)),
        CommandSpec::new("file.export_cells", "Export cell measurements", IconName::Sheet, st(|s, cx| s.hist_export(cx))).group("File").hint("Cells, tiles, labels and provenance to the output store").enabled(needs_image(s)),
        CommandSpec::new("app.quit", "Quit", IconName::Power, |_, cx| cx.quit()).group("File").hint("Close lumen").key("secondary-q"),
    ];
    // Edit
    let (u1, u2, d1) = (s.clone(), s.clone(), s.clone());
    list.extend([
        CommandSpec::new("edit.undo", "Undo", IconName::Undo2, st(|s, cx| s.undo(false, cx))).group("Edit").hint("Undo the last change").key("secondary-z").enabled(move |cx| if u1.read(cx).can_undo() { Ok(()) } else { Err("Nothing to undo".into()) }),
        CommandSpec::new("edit.redo", "Redo", IconName::Redo2, st(|s, cx| s.undo(true, cx))).group("Edit").hint("Redo what was undone").key("secondary-shift-z").enabled(move |cx| if u2.read(cx).can_redo() { Ok(()) } else { Err("Nothing to redo".into()) }),
        CommandSpec::new("edit.delete_selection", "Delete selected", IconName::Trash, st(|s, cx| s.delete_selection(cx))).group("Edit").hint("The selected shapes (Del on the image)").enabled(move |cx| if d1.read(cx).sel.shapes.is_empty() { Err("Select shapes first".into()) } else { Ok(()) }),
    ]);
    // View: what is drawn over the image, the views, the panels
    list.extend([
        CommandSpec::new("view.shapes", "Shapes", IconName::Shapes, tool(|t| t.show_annotations = !t.show_annotations)).group("View").hint("Show the shapes over the image (A on the image)").checked(tool_on(|t| t.show_annotations)),
        CommandSpec::new("view.fill", "Fill regions", IconName::PaintBucket, tool(|t| t.fill_annotations = !t.fill_annotations)).group("View").hint("Fill closed shapes (Shift+F on the image)").checked(tool_on(|t| t.fill_annotations)),
        CommandSpec::new("view.names", "Names", IconName::Type, tool(|t| t.show_names = !t.show_names)).group("View").hint("Shapes' names (Shift+N on the image)").checked(tool_on(|t| t.show_names)),
        CommandSpec::new("view.results", "Results over the image", IconName::ChartColumn, tool(|t| t.show_results = !t.show_results)).group("View").hint("Show or hide the pipeline's results").checked(tool_on(|t| t.show_results)),
        CommandSpec::new("view.crosshair", "Crosshair (3D cursor)", IconName::Crosshair, tool(|t| t.crosshair = !t.crosshair)).group("View").hint("The 3D cursor drawn across planes").checked(tool_on(|t| t.crosshair)),
        CommandSpec::new("view.rulers", "Rulers", IconName::Ruler, tool(|t| t.rulers = !t.rulers)).group("View").hint("Rulers along the edges of a plane").key("secondary-r").checked(tool_on(|t| t.rulers)),
        CommandSpec::new("view.link", "Link views", IconName::Link, view(ViewRequest::ToggleLink)).group("View").hint("Views of the same image move and zoom together").checked({
            let s = s.clone();
            move |cx| s.read(cx).prefs.link_views
        }),
        CommandSpec::new("view.pipeline_dock", "Pipeline under the views", IconName::Workflow, view(ViewRequest::TogglePipeline)).group("View").hint("The Pipeline as a panel of Explore").checked({
            let s = s.clone();
            move |cx| s.read(cx).prefs.pipeline_dock
        }),
        CommandSpec::new("view.left", "Layers panel", IconName::Layers2, view(ViewRequest::ToggleLeft)).group("View").hint("Show or hide the Layers panel").key("secondary-["),
        CommandSpec::new("view.right", "Side panel", IconName::PanelRight, view(ViewRequest::ToggleRight)).group("View").hint("Show or hide the studio panels").key("secondary-]"),
        CommandSpec::new("view.reset_layout", "Reset panel layout", IconName::RotateCcw, view(ViewRequest::ResetLayout)).group("View").hint("The panels back where they start"),
        CommandSpec::new("view.close", "Close this view", IconName::Close, view(ViewRequest::ClosePane)).group("View").hint("Close the focused view").enabled(needs_image(s)),
    ]);
    let (v1, v2, v3) = (s.clone(), s.clone(), s.clone());
    list.extend([
        CommandSpec::with_args::<ViewArgs>("view.show", "This view shows", IconName::LayoutGrid, move |a, _, cx| {
            let k = kind_named(&a.kind)?;
            v1.update(cx, |s, cx| s.request_view(ViewRequest::Show(k), cx));
            Ok(Value::Null)
        })
        .group("View")
        .hint("What the focused view shows: a plane, three planes, 3D or the channels side by side")
        .enabled(needs_image(s)),
        CommandSpec::with_args::<ViewArgs>("view.add", "Add a view", IconName::SquarePlus, move |a, _, cx| {
            let k = kind_named(&a.kind)?;
            v2.update(cx, |s, cx| s.request_view(ViewRequest::Add(k), cx));
            Ok(Value::Null)
        })
        .group("View")
        .hint("A new view of the image beside the others")
        .enabled(needs_image(s)),
        CommandSpec::with_args::<ArrangeArgs>("view.arrange", "Arrange the views", IconName::LayoutDashboard, move |a, _, cx| {
            let how = Arrangement::ALL.into_iter().find(|h| h.title().eq_ignore_ascii_case(&a.how)).ok_or_else(|| format!("no arrangement {:?}", a.how))?;
            v3.update(cx, |s, cx| s.request_view(ViewRequest::Arrange(how), cx));
            Ok(Value::Null)
        })
        .group("View")
        .hint("Tabs, side by side, stacked or a grid")
        .enabled(needs_image(s)),
    ]);
    // Image: channels and the project's images
    let (i1, i2, i3) = (s.clone(), s.clone(), s.clone());
    list.extend([
        CommandSpec::with_args::<ChannelArgs>("image.channel", "Show or hide a channel", IconName::Eye, move |a, _, cx| {
            i1.update(cx, |s, cx| {
                let image = s.doc.active_image.clone().ok_or("Open an image first")?;
                let mut display = s.doc.image(&image).and_then(|i| i.channels.get(a.index).cloned()).ok_or_else(|| format!("no channel {}", a.index))?;
                display.visible = !display.visible;
                s.execute(Command::SetChannel { image, channel: a.index, display }, cx).map_err(|e| e.to_string())?;
                Ok(Value::Null)
            })
        })
        .group("Image")
        .hint("Toggle a channel of the shown image")
        .enabled(needs_image(s)),
        CommandSpec::with_args::<ImageArgs>("image.select", "Show an image", IconName::Image, move |a, _, cx| {
            i2.update(cx, |s, cx| s.execute(Command::SelectImage { id: a.id }, cx).map(|_| Value::Null).map_err(|e| e.to_string()))
        })
        .group("Image")
        .hint("Show another image of the project"),
        CommandSpec::new("image.remove", "Remove from the project (the file stays)", IconName::Trash, move |_, cx| {
            i3.update(cx, |s, cx| {
                if let Some(id) = s.doc.active_image.clone() {
                    s.remove_image(id, cx);
                }
            })
        })
        .group("Image")
        .hint("Take the shown image out of the project; its file is not touched")
        .enabled(needs_image(s)),
    ]);
    // Tools: one command each (their keys work on the image, so typing never switches tools)
    for t in Tool::BAR.iter().flatten().copied() {
        let (store, on) = (s.clone(), s.clone());
        let mut cmd = CommandSpec::new(tool_id(t), t.name(), t.icon(), move |_, cx| {
            store.update(cx, |s, cx| {
                s.set_tool(t, cx);
                s.set_workspace(Workspace::Explore, cx);
            })
        })
        .group("Tools")
        .hint(format!("{} · key {} on the image", t.hint(), t.key().to_uppercase()))
        .checked(move |cx| on.read(cx).tool.tool == t);
        if t == Tool::Segment {
            let store = s.clone();
            cmd = cmd.enabled(move |cx| {
                let s = store.read(cx);
                if s.ml.segmenter(s.prefs.ml_model.as_deref()).is_some() { Ok(()) } else { Err("Install Segment Anything in Settings › Models".into()) }
            });
        }
        list.push(cmd);
    }
    // Shapes: combine the selected regions; masks
    for op in ShapeOp::ALL {
        let (store, sel) = (s.clone(), s.clone());
        let icon = match op {
            ShapeOp::Add => IconName::SquaresUnite,
            ShapeOp::Subtract => IconName::SquaresSubtract,
            ShapeOp::Intersect => IconName::SquaresIntersect,
            ShapeOp::Xor => IconName::SquaresExclude,
        };
        list.push(
            CommandSpec::new(format!("shapes.{}", op.title().to_lowercase()), op.hint(), icon, move |_, cx| store.update(cx, |s, cx| s.combine_shapes(op, cx)))
                .group("Shapes")
                .hint("The selected regions as one")
                .enabled(move |cx| if sel.read(cx).sel.shapes.len() >= 2 { Ok(()) } else { Err("Select two regions or more".into()) }),
        );
    }
    let masks: [(&str, &str, usize, IconName, MaskOperation); 6] = [
        ("union", "Union", 2, IconName::SquaresUnite, MaskOperation::Union),
        ("intersect", "Intersect", 2, IconName::SquaresIntersect, MaskOperation::Intersect),
        ("subtract", "Subtract", 2, IconName::SquaresSubtract, MaskOperation::Subtract),
        ("invert", "Invert", 1, IconName::Contrast, MaskOperation::Invert),
        ("grow", "Grow 2 px", 1, IconName::Expand, MaskOperation::Op(lumen_pipeline::Op::Dilate { radius: 2 })),
        ("shrink", "Shrink 2 px", 1, IconName::Shrink, MaskOperation::Op(lumen_pipeline::Op::Erode { radius: 2 })),
    ];
    for (id, title, need, icon, op) in masks {
        let (store, sel) = (s.clone(), s.clone());
        list.push(
            CommandSpec::new(format!("mask.{id}"), title, icon, move |_, cx| store.update(cx, |s, cx| s.apply_mask_operation(op.clone(), cx)))
                .group("Masks")
                .hint("Of the selected masks")
                .enabled(move |cx| if sel.read(cx).selected_masks().len() >= need { Ok(()) } else { Err(format!("Select {need} mask{} first", if need > 1 { "s" } else { "" })) }),
        );
    }
    // Process: every operation (a step on the shown channel, or a node on the Pipeline's canvas)
    let pipeline = c.pipeline.clone();
    for cat in operations() {
        for it in cat.items {
            let Template::Op(op) = it.template.clone() else { continue };
            let (store, pipeline, template) = (s.clone(), pipeline.clone(), it.template.clone());
            list.push(
                CommandSpec::with_args::<OpArgs>(op_id(&it.label), it.label.clone(), cat.icon, move |a, _, cx| {
                    let on_canvas = a.node || store.read(cx).workspace == Workspace::Pipeline;
                    if on_canvas {
                        let _ = pipeline.update(cx, |p, cx| p.canvas.update(cx, |c, cx| c.add(&template, None, cx)));
                    } else {
                        apply_op(&store, op.clone(), cx);
                    }
                    Ok(Value::Null)
                })
                .group(cat.name)
                .hint(format!("{} · a step on the shown channel, or a node in the Pipeline", it.hint)),
            );
        }
    }
    let (f1, f2) = (c.clone(), s.clone());
    list.push(
        CommandSpec::new("process.find", "Find an operation…", IconName::Search, move |window, cx| {
            let pipeline = f2.read(cx).workspace == Workspace::Pipeline;
            let filter: kitchen_shell::search::Filter = Rc::new(|c: &CommandSpec| c.id.starts_with("process.op."));
            let title = if pipeline { "Add a node" } else { "Find an operation" };
            f1.shell(cx, |sh, cx| sh.open_search_with(Some((title.into(), filter)), window, cx));
        })
        .group("Process")
        .hint("Search the operations only")
        .key("f3")
        .read_only(),
    );
    // Analyze: find and measure (in the selected regions)
    let hist = |id: &str, title: &str, icon: IconName, hint: &str, f: fn(&mut Store, &mut gpui_kit::Context<Store>)| CommandSpec::new(id.to_string(), title.to_string(), icon, st(f)).group("Analyze").hint(hint.to_string()).enabled(needs_image(s));
    let (b1, a1) = (s.clone(), s.clone());
    list.extend([
        hist("hist.find_tissue", "Find tissue", IconName::ScanSearch, "Regions of class Tissue", |s, cx| s.hist_detect_tissue(cx)),
        hist("hist.find_cells", "Find cells", IconName::Sparkles, "In the selected regions (or the tissue)", |s, cx| s.hist_detect_cells(cx)).enabled(move |cx| {
            let s = b1.read(cx);
            if s.doc.active().is_some() { Ok(()) } else { Err("Open an image first".into()) }
        }),
        hist("hist.superpixels", "Superpixels", IconName::Grid3x3, "Tiles inside the selected regions, measured", |s, cx| s.hist_superpixels(cx)),
        hist("hist.density", "Density map", IconName::Flame, "Cells per mm² (or positive %) within a radius", |s, cx| s.hist_density(cx)),
        hist("hist.hotspots", "Mark 3 hotspots", IconName::MapPin, "The three densest places as points", |s, cx| s.hist_hotspots(3, cx)),
        CommandSpec::new("analyze.table", "Measurements table", IconName::Table2, move |_, cx| a1.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx))).group("Analyze").hint("The Analyze workspace").read_only(),
        CommandSpec::with_args::<DetectorArgs>("hist.detector", "Find cells with", IconName::Microscope, {
            let s = s.clone();
            move |a, _, cx| {
                s.update(cx, |s, cx| s.hist_set_detector(a.model, cx));
                Ok(Value::Null)
            }
        })
        .group("Analyze")
        .hint("The threshold (built in) or a nucleus model"),
    ]);
    // Classify
    let classify = |id: &str, title: &str, icon: IconName, hint: &str, f: fn(&mut Store, &mut gpui_kit::Context<Store>)| CommandSpec::new(id.to_string(), title.to_string(), icon, st(f)).group("Classify").hint(hint.to_string()).enabled(needs_image(s));
    list.extend([
        classify("hist.train", "Train from the classed shapes", IconName::BrainCircuit, "Random Trees on cells and tiles", |s, cx| s.hist_train_classifier(cx)),
        classify("hist.intensity", "Intensity classes (1+ / 2+ / 3+)", IconName::Gauge, "Cells by one measurement", |s, cx| s.hist_apply_intensity(cx)),
        classify("hist.train_pixels", "Train and classify pixels", IconName::Palette, "Painted and drawn classes → a class map", |s, cx| s.hist_train_pixels(cx)),
        classify("hist.class_regions", "Class map → shapes", IconName::Shapes, "The class map's regions as shapes", |s, cx| s.hist_class_regions(cx)),
    ]);
    // Automate: scripts
    let (p1, p2, an) = (s.clone(), s.clone(), c.analyze.clone());
    let an_store = s.clone();
    list.extend([
        CommandSpec::new("script.new_python", "New Python script node…", IconName::SquareTerminal, move |_, cx| p1.update(cx, |s, cx| s.add_script_node(crate::data::script::Language::Python, cx))).group("Automate").hint("A step written in Python"),
        CommandSpec::new("script.new_rust", "New Rust script node…", IconName::SquareTerminal, move |_, cx| p2.update(cx, |s, cx| s.add_script_node(crate::data::script::Language::Rust, cx))).group("Automate").hint("A step written in Rust (crates allowed)"),
        CommandSpec::new("analyze.code", "Code on the measurement table", IconName::Code, move |_, cx| {
            an_store.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx));
            let _ = an.update(cx, |a, cx| a.open_code(cx));
        })
        .group("Automate")
        .hint("Python or Rust over the table"),
    ]);
    // Help, Settings sections, where lumen runs
    let (h, m1) = (c.toggle_help.clone(), c.clone());
    list.extend([
        CommandSpec::new("help.keys", "Keyboard & mouse", IconName::Keyboard, move |w, cx| h(w, cx)).group("Help").hint("Every key and gesture").key("f1").read_only(),
        CommandSpec::new("settings.models", "Models…", IconName::Sparkles, move |window, cx| {
            m1.shell(cx, |sh, cx| {
                let i = sh.settings_index("Models");
                sh.show_settings(i, window, cx)
            })
        })
        .group("App")
        .hint("Install and choose the model plugins (Settings › Models)")
        .read_only(),
        CommandSpec::new("remote.this", "This computer", IconName::Monitor, |_, _| {}).group("App").hint("lumen runs on this computer").checked(|_| true).read_only(),
        CommandSpec::new("remote.connect", "Connect to a lumen server…", IconName::Server, |_, _| {}).group("App").hint("Images and work on a paired server").enabled(|_| Err("Pairing in the app is next: `lumen pair` pairs from the command line".into())),
    ]);
    registry.extend(list);
    providers(registry, s);
}

/// The menu entries that follow what the project holds.
fn providers(registry: &Registry, s: &Entity<Store>) {
    let store = s.clone();
    registry.provide("file.recent", move |cx| {
        let recent = store.read(cx).prefs.recent.clone();
        if recent.is_empty() {
            return vec![];
        }
        let mut out = vec![MenuItem::Separator, MenuItem::label("Recent")];
        out.extend(recent.iter().take(8).map(|p| MenuItem::run(crate::data::store::display_name(p), "file.open", json!({"path": p}))));
        out
    });
    let store = s.clone();
    registry.provide("image.channels", move |cx| {
        let s = store.read(cx);
        let Some(img) = s.doc.active() else { return vec![MenuItem::label("No image open")] };
        img.channels.iter().enumerate().map(|(k, c)| MenuItem::run(c.label.clone(), "image.channel", json!({"index": k})).checked(c.visible)).collect()
    });
    let store = s.clone();
    registry.provide("image.images", move |cx| {
        let s = store.read(cx);
        s.doc.images.iter().map(|i| MenuItem::run(i.name.clone(), "image.select", json!({"id": i.id})).icon(IconName::Image).checked(s.doc.active_image.as_deref() == Some(i.id.as_str()))).collect()
    });
    let store = s.clone();
    registry.provide("hist.detectors", move |cx| {
        let s = store.read(cx);
        let current = s.hist.detector.clone();
        let mut out = vec![MenuItem::run("Threshold (built in)", "hist.detector", json!({})).checked(current.is_none())];
        let brightfield = s.doc.active_image.as_deref().map(|i| s.hist_is_brightfield(i));
        if let Some(b) = brightfield {
            for (_, m) in s.ml.nucleus_models_for(b) {
                if m.installed {
                    out.push(MenuItem::run(m.name.clone(), "hist.detector", json!({"model": m.id})).icon(IconName::BrainCircuit).checked(current.as_deref() == Some(m.id.as_str())));
                } else {
                    out.push(MenuItem::label(format!("{} (install in Settings › Models)", m.name)));
                }
            }
        }
        out
    });
    let store = s.clone();
    registry.provide("hist.classifiers", move |cx| {
        let s = store.read(cx);
        let models: Vec<MenuItem> = s.ml.plugins.iter().flat_map(|p| p.models.iter()).filter(|m| m.task == lumen_ml::Task::Classification).map(|m| MenuItem::label(if m.installed { m.name.clone() } else { format!("{} (install in Settings › Models)", m.name) })).collect();
        if models.is_empty() {
            return vec![];
        }
        let mut out = vec![MenuItem::label("Models")];
        out.extend(models);
        out
    });
    registry.provide("view.kinds", |_| ViewKind::ALL.iter().map(|k| MenuItem::run(format!("{}: {}", k.title(), k.hint()), "view.show", json!({"kind": k.title()})).icon(k.icon())).collect());
    registry.provide("view.add_kinds", |_| {
        let mut out: Vec<MenuItem> = ViewKind::ALL.iter().map(|k| MenuItem::run(format!("Add a {} view", k.title()), "view.add", json!({"kind": k.title()})).icon(k.icon())).collect();
        out.push(MenuItem::command("view.close"));
        out.push(MenuItem::Separator);
        out.extend(Arrangement::ALL.iter().map(|h| MenuItem::run(format!("Arrange: {}", h.title()), "view.arrange", json!({"how": h.title()}))));
        out
    });
    registry.provide("process.families", |_| {
        operations()
            .into_iter()
            .map(|c| {
                let items = c.items.iter().filter(|i| matches!(i.template, Template::Op(_))).map(|i| MenuItem::command(op_id(&i.label))).collect();
                MenuItem::submenu_with_icon(c.name, c.icon, items)
            })
            .collect()
    });
}

/// The app menu (named after the app): files, projects, shapes; Settings comes last (the frame's).
pub fn app_menu() -> Vec<MenuItem> {
    vec![
        MenuItem::command("file.new"),
        MenuItem::command("file.open_image"),
        MenuItem::command("file.open_project"),
        MenuItem::command("file.save_in"),
        MenuItem::Separator,
        MenuItem::label("Shapes"),
        MenuItem::command("file.import_geojson"),
        MenuItem::command("file.export_geojson"),
        MenuItem::command("file.export_cells"),
        MenuItem::provided("file.recent"),
        MenuItem::Separator,
        MenuItem::command("app.quit"),
    ]
}

/// Explore's menu row, by purpose: Edit, View, Image, Shapes, Process, Analyze, Classify,
/// Automate.
pub fn explore_menus() -> Vec<MenuSpec> {
    let c = |id: &str| MenuItem::command(id.to_string());
    let tools = |ts: &[Tool]| ts.iter().map(|t| c(&tool_id(*t))).collect::<Vec<_>>();
    let mut shapes = vec![MenuItem::label("Select")];
    shapes.extend(tools(&[Tool::Select, Tool::Marquee]));
    shapes.push(MenuItem::submenu_with_icon("Combine the selected shapes", IconName::SquaresUnite, ShapeOp::ALL.iter().map(|o| c(&format!("shapes.{}", o.title().to_lowercase()))).collect()));
    shapes.push(MenuItem::Separator);
    shapes.push(MenuItem::label("Draw"));
    shapes.extend(tools(&[Tool::Rect, Tool::Ellipse, Tool::Polygon, Tool::Lasso, Tool::Point, Tool::Brush, Tool::Eraser, Tool::Wand, Tool::Threshold, Tool::Segment]));
    shapes.push(MenuItem::Separator);
    shapes.push(MenuItem::submenu_with_icon("Combine the selected masks", IconName::Combine, ["union", "intersect", "subtract", "invert", "grow", "shrink"].iter().map(|m| c(&format!("mask.{m}"))).collect()));
    shapes.extend([MenuItem::Separator, MenuItem::label("Measure")]);
    shapes.extend(tools(&[Tool::Ruler, Tool::Probe]));
    shapes.extend([MenuItem::Separator, c("edit.delete_selection")]);
    vec![
        MenuSpec::new("Edit", vec![c("edit.undo"), c("edit.redo"), MenuItem::Separator, c("edit.delete_selection")]),
        MenuSpec::new(
            "View",
            vec![
                MenuItem::label("Show over the image"),
                c("view.shapes"),
                c("view.fill"),
                c("view.names"),
                c("view.results"),
                c("view.crosshair"),
                c("view.rulers"),
                MenuItem::Separator,
                MenuItem::submenu_with_icon("This view shows", IconName::LayoutGrid, vec![MenuItem::provided("view.kinds")]),
                MenuItem::submenu_with_icon("Views", IconName::SquarePlus, vec![MenuItem::provided("view.add_kinds")]),
                c("view.link"),
                c("view.pipeline_dock"),
                MenuItem::Separator,
                c("view.left"),
                c("view.right"),
                c("view.reset_layout"),
                MenuItem::Separator,
                c("shell.theme"),
            ],
        ),
        MenuSpec::new(
            "Image",
            vec![MenuItem::label("Channels shown"), MenuItem::provided("image.channels"), MenuItem::Separator, MenuItem::label("Images of the project"), MenuItem::provided("image.images"), MenuItem::Separator, c("file.open_image"), c("image.remove")],
        ),
        MenuSpec::new("Shapes", shapes),
        MenuSpec::new("Process", vec![c("process.find"), MenuItem::Separator, MenuItem::label("A step on the shown channel"), MenuItem::provided("process.families")]),
        MenuSpec::new(
            "Analyze",
            vec![
                MenuItem::label("Find (in the selected regions)"),
                c("hist.find_tissue"),
                c("hist.find_cells"),
                MenuItem::submenu_with_icon("Find cells with", IconName::Microscope, vec![MenuItem::provided("hist.detectors")]),
                c("hist.superpixels"),
                MenuItem::Separator,
                MenuItem::label("Density"),
                c("hist.density"),
                c("hist.hotspots"),
                MenuItem::Separator,
                MenuItem::label("Measure"),
                c("analyze.table"),
                c("file.export_cells"),
            ],
        ),
        MenuSpec::new(
            "Classify",
            vec![MenuItem::label("Cells and tiles"), c("hist.train"), c("hist.intensity"), MenuItem::provided("hist.classifiers"), MenuItem::Separator, MenuItem::label("Pixels"), c("hist.train_pixels"), c("hist.class_regions")],
        ),
        MenuSpec::new(
            "Automate",
            vec![
                MenuItem::label("Pipeline"),
                c("shell.workspace.pipeline"),
                c("view.pipeline_dock"),
                MenuItem::Separator,
                MenuItem::label("Scripts (Python, Rust)"),
                c("script.new_python"),
                c("script.new_rust"),
                c("analyze.code"),
            ],
        ),
    ]
}

/// The workspace ids the frame uses (`shell.workspace.<id>`).
pub fn workspace_id(w: Workspace) -> SharedString {
    w.title().to_lowercase().into()
}

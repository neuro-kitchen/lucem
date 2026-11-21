//! Simplified Zed-style **Code** workspace for authoring Python & Rust Lucem script ops.
//!
//! Layout inspired by Zed:
//! - Top file tab bar (`+ New .py`, `+ New .rs`, **Save**, **Add to Pipeline**, **Output** toggle)
//! - Full-size `kitchen_ui::widgets::CodeView` syntax-highlighted editor in the center
//! - Collapsible bottom Output & Header Inspector drawer (shows parsed `# lucem:` ports & logs)
//! - Right-side file tree listing `~/.config/lucem/plugins/ops` and `~/.config/lucem/scripts`
//!
//! Works identically on desktop (local filesystem) and `lucem serve --web` (via `/api/scripts/*`).

use std::path::PathBuf;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::{
    div, px, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};
use kitchen_shell::MenuSpec;
use kitchen_ui::widgets::{CodeChanged, CodeLanguage, CodeView};

use crate::data::script;
use crate::data::store::Store;
use crate::ui::theme::{self, color};

const SCRATCHPAD_PY: &str = r#"# /// script
# dependencies = ["numpy", "scipy"]
# ///
import numpy as np
from scipy.ndimage import gaussian_filter
import lucem


@lucem.op(family="Filter", halo=12)
def local_contrast_boost(
    image: np.ndarray,
    sigma: float = 3.0,
    strength: float = 1.8,
) -> np.ndarray:
    """Enhances fine cellular structures by subtracting a blurred local background."""
    img = image.astype(np.float32, copy=False)
    bg = gaussian_filter(img, sigma=max(float(sigma), 0.2))
    enhanced = img + float(strength) * (img - bg)
    return np.clip(enhanced, float(img.min()), float(img.max())).astype(np.float32)
"#;

#[derive(Clone, Debug)]
struct FileEntry {
    name: String,
    path: String,
    is_plugin: bool,
    lang: &'static str,
}

pub struct CodeWorkspaceView {
    store: Entity<Store>,
    files: Vec<FileEntry>,
    active_path: Option<String>,
    active_name: String,
    editor: Entity<CodeView>,
    saved_text: String,
    dirty: bool,
    console_open: bool,
    sidebar_open: bool,
    logs: Vec<String>,
    _store_sub: Subscription,
    _subs: Vec<Subscription>,
}

impl CodeWorkspaceView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let editor = cx.new(|cx| CodeView::new(CodeLanguage::from_name("py"), SCRATCHPAD_PY.to_string(), window, cx).editable(true));
        crate::screens::code_lsp::attach(&editor, cx);
        let sub = cx.subscribe(&editor, |this, _, e: &CodeChanged, cx| {
            this.dirty = e.0 != this.saved_text;
            cx.notify();
        });
        let store_sub = cx.subscribe_in(&store, window, |this, _, e: &crate::data::store::AppEvent, window, cx| {
            if let crate::data::store::AppEvent::OpenScriptInCode(path) = e {
                let name = std::path::Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or("script").to_string();
                this.open_file(path.clone(), name, window, cx);
            }
        });
        let mut this = Self {
            store,
            files: Vec::new(),
            active_path: None,
            active_name: "scratchpad.py".into(),
            editor,
            saved_text: SCRATCHPAD_PY.to_string(),
            dirty: false,
            console_open: true,
            sidebar_open: true,
            logs: vec!["Ready. Select a script from the right file tree or create a new .py / .rs op.".into()],
            _store_sub: store_sub,
            _subs: vec![sub],
        };
        this.refresh_files(window, cx);
        this
    }

    pub fn menu_specs() -> Vec<MenuSpec> {
        vec![]
    }

    fn log(&mut self, msg: impl Into<String>) {
        self.logs.push(msg.into());
        if self.logs.len() > 40 {
            self.logs.remove(0);
        }
    }

    fn refresh_files(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        #[cfg(not(target_family = "wasm"))]
        {
            let mut entries = Vec::new();
            for op in script::discover_plugin_ops() {
                let name = op.path.file_name().and_then(|s| s.to_str()).unwrap_or("op").to_string();
                let lang = if op.path.extension().and_then(|s| s.to_str()) == Some("rs") { "rs" } else { "py" };
                entries.push(FileEntry { name, path: op.path.to_string_lossy().into_owned(), is_plugin: true, lang });
            }
            let sdir = script::scripts_dir(self.store.read(cx).folder().as_deref());
            if let Ok(rd) = std::fs::read_dir(&sdir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if let Some(lang_enum) = script::Language::of(&p) {
                        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("script").to_string();
                        entries.push(FileEntry { name, path: p.to_string_lossy().into_owned(), is_plugin: false, lang: lang_enum.extension() });
                    }
                }
            }
            entries.sort_by(|a, b| a.is_plugin.cmp(&b.is_plugin).reverse().then(a.name.cmp(&b.name)));
            let first = self.active_path.is_none().then(|| entries.first().cloned()).flatten();
            self.files = entries;
            if let Some(f) = first {
                self.open_file(f.path, f.name, window, cx);
            }
            cx.notify();
        }
        #[cfg(target_family = "wasm")]
        {
            if crate::data::remote::runs_alone() {
                return;
            }
            cx.spawn_in(window, async move |this, cx| {
                if let Ok(info) = crate::data::server_api::fetch_env().await {
                    let mut entries: Vec<FileEntry> = info
                        .plugin_ops
                        .into_iter()
                        .map(|op| {
                            let name = std::path::Path::new(&op.path)
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or(&op.name)
                                .to_string();
                            let lang: &'static str = if op.lang == "rs" { "rs" } else { "py" };
                            FileEntry { name, path: op.path, is_plugin: true, lang }
                        })
                        .collect();
                    entries.sort_by(|a, b| a.name.cmp(&b.name));
                    let _ = this.update_in(cx, |v, window, cx| {
                        let first = v.active_path.is_none().then(|| entries.first().cloned()).flatten();
                        v.files = entries;
                        if let Some(f) = first {
                            v.open_file(f.path, f.name, window, cx);
                        }
                        cx.notify();
                    });
                }
            })
            .detach();
        }
    }

    fn open_file(&mut self, path: String, name: String, window: &mut Window, cx: &mut Context<Self>) {
        self.active_path = Some(path.clone());
        self.active_name = name;
        let ext = std::path::Path::new(&path).extension().and_then(|s| s.to_str()).unwrap_or("py");
        let lang = CodeLanguage::from_name(ext);
        #[cfg(not(target_family = "wasm"))]
        {
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            self.saved_text = text.clone();
            self.dirty = false;
            self.editor = cx.new(|cx| CodeView::new(lang, text, window, cx).editable(true));
            crate::screens::code_lsp::attach(&self.editor, cx);
            let sub = cx.subscribe(&self.editor, |this, _, e: &CodeChanged, cx| {
                this.dirty = e.0 != this.saved_text;
                cx.notify();
            });
            self._subs = vec![sub];
            self.log(format!("Opened {path}"));
            cx.notify();
        }
        #[cfg(target_family = "wasm")]
        {
            self.editor = cx.new(|cx| CodeView::new(lang, "# Loading from server…\n".to_string(), window, cx).editable(true));
            crate::screens::code_lsp::attach(&self.editor, cx);
            let sub = cx.subscribe(&self.editor, |this, _, e: &CodeChanged, cx| {
                this.dirty = e.0 != this.saved_text;
                cx.notify();
            });
            self._subs = vec![sub];
            let p = path.clone();
            cx.spawn_in(window, async move |this, cx| {
                match crate::data::server_api::read_server_script(&p).await {
                    Ok(content) => {
                        let _ = this.update_in(cx, |v, window, cx| {
                            if v.active_path.as_deref() == Some(p.as_str()) {
                                v.saved_text = content.clone();
                                v.dirty = false;
                                v.editor.update(cx, |ed, cx| ed.set_text(content, window, cx));
                                v.log(format!("Loaded {p} from server"));
                                cx.notify();
                            }
                        });
                    }
                    Err(e) => {
                        let _ = this.update(cx, |v, cx| {
                            v.log(format!("Error reading {p}: {e}"));
                            cx.notify();
                        });
                    }
                }
            })
            .detach();
            cx.notify();
        }
    }

    fn create_file(&mut self, lang: script::Language, window: &mut Window, cx: &mut Context<Self>) {
        #[cfg(not(target_family = "wasm"))]
        {
            let dir = script::scripts_dir(self.store.read(cx).folder().as_deref());
            match script::new_script(&dir, lang) {
                Ok(path) => {
                    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("script").to_string();
                    let p_str = path.to_string_lossy().into_owned();
                    self.refresh_files(window, cx);
                    self.open_file(p_str, name, window, cx);
                }
                Err(e) => {
                    self.log(format!("Could not create script: {e}"));
                    cx.notify();
                }
            }
        }
        #[cfg(target_family = "wasm")]
        {
            if crate::data::remote::runs_alone() {
                self.log("Standalone viewer has no server. Connect to lucem serve --web or use the desktop app to create files.");
                cx.notify();
                return;
            }
            let ext = lang.extension();
            cx.spawn_in(window, async move |this, cx| {
                match crate::data::server_api::new_server_script(ext).await {
                    Ok((path, content)) => {
                        let name = std::path::Path::new(&path).file_name().and_then(|s| s.to_str()).unwrap_or("script").to_string();
                        let _ = this.update_in(cx, |v, window, cx| {
                            v.files.push(FileEntry { name: name.clone(), path: path.clone(), is_plugin: false, lang: ext });
                            v.active_path = Some(path.clone());
                            v.active_name = name;
                            v.saved_text = content.clone();
                            v.dirty = false;
                            let lang_code = CodeLanguage::from_name(ext);
                            v.editor = cx.new(|cx| CodeView::new(lang_code, content, window, cx).editable(true));
                            crate::screens::code_lsp::attach(&v.editor, cx);
                            let sub = cx.subscribe(&v.editor, |this, _, e: &CodeChanged, cx| {
                                this.dirty = e.0 != this.saved_text;
                                cx.notify();
                            });
                            v._subs = vec![sub];
                            v.log(format!("Created {path} on server"));
                            cx.notify();
                        });
                    }
                    Err(e) => {
                        let _ = this.update(cx, |v, cx| {
                            v.log(format!("Could not create server script: {e}"));
                            cx.notify();
                        });
                    }
                }
            })
            .detach();
        }
    }

    fn save_active(&mut self, cx: &mut Context<Self>) {
        let Some(path) = self.active_path.clone() else {
            self.log("Scratchpad is not backed by a file — click + New .py or + New .rs to save on disk.");
            cx.notify();
            return;
        };
        let text = self.editor.read(cx).text(cx);
        #[cfg(not(target_family = "wasm"))]
        {
            match std::fs::write(&path, &text) {
                Ok(()) => {
                    self.saved_text = text;
                    self.dirty = false;
                    self.log(format!("Saved {path}"));
                    self.store.update(cx, |s, cx| s.set_status(format!("Saved {}", self.active_name), cx));
                }
                Err(e) => self.log(format!("Save failed: {e}")),
            }
            cx.notify();
        }
        #[cfg(target_family = "wasm")]
        {
            let name = self.active_name.clone();
            let store = self.store.clone();
            let ed = self.editor.clone();
            let text_check = text.clone();
            let is_py = !path.ends_with(".rs");
            cx.spawn(async move |this, cx| {
                match crate::data::server_api::write_server_script(&path, &text).await {
                    Ok(()) => {
                        let _ = this.update(cx, |v, cx| {
                            v.saved_text = text;
                            v.dirty = false;
                            v.log(format!("Saved {path} on server"));
                            cx.notify();
                        });
                        store.update(cx, |s, cx| s.set_status(format!("Saved {name} on server"), cx));
                        if is_py
                            && let Ok(diags) = crate::data::server_api::check_server_script("py", &text_check).await
                        {
                            let n = diags.len();
                            let _ = ed.update(cx, |e, cx| e.set_errors(diags, cx));
                            let _ = this.update(cx, |v, cx| {
                                if n == 0 {
                                    v.log("Syntax check: OK (0 errors)");
                                } else {
                                    v.log(format!("Syntax check: {n} error(s) marked in editor"));
                                }
                                cx.notify();
                            });
                        }
                    }
                    Err(e) => {
                        let _ = this.update(cx, |v, cx| {
                            v.log(format!("Save failed: {e}"));
                            cx.notify();
                        });
                    }
                }
            })
            .detach();
        }
    }

    fn add_to_pipeline(&mut self, cx: &mut Context<Self>) {
        let Some(path) = self.active_path.clone() else {
            self.log("Save as a .py or .rs file first to add it as a Pipeline node.");
            cx.notify();
            return;
        };
        if self.dirty {
            self.save_active(cx);
        }
        self.store.update(cx, |s, cx| s.add_script_file(PathBuf::from(path), cx));
    }
}

impl Render for CodeWorkspaceView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let text = self.editor.read(cx).text(cx);
        let header_status = match script::read_header(&text) {
            Ok(h) => {
                let inputs: Vec<String> = h.inputs.iter().map(|i| format!("{}: {}", i.name, i.port.name())).collect();
                let params: Vec<String> = h.params.iter().map(|p| format!("{}={}", p.name, p.value)).collect();
                Ok(format!(
                    "Header OK · {} · inputs [{}] · params [{}] · output {} · halo {} px",
                    h.name.as_deref().unwrap_or(&self.active_name),
                    inputs.join(", "),
                    params.join(", "),
                    h.output.name(),
                    h.halo
                ))
            }
            Err(e) => Err(format!("Header issue: {e}")),
        };
        let active_path = self.active_path.clone();
        let tab_label = if self.dirty { format!("{} •", self.active_name) } else { self.active_name.clone() };

        // Top Zed-style tab bar
        let tab_bar = h_flex()
            .flex_none()
            .h(px(34.))
            .px(px(theme::SM))
            .gap(px(theme::XS))
            .items_center()
            .border_b_1()
            .border_color(color::border())
            .bg(color::panel())
            .child(
                h_flex()
                    .px(px(theme::SM))
                    .h(px(24.))
                    .gap(px(6.))
                    .items_center()
                    .rounded(px(6.))
                    .bg(color::control())
                    .child(Icon::new(IconName::FileCode).xsmall().text_color(color::accent_mark()))
                    .child(div().text_xs().font_weight(FontWeight::MEDIUM).text_color(color::text()).child(tab_label)),
            )
            .child(div().flex_1())
            .child(
                Button::new("code-new-py")
                    .ghost()
                    .xsmall()
                    .icon(Icon::new(IconName::FilePlus))
                    .label("New .py")
                    .on_click(cx.listener(|this, _, w, cx| this.create_file(script::Language::Python, w, cx))),
            )
            .child(
                Button::new("code-new-rs")
                    .ghost()
                    .xsmall()
                    .icon(Icon::new(IconName::FilePlus))
                    .label("New .rs")
                    .on_click(cx.listener(|this, _, w, cx| this.create_file(script::Language::Rust, w, cx))),
            )
            .child({
                let b = Button::new("code-save")
                    .xsmall()
                    .icon(Icon::new(IconName::Save))
                    .label("Save")
                    .disabled(!self.dirty || active_path.is_none())
                    .on_click(cx.listener(|this, _, _, cx| this.save_active(cx)));
                if self.dirty { b.primary() } else { b.outline() }
            })
            .child(
                Button::new("code-add-pipe")
                    .outline()
                    .xsmall()
                    .icon(Icon::new(IconName::Workflow))
                    .label("Add to Pipeline")
                    .disabled(active_path.is_none())
                    .on_click(cx.listener(|this, _, _, cx| this.add_to_pipeline(cx))),
            )
            .child(
                Button::new("code-toggle-out")
                    .ghost()
                    .xsmall()
                    .selected(self.console_open)
                    .label("Output")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.console_open = !this.console_open;
                        cx.notify();
                    })),
            )
            .child(
                Button::new("code-toggle-tree")
                    .ghost()
                    .xsmall()
                    .selected(self.sidebar_open)
                    .icon(Icon::new(IconName::PanelRight))
                    .tooltip("Toggle scripts file tree")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.sidebar_open = !this.sidebar_open;
                        cx.notify();
                    })),
            );

        // Collapsible bottom Output & Header Inspector drawer (Zed-style)
        let console = self.console_open.then(|| {
            v_flex()
                .flex_none()
                .h(px(148.))
                .w_full()
                .border_t_1()
                .border_color(color::border())
                .bg(color::panel())
                .child(
                    h_flex()
                        .flex_none()
                        .h(px(28.))
                        .px(px(theme::MD))
                        .gap(px(theme::SM))
                        .items_center()
                        .border_b_1()
                        .border_color(color::border())
                        .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child("OUTPUT"))
                        .child(match &header_status {
                            Ok(s) => div().flex_1().min_w_0().truncate().text_xs().text_color(color::accent_mark()).child(s.clone()),
                            Err(e) => div().flex_1().min_w_0().truncate().text_xs().text_color(color::warning()).child(e.clone()),
                        })
                        .child(
                            Button::new("code-close-out")
                                .ghost()
                                .xsmall()
                                .icon(Icon::new(IconName::Close))
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.console_open = false;
                                    cx.notify();
                                })),
                        ),
                )
                .child(
                    v_flex()
                        .id("code-output-scroll")
                        .flex_1()
                        .min_h_0()
                        .overflow_y_scroll()
                        .px(px(theme::MD))
                        .py(px(theme::XS))
                        .gap(px(2.))
                        .children(self.logs.iter().rev().map(|line| {
                            div().text_xs().font_family("monospace").text_color(color::text_muted()).child(line.clone())
                        })),
                )
        });

        // Zed-style right file tree sidebar
        let sidebar = self.sidebar_open.then(|| {
            let plugins: Vec<FileEntry> = self.files.iter().filter(|f| f.is_plugin).cloned().collect();
            let scripts: Vec<FileEntry> = self.files.iter().filter(|f| !f.is_plugin).cloned().collect();
            let row_btn = |idx: usize, prefix: &'static str, f: FileEntry, active: Option<&str>, cx: &mut Context<Self>| {
                let selected = active == Some(f.path.as_str());
                let (p, n) = (f.path.clone(), f.name.clone());
                Button::new(SharedString::from(format!("code-file-{prefix}-{idx}")))
                    .ghost()
                    .small()
                    .selected(selected)
                    .icon(Icon::new(if f.lang == "rs" { IconName::Code } else { IconName::FileCode }))
                    .label(f.name)
                    .on_click(cx.listener(move |this, _, w, cx| this.open_file(p.clone(), n.clone(), w, cx)))
            };
            let mut tree = v_flex()
                .flex_none()
                .w(px(244.))
                .h_full()
                .rounded(px(theme::RADIUS_LG))
                .overflow_hidden()
                .bg(color::panel())
                .child(
                    h_flex()
                        .flex_none()
                        .h(px(34.))
                        .px(px(theme::MD))
                        .items_center()
                        .border_b_1()
                        .border_color(color::border())
                        .child(div().flex_1().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child("SCRIPTS"))
                        .child(
                            Button::new("code-tree-refresh")
                                .ghost()
                                .xsmall()
                                .icon(Icon::new(IconName::RotateCcw))
                                .tooltip("Refresh scripts")
                                .on_click(cx.listener(|this, _, w, cx| this.refresh_files(w, cx))),
                        ),
                );
            let mut list = v_flex().id("code-tree-scroll").flex_1().min_h_0().overflow_y_scroll().p(px(theme::SM)).gap(px(2.));
            list = list.child(div().px(px(theme::XS)).pt(px(theme::XS)).text_size(px(10.)).font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child("PLUGIN OPS"));
            if plugins.is_empty() {
                list = list.child(div().px(px(theme::XS)).py(px(2.)).text_xs().text_color(color::text_muted()).child("None in plugins/ops"));
            } else {
                for (i, f) in plugins.into_iter().enumerate() {
                    list = list.child(row_btn(i, "pl", f, active_path.as_deref(), cx));
                }
            }
            list = list.child(div().px(px(theme::XS)).pt(px(theme::MD)).text_size(px(10.)).font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child("PROJECT SCRIPTS"));
            if scripts.is_empty() {
                list = list.child(div().px(px(theme::XS)).py(px(2.)).text_xs().text_color(color::text_muted()).child("Click + New .py or .rs"));
            } else {
                for (i, f) in scripts.into_iter().enumerate() {
                    list = list.child(row_btn(i, "sc", f, active_path.as_deref(), cx));
                }
            }
            tree = tree.child(list);
            tree
        });

        #[cfg(target_family = "wasm")]
        let standalone_banner = crate::data::remote::runs_alone().then(|| {
            h_flex()
                .flex_none()
                .w_full()
                .px(px(theme::MD))
                .py(px(theme::SM))
                .gap(px(theme::SM))
                .items_center()
                .bg(color::panel())
                .border_b_1()
                .border_color(color::border())
                .child(Icon::new(IconName::Info).small().text_color(color::accent_mark()))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .text_xs()
                        .text_color(color::text_muted())
                        .child("Browser-only viewer: live Python/Rust script execution and local file saving require the desktop app (`lucem app`) or a connected GPU server (`lucem serve --web`)."),
                )
        });
        #[cfg(not(target_family = "wasm"))]
        let standalone_banner: Option<gpui_kit::Div> = None;

        h_flex()
            .id("code-workspace")
            .size_full()
            .p(px(theme::PANE_GAP))
            .gap(px(theme::PANE_GAP))
            .bg(color::chrome())
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .rounded(px(theme::RADIUS_LG))
                    .overflow_hidden()
                    .bg(color::canvas())
                    .children(standalone_banner)
                    .child(tab_bar)
                    .child(div().flex_1().min_h_0().w_full().child(self.editor.clone()))
                    .children(console),
            )
            .children(sidebar)
    }
}

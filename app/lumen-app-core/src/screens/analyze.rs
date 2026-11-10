//! The Analyze workspace: measurements as tables — the annotations of the shown image and the
//! objects of every Measure result — that people can work with:
//! - each annotation row starts with a preview (the image around it, its outline drawn); a click
//!   shows it in Explore;
//! - spreadsheet filters: sort by a column (click its name), a filter under each name (`> 10`,
//!   `10..50`, `= Tumor`, or text it contains), a search over every column, "n of m";
//! - plots of the rows shown (a histogram of a column, or two columns against each other);
//! - code on the table (VS Code's Data Wrangler): a Python or Rust cell run on the rows shown,
//!   whose result becomes the table shown; every run is a step, listed, re-run and exported.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Editor, EditorState, Input, InputEvent, InputState};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    canvas, div, fill, img, point, px, size, AppContext as _, Bounds, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, KeyDownEvent, ParentElement as _, Render, RenderImage, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Task, Window,
};
use lumen_io::Pyramid;
use lumen_view::{bgra_image, composite_bgra, ChannelDisplay};
use polars::prelude::{AnyValue, Column, DataFrame, NamedFrom as _, Series};

use crate::data::results::centroid_l0;
use crate::data::script::{self, Language};
use crate::data::store::{AppEvent, Store};
use crate::domain::annotation::{rasterize, Annotation};
use crate::domain::graph::NodeKind;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{group_digits, EmptyState, MenuSelect, Section};

/// Rows drawn at most (filters, plots, code and export use them all).
const SHOWN: usize = 1000;
/// Rows that get a preview (the first ones shown).
const PREVIEWED: usize = 120;
/// Preview side (px) and column width.
const PREVIEW: f32 = 44.;
const COL_W: f32 = 124.;

/// The table in a form the filters, the sort and the plots read quickly.
struct Model {
    names: Vec<String>,
    /// Per column: each row's text and, for numeric columns, its number.
    text: Vec<Vec<String>>,
    num: Vec<Vec<Option<f64>>>,
    numeric: Vec<bool>,
    /// The annotation of each row (the annotations table, or any table with an `id` column).
    ids: Option<Vec<String>>,
}

fn text_of(v: &AnyValue) -> String {
    match v {
        AnyValue::Null => String::new(),
        AnyValue::String(s) => s.to_string(),
        AnyValue::StringOwned(s) => s.to_string(),
        other => other.to_string().trim_matches('"').to_string(),
    }
}

impl Model {
    fn of(df: &DataFrame) -> Model {
        let cols = df.columns();
        let n = df.height();
        let mut m = Model { names: Vec::new(), text: Vec::new(), num: Vec::new(), numeric: Vec::new(), ids: None };
        for c in cols {
            let values: Vec<AnyValue> = (0..n).map(|i| c.get(i).unwrap_or(AnyValue::Null)).collect();
            let num: Vec<Option<f64>> = values.iter().map(|v| if v.is_null() { None } else { v.extract::<f64>() }).collect();
            let numeric = values.iter().zip(&num).all(|(v, x)| v.is_null() || x.is_some());
            let text: Vec<String> = values.iter().zip(&num).map(|(v, x)| match (numeric, x) {
                (true, Some(x)) => short_num(*x),
                _ => text_of(v),
            }).collect();
            if c.name().as_str() == "id" {
                m.ids = Some(text.clone());
            }
            m.names.push(c.name().to_string());
            m.text.push(text);
            m.num.push(num);
            m.numeric.push(numeric);
        }
        m
    }

    fn rows(&self) -> usize {
        self.text.first().map_or(0, Vec::len)
    }

    /// Whether row `r` passes column `c`'s filter (`> 10`, `>= 1`, `< 5`, `<= 5`, `10..50`,
    /// `= Tumor`, `!= Tumor`, or text it contains).
    fn passes(&self, c: usize, r: usize, f: &str) -> bool {
        let f = f.trim();
        if f.is_empty() {
            return true;
        }
        let text = &self.text[c][r];
        let num = self.num[c][r];
        let number = |s: &str| s.trim().parse::<f64>().ok();
        if let Some((a, b)) = f.split_once("..") {
            return match (num, number(a), number(b)) {
                (Some(x), lo, hi) => lo.is_none_or(|lo| x >= lo) && hi.is_none_or(|hi| x <= hi),
                _ => false,
            };
        }
        for (op, test) in [(">=", 0), ("<=", 1), ("!=", 2), (">", 3), ("<", 4), ("=", 5)] {
            if let Some(rest) = f.strip_prefix(op) {
                let rest = rest.trim();
                return match (test, num, number(rest)) {
                    (0, Some(x), Some(v)) => x >= v,
                    (1, Some(x), Some(v)) => x <= v,
                    (3, Some(x), Some(v)) => x > v,
                    (4, Some(x), Some(v)) => x < v,
                    (2, _, _) => !text.eq_ignore_ascii_case(rest),
                    (5, _, _) => text.eq_ignore_ascii_case(rest),
                    _ => false,
                };
            }
        }
        text.to_lowercase().contains(&f.to_lowercase())
    }

    /// The rows that pass every filter and the search, in the sort's order.
    fn order(&self, filters: &[(usize, String)], search: &str, sort: Option<(usize, bool)>) -> Vec<usize> {
        let search = search.trim().to_lowercase();
        let mut rows: Vec<usize> = (0..self.rows())
            .filter(|&r| filters.iter().all(|(c, f)| self.passes(*c, r, f)))
            .filter(|&r| search.is_empty() || self.text.iter().any(|col| col[r].to_lowercase().contains(&search)))
            .collect();
        if let Some((c, desc)) = sort {
            if self.numeric[c] {
                rows.sort_by(|&a, &b| self.num[c][a].unwrap_or(f64::NEG_INFINITY).total_cmp(&self.num[c][b].unwrap_or(f64::NEG_INFINITY)));
            } else {
                rows.sort_by(|&a, &b| self.text[c][a].cmp(&self.text[c][b]));
            }
            if desc {
                rows.reverse();
            }
        }
        rows
    }

    /// The rows in `order` as CSV (with a header).
    fn csv(&self, order: &[usize]) -> String {
        let mut out = self.names.join(",");
        out.push('\n');
        for &r in order {
            let row: Vec<String> = (0..self.names.len()).map(|c| if self.numeric[c] { self.num[c][r].map_or(String::new(), |x| x.to_string()) } else { cell(&self.text[c][r]) }).collect();
            out.push_str(&row.join(","));
            out.push('\n');
        }
        out
    }
}

/// A number shortened for the table.
fn short_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e12 {
        format!("{}", v as i64)
    } else if v.abs() >= 1000.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.3}")
    }
}

/// A CSV cell (quoted when it holds a comma or quote).
fn cell(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

/// Splits a CSV line (quotes, doubled quotes).
fn split_csv(line: &str) -> Vec<String> {
    let (mut out, mut cur, mut quoted, mut chars) = (Vec::new(), String::new(), false, line.chars().peekable());
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            c => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// A CSV text (a header row first) as a frame: a column is numeric when every non-empty cell is
/// a number.
fn parse_csv(text: &str) -> Result<DataFrame, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let names = split_csv(lines.next().ok_or("the result table is empty")?);
    let rows: Vec<Vec<String>> = lines.map(split_csv).collect();
    let n = rows.len();
    let cols: Vec<Column> = names
        .iter()
        .enumerate()
        .map(|(c, name)| {
            let v: Vec<&str> = rows.iter().map(|r| r.get(c).map_or("", |s| s.as_str())).collect();
            let blank = |s: &str| s.is_empty() || s == "null" || s == "NaN" || s == "nan";
            if v.iter().all(|s| blank(s) || s.parse::<f64>().is_ok()) {
                Column::from(Series::new(name.as_str().into(), v.iter().map(|s| if blank(s) { None } else { s.parse::<f64>().ok() }).collect::<Vec<Option<f64>>>()))
            } else {
                Column::from(Series::new(name.as_str().into(), v.iter().map(|s| (!blank(s)).then(|| s.to_string())).collect::<Vec<Option<String>>>()))
            }
        })
        .collect();
    DataFrame::new(n, cols).map_err(|e| e.to_string())
}

/// A run of code on the table.
#[derive(Clone)]
struct Step {
    lang: Language,
    code: String,
    rows: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Plot {
    Histogram,
    Scatter,
}

pub struct AnalyzeView {
    store: Entity<Store>,
    /// `None`: annotations; `Some(view)`: that view's objects.
    tab: Option<String>,
    /// The table and what it was made from.
    model: Option<(u64, Arc<Model>)>,
    sort: Option<(String, bool)>,
    filters: HashMap<String, Entity<InputState>>,
    search: Entity<InputState>,
    /// Previews by annotation id and version; what is being made.
    previews: HashMap<String, Arc<RenderImage>>,
    pending: HashSet<String>,
    preview_task: Option<Task<()>>,
    plot: Plot,
    plot_x: Option<String>,
    plot_y: Option<String>,
    // Code on the table
    code_open: bool,
    lang: Language,
    editor: Entity<EditorState>,
    steps: Vec<Step>,
    /// The last result, shown instead of the source while `show_result`.
    result: Option<(u64, DataFrame)>,
    show_result: bool,
    running: bool,
    code_error: Option<String>,
    _subs: Vec<Subscription>,
}

impl AnalyzeView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let search = cx.new(|cx| InputState::new(window, cx).placeholder("Search every column…"));
        let editor = cx.new(|cx| EditorState::new(window, cx).language("python").default_value(script::table_template(Language::Python)));
        let subs = vec![
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::Document(_) | AppEvent::Results) {
                    cx.notify();
                }
            }),
            cx.subscribe(&search, |_, _, _: &InputEvent, cx| cx.notify()),
        ];
        Self {
            store,
            tab: None,
            model: None,
            sort: None,
            filters: HashMap::new(),
            search,
            previews: HashMap::new(),
            pending: HashSet::new(),
            preview_task: None,
            plot: Plot::Histogram,
            plot_x: None,
            plot_y: None,
            code_open: false,
            lang: Language::Python,
            editor,
            steps: Vec::new(),
            result: None,
            show_result: false,
            running: false,
            code_error: None,
            _subs: subs,
        }
    }

    /// The annotations of the shown image as a frame.
    fn annotations_frame(&self, cx: &Context<Self>) -> Option<DataFrame> {
        let s = self.store.read(cx);
        let img = s.doc.active_image.clone()?;
        let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
        let (k, k2) = (um.unwrap_or(1.0), um.unwrap_or(1.0).powi(2));
        let rows: Vec<_> = s.doc.annotations_of(&img).collect();
        let (mut id, mut class, mut kind, mut area, mut length, mut cx_, mut cy_, mut z) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        for a in &rows {
            id.push(a.id.clone());
            class.push(a.class.clone());
            kind.push(a.shape.kind().to_string());
            area.push(a.shape.area().map(|v| v * k2));
            length.push(a.shape.length().map(|v| v * k));
            let (x0, y0, x1, y1) = a.shape.bounds();
            cx_.push((x0 + x1) / 2.0 * k);
            cy_.push((y0 + y1) / 2.0 * k);
            z.push(a.z.map(|v| v as f64));
        }
        let u = if um.is_some() { "um" } else { "px" };
        let mut cols = vec![
            Column::from(Series::new("id".into(), id)),
            Column::from(Series::new("class".into(), class)),
            Column::from(Series::new("shape".into(), kind)),
            Column::from(Series::new(format!("area_{u}2").into(), area)),
            Column::from(Series::new(format!("length_{u}").into(), length)),
            Column::from(Series::new(format!("x_{u}").into(), cx_)),
            Column::from(Series::new(format!("y_{u}").into(), cy_)),
        ];
        if z.iter().any(Option::is_some) {
            cols.push(Column::from(Series::new("z".into(), z)));
        }
        DataFrame::new(rows.len(), cols).ok()
    }

    /// A Measure result's objects on the whole image, positions in level-0 units (µm when known).
    fn objects_frame(&self, view: &str, cx: &Context<Self>) -> Option<DataFrame> {
        let s = self.store.read(cx);
        let a = s.analysis.as_ref()?;
        let t = a.tables.get(view)?;
        let um = s.pyramids.get(&a.image).and_then(|p| p.pixel_size_um);
        let u1 = um.unwrap_or(1.0);
        let k = u1 * t.downsample;
        let n = t.rows.len();
        let (mut label, mut area, mut x, mut y, mut diam, mut ecc, mut mean, mut max) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        for f in t.rows.iter() {
            label.push(f.label);
            area.push(f.area as f64 * k * k);
            let (px_, py_) = centroid_l0(f, t.downsample).unwrap_or((0.0, 0.0));
            x.push(px_ * u1);
            y.push(py_ * u1);
            diam.push(f.equivalent_diameter * k);
            ecc.push(f.eccentricity);
            mean.push(f.mean_intensity);
            max.push(f.max_intensity);
        }
        let u = if um.is_some() { "um" } else { "px" };
        let cols = vec![
            Column::from(Series::new("label".into(), label)),
            Column::from(Series::new(format!("area_{u}2").into(), area)),
            Column::from(Series::new(format!("x_{u}").into(), x)),
            Column::from(Series::new(format!("y_{u}").into(), y)),
            Column::from(Series::new(format!("diameter_{u}").into(), diam)),
            Column::from(Series::new("eccentricity".into(), ecc)),
            Column::from(Series::new("mean_intensity".into(), mean)),
            Column::from(Series::new("max_intensity".into(), max)),
        ];
        DataFrame::new(n, cols).ok()
    }

    /// What the shown table was made from (it is built again when this changes).
    fn source_key(&self, cx: &Context<Self>) -> u64 {
        let s = self.store.read(cx);
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (&self.tab, s.annotations_rev, s.doc.active_image.clone(), s.analysis.as_ref().map(|a| a.key), self.show_result, self.result.as_ref().map(|r| r.0)).hash(&mut h);
        h.finish()
    }

    /// The table shown: the last code result, or the source.
    fn model(&mut self, cx: &Context<Self>) -> Option<Arc<Model>> {
        let key = self.source_key(cx);
        if let Some((k, m)) = &self.model
            && *k == key {
                return Some(m.clone());
            }
        let df = match (&self.result, self.show_result, &self.tab) {
            (Some((_, df)), true, _) => Some(df.clone()),
            (_, _, None) => self.annotations_frame(cx),
            (_, _, Some(v)) => self.objects_frame(v, cx),
        }?;
        let m = Arc::new(Model::of(&df));
        self.model = Some((key, m.clone()));
        Some(m)
    }

    /// The filter of each column that has one.
    fn filter_values(&self, m: &Model, cx: &Context<Self>) -> Vec<(usize, String)> {
        m.names.iter().enumerate().filter_map(|(c, n)| self.filters.get(n).map(|i| (c, i.read(cx).value().to_string()))).filter(|(_, f)| !f.trim().is_empty()).collect()
    }

    fn order(&self, m: &Model, cx: &Context<Self>) -> Vec<usize> {
        let sort = self.sort.as_ref().and_then(|(n, d)| m.names.iter().position(|x| x == n).map(|c| (c, *d)));
        m.order(&self.filter_values(m, cx), &self.search.read(cx).value(), sort)
    }

    /// Writes the rows shown as CSV next to the project (or in the working folder).
    fn export(&self, csv: String, name: &str, rows: usize, cx: &mut Context<Self>) {
        let folder = self.store.read(cx).folder().or_else(|| std::env::current_dir().ok()).unwrap_or_default();
        let path = folder.join(format!("{}.csv", name.replace([' ', '/'], "_")));
        let msg = match std::fs::write(&path, csv) {
            Ok(()) => format!("Exported {rows} rows to {}", path.display()),
            Err(e) => format!("Export failed: {e}"),
        };
        self.store.update(cx, |s, cx| s.set_status(msg, cx));
    }

    // ------------------------------------------------------------------------
    // Previews
    // ------------------------------------------------------------------------

    /// The preview key of an annotation (its id and shape).
    fn preview_key(a: &Annotation) -> String {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{:?}{:?}", a.shape, a.z).hash(&mut h);
        format!("{}#{:x}", a.id, h.finish())
    }

    /// Makes the previews of the first rows shown that have none (off the UI thread).
    fn request_previews(&mut self, ids: &[String], cx: &mut Context<Self>) {
        if self.preview_task.is_some() {
            return;
        }
        let s = self.store.read(cx);
        let Some(img) = s.doc.active().cloned() else { return };
        let Some(py) = s.pyramids.get(&img.id).cloned() else { return };
        let cursor_z = s.cursor().map_or(0, |c| c[0]);
        let displays: Vec<ChannelDisplay> = img.channels.iter().map(|c| ChannelDisplay { visible: c.visible, color: [c.color[0] as f32 / 255.0, c.color[1] as f32 / 255.0, c.color[2] as f32 / 255.0], min: c.min, max: c.max, gamma: c.gamma }).collect();
        let todo: Vec<(String, Annotation, [u8; 3])> = ids
            .iter()
            .filter_map(|id| s.doc.annotations.iter().find(|a| &a.id == id))
            .map(|a| (Self::preview_key(a), a.clone(), s.doc.class(&a.class).map_or([255, 255, 0], |c| c.color)))
            .filter(|(k, _, _)| !self.previews.contains_key(k) && !self.pending.contains(k))
            .take(24)
            .collect();
        if todo.is_empty() {
            return;
        }
        for (k, _, _) in &todo {
            self.pending.insert(k.clone());
        }
        let work = cx.background_spawn(async move { todo.into_iter().filter_map(|(k, a, c)| preview(&py, &displays, &a, c, cursor_z).map(|i| (k, i))).collect::<Vec<_>>() });
        self.preview_task = Some(cx.spawn(async move |this, cx| {
            let done = work.await;
            let _ = this.update(cx, |v, cx| {
                for (k, i) in done {
                    v.pending.remove(&k);
                    v.previews.insert(k, i);
                }
                v.pending.clear();
                v.preview_task = None;
                cx.notify();
            });
        }));
    }

    // ------------------------------------------------------------------------
    // Code
    // ------------------------------------------------------------------------

    /// Shows the code pane (from the Analyze and Automate menus).
    pub fn open_code(&mut self, cx: &mut Context<Self>) {
        self.code_open = true;
        cx.notify();
    }

    fn set_lang(&mut self, lang: Language, window: &mut Window, cx: &mut Context<Self>) {
        if self.lang == lang {
            return;
        }
        self.lang = lang;
        let name = if lang == Language::Python { "python" } else { "rust" };
        self.editor = cx.new(|cx| EditorState::new(window, cx).language(name).default_value(script::table_template(lang)));
        cx.notify();
    }

    /// Runs the cell on the rows shown; the result becomes the table shown and a step.
    fn run_code(&mut self, cx: &mut Context<Self>) {
        if self.running {
            return;
        }
        let Some(m) = self.model(cx) else { return };
        let order = self.order(&m, cx);
        let csv = m.csv(&order);
        let code = self.editor.read(cx).value().to_string();
        let lang = self.lang;
        self.running = true;
        self.code_error = None;
        cx.notify();
        let work = cx.background_spawn(async move { script::run_table(lang, &code, &csv).and_then(|out| parse_csv(&out)).map(|df| (df, code)) });
        cx.spawn(async move |this, cx| {
            let done = work.await;
            let _ = this.update(cx, |v, cx| {
                v.running = false;
                match done {
                    Ok((df, code)) => {
                        let rows = df.height();
                        let generation = v.steps.len() as u64 + 1;
                        v.steps.push(Step { lang, code, rows });
                        v.result = Some((generation, df));
                        v.show_result = true;
                        v.store.update(cx, |s, cx| s.set_status(format!("Step {generation}: {rows} rows"), cx));
                    }
                    Err(e) => v.code_error = Some(e),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn export_steps(&self, cx: &mut Context<Self>) {
        let Some(first) = self.steps.first() else { return };
        let lang = first.lang;
        let codes: Vec<String> = self.steps.iter().map(|s| if s.lang == lang { s.code.clone() } else { format!("{} ({} step, run separately):\n{}", if lang == Language::Python { "#" } else { "//" }, s.lang.title(), s.code) }).collect();
        let folder = self.store.read(cx).folder().or_else(|| std::env::current_dir().ok()).unwrap_or_default();
        let path = folder.join(format!("table_steps.{}", lang.extension()));
        let msg = match std::fs::write(&path, script::export_steps(lang, &codes)) {
            Ok(()) => format!("Steps written to {}", path.display()),
            Err(e) => format!("Could not write the steps: {e}"),
        };
        self.store.update(cx, |s, cx| s.set_status(msg, cx));
    }

    // ------------------------------------------------------------------------
    // Drawing
    // ------------------------------------------------------------------------

    fn table(&mut self, m: &Model, order: &[usize], with_previews: bool, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let me = cx.entity().downgrade();
        // The filter box of each column, made once
        for n in &m.names {
            if !self.filters.contains_key(n) {
                let input = cx.new(|cx| InputState::new(window, cx).placeholder("filter"));
                self._subs.push(cx.subscribe(&input, |_, _, _: &InputEvent, cx| cx.notify()));
                self.filters.insert(n.clone(), input);
            }
        }
        let pw = if with_previews { PREVIEW + 8.0 } else { 0.0 };
        let header = h_flex().flex_none().bg(color::raised()).border_b_1().border_color(color::border()).when(with_previews, |d| d.child(div().flex_none().w(px(pw)))).children(m.names.iter().map(|n| {
            let me = me.clone();
            let (name, shown) = (n.clone(), self.sort.as_ref().filter(|(s, _)| s == n).map(|(_, d)| *d));
            let arrow = match shown {
                Some(true) => " ↓",
                Some(false) => " ↑",
                None => "",
            };
            div()
                .id(SharedString::from(format!("col-{n}")))
                .flex_none()
                .w(px(COL_W))
                .px(px(theme::SM))
                .py(px(theme::XS))
                .truncate()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(if shown.is_some() { color::text() } else { color::text_muted() })
                .cursor_pointer()
                .hover(|d| d.bg(color::hover()))
                .child(format!("{n}{arrow}"))
                .on_click(move |_, _, cx| {
                    let name = name.clone();
                    let _ = me.update(cx, |a, cx| {
                        // Ascending, descending, off
                        a.sort = match &a.sort {
                            Some((s, false)) if *s == name => Some((name, true)),
                            Some((s, true)) if *s == name => None,
                            _ => Some((name, false)),
                        };
                        cx.notify();
                    });
                })
        }));
        let filter_row = h_flex().flex_none().border_b_1().border_color(color::border()).when(with_previews, |d| d.child(div().flex_none().w(px(pw)))).children(m.names.iter().map(|n| div().flex_none().w(px(COL_W)).px(px(2.)).py(px(2.)).child(Input::new(&self.filters[n]).xsmall())));
        let s = self.store.read(cx);
        let selected = s.selection.clone();
        let ann_by_id: HashMap<&str, &Annotation> = s.doc.annotations.iter().map(|a| (a.id.as_str(), a)).collect();
        let rows = order.iter().take(SHOWN).enumerate().map(|(i, &r)| {
            let id = m.ids.as_ref().map(|ids| ids[r].clone());
            let ann = id.as_deref().and_then(|i| ann_by_id.get(i).copied());
            let thumb = with_previews.then(|| {
                let p = ann.and_then(|a| self.previews.get(&Self::preview_key(a)).cloned());
                div().flex_none().w(px(pw)).p(px(4.)).child(match p {
                    Some(p) => div().size(px(PREVIEW)).rounded(px(3.)).overflow_hidden().bg(gpui_kit::black()).child(img(p).size_full()).into_any_element(),
                    None => div().size(px(PREVIEW)).rounded(px(3.)).bg(color::control()).into_any_element(),
                })
            });
            let picked = id.as_ref().is_some_and(|i| selected.contains(i));
            let store = self.store.clone();
            div()
                .id(("analyze-row", i))
                .flex()
                .flex_row()
                .flex_none()
                .items_center()
                .when(i % 2 == 1, |d| d.bg(color::raised()))
                .when(picked, |d| d.bg(color::accent().opacity(0.3)))
                .when(ann.is_some(), |d| d.cursor_pointer().hover(|d| d.bg(color::hover())))
                .children(thumb)
                .children((0..m.names.len()).map(|c| div().flex_none().w(px(COL_W)).px(px(theme::SM)).py(px(3.)).truncate().text_sm().font_family("monospace").child(m.text[c][r].clone())))
                .on_click(move |_, _, cx| {
                    if let Some(id) = &id {
                        let id = id.clone();
                        store.update(cx, |s, cx| s.reveal_annotation(&id, cx));
                    }
                })
        });
        let rows: Vec<_> = rows.collect();
        v_flex().id("analyze-table").flex_1().min_h_0().overflow_scroll().child(header).child(filter_row).children(rows)
    }

    /// A histogram of a column, or two columns against each other, over the rows shown.
    fn plot(&self, m: &Model, order: &[usize], cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let numeric: Vec<String> = m.names.iter().zip(&m.numeric).filter(|(_, n)| **n).map(|(n, _)| n.clone()).collect();
        let x = self.plot_x.clone().filter(|c| numeric.contains(c)).or_else(|| numeric.iter().find(|n| n.starts_with("area")).cloned()).or_else(|| numeric.first().cloned());
        let y = self.plot_y.clone().filter(|c| numeric.contains(c)).or_else(|| numeric.get(1).cloned());
        let col = |name: &Option<String>| name.as_ref().and_then(|n| m.names.iter().position(|x| x == n));
        let (xs, ys): (Vec<f64>, Vec<f64>) = match (col(&x), col(&y)) {
            (Some(cx_), cy) => order.iter().filter_map(|&r| Some((m.num[cx_][r]?, cy.map_or(Some(0.0), |c| m.num[c][r])?))).filter(|(a, b)| a.is_finite() && b.is_finite()).unzip(),
            _ => (Vec::new(), Vec::new()),
        };
        let me = cx.entity().downgrade();
        let pick = |id: &'static str, current: Option<String>, set: fn(&mut AnalyzeView, String)| {
            let me = me.clone();
            let names = numeric.clone();
            MenuSelect::new(id, numeric.iter().map(|n| SharedString::from(n.clone())).collect(), current.and_then(|c| numeric.iter().position(|n| *n == c)), move |i, _, cx| {
                let name = names[i].clone();
                let _ = me.update(cx, |a, cx| {
                    set(a, name);
                    cx.notify();
                });
            })
        };
        let kind = self.plot;
        let (m1, m2) = (me.clone(), me.clone());
        let switch = h_flex()
            .gap(px(2.))
            .child(Button::new("plot-hist").ghost().xsmall().label("Histogram").selected(kind == Plot::Histogram).on_click(move |_, _, cx| {
                let _ = m1.update(cx, |a, cx| {
                    a.plot = Plot::Histogram;
                    cx.notify();
                });
            }))
            .child(Button::new("plot-scatter").ghost().xsmall().label("Scatter").selected(kind == Plot::Scatter).on_click(move |_, _, cx| {
                let _ = m2.update(cx, |a, cx| {
                    a.plot = Plot::Scatter;
                    cx.notify();
                });
            }));
        let accent = color::accent();
        let picture = canvas(
            |_, _, _| {},
            move |b, _, window, _| {
                window.paint_quad(fill(b, color::raised()));
                if xs.is_empty() {
                    return;
                }
                let (lo, hi) = xs.iter().fold((f64::MAX, f64::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
                let span = (hi - lo).max(1e-12);
                let (w, h) = (b.size.width.as_f32(), b.size.height.as_f32());
                match kind {
                    Plot::Histogram => {
                        let bins = 30usize;
                        let mut count = vec![0usize; bins];
                        for &v in &xs {
                            count[(((v - lo) / span) * bins as f64).floor().clamp(0.0, bins as f64 - 1.0) as usize] += 1;
                        }
                        let top = *count.iter().max().unwrap_or(&1) as f32;
                        let bw = w / bins as f32;
                        for (i, &c) in count.iter().enumerate() {
                            let bh = (c as f32 / top.max(1.0)) * (h - 4.0);
                            window.paint_quad(fill(Bounds { origin: point(b.origin.x + px(i as f32 * bw + 1.0), b.origin.y + px(h - bh)), size: size(px((bw - 2.0).max(1.0)), px(bh)) }, accent));
                        }
                    }
                    Plot::Scatter => {
                        let (ylo, yhi) = ys.iter().fold((f64::MAX, f64::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
                        let yspan = (yhi - ylo).max(1e-12);
                        for (&vx, &vy) in xs.iter().zip(&ys).take(20_000) {
                            let px_ = ((vx - lo) / span) as f32 * (w - 6.0) + 3.0;
                            let py_ = h - 3.0 - ((vy - ylo) / yspan) as f32 * (h - 6.0);
                            window.paint_quad(fill(Bounds { origin: point(b.origin.x + px(px_ - 1.5), b.origin.y + px(py_ - 1.5)), size: size(px(3.), px(3.)) }, accent.opacity(0.75)));
                        }
                    }
                }
            },
        )
        .w_full()
        .h(px(170.));
        Section::new("Plot")
            .child(switch)
            .child(h_flex().gap(px(theme::SM)).child(div().w(px(16.)).text_xs().text_color(color::text_muted()).child("x")).child(div().flex_1().child(pick("plot-x", x, |a, n| a.plot_x = Some(n)))))
            .when(kind == Plot::Scatter, |d| d.child(h_flex().gap(px(theme::SM)).child(div().w(px(16.)).text_xs().text_color(color::text_muted()).child("y")).child(div().flex_1().child(pick("plot-y", y, |a, n| a.plot_y = Some(n))))))
            .child(picture)
    }

    /// The code pane under the table.
    fn code_pane(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let me = cx.entity().downgrade();
        let langs = Language::ALL.into_iter().map(|l| {
            let me = me.clone();
            Button::new(SharedString::from(format!("code-lang-{}", l.title()))).ghost().xsmall().label(l.title()).selected(self.lang == l).on_click(move |_, window, cx| {
                let _ = me.update(cx, |a, cx| a.set_lang(l, window, cx));
            })
        });
        let (run, tmpl, exp) = (me.clone(), me.clone(), me.clone());
        let lang = self.lang;
        v_flex()
            .flex_none()
            .h(px(250.))
            .border_t_1()
            .border_color(color::border())
            .bg(color::chrome())
            .capture_key_down(cx.listener(|a, e: &KeyDownEvent, _, cx| {
                if e.keystroke.key == "enter" && (e.keystroke.modifiers.control || e.keystroke.modifiers.platform) {
                    a.run_code(cx);
                    cx.stop_propagation();
                }
            }))
            .child(
                h_flex()
                    .flex_none()
                    .h(px(32.))
                    .px(px(theme::SM))
                    .gap(px(theme::XS))
                    .child(Icon::new(IconName::SquareTerminal).small())
                    .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).child("Code on the table"))
                    .children(langs)
                    .child(Button::new("code-run").primary().xsmall().icon(Icon::new(IconName::Play)).label(if self.running { "Running…" } else { "Run  Ctrl+Enter" }).disabled(self.running).on_click(move |_, _, cx| {
                        let _ = run.update(cx, |a, cx| a.run_code(cx));
                    }))
                    .child(Button::new("code-template").ghost().xsmall().label("Example").on_click(move |_, window, cx| {
                        let _ = tmpl.update(cx, |a, cx| {
                            let t = script::table_template(lang);
                            a.editor.update(cx, |e, cx| e.set_value(t, window, cx));
                        });
                    }))
                    .child(Button::new("code-export").ghost().xsmall().icon(Icon::new(IconName::Download)).label("Export steps").disabled(self.steps.is_empty()).on_click(move |_, _, cx| {
                        let _ = exp.update(cx, |a, cx| a.export_steps(cx));
                    }))
                    .child(div().flex_1())
                    .child(div().text_xs().text_color(color::text_muted()).child(match lang {
                        Language::Python => "`df`: Polars (else pandas); leave the result in `df` or `result`",
                        Language::Rust => "`df`: Table — num(), text(), filter(), with_num(), sort_by()",
                    })),
            )
            .children(self.code_error.clone().map(|e| div().flex_none().max_h(px(64.)).overflow_hidden().px(px(theme::SM)).text_xs().font_family("monospace").text_color(color::danger()).child(e)))
            .child(div().flex_1().min_h_0().px(px(theme::SM)).pb(px(theme::SM)).child(Editor::new(&self.editor).h_full()))
    }
}

/// The image around an annotation, its outline drawn in its class colour (≈ 64 px a side).
fn preview(py: &Pyramid, displays: &[ChannelDisplay], a: &Annotation, outline: [u8; 3], cursor_z: usize) -> Option<Arc<RenderImage>> {
    let (x0, y0, x1, y1) = a.shape.bounds();
    let pad = (x1 - x0).max(y1 - y0) * 0.15 + 4.0;
    let (rx0, ry0) = ((x0 - pad).max(0.0), (y0 - pad).max(0.0));
    let (rx1, ry1) = ((x1 + pad).min(py.width() as f64), (y1 + pad).min(py.height() as f64));
    if rx1 <= rx0 || ry1 <= ry0 {
        return None;
    }
    let side = (rx1 - rx0).max(ry1 - ry0);
    let level = (0..py.levels.len()).find(|&l| side / py.levels[l].downsample <= 64.0).unwrap_or(py.coarsest());
    let l = &py.levels[level];
    let ds = l.downsample;
    let (lx0, ly0) = ((rx0 / ds) as usize, (ry0 / ds) as usize);
    let lw = (((rx1 - rx0) / ds).ceil() as usize).max(1).min(l.width.saturating_sub(lx0)).max(1);
    let lh = (((ry1 - ry0) / ds).ceil() as usize).max(1).min(l.height.saturating_sub(ly0)).max(1);
    let plane = ((a.z.unwrap_or(cursor_z) as f64 / l.z_downsample) as usize).min(l.depth - 1);
    let mut planes = Vec::new();
    for (c, d) in displays.iter().enumerate().filter(|(_, d)| d.visible) {
        planes.push((py.read_region_at(level, c, plane, lx0, ly0, lw, lh).ok()?, *d));
    }
    let refs: Vec<(&[f32], ChannelDisplay)> = planes.iter().map(|(v, d)| (v.as_slice(), *d)).collect();
    let mut bgra = if refs.is_empty() { vec![0u8; lw * lh * 4] } else { composite_bgra(&refs, lw * lh) };
    // The outline: pixels of the shape next to one outside it
    let mask = rasterize(&[a], None, lx0, ly0, lw, lh, ds);
    for y in 0..lh {
        for x in 0..lw {
            let i = y * lw + x;
            if mask[i] == 0 {
                continue;
            }
            let edge = x == 0 || y == 0 || x + 1 == lw || y + 1 == lh || mask[i - 1] == 0 || mask[i + 1] == 0 || mask[i - lw] == 0 || mask[i + lw] == 0;
            if edge {
                bgra[i * 4..i * 4 + 4].copy_from_slice(&[outline[2], outline[1], outline[0], 255]);
            }
        }
    }
    Some(bgra_image(lw, lh, bgra))
}

impl Render for AnalyzeView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let s = self.store.read(cx);
        if s.doc.images.is_empty() {
            return div().size_full().child(EmptyState::new(IconName::Table2, "Nothing to measure yet", "Open an image, draw annotations or add a Measure operation: their tables show here.")).into_any_element();
        }
        // Tabs: annotations, then each view with a table
        let tables: Vec<(String, String)> = s
            .doc
            .graph
            .views()
            .filter(|v| s.analysis.as_ref().is_some_and(|a| a.tables.contains_key(&v.id)))
            .map(|v| (v.id.clone(), match &v.kind {
                NodeKind::View { name, .. } => name.clone(),
                _ => v.id.clone(),
            }))
            .collect();
        if self.tab.as_ref().is_some_and(|t| !tables.iter().any(|(id, _)| id == t)) {
            self.tab = None;
        }
        let me = cx.entity().downgrade();
        let tab_button = |id: SharedString, icon: IconName, label: String, on: bool, tab: Option<String>| {
            let me = me.clone();
            Button::new(id).ghost().small().icon(icon).label(label).selected(on).on_click(move |_, _, cx| {
                let tab = tab.clone();
                let _ = me.update(cx, |a, cx| {
                    a.tab = tab;
                    a.show_result = false;
                    a.sort = None;
                    cx.notify();
                });
            })
        };
        let mut tabs = h_flex().gap(px(2.)).child(tab_button("analyze-tab-annotations".into(), IconName::Shapes, "Annotations".into(), self.tab.is_none() && !self.show_result, None));
        for (id, name) in &tables {
            tabs = tabs.child(tab_button(SharedString::from(format!("analyze-tab-{id}")), IconName::Grid2x2, name.clone(), self.tab.as_deref() == Some(id.as_str()) && !self.show_result, Some(id.clone())));
        }
        if self.result.is_some() {
            let me = me.clone();
            let n = self.steps.len();
            tabs = tabs.child(Button::new("analyze-tab-result").ghost().small().icon(IconName::SquareTerminal).label(format!("Result of step {n}")).selected(self.show_result).on_click(move |_, _, cx| {
                let _ = me.update(cx, |a, cx| {
                    a.show_result = true;
                    cx.notify();
                });
            }));
        }
        let title = match (&self.tab, self.show_result) {
            (_, true) => "result".to_string(),
            (None, _) => "annotations".to_string(),
            (Some(v), _) => tables.iter().find(|(id, _)| id == v).map_or(v.clone(), |(_, n)| n.clone()),
        };
        let note = match (&self.tab, self.show_result) {
            (_, true) => "The result of the last step (the source table is unchanged).".to_string(),
            (None, _) => "The annotations of the image shown in Explore. Click a row to show it there.".to_string(),
            (Some(_), _) => format!("Objects of the whole image, measured at pyramid level {}.", self.store.read(cx).analysis.as_ref().map_or(0, |a| a.level)),
        };
        let model = self.model(cx);
        let order = model.as_ref().map(|m| self.order(m, cx)).unwrap_or_default();
        let total = model.as_ref().map_or(0, |m| m.rows());
        let with_previews = self.tab.is_none() && !self.show_result;
        if with_previews
            && let Some(ids) = model.as_ref().and_then(|m| m.ids.clone()) {
                let first: Vec<String> = order.iter().take(PREVIEWED).map(|&r| ids[r].clone()).collect();
                self.request_previews(&first, cx);
            }
        let table = model.as_ref().map(|m| self.table(m, &order, with_previews, window, cx).into_any_element());
        let plot = model.as_ref().map(|m| self.plot(m, &order, cx));
        // Summary per class (annotations)
        let summary = (self.tab.is_none() && !self.show_result).then(|| {
            let s = self.store.read(cx);
            let img = s.doc.active_image.clone().unwrap_or_default();
            let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
            let mut sec = Section::new("Per class");
            for c in &s.doc.classes {
                let list: Vec<_> = s.doc.annotations_of(&img).filter(|a| a.class == c.name).collect();
                if list.is_empty() {
                    continue;
                }
                let area: f64 = list.iter().filter_map(|a| a.shape.area()).sum();
                sec = sec.child(
                    h_flex()
                        .gap(px(theme::MD))
                        .child(crate::ui::widgets::swatch(c.color, 12.))
                        .child(div().w(px(100.)).text_sm().child(c.name.clone()))
                        .child(div().w(px(56.)).text_sm().text_color(color::text_muted()).child(group_digits(list.len())))
                        .child(div().text_sm().text_color(color::text_muted()).child(if area > 0.0 { crate::ui::widgets::format_area(area, um) } else { String::new() })),
                );
            }
            sec
        });
        // The steps run so far
        let steps = (!self.steps.is_empty()).then(|| {
            let mut sec = Section::new("Steps");
            for (i, st) in self.steps.iter().enumerate() {
                let (me, code, lang) = (me.clone(), st.code.clone(), st.lang);
                sec = sec.child(
                    div()
                        .id(("step", i))
                        .px(px(theme::XS))
                        .py(px(2.))
                        .rounded(px(4.))
                        .cursor_pointer()
                        .hover(|d| d.bg(color::control()))
                        .text_sm()
                        .child(format!("{} · {} · {} rows", i + 1, st.lang.title(), group_digits(st.rows)))
                        .on_click(move |_, window, cx| {
                            let code = code.clone();
                            let _ = me.update(cx, |a, cx| {
                                a.set_lang(lang, window, cx);
                                a.editor.update(cx, |e, cx| e.set_value(code, window, cx));
                                a.code_open = true;
                                cx.notify();
                            });
                        }),
                );
            }
            sec
        });
        let export = model.as_ref().map(|m| {
            let (me, title, csv, n) = (me.clone(), title.clone(), m.csv(&order), order.len());
            Button::new("analyze-export").outline().small().icon(IconName::Download).label("Export CSV").on_click(move |_, _, cx| {
                let csv = csv.clone();
                let _ = me.update(cx, |a, cx| a.export(csv, &title, n, cx));
            })
        });
        let code_toggle = {
            let me = me.clone();
            Button::new("analyze-code").ghost().small().icon(IconName::SquareTerminal).label("Code").selected(self.code_open).tooltip("Run Python or Rust on the table (Data Wrangler style)").on_click(move |_, _, cx| {
                let _ = me.update(cx, |a, cx| {
                    a.code_open = !a.code_open;
                    cx.notify();
                });
            })
        };
        let count = if order.len() == total { format!("{} rows", group_digits(total)) } else { format!("{} of {} rows", group_digits(order.len()), group_digits(total)) };
        v_flex()
            .size_full()
            .bg(color::panel())
            .child(
                h_flex()
                    .flex_none()
                    .h(px(theme::OPTIONS_BAR_H))
                    .px(px(theme::MD))
                    .gap(px(theme::MD))
                    .border_b_1()
                    .border_color(color::border())
                    .child(tabs)
                    .child(div().flex_1())
                    .child(div().w(px(220.)).child(Input::new(&self.search).small()))
                    .child(div().text_xs().text_color(color::text_muted()).child(count))
                    .child(code_toggle)
                    .children(export),
            )
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .child(div().px(px(theme::MD)).py(px(theme::SM)).text_xs().text_color(color::text_muted()).child(note))
                            .children(table)
                            .when(self.code_open, |d| d.child(self.code_pane(cx))),
                    )
                    .child(v_flex().id("analyze-side").flex_none().w(px(300.)).h_full().overflow_y_scroll().p(px(theme::MD)).gap(px(theme::XL)).border_l_1().border_color(color::border()).children(summary).children(plot).children(steps)),
            )
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_read_ranges_comparisons_and_text() {
        let df = parse_csv("id,class,area\na1,Tumor,12.5\na2,Stroma,3\na3,Tumor,40\n").unwrap();
        let m = Model::of(&df);
        assert!(m.numeric[2] && !m.numeric[1]);
        assert_eq!(m.order(&[(2, "> 10".into())], "", None), vec![0, 2]);
        assert_eq!(m.order(&[(2, "1..20".into())], "", None), vec![0, 1]);
        assert_eq!(m.order(&[(1, "= stroma".into())], "", None), vec![1]);
        assert_eq!(m.order(&[], "tum", Some((2, true))), vec![2, 0]);
        assert_eq!(split_csv("a,\"b,c\",\"d\"\"e\""), vec!["a", "b,c", "d\"e"]);
    }
}

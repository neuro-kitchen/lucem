//! The Analyze workspace: measurements as tables — the regions and shapes of the shown image,
//! its detected cells and tiles (with every measurement and the region each belongs to), and
//! the objects of every Measure result — that people can work with. On the left, the object
//! hierarchy (image ▸ regions ▸ the regions inside them, each with its cells): a click shows that
//! region's cells.
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
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::base::dock::{DockArea, DockLayout, Panel as BasePanel, PanelEvent};
use gpui_kit::base::Placement;
use gpui_kit::component::dock::{panel_handle, Panel};

use crate::ui::container::{self, AddViewDialog, DialogEvent, ViewOption};
use gpui_kit::EventEmitter;
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
    /// The detected object of each row (`c12`, `t5`: the Cells and Tiles tables).
    objects: Option<Vec<String>>,
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
        let mut m = Model { names: Vec::new(), text: Vec::new(), num: Vec::new(), numeric: Vec::new(), ids: None, objects: None };
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
            if c.name().as_str() == "object" {
                m.objects = Some(text.clone());
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

/// A kind of plot view.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Plot {
    Histogram,
    Scatter,
    Box,
    Counts,
}

impl Plot {
    const ALL: [Plot; 4] = [Plot::Histogram, Plot::Scatter, Plot::Box, Plot::Counts];

    fn title(self) -> &'static str {
        match self {
            Plot::Histogram => "Histogram",
            Plot::Scatter => "Scatter",
            Plot::Box => "Box plot",
            Plot::Counts => "Counts",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Plot::Histogram => "How one measurement is spread",
            Plot::Scatter => "One measurement against another",
            Plot::Box => "A measurement compared across groups (class…)",
            Plot::Counts => "How many rows each group has",
        }
    }

    fn icon(self) -> IconName {
        match self {
            Plot::Histogram => IconName::ChartColumn,
            Plot::Scatter => IconName::ChartScatter,
            Plot::Box => IconName::ChartCandlestick,
            Plot::Counts => IconName::ChartBar,
        }
    }

    /// Whether it splits the rows by a text column (always for box and counts; histograms and
    /// scatters only colour by one, when chosen).
    fn needs_group(self) -> bool {
        matches!(self, Plot::Box | Plot::Counts)
    }
}

/// A plot view of the table: it draws the rows the table shows (filters and search applied).
#[derive(Clone)]
pub(crate) struct PlotView {
    id: usize,
    kind: Plot,
    x: Option<String>,
    y: Option<String>,
    /// The text column the rows are split (or coloured) by; `None`: one group.
    group: Option<String>,
}

/// Groups drawn at most (the rest are "other").
const MAX_GROUPS: usize = 12;

/// The tabs of the detected objects (the others are Measure views' ids).
const CELLS: &str = "@cells";
const TILES: &str = "@tiles";

pub struct AnalyzeView {
    store: Entity<Store>,
    /// `None`: regions and shapes; `@cells` / `@tiles`: detections; `Some(view)`: that view's
    /// objects.
    tab: Option<String>,
    /// The region whose objects (with those of the regions inside it) the Cells / Tiles tables
    /// show; `None`: the whole image.
    scope: Option<String>,
    /// The table and what it was made from; the one being made (off the UI thread).
    model: Option<(u64, Arc<Model>)>,
    building: Option<(u64, Task<()>)>,
    /// The rows on screen: the table, its order, whether rows carry previews (read by the
    /// virtual list).
    shown: Option<(Arc<Model>, Arc<Vec<usize>>, bool)>,
    /// The rows' order (filters, search, sort applied), kept until one of them or the table
    /// changes: never worked out again on a frame where nothing changed.
    order_cache: Option<(u64, Arc<Vec<usize>>)>,
    /// The rows' scroll (both ways); the header follows its sideways offset.
    rows_scroll: gpui_kit::UniformListScrollHandle,
    sort: Option<(String, bool)>,
    filters: HashMap<String, Entity<InputState>>,
    search: Entity<InputState>,
    /// Previews by annotation id and version; what is being made.
    previews: HashMap<String, Arc<RenderImage>>,
    pending: HashSet<String>,
    preview_task: Option<Task<()>>,
    /// The plot views (each a panel of the dock).
    pub(crate) plots: Vec<PlotView>,
    next_plot: usize,
    /// Explore's container: the table and the plots are views in this dock, dragged into tabs
    /// and splits; the table's panel (when shown) and each plot's.
    pub(crate) dock: Entity<DockArea>,
    pub(crate) table_panel: Option<Entity<AnalyzePanel>>,
    pub(crate) plot_panels: Vec<(usize, Entity<AnalyzePanel>)>,
    /// The hierarchy (left) and the side pane (right) shown.
    show_hierarchy: bool,
    show_side: bool,
    /// What the table view shows above its rows (set by `render`).
    note: String,
    /// "Add view or container" open.
    dialog: Option<(Entity<AddViewDialog>, Subscription)>,
    /// The project layout restored last (the Store's `layout_rev`), and the views saved last.
    layout_seen: u64,
    layout_saved: Option<crate::data::layout::AnalyzeLayout>,
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
        let me = cx.weak_entity();
        let dock = container::new_dock("analyze", window, cx);
        let table_panel = cx.new(|cx| AnalyzePanel::new(me, PanelKind::Table, cx));
        dock.update(cx, |d, cx| d.set_center(DockLayout::tabs().panel_view(panel_handle(table_panel.clone()), cx), window, cx));
        Self {
            dock,
            table_panel: Some(table_panel),
            plot_panels: Vec::new(),
            show_hierarchy: true,
            show_side: true,
            note: String::new(),
            dialog: None,
            layout_seen: 0,
            layout_saved: None,
            store,
            tab: None,
            scope: None,
            model: None,
            building: None,
            shown: None,
            order_cache: None,
            rows_scroll: gpui_kit::UniformListScrollHandle::new(),
            sort: None,
            filters: HashMap::new(),
            search,
            previews: HashMap::new(),
            pending: HashSet::new(),
            preview_task: None,
            plots: Vec::new(),
            next_plot: 1,
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
        // The hierarchy: each region's parent, and the cells it holds (with those inside it)
        let cells = s.hist.cells.get(&img).cloned().unwrap_or_default();
        let h = s.hierarchy(&img);
        let (mut parent, mut n_cells, mut positive) = (vec![], vec![], vec![]);
        let (mut id, mut class, mut kind, mut area, mut length, mut cx_, mut cy_, mut z) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        for a in &rows {
            if a.shape.is_region() {
                let sub = h.subtree(&a.id);
                let inside: Vec<usize> = h.object_parent.iter().enumerate().filter(|(_, p)| p.as_ref().is_some_and(|p| sub.contains(p))).map(|(i, _)| i).collect();
                parent.push(h.region_parent.get(&a.id).cloned().flatten().unwrap_or_default());
                n_cells.push(Some(inside.len() as f64));
                let scored: Vec<&lumen_hist::HistObject> = inside.iter().map(|&i| &cells[i]).filter(|c| c.intensity.is_some()).collect();
                positive.push((!scored.is_empty()).then(|| 100.0 * scored.iter().filter(|c| c.intensity.is_some_and(|k| k.is_positive())).count() as f64 / scored.len() as f64));
            } else {
                parent.push(String::new());
                n_cells.push(None);
                positive.push(None);
            }
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
        cols.push(Column::from(Series::new("parent".into(), parent)));
        if !cells.is_empty() {
            cols.push(Column::from(Series::new("cells".into(), n_cells)));
            if positive.iter().any(Option::is_some) {
                cols.push(Column::from(Series::new("positive_pct".into(), positive)));
            }
        }
        DataFrame::new(rows.len(), cols).ok()
    }

    /// The detected cells (`tiles`: the superpixel tiles) of the shown image, those in the
    /// scope's region (and the regions inside it): class, intensity, the region each belongs to,
    /// position, then every measurement.
    fn detections_frame(&self, tiles: bool, cx: &Context<Self>) -> Option<DataFrame> {
        let s = self.store.read(cx);
        let img = s.doc.active_image.clone()?;
        let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
        let k = um.unwrap_or(1.0);
        let objects = if tiles { s.hist.tiles.get(&img) } else { s.hist.cells.get(&img) }?.clone();
        // Cells: the cached hierarchy; tiles: their own (fewer, built with the table)
        let h = if tiles {
            let anns: Vec<&Annotation> = s.doc.annotations_of(&img).collect();
            Arc::new(crate::domain::hierarchy::Hierarchy::build(anns.iter().copied(), &objects))
        } else {
            s.hierarchy(&img)
        };
        let scope = self.scope.as_ref().map(|r| h.subtree(r));
        let class_of: HashMap<String, String> = h.regions.iter().map(|r| (r.id.clone(), r.class.clone())).collect();
        let keep: Vec<usize> = (0..objects.len()).filter(|&i| scope.as_ref().is_none_or(|sub| h.object_parent[i].as_ref().is_some_and(|p| sub.contains(p)))).collect();
        let names: Vec<String> = keep.iter().flat_map(|&i| objects[i].measurements.keys().cloned()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let u = if um.is_some() { "um" } else { "px" };
        let mut cols = vec![
            Column::from(Series::new("object".into(), keep.iter().map(|&i| format!("{}{}", if tiles { "t" } else { "c" }, objects[i].id)).collect::<Vec<_>>())),
            Column::from(Series::new("class".into(), keep.iter().map(|&i| objects[i].class.clone().unwrap_or_default()).collect::<Vec<_>>())),
            Column::from(Series::new("intensity".into(), keep.iter().map(|&i| objects[i].intensity.map_or(String::new(), |c| c.name().to_string())).collect::<Vec<_>>())),
            Column::from(Series::new("region".into(), keep.iter().map(|&i| h.object_parent[i].clone().unwrap_or_default()).collect::<Vec<_>>())),
            Column::from(Series::new("region_class".into(), keep.iter().map(|&i| h.object_parent[i].as_ref().and_then(|p| class_of.get(p).cloned()).unwrap_or_default()).collect::<Vec<_>>())),
            Column::from(Series::new(format!("x_{u}").into(), keep.iter().map(|&i| objects[i].centroid[0] * k).collect::<Vec<_>>())),
            Column::from(Series::new(format!("y_{u}").into(), keep.iter().map(|&i| objects[i].centroid[1] * k).collect::<Vec<_>>())),
        ];
        for n in &names {
            cols.push(Column::from(Series::new(n.as_str().into(), keep.iter().map(|&i| objects[i].measurements.get(n).copied()).collect::<Vec<_>>())));
        }
        DataFrame::new(keep.len(), cols).ok()
    }

    /// The hierarchy as a tree: the image, then its regions and those inside them, each with
    /// the cells it holds; a click shows that region's cells.
    fn hierarchy_pane(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let s = self.store.read(cx);
        let img = s.doc.active_image.clone().unwrap_or_default();
        let cells = s.hist.cells.get(&img).cloned().unwrap_or_default();
        let anns: Vec<Annotation> = s.doc.annotations_of(&img).cloned().collect();
        let colors: HashMap<String, [u8; 3]> = s.doc.classes.iter().map(|c| (c.name.clone(), c.color)).collect();
        let notes: HashMap<String, String> = anns.iter().map(|a| (a.id.clone(), if a.note.is_empty() { format!("{} {}", a.shape.kind(), a.id) } else { a.note.lines().next().unwrap_or_default().to_string() })).collect();
        let h = s.hierarchy(&img);
        let in_sub = |id: &str| h.in_subtree.get(id).copied().unwrap_or(0);
        let me = cx.entity().downgrade();
        let scope = self.scope.clone();
        let on_cells = self.tab.as_deref() == Some(CELLS);
        let row = |key: Option<String>, depth: usize, color: Option<[u8; 3]>, name: String, count: usize| {
            let me = me.clone();
            let selected = on_cells && scope == key;
            h_flex()
                .id(SharedString::from(format!("hier-{}", key.clone().unwrap_or_else(|| "image".into()))))
                .h(px(26.))
                .pl(px(theme::XS + depth as f32 * 14.0))
                .pr(px(theme::XS))
                .gap(px(theme::SM))
                .rounded(px(4.))
                .cursor_pointer()
                .when(selected, |d| d.bg(color::accent().opacity(0.3)))
                .when(!selected, |d| d.hover(|d| d.bg(color::control())))
                .child(match color {
                    Some(c) => crate::ui::widgets::swatch(c, 10.).into_any_element(),
                    None => Icon::new(IconName::Image).small().into_any_element(),
                })
                .child(div().flex_1().min_w_0().truncate().text_sm().child(name))
                .child(div().flex_none().text_xs().text_color(color::text_muted()).child(if count > 0 { format!("{} cells", group_digits(count)) } else { String::new() }))
                .on_click(move |_, _, cx| {
                    let key = key.clone();
                    let _ = me.update(cx, |a, cx| {
                        a.tab = Some(CELLS.to_string());
                        a.scope = key;
                        a.show_result = false;
                        cx.notify();
                    });
                })
        };
        let mut col = v_flex().gap(px(1.)).child(div().text_base().font_weight(FontWeight::SEMIBOLD).pb(px(theme::XS)).child("Hierarchy")).child(row(None, 0, None, s.doc.active().map_or("Image".to_string(), |i| i.name.clone()), cells.len()));
        // Depth first, largest first at each level
        let mut stack: Vec<(String, usize)> = h.children(None).iter().rev().map(|r| (r.id.clone(), 1)).collect();
        let mut shown = 0;
        while let Some((id, depth)) = stack.pop() {
            if shown > 300 {
                col = col.child(div().text_xs().text_color(color::text_muted()).child("… more regions (filter the Regions table)"));
                break;
            }
            shown += 1;
            let class = h.regions.iter().find(|r| r.id == id).map(|r| r.class.clone()).unwrap_or_default();
            col = col.child(row(Some(id.clone()), depth, Some(colors.get(&class).copied().unwrap_or([150, 150, 150])), notes.get(&id).cloned().unwrap_or(id.clone()), in_sub(&id)));
            for c in h.children(Some(&id)).iter().rev() {
                stack.push((c.id.clone(), depth + 1));
            }
        }
        if h.regions.is_empty() {
            col = col.child(div().text_xs().text_color(color::text_muted()).child("Draw regions to group the cells."));
        }
        col
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
        (&self.tab, &self.scope, s.hist.rev, s.annotations_rev, s.doc.active_image.clone(), s.analysis.as_ref().map(|a| a.key), self.show_result, self.result.as_ref().map(|r| r.0)).hash(&mut h);
        h.finish()
    }

    /// The table shown: the last code result, or the source. A new table is made off the UI
    /// thread; the one before stays on screen meanwhile.
    fn model(&mut self, cx: &mut Context<Self>) -> Option<Arc<Model>> {
        let key = self.source_key(cx);
        if let Some((k, m)) = &self.model
            && *k == key {
                return Some(m.clone());
            }
        if self.building.as_ref().is_some_and(|(k, _)| *k == key) {
            return self.model.as_ref().map(|(_, m)| m.clone());
        }
        let df = match (&self.result, self.show_result, &self.tab) {
            (Some((_, df)), true, _) => Some(df.clone()),
            (_, _, None) => self.annotations_frame(cx),
            (_, _, Some(v)) if v == CELLS => self.detections_frame(false, cx),
            (_, _, Some(v)) if v == TILES => self.detections_frame(true, cx),
            (_, _, Some(v)) => self.objects_frame(v, cx),
        };
        let Some(df) = df else {
            self.model = None;
            return None;
        };
        let work = cx.background_spawn(async move { Arc::new(Model::of(&df)) });
        let task = cx.spawn(async move |this, cx| {
            let m = work.await;
            let _ = this.update(cx, |a, cx| {
                a.model = Some((key, m));
                a.building = None;
                cx.notify();
            });
        });
        self.building = Some((key, task));
        self.model.as_ref().map(|(_, m)| m.clone())
    }

    /// The filter of each column that has one.
    fn filter_values(&self, m: &Model, cx: &Context<Self>) -> Vec<(usize, String)> {
        m.names.iter().enumerate().filter_map(|(c, n)| self.filters.get(n).map(|i| (c, i.read(cx).value().to_string()))).filter(|(_, f)| !f.trim().is_empty()).collect()
    }

    /// The order of `m`'s rows, from the cache when the table, filters, search and sort are
    /// those it was made for.
    fn cached_order(&mut self, m: &Arc<Model>, cx: &Context<Self>) -> Arc<Vec<usize>> {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (Arc::as_ptr(m) as usize, self.filter_values(m, cx), self.search.read(cx).value().to_string(), &self.sort).hash(&mut h);
        let key = h.finish();
        if let Some((k, o)) = &self.order_cache
            && *k == key
        {
            return o.clone();
        }
        let o = Arc::new(self.order(m, cx));
        self.order_cache = Some((key, o.clone()));
        o
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

    /// The shape a row's preview outlines: its annotation, or a detected object's outline.
    fn row_shape(&self, m: &Model, r: usize, cx: &Context<Self>) -> Option<(Annotation, [u8; 3])> {
        let s = self.store.read(cx);
        if let Some(id) = m.ids.as_ref().map(|ids| &ids[r]) {
            let a = s.doc.annotations.iter().find(|a| &a.id == id)?;
            return Some((a.clone(), s.doc.class(&a.class).map_or([255, 255, 0], |c| c.color)));
        }
        let key = m.objects.as_ref()?.get(r)?;
        let img = s.doc.active_image.clone()?;
        let (list, n) = match key.split_at(1) {
            ("c", n) => (s.hist.cells.get(&img)?, n),
            ("t", n) => (s.hist.tiles.get(&img)?, n),
            _ => return None,
        };
        let id: u32 = n.parse().ok()?;
        let o = list.iter().find(|o| o.id == id)?;
        let class = o.class.clone().unwrap_or_else(|| "Cell".into());
        let color = s.doc.class(&class).map_or([90, 200, 255], |c| c.color);
        Some((Annotation { id: key.clone(), image: img, shape: crate::domain::annotation::Shape::Polygon { points: o.nucleus.clone().unwrap_or_else(|| o.roi.clone()) }, class, note: String::new(), z: None }, color))
    }

    /// Makes the previews of the rows on screen that have none (off the UI thread).
    fn request_previews(&mut self, shapes: Vec<(Annotation, [u8; 3])>, cx: &mut Context<Self>) {
        if self.preview_task.is_some() {
            return;
        }
        let s = self.store.read(cx);
        let Some(img) = s.doc.active().cloned() else { return };
        let Some(py) = s.pyramids.get(&img.id).cloned() else { return };
        let cursor_z = s.cursor().map_or(0, |c| c[0]);
        let displays: Vec<ChannelDisplay> = img.channels.iter().map(|c| ChannelDisplay { visible: c.visible, color: [c.color[0] as f32 / 255.0, c.color[1] as f32 / 255.0, c.color[2] as f32 / 255.0], min: c.min, max: c.max, gamma: c.gamma }).collect();
        let todo: Vec<(String, Annotation, [u8; 3])> = shapes
            .into_iter()
            .map(|(a, c)| (Self::preview_key(&a), a, c))
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
        // Every row reachable: a virtual list (only the rows on screen are made), in a grid as wide
        // as its columns that scrolls sideways with its header
        let width = pw + m.names.len() as f32 * COL_W;
        // The list scrolls both ways itself: the wheel scrolls up and down, Shift+wheel or a
        // sideways swipe across. (Inside a sideways-only scroller, GPUI turns the wheel into a
        // sideways scroll and the rows never move.)
        let list = gpui_kit::uniform_list("analyze-rows", order.len(), cx.processor(|this, range: std::ops::Range<usize>, _, cx| this.rows(range, cx)))
            .with_horizontal_sizing_behavior(gpui_kit::ListHorizontalSizingBehavior::Unconstrained)
            .track_scroll(&self.rows_scroll)
            .flex_1()
            .min_h_0()
            .w_full();
        // The header and the filters move sideways with the rows
        let dx = self.rows_scroll.0.borrow().base_handle.offset().x;
        let head = div().flex_none().w_full().overflow_hidden().child(v_flex().w(px(width)).ml(dx).child(header).child(filter_row));
        v_flex().id("analyze-table").flex_1().min_h_0().w_full().child(head).child(list)
    }

    /// The rows `range` of the table on screen.
    fn rows(&mut self, range: std::ops::Range<usize>, cx: &mut Context<Self>) -> Vec<gpui_kit::AnyElement> {
        let Some((m, order, with_previews)) = self.shown.clone() else { return Vec::new() };
        let pw = if with_previews { PREVIEW + 8.0 } else { 0.0 };
        // Previews of the rows on screen
        if with_previews {
            let shapes: Vec<(Annotation, [u8; 3])> = range.clone().filter_map(|i| order.get(i)).filter_map(|&r| self.row_shape(&m, r, cx)).collect();
            self.request_previews(shapes, cx);
        }
        let s = self.store.read(cx);
        let selected = s.sel.shapes.clone();
        let picked_objects = s.sel.objects.clone();
        let row_w = pw + m.names.len() as f32 * COL_W;
        let mut out = Vec::new();
        for i in range {
            let Some(&r) = order.get(i) else { break };
            let id = m.ids.as_ref().map(|ids| ids[r].clone());
            let object = m.objects.as_ref().map(|o| o[r].clone());
            let thumb = with_previews.then(|| {
                let p = self.row_shape(&m, r, cx).and_then(|(a, _)| self.previews.get(&Self::preview_key(&a)).cloned());
                div().flex_none().w(px(pw)).p(px(4.)).child(match p {
                    Some(p) => div().size(px(PREVIEW)).rounded(px(3.)).overflow_hidden().bg(gpui_kit::black()).child(img(p).size_full()).into_any_element(),
                    None => div().size(px(PREVIEW)).rounded(px(3.)).bg(color::control()).into_any_element(),
                })
            });
            let picked = id.as_ref().is_some_and(|i| selected.contains(i)) || object.as_ref().is_some_and(|o| picked_objects.contains(o));
            let store = self.store.clone();
            out.push(
                div()
                    .id(("analyze-row", i))
                    .flex()
                    .flex_row()
                    .w(px(row_w))
                    .h(px(if with_previews { PREVIEW + 8.0 } else { 28.0 }))
                    .items_center()
                    .when(i % 2 == 1, |d| d.bg(color::raised()))
                    .when(picked, |d| d.bg(color::accent().opacity(0.3)))
                    .cursor_pointer()
                    .hover(|d| d.bg(color::hover()))
                    .children(thumb)
                    .children((0..m.names.len()).map(|c| div().flex_none().w(px(COL_W)).px(px(theme::SM)).truncate().text_sm().font_family("monospace").child(m.text[c][r].clone())))
                    .on_click(move |e, _, cx| {
                        let additive = e.modifiers().shift || e.modifiers().control || e.modifiers().platform;
                        if let Some(id) = &id {
                            let id = id.clone();
                            store.update(cx, |s, cx| s.reveal_annotation(&id, cx));
                        } else if let Some(o) = &object {
                            let o = o.clone();
                            store.update(cx, |s, cx| s.select_object(o, additive, cx));
                        }
                    })
                    .into_any_element(),
            );
        }
        out
    }

    /// A histogram of a column, or two columns against each other, over the rows shown.
    /// Adds a plot view of `kind`, its columns chosen from the table.
    /// A new plot view: under the table for the first, then to the right of the last plot.
    pub(crate) fn add_plot(&mut self, kind: Plot, window: &mut Window, cx: &mut Context<Self>) {
        let id = self.next_plot;
        self.next_plot += 1;
        self.plots.push(PlotView { id, kind, x: None, y: None, group: None });
        let me = cx.weak_entity();
        let panel = cx.new(|cx| AnalyzePanel::new(me, PanelKind::Plot(id), cx));
        let next_to = self.plot_panels.last().map(|(_, p)| (p.entity_id(), Placement::Right)).or_else(|| self.table_panel.as_ref().map(|p| (p.entity_id(), Placement::Bottom)));
        self.plot_panels.push((id, panel.clone()));
        self.place(panel, next_to, window, cx);
    }

    /// The table view back (after it was closed), to the left of the plots.
    pub(crate) fn add_table(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.table_panel.is_some() {
            return;
        }
        let me = cx.weak_entity();
        let panel = cx.new(|cx| AnalyzePanel::new(me, PanelKind::Table, cx));
        let next_to = self.plot_panels.first().map(|(_, p)| (p.entity_id(), Placement::Left));
        self.table_panel = Some(panel.clone());
        self.place(panel, next_to, window, cx);
    }

    fn place(&mut self, panel: Entity<AnalyzePanel>, next_to: Option<(gpui_kit::EntityId, Placement)>, window: &mut Window, cx: &mut Context<Self>) {
        container::add_beside(&self.dock, panel, next_to, window, cx);
        cx.notify();
    }

    /// Analyze's views for the dialog: the table (once), then the plots.
    fn view_options(&self) -> Vec<ViewOption> {
        let table = ViewOption::new("Table", IconName::Table2, "The rows, filtered, searched and sorted");
        let mut out = vec![if self.table_panel.is_some() { table.unavailable("Already shown") } else { table }];
        out.extend(Plot::ALL.iter().map(|k| ViewOption::new(k.title(), k.icon(), k.hint())));
        out
    }

    /// The shared "Add view or container" dialog.
    pub fn open_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let options = self.view_options();
        let dialog = cx.new(|cx| AddViewDialog::new(options, true, window, cx));
        let sub = cx.subscribe_in(&dialog, window, |this, _, e: &DialogEvent, window, cx| {
            this.dialog = None;
            match *e {
                DialogEvent::Arrange(how) => this.arrange(how, window, cx),
                DialogEvent::Add(0) => this.add_table(window, cx),
                DialogEvent::Add(i) => this.add_plot(Plot::ALL[i - 1], window, cx),
                DialogEvent::Close => {}
            }
            cx.notify();
        });
        self.dialog = Some((dialog, sub));
        cx.notify();
    }

    /// Lays the views out as `how`: the table, then the plots.
    pub(crate) fn arrange(&mut self, how: crate::data::store::Arrangement, window: &mut Window, cx: &mut Context<Self>) {
        let panels: Vec<Entity<AnalyzePanel>> = self.table_panel.iter().cloned().chain(self.plot_panels.iter().map(|(_, p)| p.clone())).collect();
        if panels.is_empty() {
            return;
        }
        let layout = container::arranged(&panels, how, cx);
        self.dock.update(cx, |d, cx| d.set_center(layout, window, cx));
        cx.notify();
    }

    /// A view left the dock (closed from its tab).
    pub(crate) fn removed(&mut self, kind: PanelKind, cx: &mut Context<Self>) {
        match kind {
            PanelKind::Table => self.table_panel = None,
            PanelKind::Plot(id) => {
                self.plots.retain(|p| p.id != id);
                self.plot_panels.retain(|(i, _)| *i != id);
            }
        }
        cx.notify();
    }

    /// The table view's content: the note, then the table.
    fn table_view(&mut self, window: &mut Window, cx: &mut Context<Self>) -> gpui_kit::AnyElement {
        let Some((m, order, previews)) = self.shown.clone() else {
            return div().size_full().p(px(theme::MD)).text_sm().text_color(color::text_muted()).child("Making the table…").into_any_element();
        };
        let note = self.note.clone();
        let table = self.table(&m, &order, previews, window, cx).into_any_element();
        v_flex().size_full().bg(color::panel()).child(div().flex_none().px(px(theme::MD)).py(px(theme::SM)).text_xs().text_color(color::text_muted()).child(note)).child(table).into_any_element()
    }

    /// A plot view's content.
    fn plot_view(&mut self, id: usize, cx: &mut Context<Self>) -> gpui_kit::AnyElement {
        let (Some((m, order, _)), Some(v)) = (self.shown.clone(), self.plots.iter().find(|p| p.id == id).cloned()) else {
            return div().into_any_element();
        };
        self.plot_card(&v, &m, &order, cx)
    }

    /// Analyze's menus (in the app's menu row while Analyze is shown): Edit, Table (what it
    /// shows, export, code), View (add a view, the side panels).
    pub fn menus(&self, cx: &mut Context<Self>) -> gpui_kit::AnyElement {
        let me = cx.entity().downgrade();
        let store = self.store.clone();
        let (can_undo, can_redo) = (self.store.read(cx).can_undo(), self.store.read(cx).can_redo());
        let img = self.store.read(cx).doc.active_image.clone().unwrap_or_default();
        let (n_cells, n_tiles) = (self.store.read(cx).hist.cells.get(&img).map_or(0, Vec::len), self.store.read(cx).hist.tiles.get(&img).map_or(0, Vec::len));
        let tab = self.tab.clone();
        let item = |label: &str, icon: IconName| PopupMenuItem::new(label.to_string()).icon(Icon::new(icon));
        let edit = {
            let (a, b) = (store.clone(), store.clone());
            Button::new("analyze-menu-edit").ghost().small().label("Edit").dropdown_menu(move |m: PopupMenu, _, _| {
                let (a, b) = (a.clone(), b.clone());
                m.item(item("Undo  Ctrl+Z", IconName::Undo2).disabled(!can_undo).on_click(move |_, _, cx| a.update(cx, |s, cx| s.undo(false, cx))))
                    .item(item("Redo  Ctrl+Shift+Z", IconName::Redo2).disabled(!can_redo).on_click(move |_, _, cx| b.update(cx, |s, cx| s.undo(true, cx))))
            })
        };
        let table = {
            let me = me.clone();
            Button::new("analyze-menu-table").ghost().small().label("Table").dropdown_menu(move |m: PopupMenu, _, _| {
                let pick = |label: String, icon: IconName, to: Option<String>, on: bool, enabled: bool| {
                    let me = me.clone();
                    PopupMenuItem::new(label).icon(Icon::new(icon)).checked(on).disabled(!enabled).on_click(move |_, _, cx| {
                        let to = to.clone();
                        let _ = me.update(cx, |a, cx| {
                            a.tab = to;
                            a.show_result = false;
                            a.sort = None;
                            cx.notify();
                        });
                    })
                };
                let (e, c) = (me.clone(), me.clone());
                m.label("Show")
                    .item(pick("Regions and shapes".into(), IconName::Shapes, None, tab.is_none(), true))
                    .item(pick(format!("Cells · {}", group_digits(n_cells)), IconName::Sparkles, Some(CELLS.into()), tab.as_deref() == Some(CELLS), n_cells > 0))
                    .item(pick(format!("Tiles · {}", group_digits(n_tiles)), IconName::Grid3x3, Some(TILES.into()), tab.as_deref() == Some(TILES), n_tiles > 0))
                    .separator()
                    .item(item("Export CSV…", IconName::Download).on_click(move |_, _, cx| {
                        let _ = e.update(cx, |a, cx| {
                            if let Some((m, order, _)) = a.shown.clone() {
                                let csv = m.csv(&order);
                                a.export(csv, "table", order.len(), cx);
                            }
                        });
                    }))
                    .item(item("Code on the table (Python, Rust)", IconName::SquareTerminal).on_click(move |_, _, cx| {
                        let _ = c.update(cx, |a, cx| {
                            a.code_open = !a.code_open;
                            cx.notify();
                        });
                    }))
            })
        };
        let (table_shown, hier, side) = (self.table_panel.is_some(), self.show_hierarchy, self.show_side);
        let view = {
            let me = me.clone();
            Button::new("analyze-menu-view").ghost().small().label("View").dropdown_menu(move |m: PopupMenu, _, _| {
                let (t, d) = (me.clone(), me.clone());
                let mut m = m
                    .item(item("Add view or container…", IconName::Plus).on_click(move |_, window, cx| {
                        let _ = d.update(cx, |a, cx| a.open_dialog(window, cx));
                    }))
                    .separator()
                    .label("Add a view")
                    .item(item("Table", IconName::Table2).disabled(table_shown).on_click(move |_, window, cx| {
                        let _ = t.update(cx, |a, cx| a.add_table(window, cx));
                    }));
                for k in Plot::ALL {
                    let me = me.clone();
                    m = m.item(PopupMenuItem::new(format!("{}   {}", k.title(), k.hint())).icon(Icon::new(k.icon())).on_click(move |_, window, cx| {
                        let _ = me.update(cx, |a, cx| a.add_plot(k, window, cx));
                    }));
                }
                let (h, s) = (me.clone(), me.clone());
                m.separator()
                    .item(item("Hierarchy", IconName::PanelLeft).checked(hier).on_click(move |_, _, cx| {
                        let _ = h.update(cx, |a, cx| {
                            a.show_hierarchy = !a.show_hierarchy;
                            cx.notify();
                        });
                    }))
                    .item(item("Selection, summary and steps", IconName::PanelRight).checked(side).on_click(move |_, _, cx| {
                        let _ = s.update(cx, |a, cx| {
                            a.show_side = !a.show_side;
                            cx.notify();
                        });
                    }))
            })
        };
        h_flex().gap(px(2.)).child(edit).child(table).child(view).into_any_element()
    }

    /// Its views as `layout.json` keeps them.
    fn layout(&self) -> crate::data::layout::AnalyzeLayout {
        crate::data::layout::AnalyzeLayout {
            table: self.table_panel.is_some(),
            plots: self.plots.iter().map(|p| crate::data::layout::PlotLayout { kind: p.kind.title().into(), x: p.x.clone(), y: p.y.clone(), group: p.group.clone() }).collect(),
            hierarchy: self.show_hierarchy,
            side: self.show_side,
        }
    }

    /// The views a project saved: its plots (with their columns), the table, the side panels.
    fn restore_layout(&mut self, l: crate::data::layout::AnalyzeLayout, window: &mut Window, cx: &mut Context<Self>) {
        let old: Vec<Entity<AnalyzePanel>> = self.plot_panels.drain(..).map(|(_, p)| p).collect();
        self.plots.clear();
        self.dock.update(cx, |d, cx| {
            for p in old {
                d.remove_panel(p, window, cx);
            }
        });
        for spec in &l.plots {
            let Some(kind) = Plot::ALL.iter().copied().find(|k| k.title() == spec.kind) else { continue };
            self.add_plot(kind, window, cx);
            if let Some(p) = self.plots.last_mut() {
                (p.x, p.y, p.group) = (spec.x.clone(), spec.y.clone(), spec.group.clone());
            }
        }
        match (&self.table_panel, l.table) {
            (Some(t), false) if !self.plot_panels.is_empty() => {
                let t = t.clone();
                self.dock.update(cx, |d, cx| d.remove_panel(t, window, cx));
                self.table_panel = None;
            }
            (None, true) => self.add_table(window, cx),
            _ => {}
        }
        (self.show_hierarchy, self.show_side) = (l.hierarchy, l.side);
        cx.notify();
    }

    /// A plot's tab title: its kind and columns.
    fn plot_title(&self, id: usize) -> String {
        match self.plots.iter().find(|p| p.id == id) {
            Some(p) => match (p.kind, &p.x, &p.y) {
                (Plot::Scatter, Some(x), Some(y)) => format!("Scatter · {y} by {x}"),
                (Plot::Histogram, Some(x), _) => format!("Histogram · {x}"),
                (Plot::Box, _, Some(y)) => format!("Box plot · {y}"),
                (k, _, _) => k.title().to_string(),
            },
            None => "Plot".into(),
        }
    }

    fn edit_plot(&mut self, id: usize, f: impl FnOnce(&mut PlotView), cx: &mut Context<Self>) {
        if let Some(p) = self.plots.iter_mut().find(|p| p.id == id) {
            f(p);
            cx.notify();
        }
    }

    /// The colour of a group: its class's colour when the column holds classes, else one of
    /// the class presets in turn.
    fn group_colors(&self, column: Option<&str>, groups: &[String], cx: &gpui_kit::App) -> Vec<gpui_kit::Hsla> {
        let s = self.store.read(cx);
        let classes = column.is_some_and(|c| c.eq_ignore_ascii_case("class"));
        groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let rgb = classes.then(|| s.doc.classes.iter().find(|c| c.name == *g).map(|c| c.color)).flatten().unwrap_or(crate::screens::explore::panels::display::CLASS_COLORS[i % 8].1);
                gpui_kit::Hsla::from(gpui_kit::rgb((rgb[0] as u32) << 16 | (rgb[1] as u32) << 8 | rgb[2] as u32))
            })
            .collect()
    }

    /// One plot view: its kind, its columns, the picture, a legend when split by groups.
    fn plot_card(&self, v: &PlotView, m: &Model, order: &[usize], cx: &mut Context<Self>) -> gpui_kit::AnyElement {
        let id = v.id;
        let numeric: Vec<String> = m.names.iter().zip(&m.numeric).filter(|(_, n)| **n).map(|(n, _)| n.clone()).collect();
        let texts: Vec<String> = m.names.iter().zip(&m.numeric).filter(|(_, n)| !**n).map(|(n, _)| n.clone()).filter(|n| n != "id" && n != "object").collect();
        let x = v.x.clone().filter(|c| numeric.contains(c)).or_else(|| numeric.iter().find(|n| n.starts_with("area")).cloned()).or_else(|| numeric.first().cloned());
        let y = v.y.clone().filter(|c| numeric.contains(c)).or_else(|| numeric.iter().find(|n| Some(*n) != x.as_ref()).cloned());
        let default_group = || texts.iter().find(|n| n.eq_ignore_ascii_case("class")).cloned().or_else(|| texts.first().cloned());
        let group = match &v.group {
            Some(g) if texts.contains(g) => Some(g.clone()),
            _ if v.kind.needs_group() => default_group(),
            _ => None,
        };
        let col = |name: &Option<String>| name.as_ref().and_then(|n| m.names.iter().position(|x| x == n));
        let (cx_, cy, cg) = (col(&x), col(&y), col(&group));
        // The groups (most rows first), at most MAX_GROUPS; each row's group
        let step = order.len().div_ceil(20_000).max(1);
        let mut names: Vec<String> = Vec::new();
        let mut gi: Vec<usize> = Vec::new();
        if let Some(c) = cg {
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for &r in order {
                *counts.entry(m.text[c][r].as_str()).or_insert(0) += 1;
            }
            let mut by: Vec<(&str, usize)> = counts.into_iter().collect();
            by.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
            names = by.iter().take(MAX_GROUPS).map(|(g, _)| if g.is_empty() { "(none)".to_string() } else { g.to_string() }).collect();
            if by.len() > MAX_GROUPS {
                names.push("other".into());
            }
            let index: HashMap<&str, usize> = by.iter().enumerate().map(|(i, (g, _))| (*g, i.min(MAX_GROUPS))).collect();
            gi = (0..m.rows()).map(|r| index.get(m.text[c][r].as_str()).copied().unwrap_or(0)).collect();
        }
        let ng = names.len().max(1);
        let g_of = |r: usize| if gi.is_empty() { 0 } else { gi[r] };
        let colors = if names.is_empty() { vec![color::accent()] } else { self.group_colors(group.as_deref(), &names, cx) };
        let num = |c: Option<usize>, r: usize| c.and_then(|c| m.num[c][r]).filter(|v| v.is_finite());
        // What the picture draws
        enum Data {
            Hist { lo: f64, hi: f64, counts: Vec<Vec<usize>> },
            Scatter { pts: Vec<(f64, f64, usize)> },
            Box { stats: Vec<Option<[f64; 5]>>, lo: f64, hi: f64 },
            Counts { counts: Vec<usize> },
        }
        let mut axis = String::new();
        let data = match v.kind {
            Plot::Histogram => {
                let vals: Vec<(f64, usize)> = order.iter().filter_map(|&r| num(cx_, r).map(|x| (x, g_of(r)))).collect();
                let (lo, hi) = vals.iter().fold((f64::MAX, f64::MIN), |a, &(v, _)| (a.0.min(v), a.1.max(v)));
                let bins = 30usize;
                let mut counts = vec![vec![0usize; ng]; bins];
                let span = (hi - lo).max(1e-12);
                for &(v, g) in &vals {
                    counts[(((v - lo) / span) * bins as f64).floor().clamp(0.0, bins as f64 - 1.0) as usize][g] += 1;
                }
                if !vals.is_empty() {
                    axis = format!("{} · {} … {}", x.clone().unwrap_or_default(), short_num(lo), short_num(hi));
                }
                Data::Hist { lo, hi, counts }
            }
            Plot::Scatter => {
                let pts: Vec<(f64, f64, usize)> = order.iter().step_by(step).filter_map(|&r| Some((num(cx_, r)?, num(cy, r)?, g_of(r)))).collect();
                if !pts.is_empty() {
                    let (xl, xh) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
                    let (yl, yh) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
                    axis = format!("x {} … {} · y {} … {}", short_num(xl), short_num(xh), short_num(yl), short_num(yh));
                }
                Data::Scatter { pts }
            }
            Plot::Box => {
                let mut per: Vec<Vec<f64>> = vec![Vec::new(); ng];
                for &r in order {
                    if let Some(v) = num(cy, r) {
                        per[g_of(r)].push(v);
                    }
                }
                let stats: Vec<Option<[f64; 5]>> = per
                    .iter_mut()
                    .map(|v| {
                        if v.is_empty() {
                            return None;
                        }
                        v.sort_by(|a, b| a.total_cmp(b));
                        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
                        Some([q(0.05), q(0.25), q(0.5), q(0.75), q(0.95)])
                    })
                    .collect();
                let (lo, hi) = stats.iter().flatten().fold((f64::MAX, f64::MIN), |a, s| (a.0.min(s[0]), a.1.max(s[4])));
                if lo <= hi {
                    axis = format!("{} · {} … {} (5–95 %, box 25–75 %, line median)", y.clone().unwrap_or_default(), short_num(lo), short_num(hi));
                }
                Data::Box { stats, lo, hi }
            }
            Plot::Counts => {
                let mut counts = vec![0usize; ng];
                for &r in order {
                    counts[g_of(r)] += 1;
                }
                axis = format!("up to {}", group_digits(counts.iter().copied().max().unwrap_or(0)));
                Data::Counts { counts }
            }
        };
        let colors_paint = colors.clone();
        let picture = canvas(
            |_, _, _| {},
            move |b, _, window, _| {
                window.paint_quad(fill(b, color::raised()));
                let (w, h) = (b.size.width.as_f32(), b.size.height.as_f32());
                let o = b.origin;
                let rect = |x: f32, y: f32, ww: f32, hh: f32| Bounds { origin: point(o.x + px(x), o.y + px(y)), size: size(px(ww.max(1.0)), px(hh.max(1.0))) };
                let col = |g: usize| colors_paint[g.min(colors_paint.len() - 1)];
                match &data {
                    Data::Hist { counts, lo, hi } => {
                        if lo > hi {
                            return;
                        }
                        let top = counts.iter().map(|c| c.iter().sum::<usize>()).max().unwrap_or(1).max(1) as f32;
                        let bw = w / counts.len() as f32;
                        for (i, c) in counts.iter().enumerate() {
                            // Stacked by group
                            let mut y0 = h;
                            for (g, &n) in c.iter().enumerate() {
                                if n == 0 {
                                    continue;
                                }
                                let bh = n as f32 / top * (h - 6.0);
                                y0 -= bh;
                                window.paint_quad(fill(rect(i as f32 * bw + 1.0, y0, bw - 2.0, bh), col(g)));
                            }
                        }
                    }
                    Data::Scatter { pts } => {
                        if pts.is_empty() {
                            return;
                        }
                        let (xl, xh) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
                        let (yl, yh) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
                        let (xs, ys) = ((xh - xl).max(1e-12), (yh - yl).max(1e-12));
                        for &(vx, vy, g) in pts {
                            let px_ = ((vx - xl) / xs) as f32 * (w - 6.0) + 3.0;
                            let py_ = h - 3.0 - ((vy - yl) / ys) as f32 * (h - 6.0);
                            window.paint_quad(fill(rect(px_ - 1.5, py_ - 1.5, 3.0, 3.0), col(g).opacity(0.75)));
                        }
                    }
                    Data::Box { stats, lo, hi } => {
                        if lo > hi {
                            return;
                        }
                        let span = (hi - lo).max(1e-12);
                        let yy = |v: f64| h - 4.0 - ((v - lo) / span) as f32 * (h - 8.0);
                        let bw = w / stats.len().max(1) as f32;
                        for (g, st) in stats.iter().enumerate() {
                            let Some([p5, q1, med, q3, p95]) = *st else { continue };
                            let cxm = g as f32 * bw + bw / 2.0;
                            let bx = bw * 0.3;
                            window.paint_quad(fill(rect(cxm - 0.5, yy(p95), 1.0, yy(p5) - yy(p95)), color::text_muted()));
                            window.paint_quad(fill(rect(cxm - bx, yy(q3), 2.0 * bx, yy(q1) - yy(q3)), col(g).opacity(0.8)));
                            window.paint_quad(fill(rect(cxm - bx, yy(med) - 1.0, 2.0 * bx, 2.0), color::text()));
                        }
                    }
                    Data::Counts { counts } => {
                        let top = counts.iter().copied().max().unwrap_or(1).max(1) as f32;
                        let bw = w / counts.len().max(1) as f32;
                        for (g, &n) in counts.iter().enumerate() {
                            let bh = n as f32 / top * (h - 6.0);
                            window.paint_quad(fill(rect(g as f32 * bw + bw * 0.15, h - bh, bw * 0.7, bh), col(g)));
                        }
                    }
                }
            },
        )
        .flex_1()
        .min_h_0()
        .w_full();
        let me = cx.entity().downgrade();
        let pick = |key: &'static str, options: Vec<String>, current: Option<String>, none: Option<&'static str>, set: fn(&mut PlotView, Option<String>)| {
            let me = me.clone();
            let mut labels: Vec<SharedString> = none.iter().map(|n| SharedString::from(*n)).collect();
            labels.extend(options.iter().map(|n| SharedString::from(n.clone())));
            let off = none.is_some() as usize;
            let sel = match &current {
                Some(c) => options.iter().position(|n| n == c).map(|i| i + off),
                None => none.map(|_| 0),
            };
            MenuSelect::new((key, id), labels, sel, move |i, _, cx| {
                let value = if i < off { None } else { options.get(i - off).cloned() };
                let _ = me.update(cx, |a, cx| a.edit_plot(id, |p| set(p, value), cx));
            })
        };
        let field = |label: &'static str, el: MenuSelect| h_flex().gap(px(theme::XS)).child(div().flex_none().text_xs().text_color(color::text_muted()).child(label)).child(div().w(px(120.)).child(el));
        let mut fields = h_flex().flex_wrap().gap_x(px(theme::MD)).gap_y(px(theme::XS));
        match v.kind {
            Plot::Histogram => {
                fields = fields.child(field("x", pick("plot-x", numeric.clone(), x.clone(), None, |p, v| p.x = v))).child(field("colour", pick("plot-g", texts.clone(), group.clone(), Some("none"), |p, v| p.group = v)));
            }
            Plot::Scatter => {
                fields = fields
                    .child(field("x", pick("plot-x", numeric.clone(), x.clone(), None, |p, v| p.x = v)))
                    .child(field("y", pick("plot-y", numeric.clone(), y.clone(), None, |p, v| p.y = v)))
                    .child(field("colour", pick("plot-g", texts.clone(), group.clone(), Some("none"), |p, v| p.group = v)));
            }
            Plot::Box => {
                fields = fields.child(field("value", pick("plot-y", numeric.clone(), y.clone(), None, |p, v| p.y = v))).child(field("by", pick("plot-g", texts.clone(), group.clone(), None, |p, v| p.group = v)));
            }
            Plot::Counts => {
                fields = fields.child(field("by", pick("plot-g", texts.clone(), group.clone(), None, |p, v| p.group = v)));
            }
        }
        // The legend (groups in their colours), when split by groups
        let legend = (!names.is_empty()).then(|| {
            h_flex().flex_wrap().gap_x(px(theme::SM)).gap_y(px(2.)).children(names.iter().zip(&colors).map(|(n, c)| {
                h_flex().gap(px(4.)).child(div().size(px(8.)).rounded(px(2.)).bg(*c)).child(div().text_xs().max_w(px(110.)).truncate().child(n.clone()))
            }))
        });
        v_flex()
            .size_full()
            .p(px(theme::MD))
            .gap(px(theme::XS))
            .bg(color::panel())
            .child(fields)
            .child(picture)
            .child(div().text_xs().text_color(color::text_muted()).truncate().child(axis))
            .children(legend)
            .into_any_element()
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
        // A project brought its own views; otherwise the views, when changed, are saved
        let (rev, saved) = (self.store.read(cx).layout_rev, self.store.read(cx).layout.analyze.clone());
        if rev != self.layout_seen {
            self.layout_seen = rev;
            self.layout_saved = Some(saved.clone());
            self.restore_layout(saved, window, cx);
        }
        let now = self.layout();
        if self.layout_saved.as_ref() != Some(&now) {
            self.layout_saved = Some(now.clone());
            self.store.update(cx, |s, _| s.set_layout(|l| l.analyze = now));
        }
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
        let img_now = s.doc.active_image.clone().unwrap_or_default();
        let n_cells = s.hist.cells.get(&img_now).map_or(0, Vec::len);
        let n_tiles = s.hist.tiles.get(&img_now).map_or(0, Vec::len);
        let valid = |t: &String| (t == CELLS && n_cells > 0) || (t == TILES && n_tiles > 0) || tables.iter().any(|(id, _)| id == t);
        if self.tab.as_ref().is_some_and(|t| !valid(t)) {
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
        let mut tabs = h_flex().gap(px(2.)).child(tab_button("analyze-tab-annotations".into(), IconName::Shapes, "Regions and shapes".into(), self.tab.is_none() && !self.show_result, None));
        if n_cells > 0 {
            tabs = tabs.child(tab_button("analyze-tab-cells".into(), IconName::Sparkles, format!("Cells · {}", group_digits(n_cells)), self.tab.as_deref() == Some(CELLS) && !self.show_result, Some(CELLS.to_string())));
        }
        if n_tiles > 0 {
            tabs = tabs.child(tab_button("analyze-tab-tiles".into(), IconName::Grid3x3, format!("Tiles · {}", group_digits(n_tiles)), self.tab.as_deref() == Some(TILES) && !self.show_result, Some(TILES.to_string())));
        }
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
            (None, _) => "regions".to_string(),
            (Some(v), _) if v == CELLS => "cells".to_string(),
            (Some(v), _) if v == TILES => "tiles".to_string(),
            (Some(v), _) => tables.iter().find(|(id, _)| id == v).map_or(v.clone(), |(_, n)| n.clone()),
        };
        let note = match (&self.tab, self.show_result) {
            (_, true) => "The result of the last step (the source table is unchanged).".to_string(),
            (None, _) => "The regions and shapes of the image shown in Explore, with their parent region and the cells they hold. Click a row to show it there.".to_string(),
            (Some(v), _) if v == CELLS || v == TILES => match &self.scope {
                Some(r) => format!("The {} of region {r} and the regions inside it (choose another in the hierarchy).", if v == CELLS { "cells" } else { "tiles" }),
                None => format!("Every {} of the image, with the region each belongs to.", if v == CELLS { "cell" } else { "tile" }),
            },
            (Some(_), _) => format!("Objects of the whole image, measured at pyramid level {}.", self.store.read(cx).analysis.as_ref().map_or(0, |a| a.level)),
        };
        let model = self.model(cx);
        let order_arc: Arc<Vec<usize>> = match &model {
            Some(m) => self.cached_order(m, cx),
            None => Arc::new(Vec::new()),
        };
        let order: &[usize] = order_arc.as_slice();
        let total = model.as_ref().map_or(0, |m| m.rows());
        let with_previews = !self.show_result && model.as_ref().is_some_and(|m| m.ids.is_some() || m.objects.is_some());
        self.shown = model.as_ref().map(|m| (m.clone(), order_arc.clone(), with_previews));
        self.note = note;
        let _ = (order, window);
        let toggle = |id: &'static str, icon: IconName, tip: &'static str, on: bool, f: fn(&mut AnalyzeView)| {
            let me = me.clone();
            Button::new(id).ghost().small().icon(icon).tooltip(tip).selected(on).on_click(move |_, _, cx| {
                let _ = me.update(cx, |a, cx| {
                    f(a);
                    cx.notify();
                });
            })
        };
        let left_toggle = toggle("analyze-toggle-hierarchy", IconName::PanelLeft, "Hierarchy", self.show_hierarchy, |a| a.show_hierarchy = !a.show_hierarchy);
        let right_toggle = toggle("analyze-toggle-side", IconName::PanelRight, "Selection, summary and steps", self.show_side, |a| a.show_side = !a.show_side);
        // Summary per class (annotations)
        let summary = (self.tab.is_none() && !self.show_result).then(|| {
            let s = self.store.read(cx);
            let img = s.doc.active_image.clone().unwrap_or_default();
            let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
            let mut sec = Section::new("Per class");
            let mut any = false;
            for c in &s.doc.classes {
                let list: Vec<_> = s.doc.annotations_of(&img).filter(|a| a.class == c.name).collect();
                if list.is_empty() {
                    continue;
                }
                any = true;
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
            any.then_some(sec)
        }).flatten();
        let has_selection = {
            let s = self.store.read(cx);
            !s.sel.shapes.is_empty() || !s.sel.objects.is_empty()
        };
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
        // The CSV is made when Export is pressed (never on a frame)
        let export = model.as_ref().map(|m| {
            let (me, title, m, order, n) = (me.clone(), title.clone(), m.clone(), order_arc.clone(), order.len());
            Button::new("analyze-export").outline().small().icon(IconName::Download).label("Export CSV").on_click(move |_, _, cx| {
                let csv = m.csv(&order);
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
        let hierarchy = self.hierarchy_pane(cx);
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
                    .child(left_toggle)
                    .child(tabs)
                    .child(div().flex_1())
                    .child(div().w(px(220.)).child(Input::new(&self.search).small()))
                    .child(div().text_xs().text_color(color::text_muted()).child(count))
                    .child(code_toggle)
                    .children(export)
                    .child(right_toggle),
            )
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .when(self.show_hierarchy, |d| d.child(v_flex().id("analyze-hierarchy").flex_none().w(px(240.)).h_full().overflow_y_scroll().p(px(theme::MD)).border_r_1().border_color(color::border()).child(hierarchy)))
                    // The views (table, plots) in Explore's container
                    .child(v_flex().flex_1().min_w_0().h_full().child(div().relative().flex_1().min_h_0().child(self.dock.clone()).children(self.dialog.as_ref().map(|(d, _)| d.clone()))).when(self.code_open, |d| d.child(self.code_pane(cx))))
                    .when(self.show_side, |d| d.child(v_flex().id("analyze-side").flex_none().w(px(300.)).h_full().overflow_y_scroll().p(px(theme::MD)).gap(px(theme::XL)).border_l_1().border_color(color::border()).children(has_selection.then(|| crate::screens::explore::panels::selection::summary(&self.store, cx))).children(summary).children(steps))),
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

/// What a panel of Analyze's dock shows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PanelKind {
    Table,
    Plot(usize),
}

/// A view of Analyze's dock (the table, or one plot): its content is drawn by the Analyze view,
/// which holds the table and the plots' settings.
pub struct AnalyzePanel {
    analyze: gpui_kit::WeakEntity<AnalyzeView>,
    kind: PanelKind,
    focus: gpui_kit::FocusHandle,
    watching: Option<Subscription>,
}

impl AnalyzePanel {
    fn new(analyze: gpui_kit::WeakEntity<AnalyzeView>, kind: PanelKind, cx: &mut Context<Self>) -> Self {
        Self { analyze, kind, focus: cx.focus_handle(), watching: None }
    }
}

impl Render for AnalyzePanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(a) = self.analyze.upgrade() else { return div().into_any_element() };
        // Drawn again whenever the table or the plots' settings change
        if self.watching.is_none() {
            self.watching = Some(cx.observe(&a, |_, _, cx| cx.notify()));
        }
        let kind = self.kind;
        let body = a.update(cx, |a, cx| match kind {
            PanelKind::Table => a.table_view(window, cx),
            PanelKind::Plot(id) => a.plot_view(id, cx),
        });
        div().id("analyze-panel").track_focus(&self.focus).size_full().bg(color::panel()).child(body).into_any_element()
    }
}

impl EventEmitter<PanelEvent> for AnalyzePanel {}

impl gpui_kit::Focusable for AnalyzePanel {
    fn focus_handle(&self, _: &gpui_kit::App) -> gpui_kit::FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for AnalyzePanel {
    fn panel_name(&self) -> &'static str {
        "lumen-app.analyze"
    }

    /// Plots close; the table only while a plot is left (a view always stays).
    fn closable(&self, cx: &gpui_kit::App) -> bool {
        match self.kind {
            PanelKind::Plot(_) => true,
            PanelKind::Table => self.analyze.upgrade().is_some_and(|a| !a.read(cx).plot_panels.is_empty()),
        }
    }

    fn on_removed(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let (a, kind) = (self.analyze.clone(), self.kind);
        cx.defer(move |cx| {
            let _ = a.update(cx, |a, cx| a.removed(kind, cx));
        });
    }
}

impl Panel for AnalyzePanel {
    fn zoom_control(&self, _: &gpui_kit::App) -> Option<gpui_kit::component::dock::PanelControl> {
        Some(gpui_kit::component::dock::PanelControl::Toolbar)
    }

    /// The shared tab buttons: a plot's kinds (a click switches this plot), the table's icon;
    /// then "+" (add a view or arrange them).
    fn toolbar_buttons(&mut self, _: &mut Window, cx: &mut Context<Self>) -> Option<Vec<Button>> {
        let a = self.analyze.clone();
        let add = {
            let a = a.clone();
            move |window: &mut Window, cx: &mut gpui_kit::App| {
                let _ = a.update(cx, |a, cx| a.open_dialog(window, cx));
            }
        };
        let _ = cx;
        Some(match self.kind {
            PanelKind::Table => container::tab_buttons("analyze-table", &[ViewOption::new("Table", IconName::Table2, "The rows, filtered, searched and sorted")], Some(0), |_, _, _| {}, add),
            PanelKind::Plot(id) => {
                let current = a.upgrade().and_then(|a| a.read(cx).plots.iter().find(|p| p.id == id).map(|p| p.kind)).and_then(|k| Plot::ALL.iter().position(|x| *x == k));
                let options: Vec<ViewOption> = Plot::ALL.iter().map(|k| ViewOption::new(k.title(), k.icon(), k.hint())).collect();
                container::tab_buttons(&format!("analyze-plot-{id}"), &options, current, move |i, _, cx| {
                    let _ = a.update(cx, |a, cx| a.edit_plot(id, |p| p.kind = Plot::ALL[i], cx));
                }, add)
            }
        })
    }

    fn title(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let text = match (self.kind, self.analyze.upgrade()) {
            (PanelKind::Table, _) => "Table".to_string(),
            (PanelKind::Plot(id), Some(a)) => a.read(cx).plot_title(id),
            _ => String::new(),
        };
        div().max_w(px(240.)).truncate().child(text)
    }
}

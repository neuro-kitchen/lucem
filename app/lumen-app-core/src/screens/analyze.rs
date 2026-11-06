//! The Analyze workspace: measurements as tables (Polars data frames) — the annotations of the
//! shown image (area, perimeter or length, centroid, in µm when the pixel size is known) and the
//! objects of every Measure result — with a summary per class and CSV export.

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window};
use polars::prelude::{Column, DataFrame, NamedFrom as _, Series};

use crate::data::results::centroid_l0;
use crate::data::store::{AppEvent, Store};
use crate::domain::graph::NodeKind;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{group_digits, EmptyState, Section};

/// Rows drawn at most (the export has them all).
const SHOWN: usize = 1000;

pub struct AnalyzeView {
    store: Entity<Store>,
    /// `None`: annotations; `Some(view)`: that view's objects.
    tab: Option<String>,
    _subs: Vec<Subscription>,
}

impl AnalyzeView {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::Results) {
                cx.notify();
            }
        })];
        Self { store, tab: None, _subs: subs }
    }

    /// The annotations of the shown image as a frame.
    fn annotations_frame(&self, cx: &Context<Self>) -> Option<DataFrame> {
        let s = self.store.read(cx);
        let img = s.doc.active_image.clone()?;
        let um = s.active_pyramid().and_then(|p| p.pixel_size_um);
        let (k, k2) = (um.unwrap_or(1.0), um.unwrap_or(1.0).powi(2));
        let rows: Vec<_> = s.doc.annotations_of(&img).collect();
        let (mut id, mut class, mut kind, mut area, mut length, mut cx_, mut cy_) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        for a in &rows {
            id.push(a.id.clone());
            class.push(a.class.clone());
            kind.push(a.shape.kind().to_string());
            area.push(a.shape.area().map(|v| v * k2));
            length.push(a.shape.length().map(|v| v * k));
            let (x0, y0, x1, y1) = a.shape.bounds();
            cx_.push((x0 + x1) / 2.0 * k);
            cy_.push((y0 + y1) / 2.0 * k);
        }
        let u = if um.is_some() { "um" } else { "px" };
        let cols = vec![
            Column::from(Series::new("id".into(), id)),
            Column::from(Series::new("class".into(), class)),
            Column::from(Series::new("shape".into(), kind)),
            Column::from(Series::new(format!("area_{u}2").into(), area)),
            Column::from(Series::new(format!("length_{u}").into(), length)),
            Column::from(Series::new(format!("x_{u}").into(), cx_)),
            Column::from(Series::new(format!("y_{u}").into(), cy_)),
        ];
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

    /// Writes `df` as CSV next to the project (or in the working folder).
    fn export(&self, df: &DataFrame, name: &str, cx: &mut Context<Self>) {
        let folder = self.store.read(cx).folder().or_else(|| std::env::current_dir().ok()).unwrap_or_default();
        let path = folder.join(format!("{}.csv", name.replace([' ', '/'], "_")));
        let cols = df.columns();
        let mut out = cols.iter().map(|c| c.name().to_string()).collect::<Vec<_>>().join(",");
        out.push('\n');
        for i in 0..df.height() {
            let row: Vec<String> = cols.iter().map(|c| c.get(i).map(|v| cell(&v.to_string())).unwrap_or_default()).collect();
            out.push_str(&row.join(","));
            out.push('\n');
        }
        let msg = match std::fs::write(&path, out) {
            Ok(()) => format!("Exported {} rows to {}", df.height(), path.display()),
            Err(e) => format!("Export failed: {e}"),
        };
        self.store.update(cx, |s, cx| s.set_status(msg, cx));
    }

    fn table(df: &DataFrame) -> impl IntoElement + use<> {
        let cols = df.columns();
        let w = 128.0;
        let header = h_flex().flex_none().bg(color::raised()).border_b_1().border_color(color::border()).children(cols.iter().map(|c| {
            div().flex_none().w(px(w)).px(px(theme::SM)).py(px(theme::XS)).truncate().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child(c.name().to_string())
        }));
        let rows = (0..df.height().min(SHOWN)).map(|i| {
            h_flex().flex_none().when(i % 2 == 1, |d| d.bg(gpui_kit::rgb(0x222222))).children(cols.iter().map(move |c| {
                let v = c.get(i).map(|v| short(&v.to_string())).unwrap_or_default();
                div().flex_none().w(px(w)).px(px(theme::SM)).py(px(3.)).truncate().text_sm().font_family("monospace").child(v)
            }))
        });
        v_flex().id("analyze-table").flex_1().min_h_0().overflow_scroll().child(header).children(rows)
    }
}

/// `"a"` stripped of quotes, numbers shortened.
fn short(s: &str) -> String {
    let s = s.trim_matches('"');
    match s.parse::<f64>() {
        Ok(v) if s.contains('.') => {
            if v.abs() >= 1000.0 { format!("{v:.1}") } else { format!("{v:.3}") }
        }
        _ => s.to_string(),
    }
}

/// A CSV cell (quoted when it holds a comma or quote).
fn cell(s: &str) -> String {
    let s = s.trim_matches('"');
    if s == "null" {
        return String::new();
    }
    if s.contains(',') || s.contains('"') { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

impl Render for AnalyzeView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
        let mut tabs = h_flex().gap(px(2.)).child({
            let me = me.clone();
            Button::new("analyze-tab-annotations").ghost().small().icon(IconName::Shapes).label("Annotations").selected(self.tab.is_none()).on_click(move |_, _, cx| {
                let _ = me.update(cx, |a, cx| {
                    a.tab = None;
                    cx.notify();
                });
            })
        });
        for (id, name) in &tables {
            let (me, id2) = (me.clone(), id.clone());
            tabs = tabs.child(Button::new(SharedString::from(format!("analyze-tab-{id}"))).ghost().small().icon(IconName::Grid2x2).label(name.clone()).selected(self.tab.as_deref() == Some(id.as_str())).on_click(move |_, _, cx| {
                let _ = me.update(cx, |a, cx| {
                    a.tab = Some(id2.clone());
                    cx.notify();
                });
            }));
        }
        let (df, title) = match &self.tab {
            None => (self.annotations_frame(cx), "annotations".to_string()),
            Some(v) => (self.objects_frame(v, cx), tables.iter().find(|(id, _)| id == v).map_or(v.clone(), |(_, n)| n.clone())),
        };
        let note = match &self.tab {
            None => "The annotations of the image shown in Explore.".to_string(),
            Some(_) => format!("Objects of the whole image, measured at pyramid level {} (Results in Explore says when it ran).", self.store.read(cx).analysis.as_ref().map_or(0, |a| a.level)),
        };
        // Summary per class (annotations)
        let summary = self.tab.is_none().then(|| {
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
                        .child(div().w(px(110.)).text_sm().child(c.name.clone()))
                        .child(div().w(px(70.)).text_sm().text_color(color::text_muted()).child(group_digits(list.len())))
                        .child(div().text_sm().text_color(color::text_muted()).child(if area > 0.0 { crate::ui::widgets::format_area(area, um) } else { String::new() })),
                );
            }
            sec
        });
        let export = df.clone().map(|df| {
            let (me, title) = (me.clone(), title.clone());
            Button::new("analyze-export").outline().small().icon(IconName::Download).label("Export CSV").on_click(move |_, _, cx| {
                let _ = me.update(cx, |a, cx| a.export(&df, &title, cx));
            })
        });
        let rows = df.as_ref().map_or(0, |d| d.height());
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
                    .child(div().text_xs().text_color(color::text_muted()).child(format!("{} rows{}", group_digits(rows), if rows > SHOWN { format!(" (first {SHOWN} shown)") } else { String::new() })))
                    .children(export),
            )
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .child(v_flex().flex_1().min_w_0().h_full().child(div().px(px(theme::MD)).py(px(theme::SM)).text_xs().text_color(color::text_muted()).child(note)).children(df.as_ref().map(Self::table)))
                    .children(summary.map(|s| v_flex().flex_none().w(px(300.)).h_full().p(px(theme::MD)).border_l_1().border_color(color::border()).child(s))),
            )
            .into_any_element()
    }
}

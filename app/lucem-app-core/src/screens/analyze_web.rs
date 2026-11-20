//! Browser **Analyze** workspace (table curation & inspection).
//!
//! Shows live in-session cell/region measurements alongside tabular measurement files served by
//! `lucem serve --web` (`/api/analyze/tables` and `/api/analyze/table`).

#![cfg(target_family = "wasm")]

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};
use kitchen_shell::MenuSpec;

use crate::data::server_api::{self, ServerTableData, ServerTableSummary};
use crate::data::store::{AppEvent, Store};
use crate::ui::theme::{self, color};

#[derive(Clone, PartialEq, Eq)]
enum Source {
    Annotations,
    Cells,
    Server(String),
}

#[derive(Clone)]
struct AnalyzeSideResize;

impl Render for AnalyzeSideResize {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
    }
}

pub struct AnalyzeWebView {
    store: Entity<Store>,
    source: Source,
    server_tables: Vec<ServerTableSummary>,
    loaded_server: Option<ServerTableData>,
    loading: bool,
    error: Option<String>,
    filter: Entity<InputState>,
    sort_col: Option<(usize, bool)>,
    sidebar_open: bool,
    sidebar_w: f32,
    _subs: Vec<Subscription>,
}

impl AnalyzeWebView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let filter = cx.new(|cx| InputState::new(window, cx).placeholder("Filter rows…"));
        let subs = vec![
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if !matches!(e, AppEvent::Status) {
                    cx.notify();
                }
            }),
            cx.subscribe(&filter, |_, _, e: &InputEvent, cx| {
                if matches!(e, InputEvent::Change) {
                    cx.notify();
                }
            }),
        ];
        let mut this = Self {
            store,
            source: Source::Annotations,
            server_tables: Vec::new(),
            loaded_server: None,
            loading: false,
            error: None,
            filter,
            sort_col: None,
            sidebar_open: true,
            sidebar_w: 240.0,
            _subs: subs,
        };
        if !crate::data::remote::runs_alone() {
            this.refresh_server_tables(cx);
        }
        this
    }

    pub fn menu_specs() -> Vec<MenuSpec> {
        vec![]
    }

    fn refresh_server_tables(&mut self, cx: &mut Context<Self>) {
        if crate::data::remote::runs_alone() {
            return;
        }
        self.loading = true;
        self.error = None;
        cx.spawn(async move |this, cx| {
            let res = server_api::list_server_tables().await;
            let _ = this.update(cx, |v, cx| {
                v.loading = false;
                match res {
                    Ok(list) => v.server_tables = list,
                    Err(e) => v.error = Some(e),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn load_server_table(&mut self, name: String, cx: &mut Context<Self>) {
        self.source = Source::Server(name.clone());
        self.loading = true;
        self.error = None;
        self.sort_col = None;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let res = server_api::fetch_server_table(&name).await;
            let _ = this.update(cx, |v, cx| {
                v.loading = false;
                match res {
                    Ok(data) => v.loaded_server = Some(data),
                    Err(e) => v.error = Some(e),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn current_table(&self, cx: &Context<Self>) -> (String, Vec<String>, Vec<Vec<String>>, usize) {
        let s = self.store.read(cx);
        match &self.source {
            Source::Annotations => {
                let cols = vec!["Name".into(), "Class".into(), "Kind".into(), "Area (px²)".into(), "Points".into()];
                let active_id = s.doc.active_image.as_deref();
                let rows: Vec<Vec<String>> = s
                    .doc
                    .annotations
                    .iter()
                    .filter(|a| active_id.is_none() || Some(a.image.as_str()) == active_id)
                    .map(|a| {
                        let class_name = if a.class.is_empty() { "Unclassified".into() } else { a.class.clone() };
                        let area = a.shape.area().map_or_else(|| "—".into(), |v| format!("{v:.1}"));
                        vec![
                            a.id.clone(),
                            class_name,
                            if a.shape.is_region() { "Region".into() } else { "Path/Point".into() },
                            area,
                            a.shape.outline().len().to_string(),
                        ]
                    })
                    .collect();
                let n = rows.len();
                ("Regions & shapes".into(), cols, rows, n)
            }
            Source::Cells => {
                let cols = vec!["Cell ID".into(), "Class".into(), "X (px)".into(), "Y (px)".into(), "Metrics".into()];
                let active_id = s.doc.active_image.as_deref().unwrap_or("");
                let cells = s.hist.cells.get(active_id);
                let rows: Vec<Vec<String>> = cells
                    .map(|list| {
                        list.iter()
                            .take(500)
                            .map(|c| {
                                vec![
                                    c.id.to_string(),
                                    c.class.clone().unwrap_or_else(|| "Unclassified".into()),
                                    format!("{:.1}", c.centroid[0]),
                                    format!("{:.1}", c.centroid[1]),
                                    c.measurements.len().to_string(),
                                ]
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let total = cells.map(|l| l.len()).unwrap_or(0);
                ("Detected cells".into(), cols, rows, total)
            }
            Source::Server(name) => match &self.loaded_server {
                Some(t) => (t.name.clone(), t.columns.clone(), t.rows.clone(), t.total_rows),
                None => (name.clone(), vec![], vec![], 0),
            },
        }
    }
}

impl Render for AnalyzeWebView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (title, cols, mut rows, total_rows) = self.current_table(cx);
        let q = self.filter.read(cx).value().trim().to_ascii_lowercase();
        if !q.is_empty() {
            rows.retain(|r| r.iter().any(|cell| cell.to_ascii_lowercase().contains(&q)));
        }
        if let Some((col, asc)) = self.sort_col {
            rows.sort_by(|a, b| {
                let va = a.get(col).map(String::as_str).unwrap_or("");
                let vb = b.get(col).map(String::as_str).unwrap_or("");
                let ord = match (va.parse::<f64>(), vb.parse::<f64>()) {
                    (Ok(na), Ok(nb)) => na.total_cmp(&nb),
                    _ => va.cmp(vb),
                };
                if asc { ord } else { ord.reverse() }
            });
        }
        let shown = rows.len();
        let alone = crate::data::remote::runs_alone();
        let src = self.source.clone();

        let mut sidebar = v_flex()
            .relative()
            .flex_none()
            .w(px(self.sidebar_w))
            .h_full()
            .p(px(theme::MD))
            .gap(px(theme::SM))
            .rounded(px(theme::RADIUS_LG))
            .bg(color::panel())
            .child(
                h_flex()
                    .items_center()
                    .child(div().flex_1().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child("SESSION TABLES"))
                    .child(
                        Button::new("an-side-collapse")
                            .ghost()
                            .xsmall()
                            .icon(Icon::new(IconName::PanelLeft))
                            .tooltip("Collapse tables sidebar")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.sidebar_open = false;
                                cx.notify();
                            })),
                    ),
            )
            .child(
                Button::new("an-src-ann")
                    .ghost()
                    .small()
                    .selected(src == Source::Annotations)
                    .icon(Icon::new(IconName::Shapes))
                    .label("Regions & shapes")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.source = Source::Annotations;
                        this.sort_col = None;
                        cx.notify();
                    })),
            )
            .child(
                Button::new("an-src-cells")
                    .ghost()
                    .small()
                    .selected(src == Source::Cells)
                    .icon(Icon::new(IconName::Table2))
                    .label("Detected cells")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.source = Source::Cells;
                        this.sort_col = None;
                        cx.notify();
                    })),
            )
            .child(div().h(px(1.)).w_full().bg(color::border()))
            .child(
                h_flex()
                    .items_center()
                    .child(div().flex_1().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child("SERVER TABLES"))
                    .when(!alone, |h| {
                        h.child(
                            Button::new("an-refresh")
                                .ghost()
                                .xsmall()
                                .icon(Icon::new(IconName::RotateCcw))
                                .tooltip("Refresh server tables")
                                .on_click(cx.listener(|this, _, _, cx| this.refresh_server_tables(cx))),
                        )
                    }),
            );

        if alone {
            sidebar = sidebar.child(div().text_xs().text_color(color::text_muted()).child("Connect to lucem serve --web or use the desktop app to query server CSV/Parquet tables."));
        } else if self.server_tables.is_empty() {
            sidebar = sidebar.child(div().text_xs().text_color(color::text_muted()).child("No CSV/TSV tables found in the server data folder yet."));
        } else {
            for (i, t) in self.server_tables.iter().enumerate() {
                let name = t.name.clone();
                let selected = src == Source::Server(name.clone());
                sidebar = sidebar.child(
                    Button::new(SharedString::from(format!("an-srv-{i}")))
                        .ghost()
                        .small()
                        .selected(selected)
                        .icon(Icon::new(IconName::Sheet))
                        .label(name.clone())
                        .on_click(cx.listener(move |this, _, _, cx| this.load_server_table(name.clone(), cx))),
                );
            }
        }

        sidebar = sidebar.child(
            div()
                .id("an-side-resize")
                .absolute()
                .right(px(-3.))
                .top_0()
                .h_full()
                .w(px(6.))
                .cursor(gpui_kit::CursorStyle::ResizeLeftRight)
                .on_drag(AnalyzeSideResize, |_, _, _, cx| cx.new(|_| AnalyzeSideResize)),
        );
        let header_row = h_flex()
            .flex_none()
            .w_full()
            .border_b_1()
            .border_color(color::border())
            .bg(color::control())
            .children(cols.iter().enumerate().map(|(ci, col)| {
                let marker = match self.sort_col {
                    Some((sc, true)) if sc == ci => " ↑",
                    Some((sc, false)) if sc == ci => " ↓",
                    _ => "",
                };
                div()
                    .id(SharedString::from(format!("an-col-{ci}")))
                    .flex_1()
                    .min_w(px(100.))
                    .px(px(theme::SM))
                    .py(px(6.))
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(color::text_muted())
                    .cursor_pointer()
                    .truncate()
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.sort_col = match this.sort_col {
                            Some((sc, true)) if sc == ci => Some((ci, false)),
                            _ => Some((ci, true)),
                        };
                        cx.notify();
                    }))
                    .child(format!("{col}{marker}"))
            }));

        let main = v_flex()
            .flex_1()
            .min_w_0()
            .h_full()
            .rounded(px(theme::RADIUS_LG))
            .overflow_hidden()
            .bg(color::panel())
            .child(
                h_flex()
                    .flex_none()
                    .px(px(theme::MD))
                    .py(px(theme::SM))
                    .gap(px(theme::MD))
                    .items_center()
                    .border_b_1()
                    .border_color(color::border())
                    .when(!self.sidebar_open, |h| {
                        h.child(
                            Button::new("an-side-expand")
                                .ghost()
                                .xsmall()
                                .icon(Icon::new(IconName::PanelLeft))
                                .tooltip("Show tables sidebar")
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.sidebar_open = true;
                                    cx.notify();
                                })),
                        )
                    })
                    .child(Icon::new(IconName::Table2).small().text_color(color::accent_mark()))
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).text_color(color::text()).child(title))
                    .child(div().text_xs().text_color(color::text_muted()).child(format!("{shown} / {total_rows} rows · {} cols", cols.len())))
                    .child(div().flex_1())
                    .child(div().w(px(220.)).child(Input::new(&self.filter).small())),
            );
        let cells_summary = (src == Source::Cells).then(|| {
            let s = self.store.read(cx);
            let active = s.doc.active_image.as_deref();
            let cells = active.and_then(|id| s.hist.cells.get(id));
            let um = active.and_then(|id| s.pyramids.get(id)).and_then(|p| p.pixel_size_um);
            let list = cells.filter(|l| !l.is_empty())?;
            let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            let mut by_class: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
            for c in list {
                x0 = x0.min(c.centroid[0]);
                y0 = y0.min(c.centroid[1]);
                x1 = x1.max(c.centroid[0]);
                y1 = y1.max(c.centroid[1]);
                *by_class.entry(c.class.clone().unwrap_or_else(|| "Unclassified".into())).or_default() += 1;
            }
            let (span_w, span_h) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));
            let (region_label, density_label) = match um {
                Some(u) if u > 0.0 => {
                    let w_mm = span_w * u / 1000.0;
                    let h_mm = span_h * u / 1000.0;
                    let area_mm2 = (w_mm * h_mm).max(1e-6);
                    (format!("{w_mm:.2} × {h_mm:.2} mm"), format!("{:.0} cells/mm²", list.len() as f64 / area_mm2))
                }
                _ => (format!("{span_w:.0} × {span_h:.0} px"), "—".to_string()),
            };
            let chips: Vec<gpui_kit::AnyElement> = by_class
                .into_iter()
                .take(6)
                .map(|(cls, n)| {
                    h_flex()
                        .px(px(8.))
                        .h(px(22.))
                        .rounded(px(999.))
                        .bg(color::panel())
                        .text_xs()
                        .text_color(color::text_muted())
                        .child(format!("{cls}: {n}"))
                        .into_any_element()
                })
                .collect();
            Some(
                v_flex()
                    .flex_none()
                    .mx(px(theme::MD))
                    .mt(px(theme::SM))
                    .p(px(theme::SM))
                    .gap(px(theme::XS))
                    .rounded(px(10.))
                    .bg(color::control())
                    .child(
                        h_flex()
                            .gap(px(theme::LG))
                            .flex_wrap()
                            .items_center()
                            .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text()).child(format!("Region LOD summary · {} cells", list.len())))
                            .child(div().text_xs().text_color(color::text_muted()).child(format!("Span: {region_label}")))
                            .child(div().text_xs().text_color(color::text_muted()).child(format!("Density: {density_label}"))),
                    )
                    .child(h_flex().gap(px(6.)).flex_wrap().children(chips)),
            )
        }).flatten();
        let main = main
            .children(cells_summary)
            .when(self.loading, |v| v.child(div().p(px(theme::LG)).text_xs().text_color(color::text_muted()).child("Loading table…")))
            .children(self.error.clone().map(|e| div().p(px(theme::MD)).text_xs().text_color(color::danger()).child(e)))
            .when(!self.loading && cols.is_empty(), |v| {
                v.child(div().p(px(theme::LG)).text_sm().text_color(color::text_muted()).child("Select a table on the left, or draw shapes / detect cells in Explore to curate measurements here."))
            })
            .when(!self.loading && !cols.is_empty(), |v| {
                v.child(header_row).child(
                    v_flex()
                        .id("an-table-body")
                        .flex_1()
                        .min_h_0()
                        .overflow_y_scroll()
                        .when(rows.is_empty(), |b| b.child(div().p(px(theme::LG)).text_xs().text_color(color::text_muted()).child("No rows match the current filter.")))
                        .children(rows.into_iter().enumerate().map(|(ri, row)| {
                            h_flex()
                                .w_full()
                                .border_b_1()
                                .border_color(color::border())
                                .when(ri % 2 == 1, |r| r.bg(color::canvas()))
                                .children(row.into_iter().map(|cell| {
                                    div()
                                        .flex_1()
                                        .min_w(px(100.))
                                        .px(px(theme::SM))
                                        .py(px(5.))
                                        .text_xs()
                                        .font_family("monospace")
                                        .text_color(color::text())
                                        .truncate()
                                        .child(cell)
                                }))
                        })),
                )
            });

        h_flex()
            .id("analyze-web-workspace")
            .size_full()
            .p(px(theme::PANE_GAP))
            .gap(px(theme::PANE_GAP))
            .bg(color::chrome())
            .on_drag_move(cx.listener(|this, e: &gpui_kit::DragMoveEvent<AnalyzeSideResize>, _, cx| {
                let w = f32::from(e.event.position.x - e.bounds.left()) - theme::PANE_GAP;
                this.sidebar_w = w.clamp(160., 420.);
                cx.notify();
            }))
            .when(self.sidebar_open, |h| h.child(sidebar))
            .child(main)
    }
}

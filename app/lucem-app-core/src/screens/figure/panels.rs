//! Figure composer UI sub-views (MVVM View layer): left panel list, center artboard, and right
//! inspector.

use std::collections::HashMap;
use std::sync::Arc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::Input;
use gpui_kit::component::scroll::ScrollableElement;
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Selectable as _, Sizable as _};
use gpui_kit::prelude::*;
use gpui_kit::{div, img, px, AnyElement, Context, FontWeight, ObjectFit, SharedString};
use lucem_io::Pyramid;

use crate::domain::figure::{panel_letter, panel_scale_bar, FigureBg, PanelSource, ScaleCorner};
use crate::screens::figure::FigureView;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{divider_v, icon_button, row, MenuSelect};

fn heading(title: impl Into<SharedString>) -> AnyElement {
    div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child(title.into()).into_any_element()
}

fn note(text: impl Into<SharedString>) -> AnyElement {
    div().text_xs().text_color(color::text_muted()).child(text.into()).into_any_element()
}

/// Top toolbar with presets, artboard background/column controls, and PNG / SVG export buttons.
pub fn top_toolbar(view: &FigureView, has_image: bool, cx: &mut Context<FigureView>) -> AnyElement {
    h_flex()
        .h(px(38.))
        .px(px(theme::MD))
        .gap(px(theme::SM))
        .items_center()
        .bg(color::panel())
        .border_b_1()
        .border_color(color::border())
        .child(
            Button::new("fig-preset-channels")
                .ghost()
                .xsmall()
                .icon(IconName::LayoutGrid)
                .label("All channels + Merge")
                .disabled(!has_image)
                .on_click(cx.listener(|this, _, _, cx| this.preset_all_channels(cx))),
        )
        .child(
            Button::new("fig-preset-stages")
                .ghost()
                .xsmall()
                .icon(IconName::Workflow)
                .label("Pipeline stages")
                .disabled(!has_image)
                .on_click(cx.listener(|this, _, _, cx| this.preset_pipeline_stages(cx))),
        )
        .child(
            Button::new("fig-add-panel")
                .ghost()
                .xsmall()
                .icon(IconName::Plus)
                .label("Add panel")
                .disabled(!has_image)
                .on_click(cx.listener(|this, _, _, cx| this.add_panel(cx))),
        )
        .child(divider_v())
        .child(
            h_flex()
                .gap(px(2.))
                .p(px(2.))
                .rounded(px(6.))
                .bg(color::control())
                .children([FigureBg::Dark, FigureBg::White, FigureBg::Black].into_iter().map(|bg| {
                    Button::new(SharedString::from(format!("fig-bg-{}", bg.label())))
                        .ghost()
                        .xsmall()
                        .selected(view.doc.bg == bg)
                        .label(bg.label())
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.doc.bg = bg;
                            cx.notify();
                        }))
                })),
        )
        .child(
            h_flex()
                .gap(px(2.))
                .p(px(2.))
                .rounded(px(6.))
                .bg(color::control())
                .children((1..=4).map(|c| {
                    Button::new(SharedString::from(format!("fig-cols-{c}")))
                        .ghost()
                        .xsmall()
                        .selected(view.doc.cols == c)
                        .label(format!("{c} col"))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.doc.cols = c;
                            cx.notify();
                        }))
                })),
        )
        .child(div().flex_1())
        .when_some(view.status_note.clone(), |d, msg| d.child(div().text_xs().text_color(color::text_muted()).child(msg)))
        .child(
            Button::new("fig-export-svg")
                .outline()
                .xsmall()
                .icon(IconName::FileCode)
                .label("Export SVG")
                .disabled(view.doc.panels.is_empty())
                .on_click(cx.listener(|this, _, _, cx| this.export_svg(cx))),
        )
        .child(
            Button::new("fig-export-png")
                .primary()
                .xsmall()
                .icon(IconName::Download)
                .label("Export PNG")
                .disabled(view.doc.panels.is_empty())
                .on_click(cx.listener(|this, _, _, cx| this.export_png(cx))),
        )
        .into_any_element()
}

/// Left sidebar: ordered panel list + one-click ROI crop sync.
pub fn left_sidebar(view: &FigureView, annotations: &[(String, String, [f64; 4])], cx: &mut Context<FigureView>) -> AnyElement {
    let mut panel_list = v_flex().gap(px(3.));
    let n_panels = view.doc.panels.len();
    for (idx, p) in view.doc.panels.iter().enumerate() {
        let is_sel = view.doc.selected == Some(idx);
        panel_list = panel_list.child(
            h_flex()
                .id(SharedString::from(format!("fig-panel-row-{}", p.id)))
                .px(px(8.))
                .py(px(5.))
                .rounded(px(6.))
                .gap(px(6.))
                .items_center()
                .bg(if is_sel { color::control() } else { gpui_kit::transparent_black() })
                .hover(|d| d.bg(color::control()))
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.doc.selected = Some(idx);
                    cx.notify();
                }))
                .child(div().w(px(18.)).text_xs().font_weight(FontWeight::BOLD).text_color(color::text_muted()).child(panel_letter(idx)))
                .child(
                    v_flex()
                        .flex_1()
                        .min_w_0()
                        .child(div().text_xs().truncate().child(p.title.clone()))
                        .child(div().text_size(px(10.)).text_color(color::text_muted()).truncate().child(p.region_label.clone())),
                )
                .child(
                    Button::new(SharedString::from(format!("fig-up-{}", p.id)))
                        .ghost()
                        .xsmall()
                        .label("↑")
                        .disabled(idx == 0)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            if idx > 0 && idx < this.doc.panels.len() {
                                this.doc.panels.swap(idx, idx - 1);
                                this.doc.selected = Some(idx - 1);
                                cx.notify();
                            }
                        })),
                )
                .child(
                    Button::new(SharedString::from(format!("fig-dn-{}", p.id)))
                        .ghost()
                        .xsmall()
                        .label("↓")
                        .disabled(idx + 1 >= n_panels)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            if idx + 1 < this.doc.panels.len() {
                                this.doc.panels.swap(idx, idx + 1);
                                this.doc.selected = Some(idx + 1);
                                cx.notify();
                            }
                        })),
                )
                .child(
                    icon_button(SharedString::from(format!("fig-del-{}", p.id)), IconName::Trash, "Remove panel").on_click(
                        cx.listener(move |this, _, _, cx| {
                            if idx < this.doc.panels.len() {
                                let removed = this.doc.panels.remove(idx);
                                this.renders.remove(&removed.id);
                                this.doc.selected = if this.doc.panels.is_empty() { None } else { Some(idx.min(this.doc.panels.len() - 1)) };
                                this.last_synced_panel = None;
                                cx.notify();
                            }
                        }),
                    ),
                ),
        );
    }

    let mut roi_chips = v_flex().gap(px(3.)).child(
        Button::new("fig-roi-all-full")
            .ghost()
            .xsmall()
            .label("Sync all → Full slide")
            .disabled(view.doc.panels.is_empty())
            .on_click(cx.listener(|this, _, _, cx| {
                for p in &mut this.doc.panels {
                    p.region = None;
                    p.region_label = "Full slide".into();
                }
                cx.notify();
            })),
    );
    for (aid, label, rect) in annotations {
        let label_c = label.clone();
        let rect_c = *rect;
        roi_chips = roi_chips.child(
            Button::new(SharedString::from(format!("fig-roi-all-{aid}")))
                .ghost()
                .xsmall()
                .label(format!("Sync all → {label}"))
                .disabled(view.doc.panels.is_empty())
                .on_click(cx.listener(move |this, _, _, cx| {
                    for p in &mut this.doc.panels {
                        p.region = Some(rect_c);
                        p.region_label = label_c.clone();
                    }
                    this.status_note = Some(format!("Synced all panels to {label_c}"));
                    cx.notify();
                })),
        );
    }

    v_flex()
        .w(px(240.))
        .h_full()
        .p(px(theme::MD))
        .gap(px(theme::MD))
        .bg(color::panel())
        .border_r_1()
        .border_color(color::border())
        .overflow_y_scrollbar()
        .child(heading("Panels"))
        .child(if view.doc.panels.is_empty() {
            note("Open an image, then click All channels + Merge or Add panel.").into_any_element()
        } else {
            panel_list.into_any_element()
        })
        .child(heading("Regions of Interest"))
        .child(roi_chips)
        .into_any_element()
}

/// Center publication artboard canvas.
pub fn center_artboard(view: &FigureView, pyramids: &HashMap<String, Arc<Pyramid>>, cx: &mut Context<FigureView>) -> AnyElement {
    let [br, bg_g, bb] = view.doc.bg.rgb();
    let artboard_bg = gpui_kit::rgba(((br as u32) << 24) | ((bg_g as u32) << 16) | ((bb as u32) << 8) | 0xff);
    let title_color = if view.doc.bg.is_light() { gpui_kit::rgba(0x111318ff) } else { gpui_kit::rgba(0xf3f5f8ff) };
    let sub_color = if view.doc.bg.is_light() { gpui_kit::rgba(0x5a6270ff) } else { gpui_kit::rgba(0x9aa3b2ff) };

    let cols = view.doc.cols.clamp(1, 4);
    let gap = px(view.doc.gap_px as f32);
    let mut grid = v_flex().gap(gap).w_full();
    for (row_idx, chunk) in view.doc.panels.chunks(cols).enumerate() {
        let mut row_el = h_flex().gap(gap).w_full().items_start();
        for (col_idx, p) in chunk.iter().enumerate() {
            let idx = row_idx * cols + col_idx;
            let is_sel = view.doc.selected == Some(idx);
            let rendered = view.renders.get(&p.id).map(|r| r.image.clone());
            let scale_info = p.show_scale_bar.then(|| panel_scale_bar(p, pyramids.get(p.source.image_id()))).flatten();

            let mut img_box = div()
                .id(SharedString::from(format!("fig-img-box-{}", p.id)))
                .relative()
                .w_full()
                .h(px(260.))
                .rounded(px(4.))
                .overflow_hidden()
                .bg(gpui_kit::rgba(0x090b0eff))
                .flex()
                .items_center()
                .justify_center()
                .cursor_grab()
                .on_scroll(cx.listener(move |this, ev: &gpui_kit::ScrollEvent, _, cx| {
                    let dy = f32::from(ev.delta.pixel_delta(px(24.)).y);
                    if dy.abs() > 0.1 {
                        this.doc.selected = Some(idx);
                        let factor = if dy > 0.0 { 0.8 } else { 1.25 };
                        this.zoom_panel(idx, factor, cx);
                        cx.stop_propagation();
                    }
                }))
                .on_mouse_down(gpui_kit::MouseButton::Left, cx.listener(move |this, ev: &gpui_kit::MouseDownEvent, _, cx| {
                    this.doc.selected = Some(idx);
                    this.drag_pan = Some((idx, ev.position));
                    cx.notify();
                }))
                .on_mouse_move(cx.listener(move |this, ev: &gpui_kit::MouseMoveEvent, _, cx| {
                    if let Some((drag_idx, last)) = this.drag_pan {
                        if drag_idx == idx && ev.pressed_button == Some(gpui_kit::MouseButton::Left) {
                            let dx = f32::from(last.x - ev.position.x) as f64 / 260.0;
                            let dy = f32::from(last.y - ev.position.y) as f64 / 260.0;
                            if dx.abs() > 0.005 || dy.abs() > 0.005 {
                                this.drag_pan = Some((idx, ev.position));
                                this.pan_panel(idx, dx, dy, cx);
                            }
                        } else if ev.pressed_button.is_none() {
                            this.drag_pan = None;
                        }
                    }
                }))
                .on_mouse_up(gpui_kit::MouseButton::Left, cx.listener(|this, _, _, _| {
                    this.drag_pan = None;
                }));

            if let Some(tex) = rendered {
                img_box = img_box.child(img(tex).w_full().h_full().object_fit(ObjectFit::Contain));
            } else {
                img_box = img_box.child(div().text_xs().text_color(color::text_muted()).child("Rendering…"));
            }

            if view.doc.show_letters {
                img_box = img_box.child(
                    div()
                        .absolute()
                        .top(px(8.))
                        .left(px(8.))
                        .px(px(6.))
                        .py(px(2.))
                        .rounded(px(4.))
                        .bg(gpui_kit::rgba(0x00000099))
                        .text_xs()
                        .font_weight(FontWeight::BOLD)
                        .text_color(gpui_kit::rgba(0xffffffff))
                        .child(panel_letter(idx)),
                );
            }

            if let Some((frac, label)) = scale_info {
                let bar_px = (frac * 240.0).clamp(18.0, 120.0) as f32;
                let sb = v_flex()
                    .gap(px(2.))
                    .px(px(6.))
                    .py(px(4.))
                    .rounded(px(4.))
                    .bg(gpui_kit::rgba(0x00000088))
                    .child(div().text_size(px(10.)).font_weight(FontWeight::SEMIBOLD).text_color(gpui_kit::rgba(0xffffffff)).child(label))
                    .child(div().w(px(bar_px)).h(px(3.)).rounded(px(1.)).bg(gpui_kit::rgba(0xffffffff)));
                let sb_pos = match p.scale_corner {
                    ScaleCorner::BottomLeft => div().absolute().bottom(px(8.)).left(px(8.)).child(sb),
                    ScaleCorner::BottomRight => div().absolute().bottom(px(8.)).right(px(8.)).child(sb),
                    ScaleCorner::TopLeft => div().absolute().top(px(8.)).left(px(34.)).child(sb),
                    ScaleCorner::TopRight => div().absolute().top(px(8.)).right(px(8.)).child(sb),
                };
                img_box = img_box.child(sb_pos);
            }

            let panel_card = v_flex()
                .id(SharedString::from(format!("fig-art-panel-{}", p.id)))
                .flex_1()
                .min_w_0()
                .gap(px(4.))
                .p(px(6.))
                .rounded(px(6.))
                .border_1()
                .border_color(if is_sel { color::text_muted() } else { gpui_kit::transparent_black() })
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.doc.selected = Some(idx);
                    cx.notify();
                }))
                .child(
                    v_flex()
                        .gap(px(1.))
                        .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).text_color(title_color).truncate().child(p.title.clone()))
                        .when(!p.subtitle.is_empty(), |d| d.child(div().text_xs().text_color(sub_color).truncate().child(p.subtitle.clone()))),
                )
                .child(img_box);

            row_el = row_el.child(panel_card);
        }
        for _ in chunk.len()..cols {
            row_el = row_el.child(div().flex_1());
        }
        grid = grid.child(row_el);
    }

    let artboard = v_flex()
        .w_full()
        .max_w(px(1100.))
        .p(px(24.))
        .gap(px(16.))
        .rounded(px(10.))
        .bg(artboard_bg)
        .border_1()
        .border_color(color::border())
        .when(!view.doc.title.is_empty() || !view.doc.subtitle.is_empty(), |d| {
            d.child(
                v_flex()
                    .gap(px(3.))
                    .when(!view.doc.title.is_empty(), |d| d.child(div().text_lg().font_weight(FontWeight::SEMIBOLD).text_color(title_color).child(view.doc.title.clone())))
                    .when(!view.doc.subtitle.is_empty(), |d| d.child(div().text_xs().text_color(sub_color).child(view.doc.subtitle.clone()))),
            )
        })
        .child(if view.doc.panels.is_empty() {
            div()
                .py(px(80.))
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(sub_color)
                .child("No panels yet — open an image and click All channels + Merge or Pipeline stages.")
                .into_any_element()
        } else {
            grid.into_any_element()
        });

    div()
        .flex_1()
        .h_full()
        .bg(color::canvas())
        .overflow_y_scrollbar()
        .p(px(24.))
        .flex()
        .justify_center()
        .items_start()
        .child(artboard)
        .into_any_element()
}

/// Right inspector: Figure settings + Selected Panel source, region, title/subtitle, and scale bar.
pub fn right_inspector(
    view: &FigureView,
    channel_names: &[String],
    stage_nodes: &[(String, String)],
    annotations: &[(String, String, [f64; 4])],
    cx: &mut Context<FigureView>,
) -> AnyElement {
    let mut inspector = v_flex()
        .w(px(290.))
        .h_full()
        .p(px(theme::MD))
        .gap(px(theme::MD))
        .bg(color::panel())
        .border_l_1()
        .border_color(color::border())
        .overflow_y_scrollbar()
        .child(heading("Figure"))
        .child(
            v_flex()
                .gap(px(theme::SM))
                .child(v_flex().gap(px(3.)).child(div().text_xs().text_color(color::text_muted()).child("Title")).child(Input::new(&view.fig_title_in).small()))
                .child(v_flex().gap(px(3.)).child(div().text_xs().text_color(color::text_muted()).child("Subtitle")).child(Input::new(&view.fig_sub_in).small()))
                .child(
                    Checkbox::new("fig-show-letters")
                        .checked(view.doc.show_letters)
                        .label("Panel letters (A, B, C…)")
                        .on_click(cx.listener(|this, on: &bool, _, cx| {
                            this.doc.show_letters = *on;
                            cx.notify();
                        })),
                ),
        );

    if let Some(sel_idx) = view.doc.selected.filter(|&i| i < view.doc.panels.len()) {
        let p = view.doc.panels[sel_idx].clone();
        let img_id = p.source.image_id().to_string();

        let mut src_labels: Vec<SharedString> = vec!["Composite (all visible)".into()];
        let mut src_values: Vec<PanelSource> = vec![PanelSource::Composite { image: img_id.clone() }];
        for (ci, cname) in channel_names.iter().enumerate() {
            src_labels.push(SharedString::from(format!("Channel · {cname}")));
            src_values.push(PanelSource::Channel { image: img_id.clone(), channel: ci });
        }
        for (nid, nlabel) in stage_nodes {
            src_labels.push(SharedString::from(format!("Stage · {nlabel}")));
            src_values.push(PanelSource::PipelineStage { image: img_id.clone(), node_id: nid.clone(), node_label: nlabel.clone() });
        }
        let cur_src = src_values.iter().position(|s| s == &p.source);
        let ch_names_c = channel_names.to_vec();
        let view_h = cx.entity().downgrade();
        let src_pick = MenuSelect::new("fig-panel-source", src_labels, cur_src, move |i, _, cx| {
            let chosen = src_values[i].clone();
            if let Some(v) = view_h.upgrade() {
                v.update(cx, |this, cx| {
                    if let Some(panel) = this.doc.panels.get_mut(sel_idx) {
                        panel.title = match &chosen {
                            PanelSource::Composite { .. } => "Composite".into(),
                            PanelSource::Channel { channel, .. } => ch_names_c.get(*channel).cloned().unwrap_or_else(|| format!("Channel {}", channel + 1)),
                            PanelSource::PipelineStage { node_label, .. } => node_label.clone(),
                        };
                        panel.source = chosen;
                        this.last_synced_panel = None;
                        cx.notify();
                    }
                });
            }
        });

        let mut reg_labels: Vec<SharedString> = vec!["Full slide".into()];
        let mut reg_values: Vec<(Option<[f64; 4]>, String)> = vec![(None, "Full slide".into())];
        for (_, label, rect) in annotations {
            reg_labels.push(SharedString::from(label.clone()));
            reg_values.push((Some(*rect), label.clone()));
        }
        let cur_reg = reg_values.iter().position(|(_, l)| l == &p.region_label).or(Some(0));
        let view_r = cx.entity().downgrade();
        let reg_pick = MenuSelect::new("fig-panel-region", reg_labels, cur_reg, move |i, _, cx| {
            let (rect, lbl) = reg_values[i].clone();
            if let Some(v) = view_r.upgrade() {
                v.update(cx, |this, cx| {
                    if let Some(panel) = this.doc.panels.get_mut(sel_idx) {
                        panel.region = rect;
                        panel.region_label = lbl;
                        cx.notify();
                    }
                });
            }
        });

        let sb_opts: [(Option<f64>, &str); 7] = [
            (None, "Auto"),
            (Some(25.0), "25 µm"),
            (Some(50.0), "50 µm"),
            (Some(100.0), "100 µm"),
            (Some(200.0), "200 µm"),
            (Some(500.0), "500 µm"),
            (Some(1000.0), "1 mm"),
        ];
        let cur_sb = sb_opts.iter().position(|(v, _)| *v == p.scale_bar_um).or(Some(0));
        let view_sb = cx.entity().downgrade();
        let sb_pick = MenuSelect::new(
            "fig-panel-sb-len",
            sb_opts.iter().map(|(_, l)| SharedString::from(*l)).collect(),
            cur_sb,
            move |i, _, cx| {
                let chosen = sb_opts[i].0;
                if let Some(v) = view_sb.upgrade() {
                    v.update(cx, |this, cx| {
                        if let Some(panel) = this.doc.panels.get_mut(sel_idx) {
                            panel.scale_bar_um = chosen;
                            cx.notify();
                        }
                    });
                }
            },
        );

        inspector = inspector.child(heading(format!("Panel {}", panel_letter(sel_idx)))).child(
            v_flex()
                .gap(px(theme::SM))
                .child(row("Source", src_pick))
                .child(row("Region", reg_pick))
                .child(row(
                    "Zoom / Pan",
                    h_flex()
                        .gap(px(3.))
                        .child(
                            Button::new("fig-zoom-out")
                                .outline()
                                .xsmall()
                                .icon(IconName::ZoomOut)
                                .tooltip("Zoom out (or scroll down on panel)")
                                .on_click(cx.listener(move |this, _, _, cx| this.zoom_panel(sel_idx, 1.35, cx))),
                        )
                        .child(
                            Button::new("fig-zoom-fit")
                                .outline()
                                .xsmall()
                                .label("Fit")
                                .tooltip("Reset to full slide")
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    if let Some(panel) = this.doc.panels.get_mut(sel_idx) {
                                        panel.region = None;
                                        panel.region_label = "Full slide".into();
                                        cx.notify();
                                    }
                                })),
                        )
                        .child(
                            Button::new("fig-zoom-in")
                                .outline()
                                .xsmall()
                                .icon(IconName::ZoomIn)
                                .tooltip("Zoom in (or scroll up on panel, drag to pan)")
                                .on_click(cx.listener(move |this, _, _, cx| this.zoom_panel(sel_idx, 0.72, cx))),
                        ),
                ))
                .child(
                    Button::new("fig-sync-this-crop")
                        .ghost()
                        .xsmall()
                        .icon(IconName::Link)
                        .label("Sync this crop to all panels")
                        .on_click(cx.listener(move |this, _, _, cx| this.sync_crop_to_all(sel_idx, cx))),
                )
                .child(v_flex().gap(px(3.)).child(div().text_xs().text_color(color::text_muted()).child("Panel title")).child(Input::new(&view.panel_title_in).small()))
                .child(v_flex().gap(px(3.)).child(div().text_xs().text_color(color::text_muted()).child("Panel subtitle")).child(Input::new(&view.panel_sub_in).small()))
                .child(
                    Checkbox::new("fig-panel-scale-bar")
                        .checked(p.show_scale_bar)
                        .label("Show scale bar")
                        .on_click(cx.listener(move |this, on: &bool, _, cx| {
                            if let Some(panel) = this.doc.panels.get_mut(sel_idx) {
                                panel.show_scale_bar = *on;
                                cx.notify();
                            }
                        })),
                )
                .when(p.show_scale_bar, |d| {
                    d.child(row("Length", sb_pick)).child(row(
                        "Corner",
                        h_flex().gap(px(2.)).p(px(2.)).rounded(px(6.)).bg(color::control()).children(ScaleCorner::ALL.into_iter().map(|corner| {
                            Button::new(SharedString::from(format!("fig-corner-{}", corner.short())))
                                .ghost()
                                .xsmall()
                                .selected(p.scale_corner == corner)
                                .label(corner.short())
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    if let Some(panel) = this.doc.panels.get_mut(sel_idx) {
                                        panel.scale_corner = corner;
                                        cx.notify();
                                    }
                                }))
                        })),
                    ))
                }),
        );
    }

    inspector.into_any_element()
}

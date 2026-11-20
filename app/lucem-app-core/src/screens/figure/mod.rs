//! Publication **Figure** composer workspace (MVVM ViewModel / orchestrator).
//!
//! Split cleanly along MVVM lines:
//! - Model: `crate::domain::figure` (`FigureDoc`, `FigurePanel`, `PanelSource`, scale-bar math)
//! - Render & Export engine: `export.rs` (async crop rasterization + PNG/SVG export)
//! - Sub-views: `panels.rs` (top toolbar, panel list, center artboard, right inspector)

pub mod export;
mod panels;

use std::collections::HashMap;
use std::sync::Arc;

use gpui_kit::component::input::{InputEvent, InputState};
use gpui_kit::component::{h_flex, v_flex};
use gpui_kit::prelude::*;
use gpui_kit::{Context, Entity, Render, Subscription, Task, Window};
use kitchen_shell::MenuSpec;
use lucem_io::Pyramid;

use crate::data::store::{AppEvent, Store};
use crate::domain::figure::{node_stage_label, pan_panel_region, panel_render_key, sanitize_filename, zoom_panel_region, FigureDoc, FigurePanel, PanelSource, ScaleCorner};
use crate::domain::project::ChannelDoc;
use crate::screens::figure::export::{compose_artboard_rgb, compose_artboard_svg, render_image_to_rgb, render_panel_rgb, save_or_download, RenderedPanel};

/// Publication Figure Composer workspace view-model.
pub struct FigureView {
    pub(crate) store: Entity<Store>,
    pub(crate) doc: FigureDoc,
    pub(crate) renders: HashMap<usize, RenderedPanel>,
    inflight: HashMap<usize, String>,
    render_tasks: HashMap<usize, Task<()>>,
    pub(crate) fig_title_in: Entity<InputState>,
    pub(crate) fig_sub_in: Entity<InputState>,
    pub(crate) panel_title_in: Entity<InputState>,
    pub(crate) panel_sub_in: Entity<InputState>,
    pub(crate) last_synced_panel: Option<usize>,
    pub(crate) status_note: Option<String>,
    /// Active drag-to-pan state on an artboard panel: `(panel_idx, last_pos)`.
    pub(crate) drag_pan: Option<(usize, gpui_kit::Point<gpui_kit::Pixels>)>,
    _subs: Vec<Subscription>,
}

impl FigureView {
    pub fn menu_specs() -> Vec<MenuSpec> {
        vec![]
    }

    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let fig_title_in = cx.new(|cx| InputState::new(window, cx).placeholder("Figure title (e.g. Multiplex Overview)"));
        let fig_sub_in = cx.new(|cx| InputState::new(window, cx).placeholder("Subtitle / caption (optional)"));
        let panel_title_in = cx.new(|cx| InputState::new(window, cx).placeholder("Panel title"));
        let panel_sub_in = cx.new(|cx| InputState::new(window, cx).placeholder("Panel subtitle (e.g. 405 nm · nuclei)"));

        let mut subs = vec![cx.subscribe(&store, |this, _, _: &AppEvent, cx| {
            this.ensure_default_panels(cx);
            cx.notify();
        })];
        subs.push(cx.subscribe(&fig_title_in, |this, inp, ev: &InputEvent, cx| {
            if matches!(ev, InputEvent::Change) {
                this.doc.title = inp.read(cx).value().to_string();
                cx.notify();
            }
        }));
        subs.push(cx.subscribe(&fig_sub_in, |this, inp, ev: &InputEvent, cx| {
            if matches!(ev, InputEvent::Change) {
                this.doc.subtitle = inp.read(cx).value().to_string();
                cx.notify();
            }
        }));
        subs.push(cx.subscribe(&panel_title_in, |this, inp, ev: &InputEvent, cx| {
            if matches!(ev, InputEvent::Change) {
                if let Some(sel) = this.doc.selected.and_then(|i| this.doc.panels.get_mut(i)) {
                    sel.title = inp.read(cx).value().to_string();
                    cx.notify();
                }
            }
        }));
        subs.push(cx.subscribe(&panel_sub_in, |this, inp, ev: &InputEvent, cx| {
            if matches!(ev, InputEvent::Change) {
                if let Some(sel) = this.doc.selected.and_then(|i| this.doc.panels.get_mut(i)) {
                    sel.subtitle = inp.read(cx).value().to_string();
                    cx.notify();
                }
            }
        }));

        let mut view = Self {
            store,
            doc: FigureDoc::default(),
            renders: HashMap::new(),
            inflight: HashMap::new(),
            render_tasks: HashMap::new(),
            fig_title_in,
            fig_sub_in,
            panel_title_in,
            panel_sub_in,
            last_synced_panel: None,
            status_note: None,
            drag_pan: None,
            _subs: subs,
        };
        view.fig_title_in.update(cx, |i, cx| i.set_value("Figure 1", window, cx));
        view.ensure_default_panels(cx);
        view
    }

    fn ensure_default_panels(&mut self, cx: &mut Context<Self>) {
        if self.doc.panels.is_empty() && self.store.read(cx).doc.active_image.is_some() {
            self.preset_all_channels(cx);
        }
    }

    pub(crate) fn preset_all_channels(&mut self, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let Some(img_id) = s.doc.active_image.clone() else { return };
        let Some(img_doc) = s.doc.image(&img_id) else { return };
        let channels: Vec<(usize, String)> = img_doc.channels.iter().enumerate().map(|(i, c)| (i, c.label.clone())).collect();
        let _ = s;

        self.doc.panels.clear();
        self.renders.clear();
        self.inflight.clear();
        for (idx, name) in channels {
            let id = self.doc.alloc_id();
            self.doc.panels.push(FigurePanel {
                id,
                title: name,
                subtitle: String::new(),
                source: PanelSource::Channel { image: img_id.clone(), channel: idx },
                region: None,
                region_label: "Full slide".into(),
                show_scale_bar: true,
                scale_bar_um: None,
                scale_corner: ScaleCorner::BottomLeft,
            });
        }
        let merge_id = self.doc.alloc_id();
        self.doc.panels.push(FigurePanel {
            id: merge_id,
            title: "Composite".into(),
            subtitle: "All visible channels".into(),
            source: PanelSource::Composite { image: img_id },
            region: None,
            region_label: "Full slide".into(),
            show_scale_bar: true,
            scale_bar_um: None,
            scale_corner: ScaleCorner::BottomLeft,
        });
        self.doc.cols = self.doc.panels.len().clamp(1, 3);
        self.doc.selected = (!self.doc.panels.is_empty()).then_some(0);
        self.last_synced_panel = None;
        self.status_note = Some(format!("Built {} channel + composite panels", self.doc.panels.len()));
        cx.notify();
    }

    pub(crate) fn preset_pipeline_stages(&mut self, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let Some(img_id) = s.doc.active_image.clone() else { return };
        let stages: Vec<(String, String)> = s
            .doc
            .graph
            .nodes
            .iter()
            .filter(|n| s.analysis.as_ref().is_some_and(|a| a.previews.contains_key(&n.id)))
            .map(|n| (n.id.clone(), node_stage_label(&n.kind)))
            .collect();
        let _ = s;

        self.doc.panels.clear();
        self.renders.clear();
        self.inflight.clear();
        let raw_id = self.doc.alloc_id();
        self.doc.panels.push(FigurePanel {
            id: raw_id,
            title: "Input".into(),
            subtitle: "Raw composite".into(),
            source: PanelSource::Composite { image: img_id.clone() },
            region: None,
            region_label: "Full slide".into(),
            show_scale_bar: true,
            scale_bar_um: None,
            scale_corner: ScaleCorner::BottomLeft,
        });
        for (nid, label) in stages {
            let id = self.doc.alloc_id();
            self.doc.panels.push(FigurePanel {
                id,
                title: label.clone(),
                subtitle: "Pipeline stage".into(),
                source: PanelSource::PipelineStage { image: img_id.clone(), node_id: nid, node_label: label },
                region: None,
                region_label: "Full slide".into(),
                show_scale_bar: true,
                scale_bar_um: None,
                scale_corner: ScaleCorner::BottomLeft,
            });
        }
        self.doc.cols = self.doc.panels.len().clamp(1, 3);
        self.doc.selected = (!self.doc.panels.is_empty()).then_some(0);
        self.last_synced_panel = None;
        self.status_note = Some(format!("Added {} processing stage panels", self.doc.panels.len()));
        cx.notify();
    }

    pub(crate) fn zoom_panel(&mut self, idx: usize, factor: f64, cx: &mut Context<Self>) {
        let Some(img_id) = self.doc.panels.get(idx).map(|p| p.source.image_id().to_string()) else { return };
        let Some(pyr) = self.store.read(cx).pyramids.get(&img_id).cloned() else { return };
        if let Some(p) = self.doc.panels.get_mut(idx) {
            zoom_panel_region(p, factor, pyr.width() as f64, pyr.height() as f64);
            cx.notify();
        }
    }

    pub(crate) fn pan_panel(&mut self, idx: usize, dx_frac: f64, dy_frac: f64, cx: &mut Context<Self>) {
        let Some(img_id) = self.doc.panels.get(idx).map(|p| p.source.image_id().to_string()) else { return };
        let Some(pyr) = self.store.read(cx).pyramids.get(&img_id).cloned() else { return };
        if let Some(p) = self.doc.panels.get_mut(idx) {
            pan_panel_region(p, dx_frac, dy_frac, pyr.width() as f64, pyr.height() as f64);
            cx.notify();
        }
    }

    pub(crate) fn sync_crop_to_all(&mut self, idx: usize, cx: &mut Context<Self>) {
        let Some((reg, lbl)) = self.doc.panels.get(idx).map(|p| (p.region, p.region_label.clone())) else { return };
        for p in &mut self.doc.panels {
            p.region = reg;
            p.region_label = lbl.clone();
        }
        self.status_note = Some(format!("Synced all panels to {lbl}"));
        cx.notify();
    }

    pub(crate) fn add_panel(&mut self, cx: &mut Context<Self>) {
        let Some(img_id) = self.store.read(cx).doc.active_image.clone() else { return };
        let id = self.doc.alloc_id();
        self.doc.panels.push(FigurePanel {
            id,
            title: "Composite".into(),
            subtitle: String::new(),
            source: PanelSource::Composite { image: img_id },
            region: None,
            region_label: "Full slide".into(),
            show_scale_bar: true,
            scale_bar_um: None,
            scale_corner: ScaleCorner::BottomLeft,
        });
        self.doc.selected = Some(self.doc.panels.len() - 1);
        self.last_synced_panel = None;
        cx.notify();
    }

    fn sync_panel_inputs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let cur_id = self.doc.selected.and_then(|i| self.doc.panels.get(i)).map(|p| p.id);
        if cur_id == self.last_synced_panel {
            return;
        }
        self.last_synced_panel = cur_id;
        let (title, sub) = self.doc.selected.and_then(|i| self.doc.panels.get(i)).map(|p| (p.title.clone(), p.subtitle.clone())).unwrap_or_default();
        self.panel_title_in.update(cx, |i, cx| i.set_value(title, window, cx));
        self.panel_sub_in.update(cx, |i, cx| i.set_value(sub, window, cx));
    }

    fn sync_panel_renders(&mut self, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let z = 0usize;
        let mut to_render: Vec<(usize, String, PanelSource, Option<[f64; 4]>, Arc<Pyramid>, Vec<ChannelDoc>, usize)> = Vec::new();
        for p in &self.doc.panels {
            let img_id = p.source.image_id();
            let Some(pyr) = s.pyramids.get(img_id).cloned() else { continue };
            let Some(img_doc) = s.doc.image(img_id) else { continue };
            let channels_doc = img_doc.channels.clone();
            let key = panel_render_key(p, &channels_doc, z);
            if !self.renders.get(&p.id).is_some_and(|r| r.key == key) && !self.inflight.get(&p.id).is_some_and(|k| k == &key) {
                to_render.push((p.id, key, p.source.clone(), p.region, pyr, channels_doc, z));
            }
        }
        let _ = s;

        for (pid, key, source, region, pyr, channels_doc, z) in to_render {
            if let PanelSource::PipelineStage { ref node_id, .. } = source {
                if let Some(preview_img) = self.store.read(cx).analysis.as_ref().and_then(|a| a.previews.get(node_id).cloned()) {
                    let (w, h, rgb) = render_image_to_rgb(&preview_img);
                    self.renders.insert(pid, RenderedPanel { key, image: preview_img, rgb: Arc::new(rgb), w, h });
                    continue;
                }
            }
            self.inflight.insert(pid, key.clone());
            let work = cx.background_spawn(async move { render_panel_rgb(&pyr, &channels_doc, &source, region, z) });
            let task = cx.spawn(async move |this, cx| {
                let res = work.await;
                let _ = this.update(cx, |v, cx| {
                    v.inflight.remove(&pid);
                    if let Some((w, h, rgb)) = res {
                        let mut bgra = Vec::with_capacity(w * h * 4);
                        for px in rgb.chunks_exact(3) {
                            bgra.extend_from_slice(&[px[2], px[1], px[0], 255]);
                        }
                        let image = lucem_view::bgra_image(w, h, bgra);
                        v.renders.insert(pid, RenderedPanel { key, image, rgb: Arc::new(rgb), w, h });
                        cx.notify();
                    }
                });
            });
            self.render_tasks.insert(pid, task);
        }
    }

    pub(crate) fn export_png(&mut self, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let Some((w, h, rgb)) = compose_artboard_rgb(&self.doc, &self.renders, &s.pyramids) else {
            self.status_note = Some("Wait for panels to finish rendering before exporting.".into());
            cx.notify();
            return;
        };
        let folder = s.folder();
        let _ = s;
        let mut png = Vec::new();
        if let Err(e) = image::ImageEncoder::write_image(image::codecs::png::PngEncoder::new(&mut png), &rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8) {
            self.status_note = Some(format!("PNG encode failed: {e}"));
            cx.notify();
            return;
        }
        let filename = sanitize_filename(&self.doc.title, "png");
        match save_or_download(&filename, "image/png", &png, folder) {
            Ok(msg) => {
                self.status_note = Some(msg.clone());
                self.store.update(cx, |s, cx| s.set_status(msg, cx));
            }
            Err(e) => self.status_note = Some(e),
        }
        cx.notify();
    }

    pub(crate) fn export_svg(&mut self, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        let Some(svg) = compose_artboard_svg(&self.doc, &self.renders, &s.pyramids) else {
            self.status_note = Some("Wait for panels to finish rendering before exporting.".into());
            cx.notify();
            return;
        };
        let folder = s.folder();
        let _ = s;
        let filename = sanitize_filename(&self.doc.title, "svg");
        match save_or_download(&filename, "image/svg+xml;charset=utf-8", svg.as_bytes(), folder) {
            Ok(msg) => {
                self.status_note = Some(msg.clone());
                self.store.update(cx, |s, cx| s.set_status(msg, cx));
            }
            Err(e) => self.status_note = Some(e),
        }
        cx.notify();
    }
}

impl Render for FigureView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.sync_panel_inputs(window, cx);
        self.sync_panel_renders(cx);

        let s = self.store.read(cx);
        let active_img = s.doc.active_image.clone();
        let has_image = active_img.is_some();
        let annotations: Vec<(String, String, [f64; 4])> = active_img
            .as_ref()
            .map(|img_id| {
                s.doc
                    .annotations_of(img_id)
                    .map(|a| {
                        let (x0, y0, x1, y1) = a.shape.bounds();
                        (a.id.clone(), format!("{} {} ({})", a.shape.kind(), a.id, a.class), [x0, y0, (x1 - x0).max(1.0), (y1 - y0).max(1.0)])
                    })
                    .collect()
            })
            .unwrap_or_default();
        let channel_names: Vec<String> = active_img
            .as_ref()
            .and_then(|id| s.doc.image(id))
            .map(|img| img.channels.iter().map(|c| c.label.clone()).collect())
            .unwrap_or_default();
        let stage_nodes: Vec<(String, String)> = s
            .doc
            .graph
            .nodes
            .iter()
            .filter(|n| s.analysis.as_ref().is_some_and(|a| a.previews.contains_key(&n.id)))
            .map(|n| (n.id.clone(), node_stage_label(&n.kind)))
            .collect();
        let pyramids = s.pyramids.clone();
        let _ = s;

        v_flex()
            .size_full()
            .child(panels::top_toolbar(self, has_image, cx))
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .child(panels::left_sidebar(self, &annotations, cx))
                    .child(panels::center_artboard(self, &pyramids, cx))
                    .child(panels::right_inspector(self, &channel_names, &stage_nodes, &annotations, cx)),
            )
    }
}

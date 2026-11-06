//! Results of the graph, in two passes (as a classifier is previewed on a constrained pyramid
//! level and runs it on the whole image):
//!
//! 1. **Analysis** ([`analyze`]): the whole image at a capped level (the finest whose longer side
//!    is ≤ [`ANALYSIS_SIDE`]). Every node is computed once there; it gives each node's preview (the
//!    thumbnails on the Pipeline canvas), the values ops need from the *whole* image (Otsu
//!    thresholds), and the tables of Measure results (every object of the image, stable while
//!    panning). It runs again only when the graph, the annotations or the image change.
//! 2. **View** ([`compute`]): each raster view over the visible region (plus the margin its ops
//!    need) at the level the zoom asks for, with the whole-image values fixed, so what is drawn
//!    agrees with the analysis at every zoom.
//!
//! Op sizes are level-0 pixels and are scaled to the level they run on.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use gpui_kit::RenderImage;
use lumen_core::cubecl::prelude::Client;
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;
use lumen_io::Pyramid;
use lumen_ops::measure::RegionFeatures;
use lumen_pipeline::{at_downsample, evaluate, evaluate_all, halo, Op, Plan, PortType, Value};
use lumen_view::{render_layer, ColormapLut, LayerStyle};

use crate::domain::annotation::rasterize;
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::domain::project::ProjectDoc;

/// Largest side of a region computed for the view (px at its level).
pub const MAX_SIDE: usize = 2048;
/// Longer side of the level the whole-image analysis runs on.
pub const ANALYSIS_SIDE: usize = 2048;
/// Longer side of a node preview.
pub const PREVIEW_SIDE: usize = 192;

/// What is on screen: an image and the visible level-0 rectangle at a zoom.
#[derive(Debug, Clone, PartialEq)]
pub struct ResultRequest {
    pub image: String,
    /// `x, y, w, h` in level-0 px.
    pub rect: [f64; 4],
    /// Screen px per level-0 px (times the display's scale).
    pub zoom: f64,
}

/// One view's layer over the visible region.
#[derive(Clone)]
pub struct LayerResult {
    pub view: String,
    pub image: String,
    /// Where the layer sits, level-0 px (`x, y, w, h`).
    pub rect: [f64; 4],
    pub layer: Option<Arc<RenderImage>>,
    pub port: Option<PortType>,
    pub error: Option<String>,
    pub ms: f64,
    pub level: usize,
}

/// A Measure result over the whole image.
#[derive(Clone)]
pub struct TableResult {
    pub rows: Arc<Vec<RegionFeatures>>,
    /// Level-0 px per pixel of the analysis level (object coordinates are in its pixels).
    pub downsample: f64,
}

/// The whole-image pass.
#[derive(Clone, Default)]
pub struct Analysis {
    pub key: u64,
    pub image: String,
    pub level: usize,
    pub downsample: f64,
    /// Each node's result, small.
    pub previews: HashMap<String, Arc<RenderImage>>,
    /// Why a node has no result.
    pub errors: HashMap<String, String>,
    /// Threshold of each Otsu node, on the whole image.
    pub thresholds: HashMap<String, f64>,
    /// Tables of views fed by a Measure.
    pub tables: HashMap<String, TableResult>,
    pub ms: f64,
}

/// What the analysis of `image` depends on: the graph, the image's annotations, the image.
pub fn analysis_key(doc: &ProjectDoc, image: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    image.hash(&mut h);
    serde_json::to_string(&doc.graph).unwrap_or_default().hash(&mut h);
    for a in doc.annotations_of(image) {
        serde_json::to_string(a).unwrap_or_default().hash(&mut h);
    }
    h.finish()
}

/// Nodes that read `image` (directly or upstream).
fn nodes_of(doc: &ProjectDoc, image: &str) -> Vec<String> {
    doc.graph.nodes.iter().filter(|n| doc.graph.images_of(&n.id).contains(image)).map(|n| n.id.clone()).collect()
}

/// The finest level whose longer side is ≤ `side` (else the coarsest).
fn level_under(py: &Pyramid, side: usize) -> usize {
    (0..py.levels.len()).find(|&l| py.levels[l].width.max(py.levels[l].height) <= side).unwrap_or(py.coarsest())
}

/// A source node's raster over a region of `level`.
#[allow(clippy::too_many_arguments)]
fn source_value(client: &Client, doc: &ProjectDoc, py: &Pyramid, image: &str, id: &str, level: usize, (x0, y0, w, h): (usize, usize, usize, usize)) -> Result<Value, String> {
    let node = doc.graph.node(id).ok_or_else(|| format!("no node {id}"))?;
    let ds = py.levels[level].downsample;
    match &node.kind {
        NodeKind::Source { image: img, channel } => {
            if img != image {
                return Err(format!("{id} reads another image ({img})"));
            }
            let mut values = py.read_region(level, *channel, x0, y0, w, h).map_err(|e| e.to_string())?;
            // Raw values to 0–1 by the type's range (thresholds and gamma mean the same on 8- and
            // 16-bit images)
            let top = py.channels.get(*channel).map_or(1.0, |c| c.range.1);
            if top > 1.0 {
                let k = 1.0 / top;
                values.iter_mut().for_each(|v| *v *= k);
            }
            DeviceArray::from_host(client, Shape::new([h, w]), values).map(Value::Image).map_err(|e| e.to_string())
        }
        NodeKind::Annotations { image: img, class } => {
            if img != image {
                return Err(format!("{id} reads another image ({img})"));
            }
            let anns: Vec<_> = doc.annotations_of(image).collect();
            let mask = rasterize(&anns, class.as_deref(), x0, y0, w, h, ds);
            DeviceArray::from_host(client, Shape::new([h, w]), mask).map(Value::Mask).map_err(|e| e.to_string())
        }
        _ => Err(format!("{id} is not a source")),
    }
}

/// The whole-image pass for `image`.
pub fn analyze(client: &Client, doc: &ProjectDoc, py: &Pyramid, image: &str) -> Analysis {
    let t0 = Instant::now();
    let level = level_under(py, ANALYSIS_SIDE);
    let l = py.levels[level].clone();
    let ds = l.downsample;
    let mut out = Analysis { key: analysis_key(doc, image), image: image.to_string(), level, downsample: ds, ..Analysis::default() };
    let ids = nodes_of(doc, image);
    if ids.is_empty() {
        return out;
    }
    let plan = at_downsample(&doc.graph.plan(), ds);
    // Views are evaluated through what feeds them
    let targets: Vec<String> = ids.iter().filter(|id| !matches!(doc.graph.node(id).map(|n| &n.kind), Some(NodeKind::View { .. }))).cloned().collect();
    let mut source = |id: &str| source_value(client, doc, py, image, id, level, (0, 0, l.width, l.height));
    let values = evaluate_all(client, &plan, &targets, &mut source);
    for (id, v) in &values {
        match v {
            Ok(v) => {
                if let Some(p) = preview(client, v) {
                    out.previews.insert(id.clone(), p);
                }
            }
            Err(e) => {
                out.errors.insert(id.clone(), e.clone());
            }
        }
    }
    // Whole-image values: the threshold of each Otsu node, from its input
    for id in &targets {
        if let Some(PlanNodeOp { op: Op::OtsuThreshold { bins }, input: Some(input) }) = op_of(&plan, id) {
            if let Some(Ok(Value::Image(a))) = values.get(&input) {
                if let Ok(t) = lumen_ops::threshold::global::otsu(client, a, bins) {
                    out.thresholds.insert(id.clone(), t);
                }
            }
        }
    }
    // Tables of the views fed by a Measure
    for v in doc.graph.views() {
        if let Some(from) = doc.graph.input(&v.id, 0) {
            match values.get(from) {
                Some(Ok(Value::Table(rows))) => {
                    out.tables.insert(v.id.clone(), TableResult { rows: Arc::new(rows.clone()), downsample: ds });
                }
                Some(Err(e)) => {
                    out.errors.insert(v.id.clone(), e.clone());
                }
                _ => {}
            }
            if let Some(p) = out.previews.get(from).cloned() {
                out.previews.insert(v.id.clone(), p);
            }
        } else {
            out.errors.insert(v.id.clone(), "Not connected: drag a result onto it".into());
        }
    }
    out.ms = t0.elapsed().as_secs_f64() * 1000.0;
    out
}

struct PlanNodeOp {
    op: Op,
    input: Option<String>,
}

fn op_of(plan: &Plan, id: &str) -> Option<PlanNodeOp> {
    let n = plan.get(id)?;
    Some(PlanNodeOp { op: n.op.clone()?, input: n.inputs.first().cloned().flatten() })
}

/// A small image of a value (rasters only).
fn preview(client: &Client, v: &Value) -> Option<Arc<RenderImage>> {
    let (h, w) = v.dims()?;
    let host = v.to_host(client)?;
    let step = (w.max(h) as f64 / PREVIEW_SIDE as f64).ceil().max(1.0) as usize;
    let (pw, ph) = (w.div_ceil(step), h.div_ceil(step));
    let mut small = Vec::with_capacity(pw * ph);
    for y in 0..ph {
        for x in 0..pw {
            // Labels and masks: a sample (never mix ids); images: the max of the block (thin
            // structures stay visible)
            let (yy, xx) = (y * step, x * step);
            let v = match v.port() {
                PortType::Image => {
                    let mut m = f32::MIN;
                    for dy in 0..step.min(h - yy) {
                        for dx in 0..step.min(w - xx) {
                            m = m.max(host[(yy + dy) * w + xx + dx]);
                        }
                    }
                    m
                }
                _ => host[yy * w + xx],
            };
            small.push(v);
        }
    }
    let style = match v.port() {
        PortType::Image => {
            let (lo, hi) = percentiles(&small, 0.01, 0.995);
            LayerStyle::Intensity { min: lo, max: hi, lut: ColormapLut::Grayscale, alpha: 1.0 }
        }
        PortType::Mask => LayerStyle::Mask { color: [0, 220, 255], alpha: 1.0 },
        _ => LayerStyle::Labels { alpha: 1.0 },
    };
    Some(render_layer(&small, pw, ph, style))
}

/// The plan for a region: sizes for `ds`, and each Otsu node fixed at its whole-image threshold.
fn region_plan(doc: &ProjectDoc, ds: f64, thresholds: &HashMap<String, f64>) -> Plan {
    let mut plan = at_downsample(&doc.graph.plan(), ds);
    for (id, t) in thresholds {
        if let Some(n) = plan.get_mut(id) {
            n.op = Some(Op::Threshold { value: *t });
        }
    }
    plan
}

/// Each raster view of `req.image` over the visible region.
pub fn compute(client: &Client, doc: &ProjectDoc, pyramid: &Pyramid, req: &ResultRequest, analysis: &Analysis) -> Vec<LayerResult> {
    let views: Vec<(String, ViewStyle)> = doc
        .graph
        .views()
        .filter(|v| doc.graph.images_of(&v.id).contains(&req.image))
        .filter_map(|v| match &v.kind {
            NodeKind::View { style, .. } => Some((v.id.clone(), style.clone())),
            _ => None,
        })
        .collect();
    let mut out = Vec::new();
    for (view, style) in views {
        let t0 = Instant::now();
        let base = LayerResult { view: view.clone(), image: req.image.clone(), rect: [0.0; 4], layer: None, port: None, error: None, ms: 0.0, level: 0 };
        let Some(target) = doc.graph.input(&view, 0).map(str::to_string) else {
            out.push(LayerResult { error: Some("Not connected: drag a result onto it in Pipeline".into()), ..base });
            continue;
        };
        let port = doc.graph.node(&target).and_then(|n| n.kind.output());
        // Tables come from the analysis; hidden layers are not computed
        if port == Some(PortType::Table) || !style.visible {
            continue;
        }
        // The level: sharp at this zoom, but never a region over MAX_SIDE a side
        let mut level = pyramid.level_for_zoom(req.zoom);
        let margin_at = |l: usize| halo(&region_plan(doc, pyramid.levels[l].downsample, &analysis.thresholds), &target);
        while level < pyramid.coarsest() && {
            let ds = pyramid.levels[level].downsample;
            let m = margin_at(level);
            ((req.rect[2] / ds).ceil() as usize + 2 * m).max((req.rect[3] / ds).ceil() as usize + 2 * m) > MAX_SIDE
        } {
            level += 1;
        }
        let l = pyramid.levels[level].clone();
        let ds = l.downsample;
        let plan = region_plan(doc, ds, &analysis.thresholds);
        let margin = halo(&plan, &target);
        let vx0 = (req.rect[0] / ds).floor().max(0.0) as usize;
        let vy0 = (req.rect[1] / ds).floor().max(0.0) as usize;
        let vx1 = (((req.rect[0] + req.rect[2]) / ds).ceil().max(0.0) as usize).min(l.width);
        let vy1 = (((req.rect[1] + req.rect[3]) / ds).ceil().max(0.0) as usize).min(l.height);
        if vx1 <= vx0 || vy1 <= vy0 {
            continue;
        }
        let (x0, y0) = (vx0.saturating_sub(margin), vy0.saturating_sub(margin));
        let (x1, y1) = ((vx1 + margin).min(l.width), (vy1 + margin).min(l.height));
        let (w, h) = (x1 - x0, y1 - y0);
        let mut source = |id: &str| source_value(client, doc, pyramid, &req.image, id, level, (x0, y0, w, h));
        let value = match evaluate(client, &plan, &target, &mut source) {
            Ok(v) => v,
            Err(e) => {
                out.push(LayerResult { error: Some(e), level, ..base });
                continue;
            }
        };
        let port = value.port();
        let Some(host) = value.to_host(client) else { continue };
        // Drop the margin: only the visible part is drawn
        let (cx0, cy0, cw, ch) = (vx0 - x0, vy0 - y0, vx1 - vx0, vy1 - vy0);
        let mut crop = Vec::with_capacity(cw * ch);
        for y in cy0..cy0 + ch {
            crop.extend_from_slice(&host[y * w + cx0..y * w + cx0 + cw]);
        }
        let layer_style = match port {
            PortType::Image => {
                let (lo, hi) = percentiles(&crop, 0.01, 0.99);
                LayerStyle::Intensity { min: lo, max: hi, lut: lut(&style.colormap), alpha: style.opacity }
            }
            PortType::Mask => LayerStyle::Mask { color: style.color, alpha: style.opacity },
            _ => LayerStyle::Labels { alpha: style.opacity },
        };
        out.push(LayerResult {
            rect: [vx0 as f64 * ds, vy0 as f64 * ds, cw as f64 * ds, ch as f64 * ds],
            layer: Some(render_layer(&crop, cw, ch, layer_style)),
            port: Some(port),
            level,
            ms: t0.elapsed().as_secs_f64() * 1000.0,
            ..base
        });
    }
    out
}

/// The colour map named in a view's style.
pub fn lut(name: &str) -> ColormapLut {
    match name {
        "grayscale" | "gray" => ColormapLut::Grayscale,
        "viridis" => ColormapLut::Viridis,
        "inferno" => ColormapLut::Inferno,
        "cyan" => ColormapLut::CyanGfp,
        _ => ColormapLut::Magma,
    }
}

/// Colour maps a view can use, as stored.
pub const COLORMAPS: [&str; 5] = ["magma", "viridis", "inferno", "grayscale", "cyan"];

/// Low and high quantiles of `v` (on a sample of at most 64k values).
pub fn percentiles(v: &[f32], lo: f64, hi: f64) -> (f32, f32) {
    let step = (v.len() / 65_536).max(1);
    let mut s: Vec<f32> = v.iter().step_by(step).copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return (0.0, 1.0);
    }
    s.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
    let (a, b) = (at(lo), at(hi));
    if b > a { (a, b) } else { (a, a + 1e-6) }
}

/// Level-0 centroid of a table row.
pub fn centroid_l0(r: &RegionFeatures, downsample: f64) -> Option<(f64, f64)> {
    let (y, x) = (*r.centroid.first()?, *r.centroid.get(1)?);
    Some((x * downsample, y * downsample))
}

/// Results by view id.
pub type Results = HashMap<String, LayerResult>;

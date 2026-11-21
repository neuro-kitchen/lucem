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
use web_time::Instant;

use gpui_kit::RenderImage;
use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::DeviceArray;
use lucem_core::Shape;
use lucem_io::Pyramid;
use lucem_ops::measure::RegionFeatures;
use lucem_pipeline::{align_region, at_downsample, crop_and_download, evaluate, evaluate_all, evaluate_cached, halo, EvalCache, EvalTimings, Op, Plan, PortType, Value};
use lucem_view::{render_layer, ColormapLut, LayerStyle};

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
    /// The plane shown, level-0 planes.
    pub z: usize,
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
    pub timings: EvalTimings,
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
    // A script saved again runs again
    for n in &doc.graph.nodes {
        if let NodeKind::Script { path, .. } = &n.kind {
            std::fs::metadata(path).and_then(|m| m.modified()).ok().hash(&mut h);
        }
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

/// A node the evaluator asks the app for: a script node (run on its inputs, Python or Rust) or a
/// source node.
#[allow(clippy::too_many_arguments)]
/// `whole`: the whole-image pass (coarse level) rather than the view's region.
fn node_value(client: &Client, doc: &ProjectDoc, plan: &Plan, py: &Pyramid, image: &str, id: &str, level: usize, z: usize, region: (usize, usize, usize, usize), whole: bool) -> Result<Value, String> {
    let Some(NodeKind::Script { path, inputs, output, params, enabled }) = doc.graph.node(id).map(|n| &n.kind) else {
        return source_value(client, doc, py, image, id, level, z, region, whole);
    };
    let name = std::path::Path::new(path).file_name().map_or_else(|| id.to_string(), |n| n.to_string_lossy().into_owned());
    let feeds = doc.graph.inputs(id);
    let mut host = Vec::with_capacity(inputs.len());
    for (i, port) in inputs.iter().enumerate() {
        let from = feeds.get(i).cloned().flatten().ok_or_else(|| format!("{name} ({id}): its {} input is not connected", port.name))?;
        let v = evaluate(client, plan, &from, &mut |sid: &str| node_value(client, doc, plan, py, image, sid, level, z, region, whole))?;
        if v.port() != port.port {
            return Err(format!("{name} ({id}): its {} input takes {}, {from} gives {}", port.name, port.port.name(), v.port().name()));
        }
        // Off: the first input passes through when it is what the script gives
        if !enabled && i == 0 && v.port() == *output {
            return Ok(v);
        }
        let (h, w) = v.dims().ok_or_else(|| format!("{name}: a table cannot go into a script"))?;
        let data = v.to_host(client).unwrap_or_default();
        host.push((port.name.clone(), port.port, crate::data::script::HostArray { width: w, height: h, data }));
    }
    if !enabled {
        return Err(format!("{name} ({id}) is off, and its first input is not a {}", output.name()));
    }
    let out = crate::data::script::run_node(std::path::Path::new(path), &host, params, *output)?;
    let (rw, rh) = host.first().map_or((region.2, region.3), |a| (a.2.width, a.2.height));
    if (out.width, out.height) != (rw, rh) {
        return Err(format!("{name} returned {} × {} px; it must keep the size of its input ({rw} × {rh})", out.width, out.height));
    }
    let shape = Shape::new([out.height, out.width]);
    match output {
        PortType::Image => DeviceArray::from_host(client, shape, out.data).map(Value::Image),
        PortType::Mask => DeviceArray::from_host(client, shape, out.data.iter().map(|&v| u32::from(v != 0.0)).collect()).map(Value::Mask),
        _ => DeviceArray::from_host(client, shape, out.data.iter().map(|&v| v.max(0.0) as u32).collect()).map(Value::Labels),
    }
    .map_err(|e| e.to_string())
}

/// A source node's raster over a region of `level`.
#[allow(clippy::too_many_arguments)]
fn source_value(client: &Client, doc: &ProjectDoc, py: &Pyramid, image: &str, id: &str, level: usize, z: usize, (x0, y0, w, h): (usize, usize, usize, usize), whole: bool) -> Result<Value, String> {
    let node = doc.graph.node(id).ok_or_else(|| format!("no node {id}"))?;
    let ds = py.levels[level].downsample;
    match &node.kind {
        NodeKind::Source { image: img, channel } => {
            if img != image {
                return Err(format!("{id} reads another image ({img})"));
            }
            let l = &py.levels[level];
            let plane = ((z as f64 / l.z_downsample) as usize).min(l.depth - 1);
            let mut values = py.read_region_at(level, *channel, plane, x0, y0, w, h).map_err(|e| e.to_string())?;
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
            // The annotations of the plane shown (and those of every plane)
            let anns: Vec<_> = doc.annotations_of(image).filter(|a| a.on_plane(z)).collect();
            let mask = rasterize(&anns, class.as_deref(), x0, y0, w, h, ds);
            DeviceArray::from_host(client, Shape::new([h, w]), mask).map(Value::Mask).map_err(|e| e.to_string())
        }
        NodeKind::Model { image: img, channels, model, roi, prompts, enabled } => {
            if img != image {
                return Err(format!("{id} reads another image ({img})"));
            }
            if !enabled {
                return Err(format!("{model} ({id}) is off"));
            }
            // Models run only when asked (Run in the Inspector); otherwise their last run is drawn
            // where it lies, and nothing runs on a pan, a zoom or another node's change
            let signature = format!("{id}|{}", serde_json::to_string(&doc.graph.node(id).map(|n| &n.kind)).unwrap_or_default());
            let asked = !whole && crate::data::model_node::take_armed(id);
            if !asked {
                let ds = py.levels[level].downsample;
                return match crate::data::model_node::last_run(&signature) {
                    Some(run) => DeviceArray::from_host(client, Shape::new([h, w]), crate::data::model_node::sample(&run, (x0, y0, w, h), ds)).map(Value::Labels).map_err(|e| e.to_string()),
                    None => Err(format!("{model} has not run: press Run in its Inspector")),
                };
            }
            let l = &py.levels[level];
            let plane = ((z as f64 / l.z_downsample) as usize).min(l.depth - 1);
            // The region of interest: only the part of the asked region its annotations cover
            let rois: Vec<&crate::domain::annotation::Annotation> = match roi {
                Some(r) => {
                    let found: Vec<_> = doc.annotations_of(image).filter(|a| r.matches(a) && a.on_plane(z)).collect();
                    if found.is_empty() {
                        return Err(format!("{model} ({id}): no region annotation for its region of interest on this plane"));
                    }
                    found
                }
                None => Vec::new(),
            };
            let (sx, sy, sw, sh) = if rois.is_empty() {
                (x0, y0, w, h)
            } else {
                let b = rois.iter().map(|a| a.shape.outline()).flatten().fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, p| (b.0.min(p[0]), b.1.min(p[1]), b.2.max(p[0]), b.3.max(p[1])));
                // In level pixels, with a margin, clipped to the asked region
                let pad = 16.0;
                let bx0 = ((b.0 / ds - pad).floor().max(x0 as f64)) as usize;
                let by0 = ((b.1 / ds - pad).floor().max(y0 as f64)) as usize;
                let bx1 = ((b.2 / ds + pad).ceil().min((x0 + w) as f64)).max(0.0) as usize;
                let by1 = ((b.3 / ds + pad).ceil().min((y0 + h) as f64)).max(0.0) as usize;
                if bx1 <= bx0 || by1 <= by0 {
                    // Outside the region of interest: nothing to run
                    return DeviceArray::from_host(client, Shape::new([h, w]), vec![0u32; w * h]).map(Value::Labels).map_err(|e| e.to_string());
                }
                (bx0, by0, bx1 - bx0, by1 - by0)
            };
            let labels: Vec<String> = py.channels.iter().map(|c| c.label.clone()).collect();
            let picked = crate::data::model_node::channels_of(channels, &labels);
            let planes: Vec<Vec<f32>> = picked
                .iter()
                .map(|&c| {
                    let top = py.channels.get(c).map_or(1.0, |ch| ch.range.1.max(1e-6));
                    py.read_region_at(level, c, plane, sx, sy, sw, sh).map(|v| v.into_iter().map(|x| x / top).collect()).map_err(|e| e.to_string())
                })
                .collect::<Result<_, _>>()?;
            let mut sub = crate::data::model_node::run(doc, image, model, prompts.as_deref(), &planes, sw, sh, z, (sx, sy, sw, sh), ds, py.pixel_size_um.map(|u| u * ds))?;
            if !rois.is_empty() {
                let inside = rasterize(&rois, None, sx, sy, sw, sh, ds);
                for (l, m) in sub.iter_mut().zip(inside) {
                    if m == 0 {
                        *l = 0;
                    }
                }
            }
            crate::data::model_node::store_run(signature, crate::data::model_node::Stored { x0: sx, y0: sy, w: sw, h: sh, ds, labels: sub.clone() });
            let labels = if (sx, sy, sw, sh) == (x0, y0, w, h) {
                sub
            } else {
                let mut full = vec![0u32; w * h];
                for row in 0..sh {
                    let (dy, dx) = (sy - y0 + row, sx - x0);
                    full[dy * w + dx..dy * w + dx + sw].copy_from_slice(&sub[row * sw..(row + 1) * sw]);
                }
                full
            };
            DeviceArray::from_host(client, Shape::new([h, w]), labels).map(Value::Labels).map_err(|e| e.to_string())
        }
        _ => Err(format!("{id} is not a source")),
    }
}

/// The whole-image pass for `image`, on level-0 plane `z` of a volume.
pub fn analyze(client: &Client, doc: &ProjectDoc, py: &Pyramid, image: &str, z: usize) -> Analysis {
    let t0 = Instant::now();
    let level = level_under(py, ANALYSIS_SIDE);
    let l = py.levels[level].clone();
    let ds = l.downsample;
    let key = analysis_key(doc, image) ^ (z as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut out = Analysis { key, image: image.to_string(), level, downsample: ds, ..Analysis::default() };
    let ids = nodes_of(doc, image);
    if ids.is_empty() {
        return out;
    }
    let plan = at_downsample(&doc.graph.plan(), ds);
    // Views are evaluated through what feeds them
    let targets: Vec<String> = ids.iter().filter(|id| !matches!(doc.graph.node(id).map(|n| &n.kind), Some(NodeKind::View { .. }))).cloned().collect();
    let zz = z.min(py.depth() - 1);
    let mut source = |id: &str| node_value(client, doc, &plan, py, image, id, level, zz, (0, 0, l.width, l.height), true);
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
    // Whole-image values: the threshold of each global threshold node, from its input
    for id in &targets {
        if let Some(PlanNodeOp { op, input: Some(input) }) = op_of(&plan, id)
            && let Some(Ok(Value::Image(a))) = values.get(&input)
        {
            let thresh = match op {
                Op::OtsuThreshold { bins } => lucem_ops::threshold::global::otsu(client, a, bins).ok(),
                Op::LiThreshold { bins } => lucem_ops::threshold::global::li(client, a, bins).ok(),
                Op::TriangleThreshold { bins } => lucem_ops::threshold::global::triangle(client, a, bins).ok(),
                _ => None,
            };
            if let Some(t) = thresh {
                out.thresholds.insert(id.clone(), t);
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
    Some(preview_from_host(&host, w, h, v.port()))
}

/// Downsamples a host raster (`w × h`) to at most [`PREVIEW_SIDE`] on its longer edge and renders
/// it as a Pipeline node thumbnail. Shared by desktop [`analyze`] and browser [`analyze_remote`].
fn preview_from_host(host: &[f32], w: usize, h: usize, port: PortType) -> Arc<RenderImage> {
    let step = (w.max(h) as f64 / PREVIEW_SIDE as f64).ceil().max(1.0) as usize;
    let (pw, ph) = (w.div_ceil(step), h.div_ceil(step));
    let mut small = Vec::with_capacity(pw * ph);
    for y in 0..ph {
        for x in 0..pw {
            // Labels and masks: a sample (never mix ids); images: the max of the block (thin
            // structures stay visible)
            let (yy, xx) = (y * step, x * step);
            let v = match port {
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
    let style = match port {
        PortType::Image => {
            let (lo, hi) = percentiles(&small, 0.01, 0.995);
            LayerStyle::Intensity { min: lo, max: hi, lut: ColormapLut::Grayscale, alpha: 1.0 }
        }
        PortType::Mask => LayerStyle::Mask { color: [0, 220, 255], alpha: 1.0 },
        _ => LayerStyle::Labels { alpha: 1.0 },
    };
    render_layer(&small, pw, ph, style)
}

/// Coarse node thumbnails for the Pipeline canvas in a served browser page: reads the coarsest
/// level ≤ `PREVIEW_SIDE * 2` for each source and op node from the server.
#[cfg(target_family = "wasm")]
pub fn analyze_remote(doc: &ProjectDoc, py: &Pyramid, image: &str, z: usize) -> Analysis {
    let t0 = Instant::now();
    let level = level_under(py, PREVIEW_SIDE * 2);
    let l = py.levels[level].clone();
    let ds = l.downsample;
    let key = analysis_key(doc, image) ^ (z as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut out = Analysis { key, image: image.to_string(), level, downsample: ds, ..Analysis::default() };
    let ids = nodes_of(doc, image);
    if ids.is_empty() {
        return out;
    }
    let plane = ((z as f64 / l.z_downsample) as usize).min(l.depth.saturating_sub(1));
    let link = crate::data::remote::page_link();
    let served = py.path.as_ref().map(|p| p.to_string_lossy().into_owned()).and_then(|p| lucem_stream::split_remote(&p).map(|(_, s)| s.to_string()));
    for id in &ids {
        let Some(node) = doc.graph.node(id) else { continue };
        match &node.kind {
            NodeKind::Source { image: img, channel } if img == image => {
                match py.read_region_at(level, *channel, plane, 0, 0, l.width, l.height) {
                    Ok(mut vals) => {
                        let top = py.channels.get(*channel).map_or(1.0, |c| c.range.1);
                        if top > 1.0 {
                            let k = 1.0 / top;
                            vals.iter_mut().for_each(|v| *v *= k);
                        }
                        out.previews.insert(id.clone(), preview_from_host(&vals, l.width, l.height, PortType::Image));
                    }
                    Err(e) => {
                        out.errors.insert(id.clone(), e.to_string());
                    }
                }
            }
            NodeKind::Op { op, .. } => {
                let port = op.output();
                if port == PortType::Table {
                    continue;
                }
                let (Some(link), Some(served)) = (&link, &served) else { continue };
                let spec = match spec_for(doc, image, id).and_then(|s| serde_json::to_string(&s).map_err(|e| e.to_string())) {
                    Ok(s) => s,
                    Err(e) => {
                        out.errors.insert(id.clone(), e);
                        continue;
                    }
                };
                match link.open(served, Some(spec)).map_err(|e| e.to_string()).and_then(|p| p.read_region_at(level, 0, plane, 0, 0, l.width, l.height).map_err(|e| e.to_string())) {
                    Ok(vals) => {
                        out.previews.insert(id.clone(), preview_from_host(&vals, l.width, l.height, port));
                    }
                    Err(e) => {
                        out.errors.insert(id.clone(), e);
                    }
                }
            }
            NodeKind::Script { output, .. } => {
                let port = *output;
                if port == PortType::Table {
                    continue;
                }
                let (Some(link), Some(served)) = (&link, &served) else { continue };
                let Some(spec_res) = remote_script_spec(doc, image, id) else { continue };
                let spec = match spec_res {
                    Ok(s) => s,
                    Err(e) => {
                        out.errors.insert(id.clone(), e);
                        continue;
                    }
                };
                match link.open(served, Some(spec)).map_err(|e| e.to_string()).and_then(|p| p.read_region_at(level, 0, plane, 0, 0, l.width, l.height).map_err(|e| e.to_string())) {
                    Ok(vals) => {
                        out.previews.insert(id.clone(), preview_from_host(&vals, l.width, l.height, port));
                    }
                    Err(e) => {
                        out.errors.insert(id.clone(), e);
                    }
                }
            }
            _ => {}
        }
    }
    for v in doc.graph.views() {
        if let Some(from) = doc.graph.input(&v.id, 0) {
            if let Some(p) = out.previews.get(from).cloned() {
                out.previews.insert(v.id.clone(), p);
            }
        }
    }
    out.ms = t0.elapsed().as_secs_f64() * 1000.0;
    out
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

/// Each raster view of `req.image` over the visible region (with a fresh cache; interactive views
/// pass a persistent [`EvalCache`] through [`compute_cached`]).
pub fn compute(client: &Client, doc: &ProjectDoc, pyramid: &Pyramid, req: &ResultRequest, analysis: &Analysis) -> Vec<LayerResult> {
    let cache = EvalCache::default();
    compute_cached(client, doc, pyramid, req, analysis, &cache)
}

/// Each raster view of `req.image` over the visible region, reusing uploaded inputs and unchanged
/// upstream node outputs from `cache` on the device and cropping halos on the GPU before read-back.
pub fn compute_cached(client: &Client, doc: &ProjectDoc, pyramid: &Pyramid, req: &ResultRequest, analysis: &Analysis, cache: &EvalCache) -> Vec<LayerResult> {
    let views: Vec<(String, ViewStyle)> = doc
        .graph
        .views()
        .filter(|v| doc.graph.images_of(&v.id).contains(&req.image))
        .filter_map(|v| match &v.kind {
            NodeKind::View { style, .. } => Some((v.id.clone(), style.clone())),
            _ => None,
        })
        .collect();
    let n_channels = doc.image(&req.image).map_or(0, |i| i.channels.len());
    let processed = doc.graph.processed_channels(&req.image, n_channels);
    // The view of a channel's chain with every step off (or Raw): the channel draws as it is;
    // a hidden channel hides its processing too (its eye in Layers acts on both)
    let channel_visible = |c: usize| doc.image(&req.image).and_then(|i| i.channels.get(c)).is_none_or(|d| d.visible);
    let raw_views: std::collections::HashSet<String> = (0..n_channels)
        .filter_map(|c| doc.graph.channel_chain(&req.image, c).1.map(|v| (c, v)))
        .filter(|(c, v)| !processed.values().any(|p| p == v) || !channel_visible(*c))
        .map(|(_, v)| v)
        .collect();
    let mut out = Vec::new();
    for (view, style) in views {
        if raw_views.contains(&view) {
            continue;
        }
        let t0 = Instant::now();
        let base = LayerResult { view: view.clone(), image: req.image.clone(), rect: [0.0; 4], layer: None, port: None, error: None, ms: 0.0, level: 0, timings: EvalTimings::default() };
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
        let (x0, y0, w, h) = align_region(
            vx0.saturating_sub(margin),
            vy0.saturating_sub(margin),
            (vx1 + margin).min(l.width),
            (vy1 + margin).min(l.height),
            l.width,
            l.height,
        );
        let region_tag = format!("{}|L{level}|Z{}|{x0},{y0},{w},{h}", req.image, req.z);
        let mut source = |id: &str| node_value(client, doc, &plan, pyramid, &req.image, id, level, req.z, (x0, y0, w, h), false);
        let (value, mut timings) = match evaluate_cached(client, &plan, &target, &region_tag, cache, &mut source) {
            Ok(v) => v,
            Err(e) => {
                out.push(LayerResult { error: Some(e), level, ..base });
                continue;
            }
        };
        let port = value.port();
        let (cw, ch) = (vx1 - vx0, vy1 - vy0);
        let (crop, readback_ms) = match crop_and_download(client, &value, x0, y0, w, h, vx0, vy0, cw, ch) {
            Ok(v) => v,
            Err(_) => continue,
        };
        timings.readback_ms = readback_ms;
        let base = LayerResult { timings, ..base };
        let rect = [vx0 as f64 * ds, vy0 as f64 * ds, cw as f64 * ds, ch as f64 * ds];
        out.push(drawn(doc, pyramid, req, &processed, &style, base, port, &crop, (cw, ch), rect, level, t0));
    }
    out
}

/// A view's values over its region (`crop`, `cw × ch`) drawn as its layer: a channel's
/// processing in that channel's colour and window, else by the view's style.
#[allow(clippy::too_many_arguments)]
fn drawn(doc: &ProjectDoc, pyramid: &Pyramid, req: &ResultRequest, processed: &HashMap<usize, String>, style: &ViewStyle, base: LayerResult, port: PortType, crop: &[f32], (cw, ch): (usize, usize), rect: [f64; 4], level: usize, t0: Instant) -> LayerResult {
    if let Some(c) = processed.iter().find(|(_, v)| **v == base.view).map(|(c, _)| *c)
        && port == PortType::Image
        && let Some(chan) = doc.image(&req.image).and_then(|i| i.channels.get(c))
    {
        let top = pyramid.channels.get(c).map_or(1.0, |i| i.range.1.max(1e-6));
        let scale = if top > 1.0 { 1.0 / top } else { 1.0 };
        let image = channel_layer(crop, cw, chan.min * scale, chan.max * scale, chan.gamma, chan.color);
        return LayerResult { rect, layer: Some(image), port: Some(port), level, ms: t0.elapsed().as_secs_f64() * 1000.0, ..base };
    }
    let layer_style = match port {
        PortType::Image => {
            let (lo, hi) = percentiles(crop, 0.01, 0.99);
            LayerStyle::Intensity { min: lo, max: hi, lut: lut(&style.colormap), alpha: style.opacity }
        }
        PortType::Mask => LayerStyle::Mask { color: style.color, alpha: style.opacity },
        _ => LayerStyle::Labels { alpha: style.opacity },
    };
    LayerResult { rect, layer: Some(render_layer(crop, cw, ch, layer_style)), port: Some(port), level, ms: t0.elapsed().as_secs_f64() * 1000.0, ..base }
}

/// The pipeline that computes `target` from the channels of `image`, as data the server runs on
/// tiles (`lucem_pipeline::PipelineSpec`): its sources are channels, its nodes the operators.
/// Script, model and annotation nodes run in the desktop app (a server cannot run them on tiles
/// yet): a chain that needs one says so.
pub fn spec_for(doc: &ProjectDoc, image: &str, target: &str) -> Result<lucem_pipeline::PipelineSpec, String> {
    let mut sources = std::collections::BTreeMap::new();
    let mut nodes = std::collections::BTreeMap::new();
    for n in &doc.graph.nodes {
        match &n.kind {
            NodeKind::Source { image: i, channel } if i == image => {
                sources.insert(n.id.clone(), *channel);
            }
            NodeKind::Op { op, enabled } => {
                let inputs = doc.graph.inputs(&n.id).into_iter().map(Option::unwrap_or_default).collect();
                nodes.insert(n.id.clone(), lucem_pipeline::SpecNode { op: op.clone(), inputs, enabled: *enabled });
            }
            _ => {}
        }
    }
    let spec = lucem_pipeline::PipelineSpec { sources, nodes, target: target.to_string(), level: 0, normalize: true };
    spec.validate().map_err(|e| format!("{e} (scripts, models and annotations as inputs run in the desktop app for now)"))?;
    Ok(spec)
}

/// If `target` is a `NodeKind::Script` fed by a channel source of `image`, encodes a
/// `{"script_node": ...}` tile spec for `lucem serve --web` to execute on the server.
#[cfg(target_family = "wasm")]
fn remote_script_spec(doc: &ProjectDoc, image: &str, target: &str) -> Option<Result<String, String>> {
    let node = doc.graph.node(target)?;
    let NodeKind::Script { path, inputs, params, enabled, .. } = &node.kind else {
        return None;
    };
    let src_id = doc.graph.input(target, 0)?;
    let channel = match doc.graph.node(src_id).map(|n| &n.kind) {
        Some(NodeKind::Source { image: i, channel }) if i == image => *channel,
        _ => return Some(Err("Connect a channel Source node directly into this script node on the server".into())),
    };
    let input_name = inputs.first().map(|i| i.name.clone()).unwrap_or_else(|| "image".into());
    let param_pairs: Vec<(&str, f64)> = params.iter().map(|p| (p.name.as_str(), p.value)).collect();
    let payload = serde_json::json!({
        "script_node": {
            "path": path,
            "source_channel": channel,
            "input_name": input_name,
            "params": param_pairs,
            "enabled": enabled,
            "halo": 12usize,
            "normalize": true,
        }
    });
    Some(serde_json::to_string(&payload).map_err(|e| e.to_string()))
}

/// Each raster view of `req.image`, computed by the page's server: the view's chain as a
/// pipeline the server runs on the tiles of the region (their halo read there), drawn here as
/// the desktop draws its own results. A browser page has no compute of its own.
#[cfg(target_family = "wasm")]
pub fn compute_remote(
    doc: &ProjectDoc,
    pyramid: &Pyramid,
    req: &ResultRequest,
    latest: &std::sync::atomic::AtomicU64,
    generation: u64,
) -> Vec<LayerResult> {
    let mut out = Vec::new();
    let Some(link) = crate::data::remote::page_link() else { return out };
    let Some(path) = pyramid.path.as_ref().map(|p| p.to_string_lossy().into_owned()) else { return out };
    let Some((_, served)) = lucem_stream::split_remote(&path).map(|(a, p)| (a.to_string(), p.to_string())) else { return out };
    let n_channels = doc.image(&req.image).map_or(0, |i| i.channels.len());
    let processed = doc.graph.processed_channels(&req.image, n_channels);
    // Skip channel views whose steps are all off (or hidden channel), so disabling an op shows
    // the clean original channel view instead of drawing a cyan overlay on top.
    let channel_visible = |c: usize| doc.image(&req.image).and_then(|i| i.channels.get(c)).is_none_or(|d| d.visible);
    let raw_views: std::collections::HashSet<String> = (0..n_channels)
        .filter_map(|c| doc.graph.channel_chain(&req.image, c).1.map(|v| (c, v)))
        .filter(|(c, v)| !processed.values().any(|p| p == v) || !channel_visible(*c))
        .map(|(_, v)| v)
        .collect();
    for v in doc.graph.views().filter(|v| doc.graph.images_of(&v.id).contains(&req.image)) {
        if latest.load(std::sync::atomic::Ordering::Acquire) != generation {
            break;
        }
        if raw_views.contains(&v.id) {
            continue;
        }
        let NodeKind::View { style, .. } = &v.kind else { continue };
        let t0 = Instant::now();
        let base = LayerResult { view: v.id.clone(), image: req.image.clone(), rect: [0.0; 4], layer: None, port: None, error: None, ms: 0.0, level: 0, timings: EvalTimings::default() };
        let Some(target) = doc.graph.input(&v.id, 0).map(str::to_string) else { continue };
        let Some(port) = doc.graph.node(&target).and_then(|n| n.kind.output()) else { continue };
        if port == PortType::Table || !style.visible {
            continue;
        }
        let spec = match remote_script_spec(doc, &req.image, &target) {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                out.push(LayerResult { error: Some(e), ..base });
                continue;
            }
            None => match spec_for(doc, &req.image, &target).and_then(|s| serde_json::to_string(&s).map_err(|e| e.to_string())) {
                Ok(s) => s,
                Err(e) => {
                    out.push(LayerResult { error: Some(e), ..base });
                    continue;
                }
            },
        };
        let mut level = pyramid.level_for_zoom(req.zoom);
        while level < pyramid.coarsest() && ((req.rect[2] / pyramid.levels[level].downsample) as usize).max((req.rect[3] / pyramid.levels[level].downsample) as usize) > MAX_SIDE {
            level += 1;
        }
        let l = pyramid.levels[level].clone();
        let ds = l.downsample;
        let vx0 = (req.rect[0] / ds).floor().max(0.0) as usize;
        let vy0 = (req.rect[1] / ds).floor().max(0.0) as usize;
        let vx1 = (((req.rect[0] + req.rect[2]) / ds).ceil().max(0.0) as usize).min(l.width);
        let vy1 = (((req.rect[1] + req.rect[3]) / ds).ceil().max(0.0) as usize).min(l.height);
        if vx1 <= vx0 || vy1 <= vy0 {
            continue;
        }
        let plane = ((req.z as f64 / l.z_downsample) as usize).min(l.depth - 1);
        let (cw, ch) = (vx1 - vx0, vy1 - vy0);
        let values = link.open(&served, Some(spec)).map_err(|e| e.to_string()).and_then(|p| {
            let (tx0, tx1) = (vx0 / l.tile_w, (vx1 - 1) / l.tile_w);
            let (ty0, ty1) = (vy0 / l.tile_h, (vy1 - 1) / l.tile_h);
            let mut buf = vec![0.0f32; cw * ch];
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    if latest.load(std::sync::atomic::Ordering::Acquire) != generation {
                        return Err(String::new());
                    }
                    let tile = p.read_tile_at(level, 0, plane, tx, ty).map_err(|e| e.to_string())?;
                    let (gx0, gy0) = (tx * l.tile_w, ty * l.tile_h);
                    let (ix0, iy0) = (gx0.max(vx0), gy0.max(vy0));
                    let (ix1, iy1) = ((gx0 + tile.width).min(vx1), (gy0 + tile.height).min(vy1));
                    for y in iy0..iy1 {
                        let src_row = (y - gy0) * tile.width + (ix0 - gx0);
                        let dst_row = (y - vy0) * cw + (ix0 - vx0);
                        let n = ix1 - ix0;
                        buf[dst_row..dst_row + n].copy_from_slice(&tile.data[src_row..src_row + n]);
                    }
                }
            }
            Ok(buf)
        });
        match values {
            Ok(crop) => {
                let rect = [vx0 as f64 * ds, vy0 as f64 * ds, cw as f64 * ds, ch as f64 * ds];
                out.push(drawn(doc, pyramid, req, &processed, style, base, port, &crop, (cw, ch), rect, level, t0));
            }
            Err(e) if e.is_empty() => break,
            Err(e) => out.push(LayerResult { error: Some(e), level, ..base }),
        }
    }
    out
}

/// A channel's processed values drawn as the channel is: through its window and gamma, in its
/// colour, opaque where bright (over the other channels).
fn channel_layer(v: &[f32], w: usize, min: f32, max: f32, gamma: f32, color: [u8; 3]) -> std::sync::Arc<gpui_kit::RenderImage> {
    let span = (max - min).abs().max(1e-12);
    let g = 1.0 / gamma.max(1e-3);
    let mut out = vec![0u8; v.len() * 4];
    for (px, &x) in out.as_chunks_mut::<4>().0.iter_mut().zip(v) {
        let t = ((x - min) / span).clamp(0.0, 1.0).powf(g);
        px.copy_from_slice(&[(color[2] as f32 * t) as u8, (color[1] as f32 * t) as u8, (color[0] as f32 * t) as u8, (t * 255.0) as u8]);
    }
    lucem_view::bgra_image(w, v.len() / w.max(1), out)
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

//! Results of the graph over what is on screen: each view node of the shown image is computed on
//! the visible region (plus the margin its ops need) at the pyramid level the zoom asks for, on
//! the device, and comes back as a layer to draw over the image (or a table for Analyze).
//!
//! A region is never larger than [`MAX_SIDE`] px a side: zoomed far out it is computed on a
//! coarser level. Running the graph over the whole image at full resolution is a batch job.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use gpui_kit::RenderImage;
use lumen_core::cubecl::prelude::Client;
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;
use lumen_io::Pyramid;
use lumen_ops::measure::RegionFeatures;
use lumen_pipeline::{evaluate, halo, PortType, Value};
use lumen_view::{render_layer, ColormapLut, LayerStyle};

use crate::domain::annotation::rasterize;
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::domain::project::ProjectDoc;

/// Largest side of a computed region (px at its level).
pub const MAX_SIDE: usize = 2048;

/// What is on screen: an image and the visible level-0 rectangle at a zoom.
#[derive(Debug, Clone, PartialEq)]
pub struct ResultRequest {
    pub image: String,
    /// `x, y, w, h` in level-0 px.
    pub rect: [f64; 4],
    /// Screen px per level-0 px (times the display's scale).
    pub zoom: f64,
}

/// One view's result.
#[derive(Clone)]
pub struct LayerResult {
    pub view: String,
    pub image: String,
    /// Where the layer sits, level-0 px (`x, y, w, h`).
    pub rect: [f64; 4],
    pub layer: Option<Arc<RenderImage>>,
    pub table: Option<Arc<Vec<RegionFeatures>>>,
    /// Level-0 px per pixel of the level it ran on, and the region's origin there: object
    /// coordinates in the table are region pixels.
    pub downsample: f64,
    pub origin: (usize, usize),
    pub port: Option<PortType>,
    pub error: Option<String>,
    pub ms: f64,
    pub level: usize,
}

/// Computes every view of `req.image` (in graph order).
pub fn compute(client: &Client, doc: &ProjectDoc, pyramid: &Pyramid, req: &ResultRequest) -> Vec<LayerResult> {
    let plan = doc.graph.plan();
    let views: Vec<(String, String, ViewStyle)> = doc
        .graph
        .views()
        .filter(|v| doc.graph.images_of(&v.id).contains(&req.image))
        .filter_map(|v| match &v.kind {
            NodeKind::View { name, style } => Some((v.id.clone(), name.clone(), style.clone())),
            _ => None,
        })
        .collect();
    let mut out = Vec::new();
    for (view, _name, style) in views {
        let t0 = Instant::now();
        let base = LayerResult { view: view.clone(), image: req.image.clone(), rect: [0.0; 4], layer: None, table: None, downsample: 1.0, origin: (0, 0), port: None, error: None, ms: 0.0, level: 0 };
        let Some(target) = doc.graph.input(&view, 0).map(str::to_string) else {
            out.push(LayerResult { error: Some("Not connected: drag a result onto it in Pipeline".into()), ..base });
            continue;
        };
        if !style.visible && !matches!(output_of(doc, &target), Some(PortType::Table)) {
            continue;
        }
        let margin = halo(&plan, &target);
        // The level: sharp at this zoom, but never a region over MAX_SIDE a side
        let mut level = pyramid.level_for_zoom(req.zoom);
        let size_at = |l: usize| {
            let ds = pyramid.levels[l].downsample;
            ((req.rect[2] / ds).ceil() as usize + 2 * margin, (req.rect[3] / ds).ceil() as usize + 2 * margin)
        };
        while level < pyramid.coarsest() && {
            let (w, h) = size_at(level);
            w.max(h) > MAX_SIDE
        } {
            level += 1;
        }
        let l = pyramid.levels[level].clone();
        let ds = l.downsample;
        // Visible region at the level, then with its margin, clipped to the image
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
        let annotations: Vec<_> = doc.annotations_of(&req.image).collect();
        let mut source = |id: &str| -> Result<Value, String> {
            let node = doc.graph.node(id).ok_or_else(|| format!("no node {id}"))?;
            match &node.kind {
                NodeKind::Source { image, channel } => {
                    if image != &req.image {
                        return Err(format!("{id} reads another image ({image}); show that image to see this result"));
                    }
                    let mut values = pyramid.read_region(level, *channel, x0, y0, w, h).map_err(|e| e.to_string())?;
                    // Raw values to 0–1 by the type's range (so thresholds and gamma mean the same
                    // on 8- and 16-bit images)
                    let top = pyramid.channels.get(*channel).map_or(1.0, |c| c.range.1);
                    if top > 1.0 {
                        let k = 1.0 / top;
                        values.iter_mut().for_each(|v| *v *= k);
                    }
                    let a = DeviceArray::from_host(client, Shape::new([h, w]), values).map_err(|e| e.to_string())?;
                    Ok(Value::Image(a))
                }
                NodeKind::Annotations { image, class } => {
                    if image != &req.image {
                        return Err(format!("{id} reads another image ({image})"));
                    }
                    let mask = rasterize(&annotations, class.as_deref(), x0, y0, w, h, ds);
                    let a = DeviceArray::from_host(client, Shape::new([h, w]), mask).map_err(|e| e.to_string())?;
                    Ok(Value::Mask(a))
                }
                _ => Err(format!("{id} is not a source")),
            }
        };
        let value = match evaluate(client, &plan, &target, &mut source) {
            Ok(v) => v,
            Err(e) => {
                out.push(LayerResult { error: Some(e), level, ..base });
                continue;
            }
        };
        let port = value.port();
        if let Value::Table(rows) = value {
            out.push(LayerResult { table: Some(Arc::new(rows)), port: Some(port), downsample: ds, origin: (x0, y0), level, rect: [x0 as f64 * ds, y0 as f64 * ds, w as f64 * ds, h as f64 * ds], ms: t0.elapsed().as_secs_f64() * 1000.0, ..base });
            continue;
        }
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
        let layer = render_layer(&crop, cw, ch, layer_style);
        out.push(LayerResult {
            rect: [vx0 as f64 * ds, vy0 as f64 * ds, cw as f64 * ds, ch as f64 * ds],
            layer: Some(layer),
            port: Some(port),
            downsample: ds,
            origin: (x0, y0),
            level,
            ms: t0.elapsed().as_secs_f64() * 1000.0,
            ..base
        });
    }
    out
}

/// What node `id` gives.
fn output_of(doc: &ProjectDoc, id: &str) -> Option<PortType> {
    doc.graph.node(id).and_then(|n| n.kind.output())
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
fn percentiles(v: &[f32], lo: f64, hi: f64) -> (f32, f32) {
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
pub fn centroid_l0(r: &RegionFeatures, downsample: f64, origin: (usize, usize)) -> Option<(f64, f64)> {
    let (y, x) = (*r.centroid.first()?, *r.centroid.get(1)?);
    Some(((x + origin.0 as f64) * downsample, (y + origin.1 as f64) * downsample))
}

/// Results by view id.
pub type Results = HashMap<String, LayerResult>;

//! Histology as a service: what the app computes, answered by method with JSON, transport-free.
//! A server registers it as its `hist` plugin (`lucem serve`) so a page computes on the server;
//! the app calls the same functions locally. As dsp-kitchen's `curation::service`.
//!
//! Each image keeps its [`Analysis`] (what the steps so far left) for the server's life.
//!
//! | method | body | answer |
//! |---|---|---|
//! | `apply` | `{path, z?, action}` (a session [`Action`]) | `{outcome, tissue, cells, tiles}` |
//! | `summary` | `{path}` | `{tissue, cells, tiles}` |
//! | `objects` | `{path, kind: "cells" \| "tiles", offset?, limit?}` | `{total, objects: [HistObject…]}` |
//! | `reset` | `{path}` | `{}`: the image's analysis is forgotten |

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::engine::{apply, Analysis, Ctx, PluginModels};
use crate::session::Action;

/// The plugin id a server registers it under.
pub const PLUGIN: &str = "hist";

/// Objects per `objects` answer, at most.
pub const OBJECTS_MAX: usize = 50_000;

/// Who calls: their name as the server knows it, and whether they may write.
#[derive(Debug, Clone, Copy)]
pub struct Caller<'a> {
    pub name: &'a str,
    pub may_write: bool,
}

#[derive(Deserialize)]
struct ApplyCall {
    path: String,
    #[serde(default)]
    z: usize,
    action: Action,
    /// The caller's tissue regions (level-0 outlines), when it keeps them (the app's `Tissue`
    /// annotations, edited by people): they replace what the server kept.
    #[serde(default)]
    tissue: Option<Vec<crate::objects::Polygon>>,
}

#[derive(Deserialize)]
struct PathCall {
    path: String,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Cells,
    Tiles,
}

#[derive(Deserialize)]
struct ObjectsCall {
    path: String,
    kind: Kind,
    #[serde(default)]
    offset: usize,
    #[serde(default = "objects_max")]
    limit: usize,
    /// When true, decimates polygon contours to <= 12 vertices for lightweight browser transfer.
    #[serde(default)]
    simplify: bool,
}

#[derive(Deserialize)]
struct RasterCall {
    path: String,
    /// Level-0 `[x0, y0, w, h]` region to rasterize.
    rect: [f64; 4],
    /// Level-0 pixels per output pixel.
    ds: f64,
    /// Overlay opacity `0..=1`.
    #[serde(default = "default_alpha")]
    alpha: f32,
    /// Selected cell/tile tags (e.g. `"c42"`, `"t7"`).
    #[serde(default)]
    selected: Vec<String>,
}

fn default_alpha() -> f32 {
    1.0
}

fn simplify_poly(poly: &[[f64; 2]], max_pts: usize) -> Vec<[f64; 2]> {
    if poly.len() <= max_pts {
        return poly.iter().map(|&[x, y]| [(x * 10.0).round() / 10.0, (y * 10.0).round() / 10.0]).collect();
    }
    let step = (poly.len() as f64 / max_pts as f64).max(1.0);
    (0..max_pts)
        .map(|k| {
            let [x, y] = poly[((k as f64 * step) as usize).min(poly.len() - 1)];
            [(x * 10.0).round() / 10.0, (y * 10.0).round() / 10.0]
        })
        .collect()
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = ((chunk[0] as u32) << 16)
            | ((chunk.get(1).copied().unwrap_or(0) as u32) << 8)
            | (chunk.get(2).copied().unwrap_or(0) as u32);
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { TABLE[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[(n & 63) as usize] as char } else { '=' });
    }
    out
}

fn object_color(c: &crate::HistObject) -> [u8; 3] {
    match (c.intensity, &c.class) {
        (Some(crate::IntensityClass::Negative), _) => [80, 120, 235],
        (Some(crate::IntensityClass::OnePlus), _) => [250, 210, 60],
        (Some(crate::IntensityClass::TwoPlus), _) => [245, 140, 40],
        (Some(crate::IntensityClass::Positive | crate::IntensityClass::ThreePlus), _) => [225, 50, 50],
        (None, Some(name)) => {
            let mut h = 0x811c9dc5u32;
            for b in name.bytes() {
                h = (h ^ b as u32).wrapping_mul(0x01000193);
            }
            let r = 80 + ((h & 0xff) as u8 % 160);
            let g = 80 + (((h >> 8) & 0xff) as u8 % 160);
            let b = 80 + (((h >> 16) & 0xff) as u8 % 160);
            [r, g, b]
        }
        (None, None) => [90, 200, 255],
    }
}

/// Server-side viewport rasterizer: renders any number of detected cells/tiles intersecting `rect`
/// natively into a compact RGBA PNG (QuPath / Cytomine server-tile architecture).
fn raster_viewport_png(analysis: &Analysis, call: &RasterCall) -> Result<Value, String> {
    let rect = call.rect;
    if rect[2] <= 0.0 || rect[3] <= 0.0 {
        return Ok(json!({"empty": true}));
    }
    let max_side = 768.0_f64;
    let ds = call.ds.max(1e-6).max(rect[2].max(rect[3]) / max_side);
    let w = ((rect[2] / ds).ceil() as usize).clamp(1, max_side as usize);
    let h = ((rect[3] / ds).ceil() as usize).clamp(1, max_side as usize);
    let origin = [rect[0], rect[1]];
    let rx1 = rect[0] + rect[2];
    let ry1 = rect[1] + rect[3];
    let sel: std::collections::HashSet<&str> = call.selected.iter().map(String::as_str).collect();

    // Gather pointers to visible objects (zero cloning)
    let mut visible: Vec<(bool, &crate::HistObject, [f64; 4])> = Vec::new();
    for o in &analysis.cells {
        let (x0, y0, x1, y1) = crate::polygon::bounds(&o.roi);
        if x1 >= rect[0] && y1 >= rect[1] && x0 <= rx1 && y0 <= ry1 {
            visible.push((true, o, [x0, y0, x1, y1]));
        }
    }
    for o in &analysis.tiles {
        let (x0, y0, x1, y1) = crate::polygon::bounds(&o.roi);
        if x1 >= rect[0] && y1 >= rect[1] && x0 <= rx1 && y0 <= ry1 {
            visible.push((false, o, [x0, y0, x1, y1]));
        }
    }
    if visible.is_empty() {
        return Ok(json!({"empty": true}));
    }

    let mut nuc = vec![0u32; w * h];
    let mut cell = vec![0u32; w * h];
    let (mut min_px, mut min_py, mut max_px, mut max_py) = (w, h, 0usize, 0usize);

    for (idx, &(_, o, [bx0, by0, bx1, by1])) in visible.iter().enumerate() {
        let id = idx as u32 + 1;
        let px0 = (((bx0 - origin[0]) / ds).floor() as isize).clamp(0, w as isize - 1) as usize;
        let py0 = (((by0 - origin[1]) / ds).floor() as isize).clamp(0, h as isize - 1) as usize;
        let px1 = (((bx1 - origin[0]) / ds).ceil() as isize).clamp(0, w as isize - 1) as usize;
        let py1 = (((by1 - origin[1]) / ds).ceil() as isize).clamp(0, h as isize - 1) as usize;
        min_px = min_px.min(px0);
        min_py = min_py.min(py0);
        max_px = max_px.max(px1);
        max_py = max_py.max(py1);

        // Fast stamp when a cell is <= 3 screen pixels wide/tall (10x faster at overview zoom)
        if px1.saturating_sub(px0) <= 3 && py1.saturating_sub(py0) <= 3 {
            for py in py0..=py1 {
                let row = py * w;
                for px in px0..=px1 {
                    cell[row + px] = id;
                    nuc[row + px] = id;
                }
            }
            continue;
        }
        crate::polygon::fill(&o.roi, &mut cell, w, h, origin, ds, id);
        match &o.nucleus {
            Some(n) => crate::polygon::fill(n, &mut nuc, w, h, origin, ds, id),
            None if o.class.is_some() || o.intensity.is_some() => crate::polygon::fill(&o.roi, &mut nuc, w, h, origin, ds, id),
            None => {}
        }
    }

    if min_px > max_px || min_py > max_py {
        return Ok(json!({"empty": true}));
    }

    let a_fill = (call.alpha.clamp(0.0, 1.0) * 145.0) as u8;
    let a_line = (call.alpha.clamp(0.0, 1.0) * 235.0) as u8;
    let mut rgba = vec![0u8; w * h * 4];
    for y in min_py..=max_py {
        let row = y * w;
        for x in min_px..=max_px {
            let i = row + x;
            let (n, c) = (nuc[i], cell[i]);
            if c == 0 && n == 0 {
                continue;
            }
            let edge = c != 0
                && (x == 0
                    || y == 0
                    || x + 1 == w
                    || y + 1 == h
                    || cell[i - 1] != c
                    || cell[i + 1] != c
                    || cell[i - w] != c
                    || cell[i + w] != c);
            let (id, a) = if edge { (c, a_line) } else if n != 0 { (n, a_fill) } else { continue };
            let (is_cell, obj, _) = visible[id as usize - 1];
            let picked = !sel.is_empty() && sel.contains(format!("{}{}", if is_cell { "c" } else { "t" }, obj.id).as_str());
            let (rgb, a) = if picked && edge { ([255, 255, 255], 255) } else { (object_color(obj), a) };
            let dst = i * 4;
            rgba[dst..dst + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], a]);
        }
    }

    let mut png_bytes = Vec::with_capacity(w * h / 4);
    image::ImageEncoder::write_image(
        image::codecs::png::PngEncoder::new(&mut png_bytes),
        &rgba,
        w as u32,
        h as u32,
        image::ExtendedColorType::Rgba8,
    )
    .map_err(|e| format!("raster png encode: {e}"))?;
    Ok(json!({
        "empty": false,
        "width": w,
        "height": h,
        "png": base64_encode(&png_bytes),
    }))
}

fn objects_max() -> usize {
    OBJECTS_MAX
}

/// The images' analyses and warm ML model plugins, kept between calls.
#[derive(Default)]
pub struct Service {
    analyses: Mutex<HashMap<PathBuf, Analysis>>,
    models: Mutex<PluginModels>,
}

/// A body that does not read, said with the method's name.
fn bad(method: &'static str) -> impl Fn(serde_json::Error) -> String {
    move |e| format!("{method}: {e}")
}

fn counts(a: &Analysis) -> Value {
    json!({"tissue": a.tissue.len(), "cells": a.cells.len(), "tiles": a.tiles.len()})
}

impl Service {
    /// Answers `method` with `body`; `resolve` turns a path the caller names into one it may read
    /// (the server's data folder; refused outside it).
    pub fn call(&self, method: &str, body: Value, resolve: &dyn Fn(&str) -> Result<PathBuf, String>, _caller: Caller) -> Result<Value, String> {
        match method {
            "apply" => {
                let t0 = std::time::Instant::now();
                let call: ApplyCall = serde_json::from_value(body).map_err(bad("apply"))?;
                let path = resolve(&call.path)?;
                let pyramid = lucem_io::Pyramid::open(&path).map_err(|e| format!("{}: {e}", call.path))?;
                let client = lucem_core::ComputeTarget::from_env().and_then(|t| t.client()).map_err(|e| e.to_string())?;
                // The analysis is taken out while the step runs: other images go on meanwhile
                let mut analysis = self.analyses.lock().unwrap_or_else(|e| e.into_inner()).remove(&path).unwrap_or_default();
                if let Some(tissue) = call.tissue {
                    analysis.tissue = tissue;
                }
                let cancel = AtomicBool::new(false);
                let mut models = self.models.lock().unwrap_or_else(|e| e.into_inner());
                let mut progress = |_: &str, _: Option<f32>| {};
                let mut ctx = Ctx { client: &client, pyramid: &pyramid, z: call.z, models: &mut *models, progress: &mut progress, cancel: &cancel };
                let outcome = apply(&mut ctx, &mut analysis, &call.action);
                let mut answer = counts(&analysis);
                // The regions themselves (few, small): the caller shows them as annotations
                answer["regions"] = json!(analysis.tissue);
                let n_cells = analysis.cells.len();
                self.analyses.lock().unwrap_or_else(|e| e.into_inner()).insert(path, analysis);
                answer["outcome"] = serde_json::to_value(outcome.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
                eprintln!("hist::service: apply finished in {:.0} ms (cells={n_cells})", t0.elapsed().as_secs_f64() * 1e3);
                Ok(answer)
            }
            "summary" => {
                let call: PathCall = serde_json::from_value(body).map_err(bad("summary"))?;
                let path = resolve(&call.path)?;
                Ok(self.analyses.lock().unwrap_or_else(|e| e.into_inner()).get(&path).map_or_else(|| counts(&Analysis::default()), counts))
            }
            "objects" => {
                let call: ObjectsCall = serde_json::from_value(body).map_err(bad("objects"))?;
                let path = resolve(&call.path)?;
                let analyses = self.analyses.lock().unwrap_or_else(|e| e.into_inner());
                let all = analyses.get(&path).map_or(&[][..], |a| if call.kind == Kind::Cells { &a.cells[..] } else { &a.tiles[..] });
                let page = all.iter().skip(call.offset).take(call.limit.min(OBJECTS_MAX));
                if call.simplify {
                    let slim: Vec<crate::HistObject> = page
                        .map(|o| {
                            let mut c = o.clone();
                            c.roi = simplify_poly(&o.roi, 10);
                            c.nucleus = o.nucleus.as_ref().map(|n| simplify_poly(n, 8));
                            c
                        })
                        .collect();
                    Ok(json!({"total": all.len(), "objects": slim}))
                } else {
                    Ok(json!({"total": all.len(), "objects": page.collect::<Vec<_>>()}))
                }
            }
            "raster" => {
                let t0 = std::time::Instant::now();
                let call: RasterCall = serde_json::from_value(body).map_err(bad("raster"))?;
                let path = resolve(&call.path)?;
                let analyses = self.analyses.lock().unwrap_or_else(|e| e.into_inner());
                let Some(analysis) = analyses.get(&path) else {
                    return Ok(json!({"empty": true}));
                };
                let res = raster_viewport_png(analysis, &call);
                eprintln!("hist::service: raster finished in {:.0} ms (total_cells={})", t0.elapsed().as_secs_f64() * 1e3, analysis.cells.len());
                res
            }
            "reset" => {
                let call: PathCall = serde_json::from_value(body).map_err(bad("reset"))?;
                let path = resolve(&call.path)?;
                self.analyses.lock().unwrap_or_else(|e| e.into_inner()).remove(&path);
                Ok(json!({}))
            }
            other => Err(format!("the histology service has no method {other:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calls are read by method; an image with no analysis yet answers empty; paths go through
    /// the caller's `resolve` (refused there, refused here); unknown methods and bodies are refused.
    #[test]
    fn methods_answer_and_refuse() {
        let service = Service::default();
        let resolve = |p: &str| -> Result<PathBuf, String> { if p.contains("..") { Err("not allowed".into()) } else { Ok(PathBuf::from("/data").join(p)) } };
        let caller = Caller { name: "test", may_write: false };
        assert_eq!(service.call("summary", json!({"path": "a.ome.zarr"}), &resolve, caller).unwrap(), json!({"tissue": 0, "cells": 0, "tiles": 0}));
        assert_eq!(service.call("objects", json!({"path": "a.ome.zarr", "kind": "cells"}), &resolve, caller).unwrap(), json!({"total": 0, "objects": []}));
        assert_eq!(service.call("reset", json!({"path": "a.ome.zarr"}), &resolve, caller).unwrap(), json!({}));
        assert!(service.call("summary", json!({"path": "../x"}), &resolve, caller).is_err());
        assert!(service.call("objects", json!({"path": "a", "kind": "nuclei"}), &resolve, caller).is_err(), "no such kind");
        assert!(service.call("nothing", json!({}), &resolve, caller).is_err());
    }
}

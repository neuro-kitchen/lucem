//! Machine-learning plugins from the app's side (`lumen-ml`'s contract): the plugin programs found,
//! one running server per program (so a model stays loaded, and SAM keeps the embedding of the
//! region between clicks), and the jobs sent to them. Everything here blocks: the store calls it
//! off the UI thread.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lumen_ml::{PluginInfo, PluginProcess, Prompt, RasterRef, Request, SegmentResult};

use crate::domain::annotation::Shape;
use crate::domain::geometry;

/// A running plugin server, started on first use and again if it stopped.
pub type Server = Arc<Mutex<Option<PluginProcess>>>;

/// What the app knows of the plugins.
#[derive(Default)]
pub struct MlState {
    pub plugins: Vec<PluginInfo>,
    /// Programs that did not answer, and why.
    pub problems: Vec<String>,
    pub scanned: bool,
    pub scanning: bool,
    /// What a plugin is doing now (an install, a segmentation), and how far (0–1).
    pub busy: Option<(String, Option<f32>)>,
    servers: HashMap<String, Server>,
}

impl MlState {
    /// The server of `program` (not started until a request needs it).
    pub fn server(&mut self, program: &str) -> Server {
        self.servers.entry(program.to_string()).or_default().clone()
    }

    /// Drops every server (after a rescan: programs may have changed).
    pub fn stop_servers(&mut self) {
        self.servers.clear();
    }

    /// The model to segment with: `chosen` when a plugin offers it, else the first default model;
    /// its program and whether it is installed.
    pub fn segmenter(&self, chosen: Option<&str>) -> Option<(String, String, bool)> {
        let interactive = |m: &&lumen_ml::ModelInfo| m.task == lumen_ml::Task::InteractiveSegmentation;
        let pick = |p: &PluginInfo, m: &lumen_ml::ModelInfo| (p.program.clone(), m.id.clone(), m.installed);
        if let Some(id) = chosen {
            for p in &self.plugins {
                if let Some(m) = p.models.iter().filter(interactive).find(|m| m.id == id) {
                    return Some(pick(p, m));
                }
            }
        }
        self.plugins.iter().find_map(|p| p.models.iter().filter(interactive).find(|m| m.default).or_else(|| p.models.iter().find(interactive)).map(|m| pick(p, m)))
    }
}

/// Runs `req` on `program`'s server (starting it if needed); `progress` sees its progress.
pub fn request(server: &Server, program: &str, req: &Request, progress: &mut dyn FnMut(&str, Option<f32>)) -> Result<serde_json::Value, String> {
    let mut slot = server.lock().map_err(|e| e.to_string())?;
    if !slot.as_mut().is_some_and(PluginProcess::alive) {
        *slot = Some(PluginProcess::start(program)?);
    }
    let p = slot.as_mut().ok_or("the plugin did not start")?;
    let r = p.request(req, progress);
    if r.is_err() && !p.alive() {
        *slot = None;
    }
    r
}

/// A region to segment and what was clicked on it.
#[derive(Clone)]
pub struct SegmentJob {
    pub image: String,
    /// The plane the annotations belong to (a volume), else `None`.
    pub z: Option<usize>,
    pub class: String,
    /// Level-0 position of the region's first pixel, and level-0 px per region pixel.
    pub origin: [f64; 2],
    pub ds: f64,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    /// In region pixels.
    pub prompts: Vec<Prompt>,
    /// Names the region's content (the plugin reuses its embedding for the same key).
    pub key: String,
}

/// Sends the job, reads the best mask and traces it into level-0 regions (pieces under 5 % of
/// the largest are dropped); the model's score.
pub fn segment(server: &Server, program: &str, model: &str, job: &SegmentJob, progress: &mut dyn FnMut(&str, Option<f32>)) -> Result<(Vec<Shape>, f32, SegmentResult), String> {
    let dir = std::env::temp_dir().join(format!("lumen-sam-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let image = dir.join("region.rgb");
    std::fs::write(&image, &job.rgb).map_err(|e| e.to_string())?;
    let req = Request::Segment {
        model: model.to_string(),
        image: RasterRef { path: image.to_string_lossy().into_owned(), width: job.width, height: job.height, channels: 3 },
        image_key: job.key.clone(),
        prompts: job.prompts.clone(),
        multimask: false,
        out_dir: dir.to_string_lossy().into_owned(),
    };
    let value = request(server, program, &req, progress)?;
    let result: SegmentResult = serde_json::from_value(value).map_err(|e| format!("the plugin's answer does not read: {e}"))?;
    let best = result.masks.first().ok_or("the model gave no mask")?;
    let bytes = std::fs::read(&best.raster.path).map_err(|e| e.to_string())?;
    let (w, h) = (best.raster.width as usize, best.raster.height as usize);
    if bytes.len() != w * h {
        return Err(format!("the mask holds {} bytes, not {w} × {h}", bytes.len()));
    }
    let mask: Vec<u8> = bytes.iter().map(|&v| u8::from(v != 0)).collect();
    let shapes = mask_to_regions(&mask, w, h, job.origin, job.ds);
    if shapes.is_empty() {
        return Err("the mask is empty".into());
    }
    Ok((shapes, best.score, result))
}

/// What carrying an object through z needs: the region of the first job on the other planes of
/// its level, read through the shared tile cache.
#[derive(Clone)]
pub struct ZRun {
    pub reader: lumen_view::RegionReader,
    pub level: usize,
    /// The plane of `level` the first job was on, and how many the level has.
    pub plane: usize,
    pub depth: usize,
    /// Level-0 planes per plane of `level`.
    pub z_downsample: f64,
    /// The region (level px) and the channels shown, in their display.
    pub x0: usize,
    pub y0: usize,
    pub displays: Vec<(usize, lumen_view::ChannelDisplay)>,
}

/// The most planes followed each way.
pub const MAX_Z_STEPS: usize = 256;
/// Below this score the model is not sure the object is there any more.
pub const MIN_Z_SCORE: f32 = 0.5;

/// The bounding box of level-0 shapes in the region pixels of `job`.
fn region_box(shapes: &[Shape], job: &SegmentJob) -> Option<[f32; 4]> {
    let b = shapes.iter().map(Shape::bounds).reduce(|a, b| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)))?;
    let to = |x: f64, o: f64| ((x - o) / job.ds) as f32;
    Some([to(b.0, job.origin[0]), to(b.1, job.origin[1]), to(b.2, job.origin[0]), to(b.3, job.origin[1])])
}

/// Follows the object `first` outlined on the first job's plane through the planes above and
/// below: each plane is prompted with the box of the one before it (10 % larger), and the run
/// stops where the model's score falls under [`MIN_Z_SCORE`], the mask empties, or its box grows
/// or shrinks more than three times. Returns the regions of each plane (level-0 plane, shapes).
pub fn carry_through_z(server: &Server, program: &str, model: &str, job: &SegmentJob, first: &[Shape], run: &ZRun, progress: &mut dyn FnMut(usize)) -> Result<Vec<(usize, Vec<Shape>)>, String> {
    let start = region_box(first, job).ok_or("the first plane gave no region")?;
    let (w, h) = (job.width as usize, job.height as usize);
    let area = |b: &[f32; 4]| ((b[2] - b[0]) * (b[3] - b[1])).max(1.0);
    let mut out = Vec::new();
    for dir in [1i64, -1] {
        let mut bx = start;
        let mut p = run.plane as i64 + dir;
        let mut steps = 0;
        while p >= 0 && (p as usize) < run.depth && steps < MAX_Z_STEPS {
            let plane = p as usize;
            let mut planes = Vec::new();
            for (c, d) in &run.displays {
                planes.push((run.reader.read_region(run.level, plane, *c, run.x0, run.y0, w, h)?, *d));
            }
            let refs: Vec<(&[f32], lumen_view::ChannelDisplay)> = planes.iter().map(|(p, d)| (p.as_slice(), *d)).collect();
            let bgra = lumen_view::composite_bgra(&refs, w * h);
            let pad = 0.1 * (bx[2] - bx[0]).max(bx[3] - bx[1]);
            let prompt = Prompt::Box { x0: (bx[0] - pad).max(0.0), y0: (bx[1] - pad).max(0.0), x1: (bx[2] + pad).min(w as f32), y1: (bx[3] + pad).min(h as f32) };
            let z = (plane as f64 * run.z_downsample).round() as usize;
            let next = SegmentJob { rgb: bgra.as_chunks::<4>().0.iter().flat_map(|p| [p[2], p[1], p[0]]).collect(), prompts: vec![prompt], key: format!("{}-z{plane}", job.key), z: Some(z), ..job.clone() };
            let Ok((shapes, score, _)) = segment(server, program, model, &next, &mut |_, _| {}) else { break };
            let Some(nb) = region_box(&shapes, &next) else { break };
            let ratio = area(&nb) / area(&bx);
            if score < MIN_Z_SCORE || !(1.0 / 3.0..=3.0).contains(&ratio) {
                break;
            }
            out.push((z, shapes));
            progress(out.len());
            bx = nb;
            p += dir;
            steps += 1;
        }
    }
    Ok(out)
}

/// A 0 / 1 mask (region pixels) as level-0 polygons; pieces under 5 % of the largest are dropped.
pub fn mask_to_regions(mask: &[u8], w: usize, h: usize, origin: [f64; 2], ds: f64) -> Vec<Shape> {
    let area = |p: &Vec<[f64; 2]>| p.iter().zip(p.iter().cycle().skip(1)).map(|(a, b)| a[0] * b[1] - b[0] * a[1]).sum::<f64>().abs() / 2.0;
    let polys = geometry::mask_to_polygons(mask, w, h, 8);
    let largest = polys.iter().map(area).fold(0.0, f64::max);
    polys
        .into_iter()
        .filter(|p| area(p) >= largest * 0.05)
        .map(|p| geometry::simplify(&p, 0.75))
        .filter(|p| p.len() >= 3)
        .map(|p| Shape::Polygon { points: p.into_iter().map(|[x, y]| [origin[0] + x * ds, origin[1] + y * ds]).collect() })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SAM follows a sphere of the sample volume through z (needs `lumen-ml-sam` built, ONNX
    /// Runtime and sam2-hiera-tiny installed: run with `--ignored`).
    #[test]
    #[ignore = "runs the SAM plugin"]
    fn sam_follows_an_object_through_z() {
        let program = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/lumen-ml-sam");
        assert!(program.exists(), "build lumen-ml-sam first");
        let program = program.to_string_lossy().into_owned();
        let py = Arc::new(lumen_io::sample_volume(256, 32).unwrap());
        let (service, _rx) = lumen_view::TileService::new(py.clone(), 1);
        let reader = service.region_reader();
        let (w, h, plane) = (256usize, 256usize, 16usize);
        let display = lumen_view::ChannelDisplay::new([255, 255, 255], (0.0, 1.0));
        let nuclei = reader.read_region(0, plane, 0, 0, 0, w, h).unwrap();
        let (best, _) = nuclei.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        let (px, py_) = ((best % w) as f32, (best / w) as f32);
        let bgra = lumen_view::composite_bgra(&[(nuclei.as_slice(), display)], w * h);
        let job = SegmentJob {
            image: "img1".into(),
            z: Some(plane),
            class: "Region".into(),
            origin: [0.0, 0.0],
            ds: 1.0,
            width: w as u32,
            height: h as u32,
            rgb: bgra.as_chunks::<4>().0.iter().flat_map(|p| [p[2], p[1], p[0]]).collect(),
            prompts: vec![Prompt::Point { x: px, y: py_, positive: true }],
            key: "test-through-z".into(),
        };
        let server: Server = Arc::new(Mutex::new(None));
        let (first, score, _) = segment(&server, &program, "sam2-hiera-tiny", &job, &mut |_, _| {}).unwrap();
        assert!(score > 0.5, "first plane: score {score}");
        let run = ZRun { reader, level: 0, plane, depth: 32, z_downsample: 1.0, x0: 0, y0: 0, displays: vec![(0, display)] };
        let planes = carry_through_z(&server, &program, "sam2-hiera-tiny", &job, &first, &run, &mut |_| {}).unwrap();
        let zs: Vec<usize> = planes.iter().map(|(z, _)| *z).collect();
        eprintln!("object at ({px}, {py_}) z {plane}: followed on planes {zs:?}");
        assert!(!zs.is_empty() && zs.len() < 31, "{zs:?}");
        assert!(zs.iter().all(|&z| z != plane) && zs.iter().any(|&z| z > plane) && zs.iter().any(|&z| z < plane), "{zs:?}");
        let _ = server.lock().map(|mut s| s.take());
    }

    #[test]
    fn a_mask_becomes_regions_in_level0_pixels() {
        let (w, h) = (20, 10);
        let mut mask = vec![0u8; w * h];
        for y in 2..8 {
            for x in 4..14 {
                mask[y * w + x] = 1;
            }
        }
        // A speck far smaller than the region is dropped
        mask[w - 1] = 1;
        let shapes = mask_to_regions(&mask, w, h, [100.0, 50.0], 2.0);
        assert_eq!(shapes.len(), 1);
        let (x0, y0, x1, y1) = shapes[0].bounds();
        assert!((x0 - 108.0).abs() <= 2.0 && (y0 - 54.0).abs() <= 2.0 && (x1 - 128.0).abs() <= 2.0 && (y1 - 66.0).abs() <= 2.0, "{:?}", shapes[0].bounds());
    }

    #[test]
    fn the_chosen_model_wins_over_the_default() {
        let model = |id: &str, default: bool| lumen_ml::ModelInfo { id: id.into(), name: id.into(), family: "sam".into(), task: lumen_ml::Task::InteractiveSegmentation, domains: vec![], description: String::new(), licence: "Apache-2.0".into(), paper: None, code: None, weights: vec![], installed: id == "b", default };
        let plugin = PluginInfo { name: "SAM".into(), version: "0".into(), licence: "Apache-2.0".into(), runtime: lumen_ml::RuntimeStatus { ready: true, detail: String::new() }, models: vec![model("a", true), model("b", false)], program: "/p".into() };
        let ml = MlState { plugins: vec![plugin], ..Default::default() };
        assert_eq!(ml.segmenter(None), Some(("/p".into(), "a".into(), false)));
        assert_eq!(ml.segmenter(Some("b")), Some(("/p".into(), "b".into(), true)));
        assert_eq!(ml.segmenter(Some("zzz")).map(|s| s.1), Some("a".into()));
    }
}

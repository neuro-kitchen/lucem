//! The pipeline's Model node: a model plugin run on the node's inputs over the region evaluated.
//!
//! - Nucleus models (`Task::Segmentation`): every nucleus of the region, as labels.
//! - Segment Anything (`Task::InteractiveSegmentation`): one object per box or point annotation
//!   of the node's prompt class on the plane shown (regions give their bounding box), label `k`
//!   for the `k`-th prompt; later objects do not cover earlier ones.
//!
//! The node reads its image itself: the channels chosen (by default the image's three colour
//! channels, else its first) become 8-bit RGB, each from its 1st to 99.8th percentile; one
//! channel fills red, green and blue. Answers are cached by what was sent, so panning back over a
//! region, or a downstream change, does not run the model again.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use lumen_ml::{InstancesResult, Prompt, RasterRef, Request, SegmentResult, Task};

use crate::domain::annotation::Shape;
use crate::domain::project::ProjectDoc;

/// Answers kept (labels by request hash).
const CACHE: usize = 12;

/// A model node's last run: its labels and where they lie (level `ds` pixels from `x0, y0`).
#[derive(Clone)]
pub struct Stored {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    pub ds: f64,
    pub labels: Vec<u32>,
}

/// Model nodes run only when asked (Run in the Inspector): the ids asked, and each node's last
/// run by its signature (its id and settings: changed settings, no result until Run again).
fn armed() -> &'static Mutex<std::collections::HashSet<String>> {
    static A: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    A.get_or_init(Default::default)
}

fn stored() -> &'static Mutex<HashMap<String, Stored>> {
    static S: OnceLock<Mutex<HashMap<String, Stored>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// What a running model is doing (the plugin's last progress), for the status bar.
fn progress_slot() -> &'static Mutex<Option<String>> {
    static P: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    P.get_or_init(Default::default)
}

pub fn progress() -> Option<String> {
    progress_slot().lock().ok()?.clone()
}

fn set_progress(p: Option<String>) {
    if let Ok(mut s) = progress_slot().lock() {
        *s = p;
    }
}

/// Asks node `id` to run on its next evaluation.
pub fn arm(id: &str) {
    if let Ok(mut a) = armed().lock() {
        a.insert(id.to_string());
    }
}

/// Whether node `id` was asked to run (and takes the request).
pub fn take_armed(id: &str) -> bool {
    armed().lock().is_ok_and(|mut a| a.remove(id))
}

pub fn store_run(signature: String, run: Stored) {
    if let Ok(mut s) = stored().lock() {
        s.insert(signature, run);
    }
}

/// The last run of a node with this signature, if any.
pub fn last_run(signature: &str) -> Option<Stored> {
    stored().lock().ok()?.get(signature).cloned()
}

/// A stored run sampled over a region (`x0, y0, w, h` at `ds` level-0 px per pixel); 0 outside it.
pub fn sample(run: &Stored, (x0, y0, w, h): (usize, usize, usize, usize), ds: f64) -> Vec<u32> {
    let mut out = vec![0u32; w * h];
    for y in 0..h {
        let sy = (((y0 + y) as f64 + 0.5) * ds / run.ds).floor() as isize - run.y0 as isize;
        if sy < 0 || sy as usize >= run.h {
            continue;
        }
        for x in 0..w {
            let sx = (((x0 + x) as f64 + 0.5) * ds / run.ds).floor() as isize - run.x0 as isize;
            if sx >= 0 && (sx as usize) < run.w {
                out[y * w + x] = run.labels[sy as usize * run.w + sx as usize];
            }
        }
    }
    out
}

/// Labels by request hash, and the hashes oldest first.
type Cache = (HashMap<u64, Vec<u32>>, VecDeque<u64>);

fn cache() -> &'static Mutex<Cache> {
    static C: OnceLock<Mutex<Cache>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// The channels a model node reads: those chosen (at most three), else the first three of a
/// three-channel image, else the first.
/// The channels a model node reads: those chosen (at most three), else the three of an RGB
/// image, else the nuclear channel (named DAPI, Hoechst, …), else the first.
pub fn channels_of(chosen: &[usize], labels: &[String]) -> Vec<usize> {
    let available = labels.len();
    let ok: Vec<usize> = chosen.iter().copied().filter(|&c| c < available).take(3).collect();
    let nuclear = labels.iter().position(|l| ["dapi", "hoechst", "nuc", "dna"].iter().any(|k| l.to_lowercase().contains(k)));
    if !ok.is_empty() {
        ok
    } else if available == 3 && nuclear.is_none() {
        vec![0, 1, 2]
    } else {
        vec![nuclear.unwrap_or(0)]
    }
}

fn stretch(v: &[f32]) -> Vec<u8> {
    let mut s: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return vec![0; v.len()];
    }
    s.sort_by(f32::total_cmp);
    let lo = s[(s.len() as f64 * 0.01) as usize];
    let hi = s[((s.len() - 1) as f64 * 0.998) as usize];
    let k = 255.0 / (hi - lo).max(1e-12);
    v.iter().map(|&x| ((x - lo) * k).round().clamp(0.0, 255.0) as u8).collect()
}

/// The prompts of class `class` on `image`, plane `z`, inside the region (`x0, y0` level px at
/// `ds` level-0 px per pixel, `w × h`), in region pixels.
fn prompts(doc: &ProjectDoc, image: &str, class: &str, z: usize, (x0, y0, w, h): (usize, usize, usize, usize), ds: f64) -> Vec<Prompt> {
    let to = |x: f64, y: f64| ((x / ds - x0 as f64) as f32, (y / ds - y0 as f64) as f32);
    let inside = |x: f32, y: f32| x >= 0.0 && y >= 0.0 && x < w as f32 && y < h as f32;
    doc.annotations_of(image)
        .filter(|a| a.class == class && a.on_plane(z))
        .filter_map(|a| match &a.shape {
            Shape::Point { x, y } => {
                let (px, py) = to(*x, *y);
                inside(px, py).then_some(Prompt::Point { x: px, y: py, positive: true })
            }
            s if s.is_region() => {
                let o = s.outline();
                let (bx0, by0, bx1, by1) = o.iter().fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, p| (b.0.min(p[0]), b.1.min(p[1]), b.2.max(p[0]), b.3.max(p[1])));
                let (ax, ay) = to(bx0, by0);
                let (bx, by) = to(bx1, by1);
                (inside(ax.max(0.0), ay.max(0.0)) && bx > 0.0 && by > 0.0).then_some(Prompt::Box { x0: ax.max(0.0), y0: ay.max(0.0), x1: bx.min(w as f32 - 1.0), y1: by.min(h as f32 - 1.0) })
            }
            _ => None,
        })
        .collect()
}

/// Runs model `model` on `channels` (0–1 values, `w × h` each, one to three) of the region;
/// `pixel_um`: µm per region pixel.
#[allow(clippy::too_many_arguments)]
pub fn run(doc: &ProjectDoc, image: &str, model: &str, prompt_class: Option<&str>, channels: &[Vec<f32>], w: usize, h: usize, z: usize, region: (usize, usize, usize, usize), ds: f64, pixel_um: Option<f64>) -> Result<Vec<u32>, String> {
    let (program, info) = super::ml::find_model(model).ok_or_else(|| format!("no plugin offers the model {model:?} (Settings → Models)"))?;
    if !info.installed {
        return Err(format!("{} is not installed: Settings → Models → Install", info.name));
    }
    let planes: Vec<Vec<u8>> = channels.iter().map(|c| stretch(c)).collect();
    let pick = |c: usize| &planes[c.min(planes.len() - 1)];
    // A one-channel model (fluorescence) gets the first channel chosen; others RGB
    let n_in = if info.channels == 1 { 1 } else { 3 };
    let rgb: Vec<u8> = if n_in == 1 { pick(0).clone() } else { (0..w * h).flat_map(|i| [pick(0)[i], pick(1)[i], pick(2)[i]]).collect() };
    let ps = match info.task {
        Task::InteractiveSegmentation => {
            let class = prompt_class.ok_or("choose the class whose boxes and points prompt the model (Inspector)")?;
            let p = prompts(doc, image, class, z, region, ds);
            if p.is_empty() {
                return Err(format!("no box or point of class {class} in this region"));
            }
            p
        }
        Task::Segmentation => Vec::new(),
        t => return Err(format!("{}: a {} model does not give labels", info.name, t.title())),
    };
    let mut hsh = std::collections::hash_map::DefaultHasher::new();
    (model, w, h, &rgb, format!("{ps:?}")).hash(&mut hsh);
    let key = hsh.finish();
    if let Some(l) = cache().lock().ok().and_then(|c| c.0.get(&key).cloned()) {
        return Ok(l);
    }
    // A folder per request: model nodes may run at the same time
    let dir = std::env::temp_dir().join(format!("lumen-model-node-{}", std::process::id())).join(format!("{key:016x}-{:?}", std::thread::current().id()).replace(['(', ')'], ""));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(dir.clone());
    let path = dir.join("region.rgb");
    std::fs::write(&path, &rgb).map_err(|e| e.to_string())?;
    let raster = RasterRef { path: path.to_string_lossy().into_owned(), width: w as u32, height: h as u32, channels: n_in };
    let server = super::ml::shared_server(&program);
    let out_dir = dir.to_string_lossy().into_owned();
    let labels = if ps.is_empty() {
        set_progress(Some(format!("{}: starting", info.name)));
        let name = info.name.clone();
        let v = super::ml::request(&server, &program, &Request::Instances { model: model.into(), image: raster, pixel_um, out_dir }, &mut |stage, f| {
            set_progress(Some(format!("{name}: {stage}{}", f.map_or(String::new(), |f| format!(" ({:.0} %)", f * 100.0)))));
        });
        set_progress(None);
        let v = v?;
        let r: InstancesResult = serde_json::from_value(v).map_err(|e| format!("the plugin's answer does not read: {e}"))?;
        if r.classes.is_empty() {
            vec![0; w * h]
        } else {
            let (l, lw, lh) = lumen_hist::models::labels_from_result(&r).map_err(|e| e.to_string())?;
            if (lw, lh) != (w, h) {
                return Err(format!("the model answered {lw} × {lh} for {w} × {h}"));
            }
            l
        }
    } else {
        let mut labels = vec![0u32; w * h];
        for (k, p) in ps.iter().enumerate() {
            let req = Request::Segment { model: model.into(), image: raster.clone(), image_key: format!("{key:x}"), prompts: vec![*p], multimask: false, out_dir: out_dir.clone() };
            let v = super::ml::request(&server, &program, &req, &mut |_, _| {})?;
            let r: SegmentResult = serde_json::from_value(v).map_err(|e| format!("the plugin's answer does not read: {e}"))?;
            let Some(best) = r.masks.first() else { continue };
            let m = std::fs::read(&best.raster.path).map_err(|e| e.to_string())?;
            if m.len() != w * h {
                return Err(format!("the mask holds {} bytes, not {w} × {h}", m.len()));
            }
            for (l, &v) in labels.iter_mut().zip(&m) {
                if v != 0 && *l == 0 {
                    *l = k as u32 + 1;
                }
            }
        }
        labels
    };
    if let Ok(mut c) = cache().lock() {
        c.0.insert(key, labels.clone());
        c.1.push_back(key);
        while c.1.len() > CACHE {
            if let Some(old) = c.1.pop_front() {
                c.0.remove(&old);
            }
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use crate::domain::graph::{EdgeDoc, GraphDoc, GraphNode, NodeKind, ViewStyle};
    use crate::domain::project::{ImageRef, ProjectDoc};

    /// A graph of three sources (R, G, B) → the StarDist node → a view, analysed over an H&E
    /// OME-Zarr: labels come out, and asking again is answered from the cache. Needs
    /// `lumen-ml-stardist` built, its model installed, and `LUMEN_HE_TEST` naming an RGB
    /// OME-Zarr with a pixel size.
    #[test]
    #[ignore = "needs the StarDist plugin, its model, and an H&E image"]
    fn a_model_node_labels_nuclei() {
        let path = std::path::PathBuf::from(std::env::var("LUMEN_HE_TEST").expect("LUMEN_HE_TEST"));
        let found: Vec<_> = lumen_ml::discover().into_iter().filter_map(Result::ok).collect();
        let mut ml = crate::data::ml::MlState::default();
        ml.set_plugins(found);
        assert!(crate::data::ml::find_model("stardist-he").is_some_and(|(_, m)| m.installed), "stardist-he not installed");
        let py = crate::data::store::open_pyramid(&path).unwrap();
        let mut doc = ProjectDoc::default();
        doc.images.push(ImageRef { id: "img1".into(), path: path.clone(), name: "he".into(), channels: Vec::new() });
        let node = |id: &str, kind: NodeKind| GraphNode { id: id.into(), x: 0.0, y: 0.0, kind };
        let nodes = vec![node("model", NodeKind::Model { image: "img1".into(), channels: Vec::new(), model: "stardist-he".into(), roi: None, prompts: None, enabled: true }), node("view", NodeKind::View { name: "Nuclei".into(), style: ViewStyle::default() })];
        let edges = vec![EdgeDoc { from: "model".into(), to: "view".into(), port: 0 }];
        doc.graph = GraphDoc { nodes, edges };
        let device = crate::data::device::Device::discover(None).unwrap();
        let analysis = crate::data::results::analyze(&device.client, &doc, &py, "img1", 0);
        let req = crate::data::results::ResultRequest { image: "img1".into(), rect: [0.0, 0.0, py.width() as f64, py.height() as f64], zoom: 0.5, z: 0 };
        // Not run yet: it says so, and runs nothing
        let none = crate::data::results::compute(&device.client, &doc, &py, &req, &analysis);
        assert!(none.iter().any(|r| r.error.as_deref().is_some_and(|e| e.contains("press Run"))), "{:?}", none.iter().map(|r| r.error.clone()).collect::<Vec<_>>());
        // Run: labels; afterwards drawn from that run, without running again
        super::arm("model");
        let t0 = std::time::Instant::now();
        let first = crate::data::results::compute(&device.client, &doc, &py, &req, &analysis);
        let t_first = t0.elapsed();
        assert!(first.iter().all(|r| r.error.is_none()) && first.iter().any(|r| r.layer.is_some()), "{:?}", first.iter().map(|r| r.error.clone()).collect::<Vec<_>>());
        let t1 = std::time::Instant::now();
        let again = crate::data::results::compute(&device.client, &doc, &py, &req, &analysis);
        let t_again = t1.elapsed();
        assert!(again.iter().all(|r| r.error.is_none()));
        eprintln!("run {t_first:?}, shown again {t_again:?}");
        assert!(t_again < t_first / 2, "the second evaluation ran the model again");
    }

    /// Segment Anything as a node: two boxes of class `Prompt` give labels 1 and 2. Needs
    /// `lumen-ml-sam` built, SAM 2 Tiny and ONNX Runtime installed, and `LUMEN_HE_TEST`.
    #[test]
    #[ignore = "needs the SAM plugin, its model, and an H&E image"]
    fn a_sam_node_gives_one_object_per_prompt() {
        use crate::domain::annotation::{Annotation, ClassDoc, Shape};
        let path = std::path::PathBuf::from(std::env::var("LUMEN_HE_TEST").expect("LUMEN_HE_TEST"));
        let mut ml = crate::data::ml::MlState::default();
        ml.set_plugins(lumen_ml::discover().into_iter().filter_map(Result::ok).collect());
        let py = crate::data::store::open_pyramid(&path).unwrap();
        let mut doc = ProjectDoc::default();
        doc.images.push(ImageRef { id: "img1".into(), path: path.clone(), name: "he".into(), channels: Vec::new() });
        doc.classes.push(ClassDoc { name: "Prompt".into(), color: [255, 0, 0], visible: true, opacity: 1.0 });
        for (i, (x, y)) in [(300.0, 300.0), (1500.0, 800.0)].into_iter().enumerate() {
            doc.annotations.push(Annotation { id: format!("a{i}"), image: "img1".into(), shape: Shape::Rect { x, y, w: 60.0, h: 60.0 }, class: "Prompt".into(), note: String::new(), z: None });
        }
        let node = |id: &str, kind: NodeKind| GraphNode { id: id.into(), x: 0.0, y: 0.0, kind };
        doc.graph = GraphDoc { nodes: vec![node("model", NodeKind::Model { image: "img1".into(), channels: Vec::new(), model: "sam2-hiera-tiny".into(), roi: None, prompts: Some("Prompt".into()), enabled: true })], edges: Vec::new() };
        let level = 1;
        let l = py.levels[level].clone();
        let channels: Vec<Vec<f32>> = (0..3).map(|c| py.read_region_at(level, c, 0, 0, 0, l.width, l.height).unwrap()).collect();
        let labels = super::run(&doc, "img1", "sam2-hiera-tiny", Some("Prompt"), &channels, l.width, l.height, 0, (0, 0, l.width, l.height), l.downsample, py.pixel_size_um).unwrap();
        let count = |k: u32| labels.iter().filter(|&&v| v == k).count();
        eprintln!("object 1: {} px, object 2: {} px", count(1), count(2));
        assert!(count(1) > 20 && count(2) > 20);
    }
}

//! InstanSeg (Goldsborough, Bankhead et al. 2024) for lucem, on Burn: nuclei as pixel
//! embeddings. For every pixel the network gives a 2D embedding, its spread (sigma) and a seed
//! score; seeds are the seed map's peaks, and a small trained classifier decides, pixel by pixel
//! around each seed, whether the pixel belongs to it ([`post`]).
//!
//! - Models ([`MODELS`]): the published "brightfield_nuclei" (H&E and DAB, RGB) and
//!   "single_channel_nuclei" (one fluorescence channel), both Apache-2.0, packaged for
//!   bioimage.io on InstanSeg's GitHub releases. Each package (a zip) is fetched and verified
//!   through `lucem-hub` (SHA-256 pinned); its TorchScript archive is read without PyTorch
//!   ([`torchscript`]) and written once as safetensors in lucem's cache, with the model's
//!   thresholds in the metadata ([`install`]). [`InstanSeg::load`] reads only the safetensors.
//!   (The fluorescence "nuclei and cells" model uses another network, the channel-invariant one,
//!   and is not ported yet.)
//! - Network: [`net`] on Burn (CUDA, wgpu or the CPU), BatchNorm folded into the convolutions.
//! - Input: each channel's 0.1st to 99.9th percentile to 0 and 1 (the packages' `scale_range`),
//!   clamped to −2…3 as the archive does; images resampled to the models' 0.5 µm per pixel when
//!   their pixel size is known, and the labels back.
//! - Output: [`post`] (seeds, the pixel classifier, fragments and holes, overlaps joined, the
//!   smallest dropped).

pub mod bench;
pub mod net;
pub mod post;
pub mod torchscript;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use lucem_ml::{ModelInfo, Task, Weight};
use safetensors::tensor::{Dtype, TensorView};

pub use post::{Found, Params};

/// The pixel size the published models work at (µm; their `rdf.yaml` scale).
pub const MODEL_PIXEL_UM: f64 = 0.5;

/// A published InstanSeg model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spec {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    /// The bioimage.io package as published: cache name, URL, SHA-256, bytes.
    pub package: (&'static str, &'static str, &'static str, u64),
    /// The safetensors made from it.
    pub safetensors: &'static str,
    pub default: bool,
    /// Channels of its images: 3 (RGB) or 1.
    pub channels: usize,
    /// The images it is for (`histology`, `fluorescence`).
    pub domains: &'static [&'static str],
}

pub static MODELS: [Spec; 2] = [
    Spec {
        id: "instanseg-brightfield",
        name: "InstanSeg · brightfield nuclei",
        description: "Nuclei of brightfield slides (H&E, DAB), at 0.5 µm per pixel (~15 MB)",
        package: (
            "instanseg_brightfield_nuclei_v0.1.1.zip",
            "https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.1/brightfield_nuclei.zip",
            "427aa9cfe961b6d0012e2abf6befc6a9f76398db5d28dc2302ad911327d1c448",
            15_054_316,
        ),
        safetensors: "instanseg_brightfield_nuclei_v0.1.1.safetensors",
        default: false,
        channels: 3,
        domains: &["histology"],
    },
    Spec {
        id: "instanseg-fluorescence",
        name: "InstanSeg · single-channel nuclei",
        description: "Nuclei of one fluorescence channel (DAPI, Hoechst), at 0.5 µm per pixel (~15 MB)",
        package: (
            "instanseg_single_channel_nuclei_v0.1.2.zip",
            "https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.2/single_channel_nuclei.zip",
            "6bc2e4cd8acd9bea64a04e926a74a53f922d76fd6c1dc633ca0bca335df4d4d7",
            14_585_795,
        ),
        safetensors: "instanseg_single_channel_nuclei_v0.1.2.safetensors",
        default: false,
        channels: 1,
        domains: &["fluorescence"],
    },
];

pub fn spec(id: &str) -> Option<&'static Spec> {
    MODELS.iter().find(|s| s.id == id)
}

pub fn path_of(file: &str) -> PathBuf {
    lucem_hub::cache_dir("instanseg").join(file)
}

/// Whether the safetensors are in the cache.
pub fn installed(s: &Spec) -> bool {
    path_of(s.safetensors).is_file()
}

pub fn info(s: &Spec) -> ModelInfo {
    ModelInfo {
        id: s.id.into(),
        name: s.name.into(),
        family: "instanseg".into(),
        task: Task::Segmentation,
        domains: s.domains.iter().map(|d| d.to_string()).collect(),
        description: s.description.into(),
        licence: "Apache-2.0".into(),
        paper: Some("Goldsborough, Philps, O'Callaghan, Inglis, Leplat, Filby, Bilen, Bankhead (2024) InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation. arXiv:2408.15954".into()),
        code: Some("https://github.com/instanseg/instanseg".into()),
        weights: vec![Weight { file: s.package.0.into(), url: s.package.1.into(), sha256: s.package.2.into(), bytes: s.package.3 }],
        installed: installed(s),
        default: s.default,
        channels: s.channels as u32,
    }
}

/// The settings kept in the safetensors' metadata (the archive's `default_*` attributes).
const PARAM_KEYS: [&str; 7] = ["default_mask_threshold", "default_peak_distance", "default_seed_threshold", "default_overlap_threshold", "default_window_size", "default_cleanup_fragments", "default_min_size"];

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Writes a TorchScript archive's tensors and settings as safetensors.
pub fn convert(archive: &torchscript::Archive, mut metadata: HashMap<String, String>, out: &Path) -> Result<(), String> {
    for k in PARAM_KEYS {
        let v = archive.scalars.get(k).ok_or_else(|| format!("the archive has no {k}"))?;
        metadata.insert(k.to_string(), v.to_string());
    }
    if archive.scalars.get("cells_and_nuclei").copied().unwrap_or(0.0) != 0.0 {
        return Err("this model finds cells and nuclei (the channel-invariant network): not supported yet".into());
    }
    let data: Vec<(String, Vec<usize>, Vec<u8>)> = archive.tensors.iter().filter(|(k, _)| k.starts_with("fcn.") || k.starts_with("pixel_classifier.")).map(|(k, t)| (k.clone(), t.shape.clone(), bytes(&t.data))).collect();
    let views: Vec<(String, TensorView)> = data.iter().map(|(n, s, b)| TensorView::new(Dtype::F32, s.clone(), b).map(|v| (n.clone(), v)).map_err(|e| format!("{n}: {e}"))).collect::<Result<_, _>>()?;
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let tmp = out.with_extension("tmp");
    safetensors::tensor::serialize_to_file(views, Some(metadata), &tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, out).map_err(|e| format!("{}: {e}", out.display()))
}

/// Fetches and verifies the package, then writes the safetensors next to it.
pub fn install(s: &Spec, progress: &mut dyn FnMut(u64, u64)) -> Result<(), String> {
    let (file, url, sha, bytes) = s.package;
    let zip = lucem_hub::fetch(url, &path_of(file), sha, bytes, progress)?;
    let package = std::fs::read(&zip).map_err(|e| format!("{}: {e}", zip.display()))?;
    let archive = torchscript::read(&torchscript::from_package(&package)?)?;
    let meta: HashMap<String, String> = [
        ("source", url.to_string()),
        ("source_sha256", sha.to_string()),
        ("model", s.name.to_string()),
        ("licence", "Apache-2.0 (InstanSeg, Goldsborough, Bankhead et al.)".to_string()),
        ("layout", "the TorchScript archive's tensors by name; BatchNorm not folded".to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    convert(&archive, meta, &path_of(s.safetensors))
}

/// The weights and settings back from the safetensors.
fn read_safetensors(path: &Path) -> Result<(BTreeMap<String, torchscript::Tensor>, Params), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (_, meta) = safetensors::SafeTensors::read_metadata(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut tensors = BTreeMap::new();
    for (name, t) in st.tensors() {
        if t.dtype() != Dtype::F32 {
            return Err(format!("{name}: {:?}, expected F32", t.dtype()));
        }
        tensors.insert(name, torchscript::Tensor { shape: t.shape().to_vec(), data: t.data().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect() });
    }
    let m = meta.metadata().clone().unwrap_or_default();
    let get = |k: &str| -> Result<f64, String> { m.get(k).ok_or_else(|| format!("{}: no {k} in the metadata", path.display()))?.parse::<f64>().map_err(|e| format!("{k}: {e}")) };
    let params = Params {
        mask_threshold: get("default_mask_threshold")? as f32,
        peak_distance: get("default_peak_distance")? as usize,
        seed_threshold: get("default_seed_threshold")? as f32,
        overlap_threshold: get("default_overlap_threshold")? as f32,
        window_size: get("default_window_size")? as usize,
        cleanup_fragments: get("default_cleanup_fragments")? != 0.0,
        min_size: get("default_min_size")? as usize,
    };
    Ok((tensors, params))
}

/// The pixel classifier's layers from the weights (`pixel_classifier.fc1..`).
fn classifier(w: &BTreeMap<String, torchscript::Tensor>) -> Result<post::PixelClassifier, String> {
    let mut layers = Vec::new();
    for k in 1.. {
        let Some(wt) = w.get(&format!("pixel_classifier.fc{k}.weight")) else { break };
        let b = w.get(&format!("pixel_classifier.fc{k}.bias")).ok_or_else(|| format!("pixel_classifier.fc{k}.bias is missing"))?;
        let [o, i] = <[usize; 2]>::try_from(wt.shape.clone()).map_err(|_| format!("pixel_classifier.fc{k}: shape {:?}", wt.shape))?;
        layers.push((wt.data.clone(), b.data.clone(), o, i));
    }
    match (layers.first(), layers.last()) {
        (Some(f), Some(l)) if f.3 == 4 && l.2 == 1 => Ok(post::PixelClassifier { layers }),
        _ => Err("the pixel classifier must take 4 features and give 1 value".into()),
    }
}

/// CubeCL's runtime configuration, once, before the first device opens: compiled kernels are
/// cached in the user cache directory (or `LUCEM_CACHE_DIR`) shared with lucem, as StarDist's.
fn configure_runtime() {
    use cubecl::config::cache::CacheConfig;
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.compilation.cache = std::env::var("LUCEM_KERNEL_CACHE").map_or(true, |v| v.trim() != "0");
        match std::env::var_os("LUCEM_CACHE_DIR") {
            Some(dir) if !dir.is_empty() => config.environment.path = CacheConfig::Directory(dir.into()),
            _ if matches!(config.environment.path, CacheConfig::Target) => config.environment.path = CacheConfig::Global,
            _ => {}
        }
        let _ = CubeClRuntimeConfig::try_set(config);
    });
}

/// Whether `LUCEM_INSTANSEG_TIMING` asks for each phase's time on stderr.
fn timing() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var_os("LUCEM_INSTANSEG_TIMING").is_some())
}

/// The Burn device: `LUCEM_RUNTIME` (`cuda`, `wgpu`, `cpu`), else wgpu's default adapter.
pub fn device() -> Result<Device, String> {
    configure_runtime();
    let want = std::env::var("LUCEM_RUNTIME").unwrap_or_default().to_lowercase();
    match want.as_str() {
        #[cfg(feature = "cuda")]
        // Opening CUDA without its driver panics on the device thread: ask the system first
        "cuda" if cfg!(target_os = "linux") && !Path::new("/proc/driver/nvidia/version").exists() => Err("LUCEM_RUNTIME=cuda: the NVIDIA driver is not loaded".into()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::cuda(0)),
        #[cfg(feature = "cpu")]
        "cpu" => Ok(Device::cpu()),
        #[cfg(feature = "wgpu")]
        "" | "auto" | "wgpu" | "vulkan" => Ok(Device::wgpu(DeviceKind::DefaultDevice)),
        other => Err(format!("LUCEM_RUNTIME={other:?}: this build runs on {}", runtimes().join(", "))),
    }
}

fn runtimes() -> Vec<&'static str> {
    [("wgpu", cfg!(feature = "wgpu")), ("cuda", cfg!(feature = "cuda")), ("cpu", cfg!(feature = "cpu"))].into_iter().filter(|(_, on)| *on).map(|(n, _)| n).collect()
}

/// The `q`-th percentile (0–100) of `n` byte values counted in `hist`, interpolated linearly
/// between order statistics (numpy's default).
fn percentile(hist: &[usize; 256], n: usize, q: f64) -> f32 {
    let pos = q / 100.0 * (n.saturating_sub(1)) as f64;
    let (lo, frac) = (pos.floor() as usize, pos - pos.floor());
    let kth = |k: usize| -> f32 {
        let mut seen = 0;
        for (v, &c) in hist.iter().enumerate() {
            seen += c;
            if seen > k {
                return v as f32;
            }
        }
        255.0
    };
    let a = kth(lo);
    let b = kth((lo + 1).min(n.saturating_sub(1)));
    a + (b - a) * frac as f32
}

/// `pixels` (`c` interleaved channels) as `[c, h, w]` planes: each channel's 0.1st to 99.9th
/// percentile to 0 and 1, then clamped to −2…3.
pub fn normalize(pixels: &[u8], w: usize, h: usize, c: usize) -> Vec<f32> {
    let n = w * h;
    let mut out = vec![0f32; c * n];
    for k in 0..c {
        let mut hist = [0usize; 256];
        for i in 0..n {
            hist[pixels[i * c + k] as usize] += 1;
        }
        let (lo, hi) = (percentile(&hist, n, 0.1), percentile(&hist, n, 99.9));
        let scale = 1.0 / (hi - lo + 1e-6);
        for i in 0..n {
            out[k * n + i] = ((pixels[i * c + k] as f32 - lo) * scale).clamp(-2.0, 3.0);
        }
    }
    out
}

/// How the network walks a large image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tiling {
    /// The core of a tile (pixels; a multiple of 16).
    pub tile: usize,
    /// Context each side (pixels; a multiple of 16).
    pub halo: usize,
}

impl Default for Tiling {
    fn default() -> Self {
        Self { tile: 1024, halo: 96 }
    }
}

/// Tile sides are rounded up to this, so CubeCL's tuned kernels come from its cache.
pub const SHAPE_STEP: usize = 256;

/// Where the time of one image went (milliseconds), and how much the network computed for it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    /// Percentiles and scaling on the host.
    pub normalize: f64,
    /// Tiles cut out and padded on the host.
    pub prepare: f64,
    /// Upload and the network, until the device is idle.
    pub compute: f64,
    /// The output back to the host.
    pub readback: f64,
    /// Seeds, the pixel classifier, labels.
    pub post: f64,
    pub tiles: usize,
    /// Pixels the network computed (padding and halos included) per pixel of the image.
    pub overhead: f64,
}

impl Timings {
    pub fn total(&self) -> f64 {
        self.normalize + self.prepare + self.compute + self.readback + self.post
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// `i` mirrored into `0..n`, as numpy's and PyTorch's `reflect`.
fn reflect(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let k = i % period;
    if k < n { k } else { period - k }
}

pub struct InstanSeg {
    pub spec: &'static Spec,
    net: net::InstanSegNet,
    classifier: post::PixelClassifier,
    pub params: Params,
}

/// Nuclei of one image.
pub struct Instances {
    /// `w × h` of the image given (0: background, `k`: nucleus `k`).
    pub labels: Vec<u32>,
    pub found: Vec<Found>,
    pub ms: f64,
}

impl InstanSeg {
    /// Loads an installed model on `device`.
    pub fn load(s: &'static Spec, device: Device) -> Result<Self, String> {
        let path = path_of(s.safetensors);
        if !path.is_file() {
            return Err(format!("{} is not installed (lucem-ml-instanseg install {})", s.name, s.id));
        }
        let (w, params) = read_safetensors(&path)?;
        Self::from_weights(s, &w, params, device)
    }

    /// The model from weights already read (the archive's, or the safetensors').
    pub fn from_weights(s: &'static Spec, w: &BTreeMap<String, torchscript::Tensor>, params: Params, device: Device) -> Result<Self, String> {
        let net = net::InstanSegNet::new(w, device)?;
        if net.channels != s.channels {
            return Err(format!("{}: the weights take {} channel(s), the model {}", s.name, net.channels, s.channels));
        }
        if net.outputs != 5 {
            return Err(format!("{}: the network gives {} channels, expected 5 (embedding, sigma, seeds)", s.name, net.outputs));
        }
        Ok(Self { spec: s, net, classifier: classifier(w)?, params })
    }

    /// The network's output (`[5, h, w]`) for `planes` (`[channels, h, w]`, normalized).
    pub fn network(&self, planes: &[f32], w: usize, h: usize, t: Tiling, progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<f32>, String> {
        self.network_timed(planes, w, h, t, progress, &mut Timings::default())
    }

    /// [`Self::network`], adding where its time went to `times`.
    pub fn network_timed(&self, planes: &[f32], w: usize, h: usize, t: Tiling, progress: &mut dyn FnMut(usize, usize), times: &mut Timings) -> Result<Vec<f32>, String> {
        let no = self.net.outputs;
        let nc = self.net.channels;
        let mut out = vec![0f32; no * w * h];
        let step = t.tile.max(net::DIVISOR);
        let (ny, nx) = (h.div_ceil(step), w.div_ceil(step));
        let mut computed = 0usize;
        for ty in 0..ny {
            for tx in 0..nx {
                let t0 = Instant::now();
                let (y0, x0) = (ty * step, tx * step);
                let (y1, x1) = ((y0 + step).min(h), (x0 + step).min(w));
                let (ry0, rx0) = (y0.saturating_sub(t.halo), x0.saturating_sub(t.halo));
                let (ry1, rx1) = ((y1 + t.halo).min(h), (x1 + t.halo).min(w));
                let (rh, rw) = (ry1 - ry0, rx1 - rx0);
                let (ph, pw) = (rh.div_ceil(SHAPE_STEP) * SHAPE_STEP, rw.div_ceil(SHAPE_STEP) * SHAPE_STEP);
                let mut input = vec![0f32; nc * ph * pw];
                for c in 0..nc {
                    for y in 0..ph {
                        let sy = ry0 + reflect(y, rh);
                        for x in 0..pw {
                            input[(c * ph + y) * pw + x] = planes[c * w * h + sy * w + rx0 + reflect(x, rw)];
                        }
                    }
                }
                computed += ph * pw;
                times.prepare += ms(t0);
                let t1 = Instant::now();
                let y = self.net.forward(Tensor::<4>::from_data(TensorData::new(input, [1, nc, ph, pw]), &self.net.device));
                self.net.device.sync().map_err(|e| format!("the device: {e:?}"))?;
                times.compute += ms(t1);
                let t2 = Instant::now();
                let y: Vec<f32> = y.try_into_data().map_err(|e| format!("reading the network's output: {e:?}"))?.try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
                times.readback += ms(t2);
                if timing() {
                    eprintln!("instanseg: tile {ph}×{pw}: network {:.0} ms, read back {:.0} ms", t2.duration_since(t1).as_secs_f64() * 1e3, ms(t2));
                }
                let t3 = Instant::now();
                for c in 0..no {
                    for yy in y0..y1 {
                        let ly = yy - ry0;
                        let src = &y[(c * ph + ly) * pw + (x0 - rx0)..(c * ph + ly) * pw + (x1 - rx0)];
                        out[c * w * h + yy * w + x0..c * w * h + yy * w + x1].copy_from_slice(src);
                    }
                }
                times.prepare += ms(t3);
                progress(ty * nx + tx + 1, ny * nx);
            }
        }
        times.tiles += ny * nx;
        times.overhead = computed as f64 / (w * h).max(1) as f64;
        if out.iter().all(|&v| v == 0.0) {
            // A device that failed quietly reads back zeros: never an answer
            return Err("the network gave nothing at all (the device failed?)".into());
        }
        Ok(out)
    }

    /// The nuclei of an image at the model's own scale (the model's channels, interleaved).
    pub fn segment(&self, pixels: &[u8], w: usize, h: usize, t: Tiling, progress: &mut dyn FnMut(usize, usize)) -> Result<(Vec<u32>, Vec<Found>), String> {
        let (labels, found, times) = self.segment_timed(pixels, w, h, t, progress)?;
        if timing() {
            eprintln!("instanseg: {times:?} ({} nuclei)", found.len());
        }
        Ok((labels, found))
    }

    /// [`Self::segment`] and where its time went.
    pub fn segment_timed(&self, pixels: &[u8], w: usize, h: usize, t: Tiling, progress: &mut dyn FnMut(usize, usize)) -> Result<(Vec<u32>, Vec<Found>, Timings), String> {
        let c = self.spec.channels;
        if pixels.len() != w * h * c {
            return Err(format!("{} bytes, not {w} × {h} × {c}", pixels.len()));
        }
        let mut times = Timings::default();
        let t0 = Instant::now();
        let planes = normalize(pixels, w, h, c);
        times.normalize = ms(t0);
        let out = self.network_timed(&planes, w, h, t, progress, &mut times)?;
        let t1 = Instant::now();
        let (labels, found) = post::instances(&post::Output { data: &out, h, w }, &self.classifier, &self.params);
        times.post = ms(t1);
        Ok((labels, found, times))
    }

    /// The nuclei of an image (`pixel_um`: its pixel size, when known: resampled to the model's
    /// 0.5 µm, and the labels back).
    pub fn instances(&self, pixels: &[u8], w: usize, h: usize, pixel_um: Option<f64>, progress: &mut dyn FnMut(usize, usize)) -> Result<Instances, String> {
        let t0 = Instant::now();
        let scale = pixel_um.filter(|p| *p > 0.0).map_or(1.0, |p| p / MODEL_PIXEL_UM);
        let resample = (scale - 1.0).abs() > 0.1;
        let c = self.spec.channels;
        let (sw, sh) = if resample { (((w as f64) * scale).round().max(64.0) as usize, ((h as f64) * scale).round().max(64.0) as usize) } else { (w, h) };
        let scaled;
        let input = if resample {
            scaled = resize_pixels(pixels, w, h, sw, sh, c);
            &scaled
        } else {
            pixels
        };
        let (labels, found) = self.segment(input, sw, sh, Tiling::default(), progress)?;
        let labels = if resample { resize_labels(&labels, sw, sh, w, h) } else { labels };
        Ok(Instances { labels, found, ms: t0.elapsed().as_secs_f64() * 1e3 })
    }
}

/// Bilinear resize of `nc` interleaved channels.
fn resize_pixels(px: &[u8], w: usize, h: usize, nw: usize, nh: usize, nc: usize) -> Vec<u8> {
    let mut out = vec![0u8; nw * nh * nc];
    let (sx, sy) = (w as f32 / nw as f32, h as f32 / nh as f32);
    for y in 0..nh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..nw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let x1 = (x0 + 1).min(w - 1);
            for k in 0..nc {
                let v = |yy: usize, xx: usize| px[(yy * w + xx) * nc + k] as f32;
                let top = v(y0, x0) * (1.0 - tx) + v(y0, x1) * tx;
                let bot = v(y1, x0) * (1.0 - tx) + v(y1, x1) * tx;
                out[(y * nw + x) * nc + k] = (top * (1.0 - ty) + bot * ty).round() as u8;
            }
        }
    }
    out
}

/// Nearest-neighbour resize of labels.
fn resize_labels(l: &[u32], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u32> {
    let mut out = vec![0u32; nw * nh];
    for y in 0..nh {
        let sy = (((y as f64 + 0.5) * h as f64 / nh as f64) as usize).min(h - 1);
        for x in 0..nw {
            let sx = (((x as f64 + 0.5) * w as f64 / nw as f64) as usize).min(w - 1);
            out[y * nw + x] = l[sy * w + sx];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A little-endian f32 `.npy` file: its shape and values.
    fn npy(path: &Path) -> (Vec<usize>, Vec<f32>) {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        let header = std::str::from_utf8(&b[10..10 + hl]).unwrap();
        assert!(header.contains("'<f4'") && header.contains("'fortran_order': False"), "{header}");
        let shape: Vec<usize> = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter_map(|s| s.trim().parse().ok()).collect();
        (shape, b[10 + hl..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    #[test]
    fn percentiles_interpolate_as_numpy() {
        let mut hist = [0usize; 256];
        for v in 0..=100u8 {
            hist[v as usize] += 1;
        }
        // numpy.percentile(range(101), 99.9) = 99.9
        assert!((percentile(&hist, 101, 99.9) - 99.9).abs() < 1e-4);
        assert_eq!(percentile(&hist, 101, 0.0), 0.0);
    }

    /// Labels `a` and `b` (same size) agree: every object of one is matched (IoU ≥ 0.5) by one of
    /// the other. Returns the fraction matched each way.
    fn agreement(a: &[u32], b: &[u32]) -> (f64, f64) {
        let mut pairs: HashMap<(u32, u32), usize> = HashMap::new();
        let mut area_a: HashMap<u32, usize> = HashMap::new();
        let mut area_b: HashMap<u32, usize> = HashMap::new();
        for (&x, &y) in a.iter().zip(b) {
            if x > 0 {
                *area_a.entry(x).or_default() += 1;
            }
            if y > 0 {
                *area_b.entry(y).or_default() += 1;
            }
            if x > 0 && y > 0 {
                *pairs.entry((x, y)).or_default() += 1;
            }
        }
        let matched: Vec<(u32, u32)> = pairs.iter().filter(|&(&(x, y), &i)| i as f64 / (area_a[&x] + area_b[&y] - i) as f64 >= 0.5).map(|(&k, _)| k).collect();
        let ma = matched.iter().map(|p| p.0).collect::<std::collections::HashSet<_>>().len();
        let mb = matched.iter().map(|p| p.1).collect::<std::collections::HashSet<_>>().len();
        (ma as f64 / area_a.len().max(1) as f64, mb as f64 / area_b.len().max(1) as f64)
    }

    /// The published test: the package's `test-input.npy` through the archive's weights must give
    /// `test-output_instance_segmentation.npy` (InstanSeg's own TorchScript run): the same nuclei,
    /// matched by IoU ≥ 0.5. `LUCEM_INSTANSEG_TEST_DIR` holds the packages unpacked, one folder
    /// per model (`brightfield_nuclei/`, `single_channel_nuclei/`).
    #[test]
    #[ignore = "needs the published packages unpacked (LUCEM_INSTANSEG_TEST_DIR)"]
    fn brightfield_matches_the_published_test_output() {
        published(&MODELS[0], "brightfield_nuclei");
    }

    #[test]
    #[ignore = "needs the published packages unpacked (LUCEM_INSTANSEG_TEST_DIR)"]
    fn single_channel_matches_the_published_test_output() {
        published(&MODELS[1], "single_channel_nuclei");
    }

    fn published(s: &'static Spec, folder: &str) {
        let dir = PathBuf::from(std::env::var("LUCEM_INSTANSEG_TEST_DIR").expect("LUCEM_INSTANSEG_TEST_DIR")).join(folder);
        let archive = torchscript::read(&std::fs::read(dir.join("instanseg.pt")).unwrap()).unwrap();
        let tmp = std::env::temp_dir().join(format!("lucem-instanseg-test-{}-{}.safetensors", folder, std::process::id()));
        convert(&archive, HashMap::new(), &tmp).unwrap();
        let (w8, params) = read_safetensors(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        let model = InstanSeg::from_weights(s, &w8, params, device().unwrap()).unwrap();
        let (si, input) = npy(&dir.join("test-input.npy"));
        let (so, expected) = npy(&dir.join("test-output_instance_segmentation.npy"));
        let (c, h, w) = (si[1], si[2], si[3]);
        assert_eq!((c, so[2], so[3]), (s.channels, h, w));
        let pixels: Vec<u8> = (0..h * w).flat_map(|i| (0..c).map(move |k| (k, i))).map(|(k, i)| input[k * h * w + i].round().clamp(0.0, 255.0) as u8).collect();
        let (labels, found) = model.segment(&pixels, w, h, Tiling::default(), &mut |_, _| {}).unwrap();
        let expected: Vec<u32> = expected.iter().map(|&v| v as u32).collect();
        let n_expected = expected.iter().copied().filter(|&l| l > 0).collect::<std::collections::HashSet<_>>().len();
        let (ours, theirs) = agreement(&labels, &expected);
        // Ours not matched by theirs: their sizes (px)
        let mut unmatched: Vec<(u32, usize)> = Vec::new();
        for l in 1..=found.len() as u32 {
            let pix: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == l).collect();
            let best = pix.iter().filter(|&&i| expected[i] > 0).count();
            if (best as f64) < 0.5 * pix.len() as f64 {
                unmatched.push((l, pix.len()));
            }
        }
        eprintln!("{folder}: ours without a match (label, px): {unmatched:?}; min_size in the archive: {:?}", archive.scalars.get("default_min_size"));
        eprintln!("{folder}: {} nuclei (published {n_expected}); matched {:.1} % of ours, {:.1} % of theirs", found.len(), ours * 100.0, theirs * 100.0);
        assert!(ours > 0.95 && theirs > 0.95, "{folder}: matched {ours:.3} / {theirs:.3}");
    }
}

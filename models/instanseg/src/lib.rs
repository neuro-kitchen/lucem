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
//! - Network: [`net`], Burn modules mirroring InstanSeg's PyTorch classes (CUDA, wgpu or the
//!   CPU), loaded with Burn's `SafetensorsStore` and its PyTorch adapter.
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

/// The shapes of the weights (for the module's configuration) and the settings, from the
/// safetensors' header; the values load into the module through Burn's store ([`net::load`]).
fn read_header(path: &Path) -> Result<(BTreeMap<String, Vec<usize>>, Params), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (_, meta) = safetensors::SafeTensors::read_metadata(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
    let shapes: BTreeMap<String, Vec<usize>> = meta.tensors().into_iter().map(|(name, info)| (name, info.shape.clone())).collect();
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
    Ok((shapes, params))
}

/// The pixel classifier's layers for the host postprocessing ([`post`]), read from the loaded
/// module (Burn's `Linear` keeps `[in, out]`; the host classifier takes `[out, in]`).
fn classifier(p: &net::ProbabilityNet) -> Result<post::PixelClassifier, String> {
    let mut layers = Vec::new();
    for fc in [&p.fc1, &p.fc2, &p.fc3] {
        let w = fc.weight.val();
        let [i, o] = w.dims();
        let w: Vec<f32> = w.try_into_data().map_err(|e| format!("the pixel classifier: {e:?}"))?.try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
        let b: Vec<f32> = match &fc.bias {
            Some(b) => b.val().try_into_data().map_err(|e| format!("the pixel classifier: {e:?}"))?.try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?,
            None => vec![0.0; o],
        };
        let t: Vec<f32> = (0..o).flat_map(|r| (0..i).map(move |c| (r, c))).map(|(r, c)| w[c * o + r]).collect();
        layers.push((t, b, o, i));
    }
    let c = post::PixelClassifier { layers };
    if !c.fits() {
        return Err(format!("the pixel classifier has a layer wider than {}", post::MAX_WIDTH));
    }
    match (c.layers.first(), c.layers.last()) {
        (Some(f), Some(l)) if f.3 == 4 && l.2 == 1 => Ok(c),
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
        // 1088 + 2 × 96 = 1280: a full tile is exactly a multiple of SHAPE_STEP (no padding
        // wasted). A 2048-wide tile (1856 + halos) is past what the matrix products of the 3 × 3
        // layers can launch on a Turing GPU ("cube count too big").
        Self { tile: 1088, halo: 96 }
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

/// The precision the network runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    /// Half precision: weights, activations and the data to and from the device (half the bytes
    /// moved), and the GPU's tensor-core convolutions.
    F16,
}

impl Precision {
    /// `LUCEM_INSTANSEG_PRECISION` (`f32`, `f16`), else half precision on a GPU and single
    /// precision on the CPU runtime (`LUCEM_RUNTIME=cpu`), where half precision is slower.
    pub fn from_env() -> Self {
        match std::env::var("LUCEM_INSTANSEG_PRECISION").unwrap_or_default().to_lowercase().as_str() {
            "f32" => Precision::F32,
            "f16" => Precision::F16,
            _ if std::env::var("LUCEM_RUNTIME").is_ok_and(|r| r.eq_ignore_ascii_case("cpu")) => Precision::F32,
            _ => Precision::F16,
        }
    }
}

pub struct InstanSeg {
    pub spec: &'static Spec,
    pub precision: Precision,
    /// The network and its pixel classifier, as Burn modules ([`net`]).
    pub model: net::InstanSegModel,
    pub config: net::UNetConfig,
    device: Device,
    /// The pixel classifier on the host, for the postprocessing.
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
        Self::load_file(s, &path, device)
    }

    /// The model from a safetensors file written by [`convert`].
    pub fn load_file(s: &'static Spec, path: &Path, device: Device) -> Result<Self, String> {
        let (shapes, params) = read_header(path)?;
        let config = net::UNetConfig::from_shapes(&shapes)?;
        if config.channels != s.channels {
            return Err(format!("{}: the weights take {} channel(s), the model {}", s.name, config.channels, s.channels));
        }
        if config.outputs() != 5 {
            return Err(format!("{}: the network gives {} channels, expected 5 (embedding, sigma, seeds)", s.name, config.outputs()));
        }
        let precision = Precision::from_env();
        // BatchNorm folded in f32, the classifier read in f32, then the network cast
        let model = net::load(path, &config, &device)?.fold_batch_norm();
        let classifier = classifier(&model.pixel_classifier)?;
        let model = match precision {
            Precision::F32 => model,
            Precision::F16 => net::InstanSegModel { fcn: model.fcn.cast(burn::tensor::FloatDType::F16), ..model },
        };
        // 3 × 3 convolutions as matrix products on the GPU in half precision (see
        // `ConvNormAct::conv_gemm`); `LUCEM_INSTANSEG_CONV=gemm|direct` chooses
        let gemm = match std::env::var("LUCEM_INSTANSEG_CONV").unwrap_or_default().to_lowercase().as_str() {
            "gemm" => true,
            "direct" => false,
            _ => precision == Precision::F16,
        };
        let model = net::InstanSegModel { fcn: model.fcn.with_gemm(gemm), ..model };
        Ok(Self { spec: s, precision, model, config, device, classifier, params })
    }

    /// The network's output (`[5, h, w]`) for `planes` (`[channels, h, w]`, normalized).
    pub fn network(&self, planes: &[f32], w: usize, h: usize, t: Tiling, progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<f32>, String> {
        self.network_timed(planes, w, h, t, progress, &mut Timings::default())
    }

    /// [`Self::network`], adding where its time went to `times`.
    pub fn network_timed(&self, planes: &[f32], w: usize, h: usize, t: Tiling, progress: &mut dyn FnMut(usize, usize), times: &mut Timings) -> Result<Vec<f32>, String> {
        let no = self.config.outputs();
        let nc = self.config.channels;
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
                // The tile through the network: its output, and the ms of compute and readback
                let attempt = || -> Result<(Vec<f32>, f64, f64), String> {
                    let t1 = Instant::now();
                    // In the network's precision on the wire: half the bytes up and back in F16
                    let data = TensorData::new(input.clone(), [1, nc, ph, pw]);
                    // Created in that dtype: `from_data` would otherwise convert to the device's default
                    let x = match self.precision {
                        Precision::F32 => Tensor::<4>::from_data(data, &self.device),
                        Precision::F16 => Tensor::<4>::from_data(data.convert::<burn::tensor::f16>(), (&self.device, burn::tensor::DType::F16)),
                    };
                    let y = self.model.fcn.forward(x);
                    self.device.sync().map_err(|e| format!("the device: {e:?}"))?;
                    let compute = ms(t1);
                    let t2 = Instant::now();
                    let y = y.try_into_data().map_err(|e| format!("reading the network's output: {e:?}"))?.convert::<f32>().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
                    Ok((y, compute, ms(t2)))
                };
                // A matrix product this device cannot launch: Burn's own convolutions from now on
                let (y, compute, readback) = match attempt() {
                    Ok(r) => r,
                    Err(e) if crate::gemm_cannot_launch(&e) && net::disable_gemm() => {
                        eprintln!("instanseg: the matrix-product convolutions failed on this device, using Burn's own ({e})");
                        attempt().map_err(crate::explain)?
                    }
                    Err(e) => return Err(crate::explain(e)),
                };
                times.compute += compute;
                times.readback += readback;
                if timing() {
                    eprintln!("instanseg: tile {ph}×{pw}: network {compute:.0} ms, read back {readback:.0} ms");
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

/// Whether a failed run was a matrix product that cannot launch on this device (its launch
/// limits), the one failure the direct convolutions are a way around. Out of memory is not: the
/// direct convolutions need memory too.
pub(crate) fn gemm_cannot_launch(e: &str) -> bool {
    e.contains("Unable to launch matmul") || e.contains("Cube count too big")
}

/// A run's failure in words a user can act on.
pub(crate) fn explain(e: String) -> String {
    if e.contains("Out of Memory") || e.contains("out of memory") || e.contains("OutOfMemory") {
        format!("the GPU ran out of memory (another program, or another runtime of the same GPU, holds much of it): close it, choose a smaller region, or set LUCEM_RUNTIME=cpu ({e})")
    } else {
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_launch_limits_switch_the_convolutions_and_out_of_memory_is_explained() {
        assert!(gemm_cannot_launch("Unable to launch matmul because a required feature is unavailable: Cube count too big (1, 131072, 1)"));
        assert!(!gemm_cannot_launch("wgpu error: Out of Memory"));
        assert!(explain("wgpu error: Out of Memory".into()).starts_with("the GPU ran out of memory"));
        assert_eq!(explain("something else".into()), "something else");
    }

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
        let model = InstanSeg::load_file(s, &tmp, device().unwrap()).unwrap();
        std::fs::remove_file(&tmp).ok();
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

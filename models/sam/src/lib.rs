//! Segment Anything 2 (Ravi et al., Meta AI, 2024) for lucem, on ONNX Runtime. Ported from
//! cv-kitchen's `vision-ml` (`models/sam/v2`): the Hiera image encoder turns an RGB image into an
//! embedding once; the prompt decoder turns points and boxes into masks in milliseconds, so an
//! image's embedding is kept for the next click ([`Sam2::segment`] reuses it by key).
//!
//! - Weights: [`MODELS`] (exported to ONNX by the community, hashes pinned), fetched and verified
//!   through `lucem-hub` ([`install`]); a model loads only files whose SHA-256 matches.
//! - Runtime: ONNX Runtime loaded at run time ([`init_runtime`]): `ORT_DYLIB_PATH`, else the
//!   `onnxruntime` Python package (`$LUCEM_PYTHON` or `python3`), else the system's library. CUDA
//!   is used when the library has it, else the CPU.
//! - Pre-processing as the ONNX export expects: the image resized (stretched) to 1024 × 1024,
//!   ImageNet mean and deviation; prompt coordinates scaled to that 1024 grid; masks come back at
//!   the image's own size.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use lucem_ml::{ModelInfo, Prompt, RuntimeStatus, Task, Weight};
use ort::session::Session;
use ort::value::Tensor;

/// The encoder's input side.
pub const INPUT: usize = 1024;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// One published SAM 2 model: its encoder and decoder files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spec {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    /// Encoder, then decoder: cache name, URL, SHA-256.
    pub files: [(&'static str, &'static str, &'static str); 2],
    pub total_bytes: u64,
    pub default: bool,
}

/// The SAM 2 models offered (hashes from cv-kitchen's registry).
pub static MODELS: [Spec; 4] = [
    Spec {
        id: "sam2-hiera-tiny",
        name: "SAM 2 · Hiera Tiny",
        description: "Fastest; good for clicking objects on a laptop (~126 MB)",
        files: [
            ("sam2_hiera_tiny_encoder.onnx", "https://huggingface.co/shubham0204/sam2-onnx-models/resolve/main/sam2_hiera_tiny_encoder.onnx", "bf8c3c0eb5943b3376261aed4c52cf335161daad0dce5af9c1a243704550ca72"),
            ("sam2_hiera_tiny_decoder.onnx", "https://huggingface.co/shubham0204/sam2-onnx-models/resolve/main/sam2_hiera_tiny_decoder.onnx", "2008192a9ff55a2a9c46813a56015315f4230495f04e7a315dbedbe35f39a339"),
        ],
        total_bytes: 126_012_913,
        default: true,
    },
    Spec {
        id: "sam2-hiera-small",
        name: "SAM 2 · Hiera Small",
        description: "Fast, a little sharper (~179 MB)",
        files: [
            ("sam2_hiera_small_encoder.onnx", "https://huggingface.co/shubham0204/sam2-onnx-models/resolve/main/sam2_hiera_small_encoder.onnx", "0edca276aeda91ac715c140266894ffa91885989ea6164be1eb24963ef3f67df"),
            ("sam2_hiera_small_decoder.onnx", "https://huggingface.co/shubham0204/sam2-onnx-models/resolve/main/sam2_hiera_small_decoder.onnx", "c619187e57d45c5b273778ab3eb37e737fed3fac92a2341a1b109d2ef166d9ff"),
        ],
        total_bytes: 179_244_476,
        default: false,
    },
    Spec {
        id: "sam2-hiera-base-plus",
        name: "SAM 2 · Hiera Base+",
        description: "Balanced (~294 MB)",
        files: [
            ("sam2_hiera_base_plus_encoder.onnx", "https://huggingface.co/shubham0204/sam2-onnx-models/resolve/main/sam2_hiera_base_plus_encoder.onnx", "0effdbb8c9be0f29bb04fe96e09b93965c49ca678b82b3f3bdaff8418be0d4cd"),
            ("sam2_hiera_base_plus_decoder.onnx", "https://huggingface.co/shubham0204/sam2-onnx-models/resolve/main/sam2_hiera_base_plus_decoder.onnx", "3a0e9232d8ee93231e9d986eba7b457ffe4bb5a9d5d43b87dce8726e00f0a27b"),
        ],
        total_bytes: 294_013_675,
        default: false,
    },
    Spec {
        id: "sam2-hiera-large",
        name: "SAM 2 · Hiera Large",
        description: "Most accurate; a GPU recommended (~910 MB)",
        files: [
            ("sam2_hiera_large_encoder.onnx", "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_large.encoder.onnx", "cb252d7b59fdeb2567f7134ed9f23d712e4f24584628913bbcb0ea72ba72b617"),
            ("sam2_hiera_large_decoder.onnx", "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_large.decoder.onnx", "2b5a3d40a017e61d2cb4fac7147ebf899d24b082753fb5049be3810d2318ca07"),
        ],
        total_bytes: 910_002_890,
        default: false,
    },
];

pub fn spec(id: &str) -> Option<&'static Spec> {
    MODELS.iter().find(|s| s.id == id)
}

/// Where a model's file lives in the cache.
pub fn path_of(file: &str) -> PathBuf {
    lucem_hub::cache_dir("sam").join(file)
}

/// Whether both files are in the cache (their hashes are checked when the model loads).
pub fn installed(s: &Spec) -> bool {
    s.files.iter().all(|(f, _, _)| path_of(f).is_file())
}

/// A model as the plugin's manifest lists it.
pub fn info(s: &Spec) -> ModelInfo {
    ModelInfo {
        id: s.id.into(),
        name: s.name.into(),
        family: "sam".into(),
        task: Task::InteractiveSegmentation,
        domains: vec!["natural".into(), "histology".into(), "fluorescence".into()],
        description: s.description.into(),
        licence: "Apache-2.0".into(),
        paper: Some("Ravi et al. (2024) SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714".into()),
        code: Some("https://github.com/facebookresearch/sam2".into()),
        weights: s.files.iter().map(|(f, url, sha)| Weight { file: (*f).into(), url: (*url).into(), sha256: (*sha).into(), bytes: 0 }).collect(),
        installed: installed(s),
        default: s.default,
        channels: 3,
    }
}

/// Downloads and verifies a model's two files; `progress(done, total)` in bytes of the model.
pub fn install(s: &Spec, progress: &mut dyn FnMut(u64, u64)) -> Result<(), String> {
    let mut before = 0u64;
    for (file, url, sha) in s.files {
        let mut last = 0u64;
        lucem_hub::fetch(url, &path_of(file), sha, 0, &mut |done, _| {
            last = done;
            progress(before + done, s.total_bytes)
        })?;
        before += last;
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// ONNX Runtime
// ----------------------------------------------------------------------------

/// ONNX Runtime as Microsoft publishes it (the latest release, CPU build), fetched and verified
/// through lucem-hub into lucem's cache, so no Python or system package is needed.
pub struct RuntimeRelease {
    pub version: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    /// The folder the archive unpacks to.
    pub folder: &'static str,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub const RUNTIME: Option<RuntimeRelease> = Some(RuntimeRelease {
    version: "1.31.1",
    url: "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz",
    sha256: "c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9",
    bytes: 11_413_655,
    folder: "onnxruntime-linux-x64-1.31.1",
});
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub const RUNTIME: Option<RuntimeRelease> = Some(RuntimeRelease {
    version: "1.31.1",
    url: "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-osx-arm64-1.31.1.tgz",
    sha256: "2e1344a1926ee50eb25b38153cb6992c3567515aef0e677a1a520a1798896d6f",
    bytes: 0,
    folder: "onnxruntime-osx-arm64-1.31.1",
});
#[cfg(not(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "macos", target_arch = "aarch64"))))]
pub const RUNTIME: Option<RuntimeRelease> = None;

/// ONNX Runtime's GPU build for CUDA 13 (the CUDA this machine family ships now); it needs an
/// NVIDIA driver and cuDNN 9 for CUDA 13 (`cudnn9-cuda-13`). Digest from the release's assets.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub const RUNTIME_CUDA: Option<RuntimeRelease> = Some(RuntimeRelease {
    version: "1.31.1",
    url: "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-gpu_cuda13-1.31.1.tgz",
    sha256: "b2146862e691f7e338f5aff9c7a9a5567c2ca57cf01442a1adb3a13bf62d9095",
    bytes: 244_573_709,
    folder: "onnxruntime-linux-x64-gpu_cuda13-1.31.1",
});
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub const RUNTIME_CUDA: Option<RuntimeRelease> = None;

/// The id `install` takes for the runtime itself, and for its GPU build.
pub const RUNTIME_ID: &str = lucem_ml::RUNTIME_ID;
pub const RUNTIME_CUDA_ID: &str = lucem_ml::RUNTIME_CUDA_ID;

fn release_dir(r: &RuntimeRelease) -> PathBuf {
    lucem_hub::cache_dir("runtime").join(r.folder)
}

/// The ONNX Runtime library inside an unpacked release (whatever its top folder is called).
fn library_in(dir: &std::path::Path) -> Option<PathBuf> {
    let is_lib = |p: &PathBuf| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| (n.starts_with("libonnxruntime.so.") || n.starts_with("libonnxruntime.") && n.ends_with(".dylib")) && !n.contains("providers"));
    let mut libs: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path().join("lib")).filter(|l| l.is_dir()).flat_map(|l| std::fs::read_dir(l).into_iter().flatten().flatten().map(|e| e.path())).filter(is_lib).collect();
    libs.sort();
    libs.pop()
}

/// The library of lucem's own ONNX Runtime (CPU build), when installed.
fn own_library() -> Option<PathBuf> {
    library_in(&release_dir(RUNTIME.as_ref()?))
}

/// The library of lucem's GPU build, when installed.
fn own_cuda_library() -> Option<PathBuf> {
    library_in(&release_dir(RUNTIME_CUDA.as_ref()?))
}

/// Whether cuDNN 9 can be loaded (the GPU build's CUDA provider needs it).
pub fn cudnn_found() -> bool {
    let in_cache = std::process::Command::new("ldconfig").arg("-p").output().is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("libcudnn.so.9"));
    let dirs = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    in_cache || dirs.split(':').chain(["/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"]).filter(|d| !d.is_empty()).any(|d| std::path::Path::new(d).join("libcudnn.so.9").exists())
}

/// The GPU build's state for the manifest.
pub fn gpu_runtime() -> Option<lucem_ml::GpuRuntime> {
    let r = RUNTIME_CUDA.as_ref()?;
    let installed = own_cuda_library().is_some();
    let cudnn = cudnn_found();
    let detail = match (installed, cudnn) {
        (false, _) => format!("ONNX Runtime {} for CUDA 13 ({:.0} MB): SAM on the GPU; needs the NVIDIA driver and cuDNN 9 (cudnn9-cuda-13)", r.version, r.bytes as f64 / 1e6),
        (true, false) => "The GPU build is installed but cuDNN 9 is not found (install cudnn9-cuda-13): the CPU build runs".into(),
        (true, true) => "The GPU build is installed and cuDNN 9 found: SAM runs on the GPU".into(),
    };
    Some(lucem_ml::GpuRuntime { installed, usable: installed && cudnn, bytes: r.bytes, detail })
}

fn install_release(r: &RuntimeRelease, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf, String> {
    let archive = lucem_hub::cache_dir("runtime").join(format!("{}.tgz", r.folder));
    lucem_hub::fetch_archive(r.url, &archive, r.sha256, r.bytes, &release_dir(r), progress)?;
    // The archive is no longer needed once unpacked (the GPU build is large)
    let _ = std::fs::remove_file(&archive);
    library_in(&release_dir(r)).ok_or_else(|| "the ONNX Runtime archive has no library".to_string())
}

/// Downloads and verifies ONNX Runtime into lucem's cache.
pub fn install_runtime(progress: &mut dyn FnMut(u64, u64)) -> Result<(), String> {
    let r = RUNTIME.as_ref().ok_or("no ONNX Runtime build is published for this platform here: set ORT_DYLIB_PATH")?;
    install_release(r, progress).map(|_| ())
}

/// Downloads and verifies the GPU build (used instead of the CPU one when cuDNN 9 is there; the
/// plugin must start again to load it).
pub fn install_runtime_cuda(progress: &mut dyn FnMut(u64, u64)) -> Result<(), String> {
    let r = RUNTIME_CUDA.as_ref().ok_or("no ONNX Runtime GPU build is published for this platform")?;
    install_release(r, progress).map(|_| ())
}

/// The ONNX Runtime library to load: `ORT_DYLIB_PATH`, else lucem's GPU build (when cuDNN 9 is
/// there and `LUCEM_ML_EP` is not `cpu`), else its CPU build, else the `onnxruntime` Python
/// package, else the system's.
fn find_library() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    let cpu_only = std::env::var("LUCEM_ML_EP").is_ok_and(|v| v.eq_ignore_ascii_case("cpu"));
    if !cpu_only
        && cudnn_found()
        && let Some(p) = own_cuda_library()
    {
        return Some(p);
    }
    if let Some(p) = own_library() {
        return Some(p);
    }
    let python = std::env::var("LUCEM_PYTHON").ok().filter(|p| !p.is_empty()).unwrap_or_else(|| "python3".into());
    let probe = "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))";
    if let Ok(out) = std::process::Command::new(&python).args(["-c", probe]).output() {
        let dir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut libs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("libonnxruntime.so") || n == "libonnxruntime.dylib" || n == "onnxruntime.dll")).collect();
            libs.sort();
            if let Some(l) = libs.into_iter().next() {
                return Some(l);
            }
        }
    }
    ["/usr/lib/libonnxruntime.so", "/usr/local/lib/libonnxruntime.so", "/usr/lib64/libonnxruntime.so", "/usr/lib/x86_64-linux-gnu/libonnxruntime.so"].iter().map(PathBuf::from).find(|p| p.is_file())
}

/// Whether the loaded ONNX Runtime has its CUDA provider (the GPU build ships
/// `libonnxruntime_providers_cuda` beside the library). The CPU build must not be asked for CUDA:
/// some builds abort instead of declining. `LUCEM_ML_EP=cpu` forces the CPU.
static CUDA: OnceLock<bool> = OnceLock::new();

fn has_cuda(lib: &std::path::Path) -> bool {
    if std::env::var("LUCEM_ML_EP").is_ok_and(|v| v.eq_ignore_ascii_case("cpu")) {
        return false;
    }
    let Some(dir) = lib.parent() else { return false };
    std::fs::read_dir(dir).is_ok_and(|entries| entries.flatten().any(|e| e.file_name().to_string_lossy().starts_with("libonnxruntime_providers_cuda")))
}

/// The ONNX Runtime environment, made once when the library loads.
static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

/// Loads ONNX Runtime once it can be found (asked again until then: it may be installed while the
/// plugin runs); what runs the models, or what is missing.
pub fn init_runtime() -> RuntimeStatus {
    static STATUS: OnceLock<RuntimeStatus> = OnceLock::new();
    if let Some(s) = STATUS.get() {
        return s.clone();
    }
    let status = match find_library() {
            None => RuntimeStatus {
                ready: false,
                detail: format!("ONNX Runtime is not installed: Install it here ({}, {:.0} MB, verified), or set ORT_DYLIB_PATH", RUNTIME.as_ref().map_or("?", |r| r.version), RUNTIME.as_ref().map_or(0.0, |r| r.bytes as f64 / 1e6)),
                gpu: gpu_runtime(),
            },
            Some(lib) => match ort::init_from(&lib).map_err(|e| e.to_string()).and_then(|b| b.with_name("lucem-ml-sam").build().map_err(|e| e.to_string())) {
                Ok(env) => {
                    let _ = ENV.set(env);
                    let cuda = *CUDA.get_or_init(|| has_cuda(&lib));
                    RuntimeStatus { ready: true, detail: format!("ONNX Runtime on the {} ({})", if cuda { "GPU (CUDA), else the CPU" } else { "CPU" }, lib.display()), gpu: gpu_runtime() }
                }
                Err(e) => RuntimeStatus { ready: false, detail: format!("ONNX Runtime at {} did not load ({e}); lucem needs ONNX Runtime 1.31 or newer: pip install -U onnxruntime", lib.display()), gpu: gpu_runtime() },
            },
    };
    if status.ready {
        let _ = STATUS.set(status.clone());
    }
    status
}

fn session(bytes: &[u8]) -> Result<Session, String> {
    let threads = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1).clamp(1, 8);
    let env = ENV.get().ok_or("ONNX Runtime is not loaded")?;
    let mut builder = Session::builder(env).map_err(|e| format!("onnx runtime: {e}"))?;
    // The CPU is ONNX Runtime's default: nothing is registered for it (registering it explicitly
    // aborts some 1.22 builds, whose device list is empty)
    if CUDA.get().copied().unwrap_or(false) {
        builder = builder.with_execution_providers([ort::ep::CUDA::default().build()]).map_err(|e| format!("onnx runtime: {e}"))?;
    }
    builder
        .with_intra_threads(threads)
        .map_err(|e| format!("onnx runtime: {e}"))?
        .commit_from_memory(bytes)
        .map_err(|e| format!("onnx runtime: {e}"))
}

/// A tensor's values (f32) and shape from a session's outputs.
fn output(outputs: &ort::session::SessionOutputs, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    let v = outputs.get(name).ok_or_else(|| format!("the model gives no {name:?}"))?;
    let (shape, data) = v.try_extract_tensor::<f32>().map_err(|e| format!("{name}: {e}"))?;
    Ok((shape.iter().map(|&d| d.max(0) as usize).collect(), data.to_vec()))
}

// ----------------------------------------------------------------------------
// The model
// ----------------------------------------------------------------------------

/// What the encoder gives for one image.
struct Embedding {
    key: String,
    width: u32,
    height: u32,
    image_embed: Vec<f32>,
    high_res_0: Vec<f32>,
    high_res_1: Vec<f32>,
}

/// One mask: one byte per pixel (0 / 255) at the image's size, and the model's confidence.
pub struct Candidate {
    pub mask: Vec<u8>,
    pub score: f32,
}

pub struct Sam2 {
    pub spec: &'static Spec,
    encoder: Session,
    decoder: Session,
    last: Option<Embedding>,
}

impl Sam2 {
    /// Loads a model whose files are installed and verified.
    pub fn load(s: &'static Spec) -> Result<Self, String> {
        let status = init_runtime();
        if !status.ready {
            return Err(status.detail);
        }
        if !installed(s) {
            return Err(format!("{} is not installed (Settings → Models)", s.name));
        }
        let enc = lucem_hub::read_verified(&path_of(s.files[0].0), s.files[0].2)?;
        let dec = lucem_hub::read_verified(&path_of(s.files[1].0), s.files[1].2)?;
        Ok(Self { spec: s, encoder: session(&enc)?, decoder: session(&dec)?, last: None })
    }

    /// The encoder's input: RGB stretched to 1024², normalised, channels first.
    fn preprocess(rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        let mut t = vec![0.0f32; 3 * INPUT * INPUT];
        for y in 0..INPUT {
            let sy = (((y as f32 + 0.5) * h as f32 / INPUT as f32) - 0.5).clamp(0.0, (h - 1) as f32);
            let (y0, fy) = (sy.floor() as usize, sy - sy.floor());
            let y1 = (y0 + 1).min(h - 1);
            for x in 0..INPUT {
                let sx = (((x as f32 + 0.5) * w as f32 / INPUT as f32) - 0.5).clamp(0.0, (w - 1) as f32);
                let (x0, fx) = (sx.floor() as usize, sx - sx.floor());
                let x1 = (x0 + 1).min(w - 1);
                for c in 0..3 {
                    let p = |xx: usize, yy: usize| rgb[(yy * w + xx) * 3 + c] as f32 / 255.0;
                    let v = (p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx) * (1.0 - fy) + (p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx) * fy;
                    t[c * INPUT * INPUT + y * INPUT + x] = (v - MEAN[c]) / STD[c];
                }
            }
        }
        t
    }

    /// Encodes the image unless `key` names the last one (then its embedding is reused).
    fn embed(&mut self, rgb: &[u8], w: u32, h: u32, key: &str) -> Result<f64, String> {
        if self.last.as_ref().is_some_and(|e| e.key == key && e.width == w && e.height == h) {
            return Ok(0.0);
        }
        if rgb.len() != w as usize * h as usize * 3 || w == 0 || h == 0 {
            return Err(format!("the image is {} bytes, not {w} × {h} RGB", rgb.len()));
        }
        let t0 = Instant::now();
        let input = Tensor::from_array((vec![1usize, 3, INPUT, INPUT], Self::preprocess(rgb, w as usize, h as usize))).map_err(|e| e.to_string())?;
        let out = self.encoder.run(ort::inputs!["image" => input]).map_err(|e| format!("encoder: {e}"))?;
        let (_, image_embed) = output(&out, "image_embed")?;
        let (_, high_res_0) = output(&out, "high_res_feats_0")?;
        let (_, high_res_1) = output(&out, "high_res_feats_1")?;
        drop(out);
        self.last = Some(Embedding { key: key.to_string(), width: w, height: h, image_embed, high_res_0, high_res_1 });
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    }

    /// Masks for `prompts` (image pixels) on the image, best first; the encode and decode times.
    pub fn segment(&mut self, rgb: &[u8], w: u32, h: u32, key: &str, prompts: &[Prompt], multimask: bool) -> Result<(Vec<Candidate>, f64, f64), String> {
        if prompts.is_empty() {
            return Err("no prompt: click on the object, or drag a box around it".into());
        }
        let encode_ms = self.embed(rgb, w, h, key)?;
        let t0 = Instant::now();
        let e = self.last.as_ref().ok_or("no embedding")?;
        // Points and box corners on the 1024 grid; labels 1 on, 0 off, 2 / 3 box corners
        let (sx, sy) = (INPUT as f32 / w as f32, INPUT as f32 / h as f32);
        let (mut coords, mut labels) = (Vec::new(), Vec::new());
        for p in prompts {
            match *p {
                Prompt::Point { x, y, positive } => {
                    coords.extend([x * sx, y * sy]);
                    labels.push(if positive { 1.0f32 } else { 0.0 });
                }
                Prompt::Box { x0, y0, x1, y1 } => {
                    coords.extend([x0.min(x1) * sx, y0.min(y1) * sy, x0.max(x1) * sx, y0.max(y1) * sy]);
                    labels.extend([2.0f32, 3.0]);
                }
            }
        }
        let n = labels.len();
        let t = |shape: Vec<usize>, v: Vec<f32>| Tensor::from_array((shape, v)).map_err(|e| e.to_string());
        let out = self
            .decoder
            .run(ort::inputs![
                "image_embed" => t(vec![1, 256, 64, 64], e.image_embed.clone())?,
                "high_res_feats_0" => t(vec![1, 32, 256, 256], e.high_res_0.clone())?,
                "high_res_feats_1" => t(vec![1, 64, 128, 128], e.high_res_1.clone())?,
                "point_coords" => t(vec![1, n, 2], coords)?,
                "point_labels" => t(vec![1, n], labels)?,
                "mask_input" => t(vec![1, 1, 256, 256], vec![0.0; 256 * 256])?,
                "has_mask_input" => t(vec![1], vec![0.0])?,
                "orig_im_size" => Tensor::from_array((vec![2usize], vec![h as i32, w as i32])).map_err(|e| e.to_string())?,
            ])
            .map_err(|e| format!("decoder: {e}"))?;
        let (shape, masks) = output(&out, "masks")?;
        let (_, scores) = output(&out, "iou_predictions")?;
        let (mh, mw) = (shape.get(2).copied().unwrap_or(h as usize), shape.get(3).copied().unwrap_or(w as usize));
        if (mw, mh) != (w as usize, h as usize) {
            return Err(format!("the decoder gave {mw} × {mh} masks for a {w} × {h} image"));
        }
        let px = mw * mh;
        let mut cands: Vec<Candidate> = scores
            .iter()
            .enumerate()
            .filter_map(|(i, &score)| masks.get(i * px..(i + 1) * px).map(|m| Candidate { mask: m.iter().map(|&v| if v > 0.0 { 255 } else { 0 }).collect(), score }))
            .collect();
        cands.sort_by(|a, b| b.score.total_cmp(&a.score));
        // An empty mask is a failure, not an answer (CubeCL #1461: zeros are not success)
        cands.retain(|c| c.mask.iter().any(|&v| v != 0));
        if cands.is_empty() {
            return Err("the model found nothing at the prompt".into());
        }
        if !multimask {
            cands.truncate(1);
        }
        Ok((cands, encode_ms, t0.elapsed().as_secs_f64() * 1000.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalogue_is_complete() {
        assert_eq!(MODELS.iter().filter(|m| m.default).count(), 1);
        for m in &MODELS {
            for (file, url, sha) in m.files {
                assert!(url.starts_with("https://huggingface.co/") && url.ends_with(".onnx"), "{file}: {url}");
                assert_eq!(sha.len(), 64);
            }
        }
    }

    #[test]
    fn preprocessing_normalises_and_stretches() {
        let rgb = vec![255u8; 2 * 2 * 3];
        let t = Sam2::preprocess(&rgb, 2, 2);
        assert_eq!(t.len(), 3 * INPUT * INPUT);
        assert!((t[0] - (1.0 - MEAN[0]) / STD[0]).abs() < 1e-5);
    }
}

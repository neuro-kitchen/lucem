//! StarDist (Schmidt, Weigert, Broaddus, Myers 2018; Weigert et al. 2020) for lumen, on Burn.
//! Nuclei as star-convex polygons: for every pixel the network gives the probability that it is
//! inside a nucleus and the distances to the nucleus' edge along 32 rays; the most probable
//! polygons that do not overlap become the nuclei.
//!
//! - Models ([`MODELS`]): the published "StarDist H&E Nuclei Segmentation" (Zenodo 6338615) for
//!   brightfield H&E (RGB), and "StarDist Fluorescence Nuclei Segmentation" (Zenodo 6348085,
//!   `2D_versatile_fluo`) for one fluorescence channel; both BSD-3-Clause, the same network but
//!   for its input channels. Its Keras weights (`stardist_weights.h5`) are fetched and verified through
//!   `lumen-hub` (SHA-256 pinned) and turned once into safetensors in lumen's cache
//!   ([`install`], [`convert`]); [`StarDist::load`] reads only the safetensors.
//! - Network: [`net`] on Burn (any of its devices: CUDA, wgpu, the CPU).
//! - Prediction: [`predict`] (normalization, tiles with a halo, thresholds and NMS as StarDist's
//!   own `predict_instances`).
//! - Scale: the model was trained on 40× H&E; images are resampled to [`MODEL_PIXEL_UM`] µm per
//!   pixel when their pixel size is known, and the labels back to the image's own pixels.

pub mod convert;
pub mod net;
pub mod predict;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::{Device, DeviceKind};
use lumen_ml::{ModelInfo, Task, Weight};

pub use predict::{Nucleus, Thresholds, Tiling, HE_THRESHOLDS};

/// The pixel size the H&E model works at (µm; StarDist's examples for it use 0.5).
pub const MODEL_PIXEL_UM: f64 = 0.5;

/// The fluorescence model's thresholds (its `rdf.yaml`).
pub const FLUO_THRESHOLDS: Thresholds = Thresholds { prob: 0.479_071_46, nms: 0.3 };

/// A published StarDist model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spec {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    /// The Keras weights as published: cache name, URL, SHA-256, bytes.
    pub keras: (&'static str, &'static str, &'static str, u64),
    /// The safetensors made from them.
    pub safetensors: &'static str,
    pub thresholds: Thresholds,
    pub default: bool,
    /// Channels of its images: 3 (RGB) or 1.
    pub channels: usize,
    /// The images it is for (`histology`, `fluorescence`).
    pub domains: &'static [&'static str],
    /// The pixel size it was trained at (µm), when it has one: images are resampled to it.
    pub pixel_um: Option<f64>,
}

pub static MODELS: [Spec; 2] = [
    Spec {
        id: "stardist-he",
        name: "StarDist · H&E nuclei",
        description: "Nuclei of H&E (brightfield) slides, trained on 40× tissue (~6 MB)",
        keras: ("stardist_he_weights.h5", "https://zenodo.org/api/records/6338615/files/stardist_weights.h5/content", "495ff069b1d3eb672ebf05046550a099d33cb00d5bdcd7d6bcc8abb59d795f7a", 5_774_704),
        safetensors: "stardist_he.safetensors",
        thresholds: HE_THRESHOLDS,
        default: true,
        channels: 3,
        domains: &["histology"],
        pixel_um: Some(MODEL_PIXEL_UM),
    },
    Spec {
        id: "stardist-fluo",
        name: "StarDist · fluorescence nuclei",
        description: "Nuclei of one fluorescence channel (DAPI, Hoechst; 2D_versatile_fluo, ~6 MB)",
        keras: ("stardist_fluo_weights.h5", "https://zenodo.org/api/records/6348085/files/stardist_weights.h5/content", "ff47d46e9aa23e52fd0c05dda624c13094664a3b6d2e715a629eb6d25e6c0184", 5_771_480),
        safetensors: "stardist_fluo.safetensors",
        thresholds: FLUO_THRESHOLDS,
        default: true,
        channels: 1,
        domains: &["fluorescence"],
        pixel_um: None,
    },
];

pub fn spec(id: &str) -> Option<&'static Spec> {
    MODELS.iter().find(|s| s.id == id)
}

pub fn path_of(file: &str) -> PathBuf {
    lumen_hub::cache_dir("stardist").join(file)
}

/// Whether the safetensors are in the cache.
pub fn installed(s: &Spec) -> bool {
    path_of(s.safetensors).is_file()
}

pub fn info(s: &Spec) -> ModelInfo {
    ModelInfo {
        id: s.id.into(),
        name: s.name.into(),
        family: "stardist".into(),
        task: Task::Segmentation,
        domains: s.domains.iter().map(|d| d.to_string()).collect(),
        description: s.description.into(),
        licence: "BSD-3-Clause".into(),
        paper: Some("Schmidt, Weigert, Broaddus, Myers (2018) Cell Detection with Star-convex Polygons. MICCAI. doi:10.1007/978-3-030-00934-2_30".into()),
        code: Some("https://github.com/stardist/stardist".into()),
        weights: vec![Weight { file: s.keras.0.into(), url: s.keras.1.into(), sha256: s.keras.2.into(), bytes: s.keras.3 }],
        installed: installed(s),
        default: s.default,
        channels: s.channels as u32,
    }
}

/// Fetches and verifies the Keras weights, then writes the safetensors next to them.
pub fn install(s: &Spec, progress: &mut dyn FnMut(u64, u64)) -> Result<(), String> {
    let (file, url, sha, bytes) = s.keras;
    let h5 = lumen_hub::fetch(url, &path_of(file), sha, bytes, progress)?;
    let layers = convert::read_keras(&h5, s.channels)?;
    let meta: HashMap<String, String> = [
        ("source", url.to_string()),
        ("source_sha256", sha.to_string()),
        ("model", s.name.to_string()),
        ("licence", "BSD-3-Clause (StarDist, Schmidt and Weigert)".to_string()),
        ("layout", "conv weight [out, in, kh, kw], from Keras [kh, kw, in, out]".to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    convert::write_safetensors(&layers, meta, &path_of(s.safetensors))
}

/// CubeCL's runtime configuration, once, before the first device opens: compiled kernels are
/// cached (CubeCL's default is off: without it every start of the plugin compiles and tunes the
/// network's kernels again, ~15 s), in the user cache directory (or `LUMEN_CACHE_DIR`) shared
/// with lumen. `cubecl.toml` and CubeCL's own variables still apply.
fn configure_runtime() {
    use cubecl::config::cache::CacheConfig;
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.compilation.cache = std::env::var("LUMEN_KERNEL_CACHE").map_or(true, |v| v.trim() != "0");
        match std::env::var_os("LUMEN_CACHE_DIR") {
            Some(dir) if !dir.is_empty() => config.environment.path = CacheConfig::Directory(dir.into()),
            _ if matches!(config.environment.path, CacheConfig::Target) => config.environment.path = CacheConfig::Global,
            _ => {}
        }
        let _ = CubeClRuntimeConfig::try_set(config);
    });
}

/// Whether `LUMEN_STARDIST_TIMING` asks for each phase's time on stderr.
pub(crate) fn timing() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var_os("LUMEN_STARDIST_TIMING").is_some())
}

/// The Burn device: `LUMEN_RUNTIME` (`cuda`, `wgpu`, `cpu`), else wgpu's default adapter.
pub fn device() -> Result<Device, String> {
    configure_runtime();
    let want = std::env::var("LUMEN_RUNTIME").unwrap_or_default().to_lowercase();
    match want.as_str() {
        #[cfg(feature = "cuda")]
        // Opening CUDA without its driver panics on the device thread: ask the system first
        "cuda" if cfg!(target_os = "linux") && !std::path::Path::new("/proc/driver/nvidia/version").exists() => Err("LUMEN_RUNTIME=cuda: the NVIDIA driver is not loaded".into()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::cuda(0)),
        #[cfg(feature = "cpu")]
        "cpu" => Ok(Device::cpu()),
        #[cfg(feature = "wgpu")]
        "" | "auto" | "wgpu" | "vulkan" => Ok(Device::wgpu(DeviceKind::DefaultDevice)),
        other => Err(format!("LUMEN_RUNTIME={other:?}: this build runs on {}", runtimes().join(", "))),
    }
}

fn runtimes() -> Vec<&'static str> {
    let mut v = Vec::new();
    if cfg!(feature = "wgpu") {
        v.push("wgpu");
    }
    if cfg!(feature = "cuda") {
        v.push("cuda");
    }
    if cfg!(feature = "cpu") {
        v.push("cpu");
    }
    v
}

pub struct StarDist {
    pub spec: &'static Spec,
    net: net::StarDistNet,
}

/// Nuclei of one image.
pub struct Instances {
    /// `w × h` of the image given (0: background, `k`: nucleus `k`).
    pub labels: Vec<u32>,
    pub nuclei: Vec<Nucleus>,
    pub ms: f64,
}

impl StarDist {
    /// Loads an installed model on `device`.
    pub fn load(s: &'static Spec, device: Device) -> Result<Self, String> {
        let path = path_of(s.safetensors);
        if !path.is_file() {
            return Err(format!("{} is not installed (lumen-ml-stardist install {})", s.name, s.id));
        }
        let layers = convert::read_safetensors(&path, s.channels)?;
        Ok(Self { spec: s, net: net::StarDistNet::new(layers, s.channels, device)? })
    }

    /// The network's output on the grid for an image (the model's channels, interleaved) at the
    /// model's own scale.
    pub fn grid(&self, pixels: &[u8], w: usize, h: usize, tiling: Tiling, progress: &mut dyn FnMut(usize, usize)) -> Result<predict::GridOutput, String> {
        let c = self.spec.channels;
        if pixels.len() != w * h * c {
            return Err(format!("{} bytes, not {w} × {h} × {c}", pixels.len()));
        }
        let planes = predict::normalize(pixels, w, h, c);
        predict::predict_grid(&self.net, &planes, w, h, tiling, progress)
    }

    /// The nuclei of an image (the model's channels, interleaved; `pixel_um`: its pixel size,
    /// when known — resampled to the model's when it has one).
    pub fn instances(&self, rgb: &[u8], w: usize, h: usize, pixel_um: Option<f64>, progress: &mut dyn FnMut(usize, usize)) -> Result<Instances, String> {
        let t0 = Instant::now();
        let scale = match (pixel_um.filter(|p| *p > 0.0), self.spec.pixel_um) {
            (Some(p), Some(m)) => p / m,
            _ => 1.0,
        };
        let resample = (scale - 1.0).abs() > 0.1;
        let (sw, sh) = if resample { (((w as f64) * scale).round().max(16.0) as usize, ((h as f64) * scale).round().max(16.0) as usize) } else { (w, h) };
        let scaled;
        let input = if resample {
            scaled = resize_pixels(rgb, w, h, sw, sh, self.spec.channels);
            &scaled
        } else {
            rgb
        };
        let t1 = Instant::now();
        let out = self.grid(input, sw, sh, Tiling::default(), progress)?;
        let t2 = Instant::now();
        let (labels, mut nuclei) = predict::instances(&out, sw, sh, self.spec.thresholds);
        if timing() {
            eprintln!("stardist: resample {:.0} ms, network {:.0} ms, NMS {:.0} ms ({} nuclei)", (t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3, t2.elapsed().as_secs_f64() * 1e3, nuclei.len());
        }
        let labels = if resample {
            for n in &mut nuclei {
                n.y = ((n.y as f64) / scale) as usize;
                n.x = ((n.x as f64) / scale) as usize;
            }
            resize_labels(&labels, sw, sh, w, h)
        } else {
            labels
        };
        Ok(Instances { labels, nuclei, ms: t0.elapsed().as_secs_f64() * 1e3 })
    }
}

/// Bilinear resize of `nc` interleaved channels.
fn resize_pixels(rgb: &[u8], w: usize, h: usize, nw: usize, nh: usize, nc: usize) -> Vec<u8> {
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
            for c in 0..nc {
                let v = |yy: usize, xx: usize| rgb[(yy * w + xx) * nc + c] as f32;
                let top = v(y0, x0) * (1.0 - tx) + v(y0, x1) * tx;
                let bot = v(y1, x0) * (1.0 - tx) + v(y1, x1) * tx;
                out[(y * nw + x) * nc + c] = (top * (1.0 - ty) + bot * ty).round() as u8;
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

    /// Reads a little-endian f32 `.npy` file: its shape and values.
    fn npy(path: &std::path::Path) -> (Vec<usize>, Vec<f32>) {
        let b = std::fs::read(path).unwrap();
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        let header = std::str::from_utf8(&b[10..10 + hl]).unwrap();
        assert!(header.contains("'<f4'") && header.contains("'fortran_order': False"), "{header}");
        let shape: Vec<usize> = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter_map(|s| s.trim().parse().ok()).collect();
        (shape, b[10 + hl..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    #[test]
    fn the_layers_add_up() {
        // Each layer takes what the one before gives (concatenations: the upsampled + the skip)
        let l = &net::LAYERS;
        assert_eq!(l[0].input, 3);
        assert_eq!(l[10].input, l[9].output + l[7].output);
        assert_eq!(l[12].input, l[11].output + l[5].output);
        assert_eq!(l[14].input, l[13].output + l[3].output);
        assert_eq!(l[17].output + l[18].output, 33);
    }

    #[test]
    fn a_star_polygon_becomes_one_nucleus() {
        // One confident point with rays of 6 px, its neighbours less sure: one disk of r ≈ 6
        let (gh, gw) = (20, 20);
        let mut prob = vec![0.0f32; gh * gw];
        let dist = vec![6.0f32; gh * gw * net::N_RAYS];
        prob[10 * gw + 10] = 0.95;
        prob[10 * gw + 11] = 0.9;
        prob[5 * gw + 5] = 0.5;
        let out = predict::GridOutput { gh, gw, prob, dist };
        let (labels, nuclei) = predict::instances(&out, 40, 40, HE_THRESHOLDS);
        assert_eq!(nuclei.len(), 1, "{nuclei:?}");
        let area = labels.iter().filter(|&&l| l == 1).count() as f32;
        let disk = std::f32::consts::PI * 36.0;
        assert!((area - disk).abs() / disk < 0.1, "area {area} vs {disk}");
        assert_eq!(labels[20 * 40 + 20], 1);
    }

    /// The published test: `test_input.npy` through the network must give `test_output.npy` (the
    /// TensorFlow export's output, upsampled to the input; compared on the grid's points). Needs
    /// the model installed and `LUMEN_STARDIST_TEST_DIR` holding the two files from Zenodo 6338615.
    #[test]
    #[ignore = "needs the model installed and the published test files"]
    fn matches_the_published_test_output() {
        published(&MODELS[0], "LUMEN_STARDIST_TEST_DIR");
    }

    /// As above for the fluorescence model (`LUMEN_STARDIST_FLUO_TEST_DIR`: Zenodo 6348085's files).
    #[test]
    #[ignore = "needs the model installed and the published test files"]
    fn fluorescence_matches_the_published_test_output() {
        published(&MODELS[1], "LUMEN_STARDIST_FLUO_TEST_DIR");
    }

    fn published(s: &'static Spec, dir_var: &str) {
        let dir = PathBuf::from(std::env::var(dir_var).expect(dir_var));
        let (si, input) = npy(&dir.join("test_input.npy"));
        let (so, expected) = npy(&dir.join("test_output.npy"));
        let (h, w) = (si[1], si[2]);
        assert_eq!((so[1], so[2], so[3], si[3]), (h, w, 33, s.channels));
        let rgb: Vec<u8> = input.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
        if !installed(s) {
            install(s, &mut |_, _| {}).unwrap();
        }
        let model = StarDist::load(s, device().unwrap()).unwrap();
        let out = model.grid(&rgb, w, h, Tiling::default(), &mut |_, _| {}).unwrap();
        let (mut worst_p, mut worst_d, mut sum_d, mut n) = (0f32, 0f32, 0f64, 0usize);
        // Within the network's reach (96 px) of an edge where the tile was padded, the padding
        // (mirrored, as StarDist's own prediction pads) differs from the export's zero padding
        // inside each layer: compared away from those edges
        let reach = 96 / net::GRID;
        let padded_y = h % 256 != 0;
        let padded_x = w % 256 != 0;
        for gy in 0..out.gh {
            if padded_y && gy + reach >= out.gh {
                continue;
            }
            for gx in 0..out.gw {
                if padded_x && gx + reach >= out.gw {
                    continue;
                }
                let (y, x) = (gy * net::GRID, gx * net::GRID);
                let e = &expected[(y * w + x) * 33..(y * w + x) * 33 + 33];
                worst_p = worst_p.max((out.prob[gy * out.gw + gx] - e[0]).abs());
                for r in 0..net::N_RAYS {
                    let d = (out.dist[(gy * out.gw + gx) * net::N_RAYS + r] - e[1 + r]).abs();
                    worst_d = worst_d.max(d);
                    sum_d += d as f64;
                    n += 1;
                }
            }
        }
        let mean_d = sum_d / n as f64;
        eprintln!("probability: worst {worst_p:.2e}; distances: worst {worst_d:.2e} px, mean {mean_d:.2e} px");
        let i = (out.gh / 2) * out.gw + out.gw / 2;
        let e = &expected[((out.gh / 2) * net::GRID * w + (out.gw / 2) * net::GRID) * 33..][..4];
        eprintln!("centre: ours prob {:.5} dist {:.4} {:.4} {:.4}; published {:.5} {:.4} {:.4} {:.4}", out.prob[i], out.dist[i * 32], out.dist[i * 32 + 1], out.dist[i * 32 + 2], e[0], e[1], e[2], e[3]);
        let positive = out.prob.iter().filter(|&&p| p > 0.5).count();
        assert!(positive > 100, "only {positive} grid points over 0.5");
        assert!(worst_p < 0.02, "probability off by {worst_p}");
        assert!(mean_d < 0.05, "distances off by {mean_d} px on average");
        let (_, nuclei) = predict::instances(&out, w, h, s.thresholds);
        eprintln!("{} nuclei", nuclei.len());
        assert!(nuclei.len() > 20);
    }
}

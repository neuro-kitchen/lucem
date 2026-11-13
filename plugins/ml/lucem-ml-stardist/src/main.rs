//! `lucem-ml-stardist`: lucem's StarDist plugin, a separate program (BSD-3-Clause, as StarDist).
//! lucem finds it by its name (`lucem-ml-*`, see `lucem-ml`) and runs it; the network runs on
//! Burn (CUDA, wgpu or the CPU: `LUCEM_RUNTIME`).
//!
//! ```text
//! lucem-ml-stardist info               the manifest (JSON): models, whether installed
//! lucem-ml-stardist install <model>    fetches the published Keras weights (verified) and turns
//!                                      them into safetensors in lucem's cache
//! lucem-ml-stardist tune [model]...    runs every installed model (or those named) on every tile
//!                                      shape, so tuned kernels are cached (`lucem autotune` runs it)
//! lucem-ml-stardist bench [--model ID] [--sizes 512,1024] [--runs N]
//!                                      times whole images, cold and warm
//! lucem-ml-stardist serve              requests on stdin, events on stdout (lucem-ml's protocol);
//!                                      `instances`: every nucleus of an RGB image as labels
//! ```

use std::collections::HashMap;
use std::io::{BufRead, Write};

use lucem_ml::{Event, InstanceClass, InstancesResult, PluginInfo, RasterRef, Request, RuntimeStatus};
use lucem_model_stardist::{self as sd, StarDist};

fn plugin_info() -> PluginInfo {
    PluginInfo {
        name: "StarDist".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        licence: "BSD-3-Clause".into(),
        runtime: match sd::device() {
            Ok(d) => RuntimeStatus { ready: true, detail: format!("Burn on {d:?}"), gpu: None },
            Err(e) => RuntimeStatus { ready: false, detail: e, gpu: None },
        },
        models: sd::MODELS.iter().map(sd::info).collect(),
        program: String::new(),
    }
}

fn emit(e: &Event) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{}", e.line());
    let _ = out.flush();
}

fn install(id: &str, events: bool) -> Result<(), String> {
    if id == lucem_ml::RUNTIME_ID {
        // Burn is built in: nothing to fetch
        return Ok(());
    }
    let s = sd::spec(id).ok_or_else(|| format!("no model {id:?}; models: {}", sd::MODELS.iter().map(|m| m.id).collect::<Vec<_>>().join(", ")))?;
    let mut last = -1i64;
    sd::install(s, &mut |done, total| {
        let pct = (done * 100 / total.max(1)) as i64;
        if pct != last {
            last = pct;
            if events {
                emit(&Event::Progress { stage: format!("downloading {}", s.name), fraction: Some(pct as f32 / 100.0) });
            } else {
                eprint!("\r{} {pct:3} %", s.name);
            }
        }
    })?;
    if !events {
        eprintln!();
    }
    Ok(())
}

fn instances(models: &mut HashMap<String, StarDist>, model: &str, image: &RasterRef, pixel_um: Option<f64>, out_dir: &str) -> Result<InstancesResult, String> {
    let s = sd::spec(model).ok_or_else(|| format!("no model {model:?}"))?;
    if !models.contains_key(model) {
        emit(&Event::Progress { stage: format!("loading {}", s.name), fraction: None });
        models.insert(model.to_string(), StarDist::load(s, sd::device()?)?);
    }
    let m = models.get(model).ok_or("model not loaded")?;
    if image.channels as usize != s.channels {
        return Err(format!("{} takes images of {} channel(s), not {}", s.name, s.channels, image.channels));
    }
    let rgb = std::fs::read(&image.path).map_err(|e| format!("{}: {e}", image.path))?;
    if rgb.len() != image.len() {
        return Err(format!("{} holds {} bytes, not {} × {} × {}", image.path, rgb.len(), image.width, image.height, image.channels));
    }
    let (w, h) = (image.width as usize, image.height as usize);
    let r = m.instances(&rgb, w, h, pixel_um, &mut |d, t| emit(&Event::Progress { stage: format!("finding nuclei: tile {d} of {t}"), fraction: Some(d as f32 / t.max(1) as f32) }))?;
    let path = std::path::Path::new(out_dir).join("labels.u32");
    let bytes: Vec<u8> = r.labels.iter().flat_map(|l| l.to_le_bytes()).collect();
    std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let classes = r.nuclei.iter().map(|n| InstanceClass { label: n.label, class: "Nucleus".into(), score: Some(n.prob) }).collect();
    Ok(InstancesResult { labels: RasterRef { path: path.to_string_lossy().into_owned(), width: image.width, height: image.height, channels: 1 }, classes, ms: r.ms })
}

fn serve() {
    let mut models: HashMap<String, StarDist> = HashMap::new();
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                emit(&Event::Error { message: format!("the request does not read: {e}") });
                continue;
            }
        };
        let answer = match req {
            Request::Quit => break,
            Request::Info => serde_json::to_value(plugin_info()).map_err(|e| e.to_string()),
            Request::Install { model } => install(&model, true).map(|_| serde_json::Value::Null),
            Request::Instances { model, image, pixel_um, out_dir } => instances(&mut models, &model, &image, pixel_um, &out_dir).and_then(|r| serde_json::to_value(r).map_err(|e| e.to_string())),
            Request::Segment { .. } => Err("StarDist finds every nucleus of an image (instances); to segment what you point at use Segment Anything".into()),
        };
        emit(&match answer {
            Ok(result) => Event::Done { result },
            Err(message) => Event::Error { message },
        });
    }
}

/// Tunes the installed models (or `ids`) on this machine's device (`LUCEM_RUNTIME`).
fn tune(ids: &[String]) -> Result<(), String> {
    let specs: Vec<&'static sd::Spec> = if ids.is_empty() { sd::MODELS.iter().filter(|s| sd::installed(s)).collect() } else { ids.iter().map(|id| sd::spec(id).ok_or_else(|| format!("no model {id:?}"))).collect::<Result<_, _>>()? };
    if specs.is_empty() {
        eprintln!("stardist: no model installed, nothing to tune (lucem-ml-stardist install <model>)");
        return Ok(());
    }
    let device = sd::device()?;
    for s in specs {
        if !sd::installed(s) {
            install(s.id, false)?;
        }
        let model = StarDist::load(s, device.clone())?;
        let r = sd::tune(&model, |d, t| eprint!("\rstardist: {}: shape {d} of {t}", s.id))?;
        eprintln!("\rstardist: {}: {} tile shapes tuned in {:.0} s", s.id, r.shapes, r.seconds);
    }
    Ok(())
}

/// A deterministic test image: round nuclei about 8–14 px across on a textured background,
/// dark purple on pink for 3 channels, bright on dark for 1.
fn synthetic(w: usize, h: usize, channels: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut amount = vec![0f32; w * h];
    for cy in (0..h).step_by(20) {
        for cx in (0..w).step_by(20) {
            if next() < 0.15 {
                continue;
            }
            let (y0, x0, r) = (cy as f64 + next() * 20.0, cx as f64 + next() * 20.0, 4.0 + next() * 3.0);
            for y in ((y0 - r - 1.0).max(0.0) as usize)..((y0 + r + 2.0) as usize).min(h) {
                for x in ((x0 - r - 1.0).max(0.0) as usize)..((x0 + r + 2.0) as usize).min(w) {
                    let d = (((y as f64 - y0).powi(2) + (x as f64 - x0).powi(2)).sqrt() - r).clamp(-1.0, 1.0);
                    let a = &mut amount[y * w + x];
                    *a = a.max((0.5 - d * 0.5) as f32);
                }
            }
        }
    }
    amount
        .into_iter()
        .flat_map(|a| {
            let a = (a + (next() - 0.5) as f32 * 0.08).clamp(0.0, 1.0);
            let px = if channels == 1 { vec![(20.0 + 200.0 * a) as u8] } else { [230.0 - 140.0 * a, 190.0 - 130.0 * a, 210.0 - 70.0 * a].map(|v| v.clamp(0.0, 255.0) as u8).to_vec() };
            px
        })
        .collect()
}

/// The value after `flag` in `args`, if given.
fn flag(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

/// Times whole images (the network and the polygons), cold and the median of `--runs` warm runs.
fn bench(args: &[String]) -> Result<(), String> {
    let id = flag(args, "--model").unwrap_or_else(|| sd::MODELS[0].id.to_string());
    let s = sd::spec(&id).ok_or_else(|| format!("no model {id:?}"))?;
    let sizes: Vec<usize> = match flag(args, "--sizes") {
        Some(v) => v.split(',').map(|x| x.trim().parse::<usize>().map_err(|e| format!("--sizes {x:?}: {e}"))).collect::<Result<_, _>>()?,
        None => vec![512, 1024, 2048, 4096],
    };
    let runs: usize = flag(args, "--runs").map(|v| v.parse().map_err(|e| format!("--runs: {e}"))).transpose()?.unwrap_or(3);
    if !sd::installed(s) {
        install(s.id, false)?;
    }
    let device = sd::device()?;
    let label = format!("{device:?}");
    let model = StarDist::load(s, device)?;
    eprintln!("{}: {label}, {:?}, {runs} warm run(s) per size", s.name, model.precision());
    println!("{:>6} {:>7} {:>9} {:>9}", "side", "nuclei", "cold ms", "warm ms");
    for side in sizes {
        let px = synthetic(side, side, s.channels);
        let t0 = std::time::Instant::now();
        let r = model.instances(&px, side, side, None, &mut |_, _| {})?;
        let cold = t0.elapsed().as_secs_f64() * 1e3;
        let mut warm: Vec<f64> = (0..runs.max(1)).map(|_| model.instances(&px, side, side, None, &mut |_, _| {}).map(|r| r.ms)).collect::<Result<_, _>>()?;
        warm.sort_by(f64::total_cmp);
        println!("{side:>6} {:>7} {cold:>9.0} {:>9.1}", r.nuclei.len(), warm[warm.len() / 2]);
    }
    Ok(())
}

fn main() {
    lucem_ml::legacy::adopt();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => println!("{}", serde_json::to_string_pretty(&plugin_info()).unwrap_or_default()),
        Some("install") => {
            let Some(id) = args.get(1) else {
                eprintln!("usage: lucem-ml-stardist install <model>");
                std::process::exit(2);
            };
            if let Err(e) = install(id, false) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        Some("bench") => {
            if let Err(e) = bench(&args[1..]) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        Some("tune") => {
            if let Err(e) = tune(&args[1..]) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        Some("serve") => serve(),
        _ => {
            eprintln!("lucem-ml-stardist {}: StarDist nuclei for lucem (BSD-3-Clause)\n\nusage:\n  lucem-ml-stardist info\n  lucem-ml-stardist install <model>\n  lucem-ml-stardist tune [model]...\n  lucem-ml-stardist bench [--model ID] [--sizes 512,1024] [--runs N]\n  lucem-ml-stardist serve", env!("CARGO_PKG_VERSION"));
            std::process::exit(2);
        }
    }
}

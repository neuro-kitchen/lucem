//! `lucem-ml-instanseg`: lucem's InstanSeg plugin, a separate program (Apache-2.0, as InstanSeg).
//! lucem finds it by its name (`lucem-ml-*`, see `lucem-ml`) and runs it; the network runs on
//! Burn (CUDA, wgpu or the CPU: `LUCEM_RUNTIME`).
//!
//! ```text
//! lucem-ml-instanseg info               the manifest (JSON): models, whether installed
//! lucem-ml-instanseg install <model>   fetches the published model package (verified) and turns
//!                                      its TorchScript weights into safetensors in lucem's cache
//! lucem-ml-instanseg serve              requests on stdin, events on stdout (lucem-ml's protocol);
//!                                      `instances`: every nucleus of an image as labels
//! ```

use std::collections::HashMap;
use std::io::{BufRead, Write};

use lucem_ml::{Event, InstanceClass, InstancesResult, PluginInfo, RasterRef, Request, RuntimeStatus};
use lucem_model_instanseg::{self as sd, InstanSeg};

fn plugin_info() -> PluginInfo {
    PluginInfo {
        name: "InstanSeg".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        licence: "Apache-2.0".into(),
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

fn instances(models: &mut HashMap<String, InstanSeg>, model: &str, image: &RasterRef, pixel_um: Option<f64>, out_dir: &str) -> Result<InstancesResult, String> {
    let s = sd::spec(model).ok_or_else(|| format!("no model {model:?}"))?;
    if !models.contains_key(model) {
        emit(&Event::Progress { stage: format!("loading {}", s.name), fraction: None });
        models.insert(model.to_string(), InstanSeg::load(s, sd::device()?)?);
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
    let classes = r.found.iter().map(|n| InstanceClass { label: n.label, class: "Nucleus".into(), score: Some(n.score) }).collect();
    Ok(InstancesResult { labels: RasterRef { path: path.to_string_lossy().into_owned(), width: image.width, height: image.height, channels: 1 }, classes, ms: r.ms })
}

fn serve() {
    let mut models: HashMap<String, InstanSeg> = HashMap::new();
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
            Request::Segment { .. } => Err("InstanSeg finds every nucleus of an image (instances); to segment what you point at use Segment Anything".into()),
        };
        emit(&match answer {
            Ok(result) => Event::Done { result },
            Err(message) => Event::Error { message },
        });
    }
}

fn main() {
    lucem_ml::legacy::adopt();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => println!("{}", serde_json::to_string_pretty(&plugin_info()).unwrap_or_default()),
        Some("install") => {
            let Some(id) = args.get(1) else {
                eprintln!("usage: lucem-ml-instanseg install <model>");
                std::process::exit(2);
            };
            if let Err(e) = install(id, false) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        Some("serve") => serve(),
        _ => {
            eprintln!("lucem-ml-instanseg {}: InstanSeg nuclei for lucem (Apache-2.0)\n\nusage:\n  lucem-ml-instanseg info\n  lucem-ml-instanseg install <model>\n  lucem-ml-instanseg serve", env!("CARGO_PKG_VERSION"));
            std::process::exit(2);
        }
    }
}

//! `lumen-ml-sam`: lumen's Segment Anything plugin, a separate program (Apache-2.0, as SAM 2).
//! lumen finds it by its name (`lumen-ml-*`, see `lumen-ml`) and runs it; it never links it, so
//! ONNX Runtime and the model's weights stay out of lumen itself.
//!
//! ```text
//! lumen-ml-sam info               the manifest (JSON): models, whether installed, the runtime
//! lumen-ml-sam install <model>    downloads and verifies a model's weights
//! lumen-ml-sam install runtime    downloads and verifies ONNX Runtime into lumen's cache
//! lumen-ml-sam install runtime-cuda   its GPU build (CUDA 13; used when cuDNN 9 is found)
//! lumen-ml-sam serve              requests on stdin, events on stdout, one JSON object per line
//!                                 (lumen-ml's protocol); models stay loaded between requests
//! ```

use std::collections::HashMap;
use std::io::{BufRead, Write};

use lumen_ml::{Event, MaskOut, PluginInfo, RasterRef, Request, SegmentResult};
use lumen_model_sam::{self as sam, Sam2};

fn plugin_info() -> PluginInfo {
    PluginInfo {
        name: "Segment Anything".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        licence: "Apache-2.0".into(),
        runtime: sam::init_runtime(),
        models: sam::MODELS.iter().map(sam::info).collect(),
        program: String::new(),
    }
}

fn emit(e: &Event) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{}", e.line());
    let _ = out.flush();
}

/// Installs a model, reporting progress as events (`serve`) or on stderr (`install`).
fn install(id: &str, events: bool) -> Result<(), String> {
    if id == sam::RUNTIME_ID || id == sam::RUNTIME_CUDA_ID {
        let gpu = id == sam::RUNTIME_CUDA_ID;
        let mut last = -1i64;
        let mut report = |done: u64, total: u64| {
            let pct = (done * 100 / total.max(1)) as i64;
            if pct != last {
                last = pct;
                if events {
                    emit(&Event::Progress { stage: if gpu { "downloading ONNX Runtime (GPU)".into() } else { "downloading ONNX Runtime".into() }, fraction: Some(pct as f32 / 100.0) });
                } else {
                    eprint!("\rONNX Runtime {pct:3} %");
                }
            }
        };
        if gpu { sam::install_runtime_cuda(&mut report) } else { sam::install_runtime(&mut report) }?;
        if !events {
            eprintln!();
        }
        return Ok(());
    }
    let s = sam::spec(id).ok_or_else(|| format!("no model {id:?}; models: {}", sam::MODELS.iter().map(|m| m.id).collect::<Vec<_>>().join(", ")))?;
    let mut last = -1i64;
    sam::install(s, &mut |done, total| {
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

fn segment(models: &mut HashMap<String, Sam2>, model: &str, image: &RasterRef, key: &str, prompts: &[lumen_ml::Prompt], multimask: bool, out_dir: &str) -> Result<SegmentResult, String> {
    let s = sam::spec(model).ok_or_else(|| format!("no model {model:?}"))?;
    if !models.contains_key(model) {
        emit(&Event::Progress { stage: format!("loading {}", s.name), fraction: None });
        models.insert(model.to_string(), Sam2::load(s)?);
    }
    let m = models.get_mut(model).ok_or("model not loaded")?;
    if image.channels != 3 {
        return Err("the image must be RGB (3 channels)".into());
    }
    let rgb = std::fs::read(&image.path).map_err(|e| format!("{}: {e}", image.path))?;
    if rgb.len() != image.len() {
        return Err(format!("{} holds {} bytes, not {} × {} × 3", image.path, rgb.len(), image.width, image.height));
    }
    emit(&Event::Progress { stage: "segmenting".into(), fraction: None });
    let (cands, encode_ms, decode_ms) = m.segment(&rgb, image.width, image.height, key, prompts, multimask)?;
    let mut masks = Vec::new();
    for (i, c) in cands.into_iter().enumerate() {
        let path = std::path::Path::new(out_dir).join(format!("mask_{i}.u8"));
        std::fs::write(&path, &c.mask).map_err(|e| format!("{}: {e}", path.display()))?;
        masks.push(MaskOut { raster: RasterRef { path: path.to_string_lossy().into_owned(), width: image.width, height: image.height, channels: 1 }, score: c.score });
    }
    Ok(SegmentResult { masks, encode_ms, decode_ms })
}

/// Requests on stdin until `quit` or the end of input.
fn serve() {
    let mut models: HashMap<String, Sam2> = HashMap::new();
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
            Request::Instances { .. } => Err("Segment Anything segments what it is pointed at; for every nucleus of a region use a nucleus model plugin".into()),
            Request::Segment { model, image, image_key, prompts, multimask, out_dir } => segment(&mut models, &model, &image, &image_key, &prompts, multimask, &out_dir).and_then(|r| serde_json::to_value(r).map_err(|e| e.to_string())),
        };
        emit(&match answer {
            Ok(result) => Event::Done { result },
            Err(message) => Event::Error { message },
        });
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => println!("{}", serde_json::to_string_pretty(&plugin_info()).unwrap_or_default()),
        Some("install") => {
            let Some(id) = args.get(1) else {
                eprintln!("usage: lumen-ml-sam install <model>");
                std::process::exit(2);
            };
            if let Err(e) = install(id, false) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        Some("serve") => serve(),
        _ => {
            eprintln!("lumen-ml-sam {}: Segment Anything for lumen (Apache-2.0)\n\nusage:\n  lumen-ml-sam info\n  lumen-ml-sam install <model>\n  lumen-ml-sam serve", env!("CARGO_PKG_VERSION"));
            std::process::exit(2);
        }
    }
}

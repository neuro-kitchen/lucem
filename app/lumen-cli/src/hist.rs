//! `lumen hist …`: the histology workflows of `lumen-hist` on an image, for batch runs.

use std::path::Path;

use lumen_core::compute::ComputeTask;
use lumen_core::cubecl::prelude::Client;
use lumen_core::ComputeTarget;
use lumen_hist::cells::CellDetection;
use lumen_hist::classify::ObjectClassifier;
use lumen_hist::export::{geojson, measurements_table, region_summary};
use lumen_hist::intensity::{classify as classify_intensity, IntensityThresholds};
use lumen_hist::raster::Grid;
use lumen_hist::slide::{detect_cells_in_region, SlideRegion, Staining, Tiling};
use lumen_hist::stains::{estimate_background, estimate_macenko, StainVectors};
use lumen_hist::tissue::{detect as detect_tissue, TissueSettings};
use lumen_hist::{HistObject, Polygon};
use lumen_io::Pyramid;

use crate::Args;

pub const USAGE: &str = "\
  lumen hist stains <image> --region X,Y,W,H [--stains h-dab|h-e] [--out FILE.json]
                                         Stain vectors fitted to a region (level-0 px)
  lumen hist tissue <image> [--threshold T] [--fluorescence] --out FILE.geojson
                                         Tissue regions, found at a coarse level
  lumen hist cells <image> [--region X,Y,W,H | --within FILE.geojson] [--level N]
              [--stains h-dab|h-e|h-e-dab|FILE.json | --nuclear CHANNEL]
              [--threshold T] [--sigma UM] [--expansion UM] [--min-area UM2] [--max-area UM2]
              [--background UM] [--tile PX] [--classifier FILE.json]
              [--intensity \"MEASUREMENT=T1[,T2,T3]\"] [--out FILE.tsv|.csv|.geojson]
              [--runtime NAME]               Cells detected, measured, classified; a summary
";

fn number<T: std::str::FromStr>(args: &mut Args, name: &str) -> Result<Option<T>, String> {
    args.take_value(name)?.map(|v| v.parse::<T>().map_err(|_| format!("{name}: {v:?} is not a number"))).transpose()
}

fn rect(v: &str) -> Result<[usize; 4], String> {
    let p: Vec<usize> = v.split(',').map(|s| s.trim().parse::<usize>()).collect::<Result<_, _>>().map_err(|_| format!("--region {v:?}: X,Y,W,H in pixels"))?;
    match p.as_slice() {
        [x, y, w, h] if *w > 0 && *h > 0 => Ok([*x, *y, *w, *h]),
        _ => Err(format!("--region {v:?}: X,Y,W,H in pixels")),
    }
}

fn stains_named(v: &str) -> Result<StainVectors, String> {
    match v {
        "h-dab" => Ok(StainVectors::h_dab()),
        "h-e" => Ok(StainVectors::h_e()),
        "h-e-dab" => Ok(StainVectors::h_e_dab()),
        file => {
            let text = std::fs::read_to_string(file).map_err(|e| format!("--stains {file}: {e}"))?;
            serde_json::from_str(&text).map_err(|e| format!("--stains {file}: {e}"))
        }
    }
}

fn pixel_um(p: &Pyramid) -> Result<f64, String> {
    p.pixel_size_um.ok_or_else(|| "the image has no pixel size (µm): histology measurements need it".to_string())
}

/// RGB of a level-0 region, interleaved 0–255.
fn read_rgb(p: &Pyramid, level: usize, [x, y, w, h]: [usize; 4]) -> Result<Vec<u8>, String> {
    if p.channels.len() < 3 {
        return Err("a brightfield image has 3 channels (RGB)".into());
    }
    let planes: Vec<Vec<f32>> = (0..3).map(|c| p.read_region_at(level, c, 0, x, y, w, h).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    let scale: Vec<f32> = (0..3).map(|c| 255.0 / p.channels[c].range.1.max(1e-6)).collect();
    Ok((0..w * h).flat_map(|i| (0..3).map(move |c| (i, c))).map(|(i, c)| (planes[c][i] * scale[c]).round().clamp(0.0, 255.0) as u8).collect())
}

fn write_out(path: &str, objects: &[HistObject], image: &str, pixel_um: f64) -> Result<(), String> {
    let text = if path.ends_with(".geojson") || path.ends_with(".json") {
        serde_json::to_string(&geojson(objects)).map_err(|e| e.to_string())?
    } else {
        measurements_table(objects, image, pixel_um).to_delimited(if path.ends_with(".csv") { ',' } else { '\t' })
    };
    std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))?;
    println!("wrote {path}");
    Ok(())
}

/// The outlines of a GeoJSON file (Polygon features; the first ring of each).
fn read_outlines(path: &str) -> Result<Vec<Polygon>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    let features = v["features"].as_array().cloned().unwrap_or_else(|| vec![v.clone()]);
    let outlines: Vec<Polygon> = features
        .iter()
        .filter_map(|f| f["geometry"]["coordinates"][0].as_array().cloned())
        .map(|ring| ring.iter().filter_map(|p| Some([p[0].as_f64()?, p[1].as_f64()?])).collect::<Polygon>())
        .filter(|p| p.len() >= 3)
        .collect();
    if outlines.is_empty() {
        return Err(format!("{path}: no Polygon feature"));
    }
    Ok(outlines)
}

pub fn hist(mut args: Args) -> Result<(), String> {
    if args.rest.is_empty() {
        return Err(format!("usage:\n{USAGE}"));
    }
    let what = args.rest.remove(0);
    match what.as_str() {
        "stains" => stains(args),
        "tissue" => tissue(args),
        "cells" => cells(args),
        other => Err(format!("unknown hist command {other:?}\n\nusage:\n{USAGE}")),
    }
}

fn stains(mut args: Args) -> Result<(), String> {
    let region = rect(&args.take_value("--region")?.ok_or("--region X,Y,W,H is needed: a region with both stains")?)?;
    let preset = stains_named(&args.take_value("--stains")?.unwrap_or_else(|| "h-dab".into()))?;
    let out = args.take_value("--out")?;
    let a = args.positional(1)?;
    let p = Pyramid::open(Path::new(&a[0])).map_err(|e| e.to_string())?;
    let rgb = read_rgb(&p, 0, region)?;
    let bg = estimate_background(&rgb);
    let names = (preset.stains[0].name.as_str(), preset.stains[1].name.as_str());
    let est = estimate_macenko(&rgb, bg, names, 0.15, 1.0).map_err(|e| e.to_string())?;
    let text = serde_json::to_string_pretty(&est).map_err(|e| e.to_string())?;
    match out {
        Some(path) => {
            std::fs::write(&path, &text).map_err(|e| format!("{path}: {e}"))?;
            println!("wrote {path}");
        }
        None => println!("{text}"),
    }
    Ok(())
}

fn tissue(mut args: Args) -> Result<(), String> {
    let threshold: Option<f64> = number(&mut args, "--threshold")?;
    let fluorescence = args.take_flag("--fluorescence");
    let out = args.take_value("--out")?.ok_or("--out FILE.geojson is needed")?;
    let a = args.positional(1)?;
    let p = Pyramid::open(Path::new(&a[0])).map_err(|e| e.to_string())?;
    let um = pixel_um(&p)?;
    // The finest level that fits 4096 px a side
    let level = (0..p.levels.len()).find(|&l| p.levels[l].width.max(p.levels[l].height) <= 4096).unwrap_or(p.coarsest());
    let l = p.levels[level].clone();
    let channels = if fluorescence { p.channels.len() } else { 3.min(p.channels.len()) };
    let planes: Vec<Vec<f32>> = (0..channels).map(|c| p.read_region_at(level, c, 0, 0, 0, l.width, l.height).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    let scale: Vec<f32> = (0..channels).map(|c| if fluorescence { 1.0 } else { 255.0 / p.channels[c].range.1.max(1e-6) }).collect();
    let n = l.width * l.height;
    let image: Vec<f32> = (0..n).flat_map(|i| (0..channels).map(move |c| (i, c))).map(|(i, c)| planes[c][i] * scale[c]).collect();
    let t = detect_tissue(&image, channels, Grid::new(l.width, l.height, [0.0, 0.0], l.downsample), um, &TissueSettings { brightfield: !fluorescence, threshold, ..TissueSettings::default() });
    println!("level {level} ({} × {} px), threshold {:.3}: {} tissue region(s), {:.2} mm²", l.width, l.height, t.threshold, t.regions.len(), t.regions.iter().filter_map(|r| r.get("Area µm^2")).sum::<f64>() / 1e6);
    write_out(&out, &t.regions, &a[0], um)
}

struct Job<F>(F);

impl<F: FnOnce(&Client) -> Result<(), String>> ComputeTask for Job<F> {
    type Output = Result<(), String>;
    fn run(self, client: Client) -> Self::Output {
        (self.0)(&client)
    }
}

fn cells(mut args: Args) -> Result<(), String> {
    let region = args.take_value("--region")?.map(|v| rect(&v)).transpose()?;
    let within = args.take_value("--within")?.map(|p| read_outlines(&p)).transpose()?;
    let level: usize = number(&mut args, "--level")?.unwrap_or(0);
    let stains = args.take_value("--stains")?;
    let nuclear: Option<usize> = number(&mut args, "--nuclear")?;
    let mut s = CellDetection::default();
    let threshold: Option<f64> = number(&mut args, "--threshold")?;
    if let Some(v) = threshold {
        s.threshold = v;
    }
    if let Some(v) = number(&mut args, "--sigma")? {
        s.sigma_um = v;
    }
    if let Some(v) = number(&mut args, "--expansion")? {
        s.cell_expansion_um = v;
    }
    if let Some(v) = number(&mut args, "--min-area")? {
        s.min_area_um2 = v;
    }
    if let Some(v) = number(&mut args, "--max-area")? {
        s.max_area_um2 = v;
    }
    if let Some(v) = number(&mut args, "--background")? {
        s.background_radius_um = v;
    }
    let tile: usize = number(&mut args, "--tile")?.unwrap_or(1024);
    let classifier = args.take_value("--classifier")?.map(|p| std::fs::read_to_string(&p).map_err(|e| format!("{p}: {e}")).and_then(|t| ObjectClassifier::from_json(&t).map_err(|e| e.to_string()))).transpose()?;
    let intensity = args
        .take_value("--intensity")?
        .map(|v| -> Result<IntensityThresholds, String> {
            let (m, t) = v.rsplit_once('=').ok_or(format!("--intensity {v:?}: MEASUREMENT=T1[,T2,T3]"))?;
            let thresholds: Vec<f64> = t.split(',').map(|x| x.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| format!("--intensity {v:?}: thresholds are numbers"))?;
            Ok(IntensityThresholds { measurement: m.trim().into(), thresholds, filter: None })
        })
        .transpose()?;
    let out = args.take_value("--out")?;
    let runtime = args.take_value("--runtime")?;
    let a = args.positional(1)?;
    let p = Pyramid::open(Path::new(&a[0])).map_err(|e| e.to_string())?;
    let um = pixel_um(&p)?;
    let l = p.levels.get(level).ok_or(format!("--level {level}: the image has {} levels", p.levels.len()))?.clone();
    let staining = match (stains, nuclear) {
        (Some(_), Some(_)) => return Err("--stains (brightfield) or --nuclear (fluorescence), not both".into()),
        (_, Some(c)) => Staining::Fluorescence { nuclear: c },
        (Some(v), None) => Staining::Brightfield { stains: stains_named(&v)? },
        (None, None) if p.channels.len() == 3 => Staining::Brightfield { stains: StainVectors::h_dab() },
        (None, None) => return Err("say how nuclei show: --stains h-dab|h-e|FILE.json (brightfield) or --nuclear CHANNEL (fluorescence)".into()),
    };
    // The region in level-0 px: given, the drawn outlines' bounds, or the whole image
    let r0 = match (&region, &within) {
        (Some(r), _) => *r,
        (None, Some(outlines)) => {
            let b = outlines.iter().map(|o| lumen_hist::polygon::bounds(o)).fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |a, b| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)));
            [b.0.max(0.0) as usize, b.1.max(0.0) as usize, (b.2 - b.0.max(0.0)).ceil() as usize, (b.3 - b.1.max(0.0)).ceil() as usize]
        }
        (None, None) => [0, 0, p.width(), p.height()],
    };
    // Fluorescence without a threshold: Otsu's of the region at a coarse level, above its median
    // (the background the rolling ball takes away)
    if let (Staining::Fluorescence { nuclear }, None) = (&staining, threshold) {
        let cl = (0..p.levels.len()).find(|&k| p.levels[k].width.max(p.levels[k].height) <= 2048).unwrap_or(p.coarsest());
        let c = &p.levels[cl];
        let (x, y) = ((r0[0] as f64 / c.downsample) as usize, (r0[1] as f64 / c.downsample) as usize);
        let (w, h) = (((r0[2] as f64 / c.downsample).ceil() as usize).max(1).min(c.width.saturating_sub(x)), ((r0[3] as f64 / c.downsample).ceil() as usize).max(1).min(c.height.saturating_sub(y)));
        let v = p.read_region_at(cl, *nuclear, 0, x, y, w, h).map_err(|e| e.to_string())?;
        let (t, otsu, median) = lumen_hist::slide::auto_threshold(&v, s.background_radius_um > 0.0);
        s.threshold = t;
        eprintln!("threshold {:.1} (Otsu {otsu:.1}, background {median:.1}; --threshold to choose)", s.threshold);
    }
    let ds = l.downsample;
    let names: Vec<String> = p.channels.iter().map(|c| c.label.clone()).collect();
    let ranges: Vec<f32> = p.channels.iter().map(|c| c.range.1).collect();
    let target = match runtime {
        Some(r) => ComputeTarget::parse(&r).and_then(ComputeTarget::checked).map_err(|e| e.to_string())?,
        None => ComputeTarget::from_env().map_err(|e| e.to_string())?,
    };
    let outlines = within.clone().unwrap_or_else(|| vec![vec![[r0[0] as f64, r0[1] as f64], [(r0[0] + r0[2]) as f64, r0[1] as f64], [(r0[0] + r0[2]) as f64, (r0[1] + r0[3]) as f64], [r0[0] as f64, (r0[1] + r0[3]) as f64]]]);
    target
        .run(Job(|client: &Client| -> Result<(), String> {
            let mut all: Vec<HistObject> = Vec::new();
            for (k, outline) in outlines.iter().enumerate() {
                let b = lumen_hist::polygon::bounds(outline);
                let (x0, y0) = ((b.0.max(0.0) / ds) as usize, (b.1.max(0.0) / ds) as usize);
                let (x1, y1) = (((b.2 / ds).ceil() as usize).min(l.width), ((b.3 / ds).ceil() as usize).min(l.height));
                let region = SlideRegion { x0, y0, width: x1.saturating_sub(x0), height: y1.saturating_sub(y0), downsample: ds, pixel_um: um, within: within.as_ref().map(|_| outline.clone()) };
                let mut read = |x: usize, y: usize, w: usize, h: usize| -> lumen_hist::HistResult<Vec<Vec<f32>>> { (0..p.channels.len()).map(|c| p.read_region_at(level, c, 0, x, y, w, h).map_err(|e| lumen_hist::HistError::Invalid(e.to_string()))).collect() };
                let mut found = detect_cells_in_region(client, &mut read, &names, &ranges, &staining, &region, &s, Tiling { tile, overlap: 64 }, &mut |d, t| {
                    if d == t || d % 8 == 0 {
                        eprintln!("region {}: {d} / {t} tiles", k + 1);
                    }
                })
                .map_err(|e| e.to_string())?;
                for o in &mut found {
                    o.id += all.len() as u32;
                }
                all.extend(found);
            }
            if let Some(c) = &classifier {
                c.apply(&mut all);
            }
            if let Some(t) = &intensity {
                classify_intensity(&mut all, t);
            }
            println!("{} cells", all.len());
            for (k, outline) in outlines.iter().enumerate() {
                let m = region_summary(outline, &all, um);
                let line: Vec<String> = m.iter().map(|(k, v)| format!("{k}: {}", if v.fract() == 0.0 { format!("{v:.0}") } else { format!("{v:.2}") })).collect();
                println!("region {}: {}", k + 1, line.join(" · "));
            }
            if let Some(path) = &out {
                write_out(path, &all, &a[0], um)?;
            }
            Ok(())
        }))
        .map_err(|e| e.to_string())?
}

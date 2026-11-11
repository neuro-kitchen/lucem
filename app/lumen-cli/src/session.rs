//! `lumen session run`: an analysis done in the app (its `session.json`) run again without it,
//! image after image — on a server, over a folder of slides.
//!
//! ```text
//! lumen session run <session.json> <image>... [--out DIR] [--runtime NAME]
//! ```
//!
//! For each image: the session's last step of each kind runs through the same engine as the
//! app (tissue, cells, classes, intensity, superpixels, pixel classes, density); the result goes to the image's output store (inside its OME-Zarr, else
//! `DIR/<image>.lumen.zarr`): `labels/cells`, `labels/nuclei`, `lumen/cells.parquet`,
//! `lumen/tiles.parquet`, `lumen/shapes.parquet` (GeoParquet), `lumen/provenance.json`.

use std::path::{Path, PathBuf};

use lumen_core::compute::ComputeTask;
use lumen_core::cubecl::prelude::Client;
use lumen_core::ComputeTarget;
use lumen_hist::output::{self, RegionShape};
use lumen_hist::session::{self, Session};
use lumen_io::Pyramid;

use crate::Args;

pub const USAGE: &str = "\
lumen session run <session.json> <image>... [--out DIR] [--runtime NAME]
    Runs a session saved by the app (its steps, with their settings) on each image; writes each
    image's output store (inside its OME-Zarr when writable, else DIR/<image>.lumen.zarr).
";

struct Job<F>(F);

impl<T, F: FnOnce(&Client) -> T> ComputeTask for Job<F> {
    type Output = T;
    fn run(self, client: Client) -> T {
        (self.0)(&client)
    }
}

pub fn session(mut args: Args) -> Result<(), String> {
    let sub = if args.rest.is_empty() { String::new() } else { args.rest.remove(0) };
    match sub.as_str() {
        "run" => run(args),
        _ => Err(USAGE.to_string()),
    }
}

fn run(mut args: Args) -> Result<(), String> {
    let out_dir = args.take_value("--out")?.map(PathBuf::from);
    let runtime = args.take_value("--runtime")?;
    if args.rest.len() < 2 {
        return Err(USAGE.to_string());
    }
    let session_path = PathBuf::from(args.rest.remove(0));
    let s = Session::load(&session_path).map_err(|e| e.to_string())?;
    let recipe = s.recipe();
    if recipe.is_empty() {
        return Err(format!("{}: no step to run", session_path.display()));
    }
    println!("{} step(s): {}", recipe.len(), recipe.iter().map(|a| a.title()).collect::<Vec<_>>().join(" → "));
    let target = match runtime {
        Some(r) => ComputeTarget::parse(&r).and_then(ComputeTarget::checked).map_err(|e| e.to_string())?,
        None => ComputeTarget::from_env().map_err(|e| e.to_string())?,
    };
    let images: Vec<String> = std::mem::take(&mut args.rest);
    let mut failed = 0;
    for image in &images {
        let t0 = std::time::Instant::now();
        match run_one(&target, Path::new(image), &s, &recipe, out_dir.as_deref()) {
            Ok((store, cells, tiles, notes)) => {
                println!("{image}: {cells} cells, {tiles} tiles in {:.1} s → {}", t0.elapsed().as_secs_f64(), store.display());
                for n in notes {
                    println!("  {n}");
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("{image}: {e}");
            }
        }
    }
    if failed > 0 {
        return Err(format!("{failed} of {} image(s) failed", images.len()));
    }
    Ok(())
}

type RunOne = (PathBuf, usize, usize, Vec<String>);

fn run_one(target: &ComputeTarget, path: &Path, s: &Session, recipe: &[&session::Action], out_dir: Option<&Path>) -> Result<RunOne, String> {
    let p = Pyramid::open(path).map_err(|e| e.to_string())?;
    let um = p.pixel_size_um.ok_or("the image has no pixel size (µm)")?;
    let r = target
        .run(Job(|client: &Client| {
            session::run(client, &p, recipe, &mut |what| {
                eprint!("\r  {what}                    ");
            })
        }))
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    eprintln!();
    let name = path.file_name().map_or_else(|| "image".to_string(), |n| n.to_string_lossy().into_owned());
    let fallback = out_dir.map(Path::to_path_buf).or_else(|| path.parent().map(Path::to_path_buf)).unwrap_or_default();
    let store = output::store_for(path, &name, &fallback);
    if store != path {
        output::ensure_group(&store).map_err(|e| e.to_string())?;
    }
    // Each object's region: the tissue region holding it
    let region_of = |o: &lumen_hist::HistObject| r.tissue.iter().position(|t| lumen_hist::polygon::contains(t, o.centroid)).map(|i| format!("tissue{}", i + 1));
    let cell_regions: Vec<Option<String>> = r.cells.iter().map(region_of).collect();
    let tile_regions: Vec<Option<String>> = r.tiles.iter().map(region_of).collect();
    if !r.cells.is_empty() {
        output::write_objects(&store, "cells", &r.cells, &cell_regions, um).map_err(|e| e.to_string())?;
        output::write_labels(&store, "cells", &r.cells, false, (p.width(), p.height()), Some(um)).map_err(|e| e.to_string())?;
        output::write_labels(&store, "nuclei", &r.cells, true, (p.width(), p.height()), Some(um)).map_err(|e| e.to_string())?;
    }
    if !r.tiles.is_empty() {
        output::write_objects(&store, "tiles", &r.tiles, &tile_regions, um).map_err(|e| e.to_string())?;
    }
    let ids: Vec<String> = (1..=r.tissue.len()).map(|i| format!("tissue{i}")).collect();
    let regions: Vec<RegionShape> = r.tissue.iter().zip(&ids).map(|(o, id)| RegionShape { id, class: "Tissue", outline: o }).collect();
    output::write_shapes(&store, &[r.cells.clone(), r.tiles.clone()].concat(), &regions).map_err(|e| e.to_string())?;
    let provenance = serde_json::json!({
        "lumen": env!("CARGO_PKG_VERSION"),
        "image": path.display().to_string(),
        "run": "lumen session run",
        "session": s,
        "notes": r.notes,
    });
    output::write_provenance(&store, &provenance).map_err(|e| e.to_string())?;
    Ok((store, r.cells.len(), r.tiles.len(), r.notes))
}

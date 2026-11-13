//! `lucem session run`: an analysis done in the app (its `session.json`) run again without it,
//! image after image — on a server, over a folder of slides.
//!
//! ```text
//! lucem session run <session.json> <image>... [--out DIR] [--runtime NAME]
//! ```
//!
//! For each image: the session's last step of each kind runs through the same engine as the
//! app (tissue, cells, classes, intensity, superpixels, pixel classes, density); the result goes to the image's output store (inside its OME-Zarr, else
//! `DIR/<image>.lucem.zarr`): `labels/cells`, `labels/nuclei`, `lucem/cells.parquet`,
//! `lucem/tiles.parquet`, `lucem/shapes.parquet` (GeoParquet), `lucem/provenance.json`.

use std::path::{Path, PathBuf};

use lucem_core::ComputeTarget;
use lucem_hist::session::{self, Session};

use crate::Args;

pub const USAGE: &str = "\
lucem session run <session.json> <image>... [--out DIR] [--runtime NAME]
    Runs a session saved by the app (its steps, with their settings) on each image; writes each
    image's output store (inside its OME-Zarr when writable, else DIR/<image>.lucem.zarr).
";

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

fn run_one(target: &ComputeTarget, path: &Path, s: &Session, _recipe: &[&session::Action], out_dir: Option<&Path>) -> Result<RunOne, String> {
    let r = lucem_mcp::work::run_session_on(target, path, s, "lucem session run", out_dir, &mut |what| eprint!("\r  {what}                    "));
    eprintln!();
    r
}

//! `lucem autotune`: tunes this machine once, for everything — lucem's operators and every model
//! plugin — on every usable runtime (or `--runtime`'s), as dsp-kitchen's `dspk autotune`. The
//! choices and compiled kernels are saved in lucem's cache folder, so later runs start at speed.

use lucem_core::ComputeTarget;

use crate::Args;

pub const USAGE: &str = "\
lucem autotune [--runtime NAME] [--sides 256,1024,2048] [--operators-only | --models-only]
    Tunes this device once for lucem's operators (every operator over a grid of image sizes)
    and every model plugin found (each installed model on every tile shape), on every usable
    runtime or --runtime's. Results are kept in lucem's cache folder for later runs.
";

pub fn autotune(mut args: Args) -> Result<(), String> {
    let runtime = args.take_value("--runtime")?;
    let sides: Vec<usize> = match args.take_value("--sides")? {
        Some(v) => v.split(',').map(|x| x.trim().parse::<usize>().map_err(|e| format!("--sides {x:?}: {e}"))).collect::<Result<_, _>>()?,
        None => lucem_pipeline::autotune::DEFAULT_SIDES.to_vec(),
    };
    let operators = !args.take_flag("--models-only");
    let models = !args.take_flag("--operators-only");
    args.positional(0).map_err(|e| format!("{e}\n\nusage:\n{USAGE}"))?;
    if !operators && !models {
        return Err(format!("--operators-only and --models-only exclude each other\n\nusage:\n{USAGE}"));
    }
    let targets: Vec<ComputeTarget> = match runtime.or_else(|| std::env::var(lucem_core::compute::RUNTIME_ENV).ok()).filter(|r| !r.trim().is_empty()) {
        Some(name) => vec![ComputeTarget::parse(&name).and_then(ComputeTarget::checked).map_err(|e| e.to_string())?],
        None => ComputeTarget::ALL.into_iter().filter(|t| t.is_available() && t.unusable_reason().is_none()).collect(),
    };
    let mut failures = 0;
    for target in targets {
        let name = target.name();
        if operators {
            // A runtime without its driver panics when opened
            let Ok(Ok(client)) = std::panic::catch_unwind(|| target.client()) else {
                println!("skip {name}: no device here");
                continue;
            };
            let s = lucem_pipeline::autotune::autotune(&client, &sides, |done, total, tag| eprint!("\r{name}: operators {done}/{total} ({tag})                    "));
            eprintln!();
            println!("ok {name}: {} operator runs in {:.0} s", s.runs - s.failed.len(), s.seconds);
            for (tag, side, why) in &s.failed {
                println!("  failed {tag} at {side} × {side}: {why}");
            }
            failures += s.failed.len();
        }
        if models {
            if target == ComputeTarget::Hip {
                println!("skip {name}: the model plugins run on CUDA, wgpu or the CPU");
                continue;
            }
            let found = lucem_ml::tune_plugins(Some(name), |r| match &r.outcome {
                lucem_ml::TuneOutcome::Tuned => println!("ok {name}: {} tuned in {:.0} s", r.name, r.seconds),
                lucem_ml::TuneOutcome::Skipped => println!("skip {name}: {} has nothing to tune", r.name),
                lucem_ml::TuneOutcome::Failed(why) => println!("failed {name}: {}: {why}", r.name),
            });
            if found.is_empty() {
                println!("{name}: no model plugins found (pip install \"lucem[ml]\", or see lucem.ml.plugin_dirs())");
            }
            failures += found.iter().filter(|r| matches!(r.outcome, lucem_ml::TuneOutcome::Failed(_))).count();
        }
    }
    if failures > 0 {
        return Err(format!("{failures} part(s) could not be tuned (above); the rest was"));
    }
    Ok(())
}

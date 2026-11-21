//! `lucem plugins` — list discovered ML model plugins and install prebuilt `lucem-ml-*`
//! binaries from GitHub Releases (verified by SHA-256 via `lucem-hub`).

use crate::Args;

const USAGE: &str = "lucem plugins — manage ML model plugin binaries

USAGE
  lucem plugins [list]
  lucem plugins install <stardist|instanseg|sam|all> [--version V]
";

pub fn run(args: Vec<String>) -> Result<(), String> {
    let mut a = Args { rest: args };
    if a.take_flag("-h") || a.take_flag("--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let sub = if a.rest.is_empty() { "list".to_string() } else { a.rest.remove(0) };
    match sub.as_str() {
        "list" => list(),
        "install" => install(a),
        other => Err(format!("unknown plugins command {other:?}\n\n{USAGE}")),
    }
}

fn list() -> Result<(), String> {
    let found = lucem_ml::discover();
    if found.is_empty() {
        println!("No ML model plugins found.");
        println!("Install them with: lucem plugins install all");
        return Ok(());
    }
    for entry in found {
        match entry {
            Ok(p) => {
                let models: Vec<String> = p
                    .models
                    .iter()
                    .map(|m| format!("{} ({})", m.id, if m.installed { "installed" } else { "weights not downloaded" }))
                    .collect();
                println!("{}  [{}]", p.name, p.program);
                for m in models {
                    println!("  - {m}");
                }
            }
            Err(e) => eprintln!("warning: {e}"),
        }
    }
    Ok(())
}

fn install(mut a: Args) -> Result<(), String> {
    let version = a.take_value("--version")?.unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
    if a.rest.is_empty() {
        return Err(format!("specify a plugin name (stardist, instanseg, sam, or all)\n\n{USAGE}"));
    }
    let mut targets: Vec<String> = Vec::new();
    for arg in a.rest {
        if arg == "all" {
            targets.extend(lucem_hub::plugins::KNOWN_PLUGINS.iter().map(|s| (*s).to_string()));
        } else {
            targets.push(arg);
        }
    }
    targets.sort();
    targets.dedup();

    for name in targets {
        let prog = lucem_hub::plugins::program_name(&name);
        eprintln!("Installing {prog} v{version}…");
        let mut last_pct = 0u64;
        let path = lucem_hub::plugins::install(&prog, &version, &mut |done, total| {
            if total > 0 {
                let pct = done.saturating_mul(100) / total;
                if pct >= last_pct + 20 || done == total {
                    last_pct = pct;
                    eprint!("\r  {prog}: {pct}% ({:.1} / {:.1} MiB)", done as f64 / 1_048_576.0, total as f64 / 1_048_576.0);
                }
            }
        })?;
        eprintln!("\r  {prog} installed → {}", path.display());
    }
    Ok(())
}

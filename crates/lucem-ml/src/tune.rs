//! Tuning every model plugin on this machine: each plugin program found ([`crate::discover`]) runs
//! `<program> tune`, which puts its installed models through every tile shape they meet, so their
//! autotuned kernels and compiled code are cached before a real image needs them. `lucem
//! autotune` runs this after lucem's own operators.
//!
//! A plugin that has nothing to tune (SAM runs on ONNX Runtime, which tunes nothing ahead) answers
//! `tune` with its usage and exit code 2: it is reported as skipped, not failed.

use std::process::{Command, Stdio};
use std::time::Instant;

/// How one plugin's tuning went.
#[derive(Debug, Clone, PartialEq)]
pub struct PluginTune {
    /// The program's name (`lucem-ml-instanseg`).
    pub name: String,
    pub outcome: TuneOutcome,
    pub seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TuneOutcome {
    Tuned,
    /// The plugin has nothing to tune.
    Skipped,
    Failed(String),
}

/// Runs `tune` on every plugin found, on `runtime` (`LUCEM_RUNTIME` for the plugin; `None`: its
/// default). The plugins' own messages go to this process's standard error; `done` is called after
/// each plugin.
pub fn tune_plugins(runtime: Option<&str>, mut done: impl FnMut(&PluginTune)) -> Vec<PluginTune> {
    let mut out = Vec::new();
    for (name, path) in crate::discover::programs() {
        let t0 = Instant::now();
        let mut cmd = Command::new(&path);
        cmd.arg("tune").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::inherit());
        if let Some(r) = runtime {
            cmd.env("LUCEM_RUNTIME", r);
        }
        let outcome = match cmd.status() {
            Ok(s) if s.success() => TuneOutcome::Tuned,
            Ok(s) if s.code() == Some(2) => TuneOutcome::Skipped,
            Ok(s) => TuneOutcome::Failed(format!("{} ended with {s}", path.display())),
            Err(e) => TuneOutcome::Failed(format!("{}: {e}", path.display())),
        };
        let r = PluginTune { name, outcome, seconds: t0.elapsed().as_secs_f64() };
        done(&r);
        out.push(r);
    }
    out
}

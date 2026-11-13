//! Where operators run: one client per process, for the runtime chosen (`set_runtime`), else
//! `LUCEM_RUNTIME`, else the first usable one.

use std::sync::{Mutex, OnceLock};

use lucem_core::cubecl::prelude::Client;
use lucem_core::ComputeTarget;
use pyo3::prelude::*;

use crate::{py_err, value_err};

static RUNTIME: OnceLock<Mutex<Option<(ComputeTarget, Client)>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<(ComputeTarget, Client)>> {
    RUNTIME.get_or_init(|| Mutex::new(None))
}

/// The client of the chosen runtime, opened on first use.
pub fn client() -> PyResult<Client> {
    let mut guard = slot().lock().map_err(py_err)?;
    if guard.is_none() {
        let target = ComputeTarget::from_env().map_err(py_err)?;
        *guard = Some((target, target.client().map_err(py_err)?));
    }
    Ok(guard.as_ref().expect("just set").1.clone())
}

/// The runtimes compiled in and usable here, in preference order.
#[pyfunction]
fn runtimes() -> Vec<String> {
    ComputeTarget::available().into_iter().map(|t| t.name().to_string()).collect()
}

/// Selects the runtime later operations use (`wgpu`, `cuda`, `cpu`, `hip`).
#[pyfunction]
fn set_runtime(name: &str) -> PyResult<()> {
    let target = ComputeTarget::parse(name).and_then(ComputeTarget::checked).map_err(value_err)?;
    let client = target.client().map_err(py_err)?;
    *slot().lock().map_err(py_err)? = Some((target, client));
    Ok(())
}

/// The runtime in use (opened now if none was chosen yet).
#[pyfunction]
fn current_runtime() -> PyResult<String> {
    client()?;
    let guard = slot().lock().map_err(py_err)?;
    Ok(guard.as_ref().expect("set above").0.name().to_string())
}

/// Tunes this device once, as `lucem autotune`: lucem's operators over a grid of image sides, then
/// every model plugin found; on `runtime`, else every usable runtime. Returns, per runtime, what
/// was done: `{"operator_runs", "operator_failures", "seconds", "plugins": {name: "tuned" | "skipped" | why}}`.
#[pyfunction]
#[pyo3(signature = (runtime=None, sides=None, operators=true, models=true))]
fn autotune(py: Python<'_>, runtime: Option<&str>, sides: Option<Vec<usize>>, operators: bool, models: bool) -> PyResult<String> {
    let targets: Vec<ComputeTarget> = match runtime {
        Some(name) => vec![ComputeTarget::parse(name).and_then(ComputeTarget::checked).map_err(value_err)?],
        None => ComputeTarget::ALL.into_iter().filter(|t| t.is_available() && t.unusable_reason().is_none()).collect(),
    };
    let sides = sides.unwrap_or_else(|| lucem_pipeline::autotune::DEFAULT_SIDES.to_vec());
    let mut report = serde_json::Map::new();
    for target in targets {
        let name = target.name();
        let entry = py.detach(|| {
            let mut e = serde_json::json!({});
            if operators {
                let opened = std::panic::catch_unwind(|| target.client());
                let Ok(Ok(client)) = opened else { return serde_json::json!({"error": "no device here"}) };
                let s = lucem_pipeline::autotune::autotune(&client, &sides, |_, _, _| {});
                e["operator_runs"] = (s.runs - s.failed.len()).into();
                e["operator_failures"] = s.failed.iter().map(|(t, side, why)| serde_json::json!({"operator": t, "side": side, "why": why})).collect::<Vec<_>>().into();
                e["seconds"] = s.seconds.into();
            }
            if models && target != ComputeTarget::Hip {
                let plugins: serde_json::Map<String, serde_json::Value> = lucem_ml::tune_plugins(Some(name), |_| {})
                    .into_iter()
                    .map(|r| {
                        let v = match r.outcome {
                            lucem_ml::TuneOutcome::Tuned => "tuned".to_string(),
                            lucem_ml::TuneOutcome::Skipped => "skipped".to_string(),
                            lucem_ml::TuneOutcome::Failed(why) => why,
                        };
                        (r.name, v.into())
                    })
                    .collect();
                e["plugins"] = plugins.into();
            }
            e
        });
        report.insert(name.to_string(), entry);
    }
    serde_json::to_string(&report).map_err(py_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(autotune, m)?)?;
    m.add_function(wrap_pyfunction!(runtimes, m)?)?;
    m.add_function(wrap_pyfunction!(set_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(current_runtime, m)?)?;
    Ok(())
}

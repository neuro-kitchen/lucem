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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(runtimes, m)?)?;
    m.add_function(wrap_pyfunction!(set_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(current_runtime, m)?)?;
    Ok(())
}

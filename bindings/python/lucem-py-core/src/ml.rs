//! The ML plugin client (`lucem-ml`): finding the plugin programs (`lucem-ml-<family>`, from
//! `lucem[stardist]`, `lucem[sam]`, … or installed by hand) and talking to one in server mode.
//! Requests and answers are the protocol's JSON.

use std::sync::Mutex;

use lucem_ml::{PluginProcess, Request};
use pyo3::prelude::*;

use crate::{py_err, value_err};

/// Every plugin found, as JSON: `[{"ok": <manifest>} | {"error": "<why>"}]`.
#[pyfunction]
fn ml_plugins(py: Python<'_>) -> PyResult<String> {
    let found = py.detach(lucem_ml::discover);
    let list: Vec<serde_json::Value> = found.into_iter().map(|r| match r {
        Ok(info) => serde_json::json!({ "ok": info }),
        Err(e) => serde_json::json!({ "error": e }),
    }).collect();
    serde_json::to_string(&list).map_err(py_err)
}

/// The folders searched for plugin programs, in order.
#[pyfunction]
fn ml_plugin_dirs() -> Vec<std::path::PathBuf> {
    lucem_ml::plugin_dirs()
}

/// A plugin program running in server mode, so its model stays loaded between requests.
#[pyclass(module = "lucem._native")]
struct Plugin {
    inner: Mutex<PluginProcess>,
    program: String,
}

#[pymethods]
impl Plugin {
    /// Starts `program serve` (a path, or a name on `PATH`).
    #[new]
    fn new(program: &str) -> PyResult<Self> {
        Ok(Self { inner: Mutex::new(PluginProcess::start(program).map_err(py_err)?), program: program.to_string() })
    }

    #[getter]
    fn program(&self) -> String {
        self.program.clone()
    }

    /// Whether the program is still running.
    fn alive(&self) -> PyResult<bool> {
        Ok(self.inner.lock().map_err(py_err)?.alive())
    }

    /// Sends a request (the protocol's JSON) and returns its answer as JSON.
    fn request(&self, py: Python<'_>, request: &str) -> PyResult<String> {
        let req: Request = serde_json::from_str(request).map_err(value_err)?;
        let answer = py.detach(|| -> Result<serde_json::Value, String> {
            let mut p = self.inner.lock().map_err(|e| e.to_string())?;
            p.request(&req, &mut |_, _| {})
        });
        serde_json::to_string(&answer.map_err(py_err)?).map_err(py_err)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ml_plugins, m)?)?;
    m.add_function(wrap_pyfunction!(ml_plugin_dirs, m)?)?;
    m.add_class::<Plugin>()?;
    Ok(())
}

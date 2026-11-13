//! The operator catalog (`lucem-pipeline`) on NumPy arrays: one operator, or a chain that stays
//! on the device between its operators (one upload, every operator, one download).

use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::DeviceArray;
use lucem_core::Shape;
use lucem_pipeline::{catalog, Op, PortType, Table, Value};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{array_of, client, float_input, py_err, uint_input, value_err};

/// Every operator as JSON: its tag, title, family, parameters with defaults, inputs and output.
#[pyfunction]
fn op_catalog() -> PyResult<String> {
    let list: Vec<serde_json::Value> = catalog()
        .into_iter()
        .map(|op| {
            let mut params = serde_json::to_value(&op).unwrap_or_default();
            if let Some(o) = params.as_object_mut() {
                o.remove("op");
            }
            serde_json::json!({
                "tag": op.tag(), "title": op.title(), "family": op.family(), "params": params, "halo": op.halo(),
                "inputs": op.inputs().iter().map(|(t, n)| serde_json::json!({"type": t.name(), "name": n})).collect::<Vec<_>>(),
                "output": op.output().name(),
            })
        })
        .collect();
    serde_json::to_string(&list).map_err(py_err)
}

fn parse_op(spec: &str) -> PyResult<Op> {
    let op: Op = serde_json::from_str(spec).map_err(value_err)?;
    op.validate().map_err(value_err)?;
    Ok(op)
}

/// Pixels of context an operator (its JSON) reads around a pixel.
#[pyfunction]
fn op_halo(spec: &str) -> PyResult<usize> {
    Ok(parse_op(spec)?.halo())
}

/// An array on the host, typed for a port.
enum HostInput {
    Image(Vec<usize>, Vec<f32>),
    Other(PortType, Vec<usize>, Vec<u32>),
}

fn host_input(port: PortType, a: &Bound<'_, PyAny>) -> PyResult<HostInput> {
    Ok(match port {
        PortType::Image => {
            let (d, v) = float_input(a)?;
            HostInput::Image(d, v)
        }
        PortType::Mask | PortType::Labels => {
            let (d, v) = uint_input(a)?;
            HostInput::Other(port, d, v)
        }
        PortType::Table => return Err(value_err("a table cannot be an input")),
    })
}

fn upload(client: &Client, input: HostInput) -> Result<Value, String> {
    let e = |e: lucem_core::CoreError| e.to_string();
    Ok(match input {
        HostInput::Image(d, v) => Value::Image(DeviceArray::from_host(client, Shape::new(d), v).map_err(e)?),
        HostInput::Other(PortType::Mask, d, v) => Value::Mask(DeviceArray::from_host(client, Shape::new(d), v).map_err(e)?),
        HostInput::Other(_, d, v) => Value::Labels(DeviceArray::from_host(client, Shape::new(d), v).map_err(e)?),
    })
}

/// A value downloaded to the host.
enum HostValue {
    Float(Vec<usize>, Vec<f32>),
    Uint(Vec<usize>, Vec<u32>),
    Table(Table),
}

fn download(client: &Client, v: Value) -> HostValue {
    match v {
        Value::Image(a) => HostValue::Float(a.dims().to_vec(), a.to_host(client)),
        Value::Mask(a) | Value::Labels(a) => HostValue::Uint(a.dims().to_vec(), a.to_host(client)),
        Value::Table(rows) => {
            let mut t = Table::new();
            for r in rows {
                let mut row: Vec<(String, f64)> = vec![
                    ("label".into(), r.label as f64),
                    ("area".into(), r.area as f64),
                    ("eccentricity".into(), r.eccentricity),
                    ("perimeter".into(), r.perimeter),
                    ("mean_intensity".into(), r.mean_intensity),
                    ("min_intensity".into(), r.min_intensity),
                    ("max_intensity".into(), r.max_intensity),
                ];
                for (axis, c) in ["z", "y", "x"][3 - r.centroid.len().min(3)..].iter().zip(&r.centroid) {
                    row.push((format!("centroid_{axis}"), *c));
                }
                t.push_row(&row);
            }
            HostValue::Table(t)
        }
    }
}

/// A table as a dict of float64 columns.
pub fn table_dict<'py>(py: Python<'py>, table: &Table) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, values) in &table.columns {
        d.set_item(name, array_of(py, &[values.len()], values.clone())?)?;
    }
    Ok(d)
}

fn to_python(py: Python<'_>, v: HostValue) -> PyResult<Py<PyAny>> {
    Ok(match v {
        HostValue::Float(d, data) => array_of(py, &d, data)?.into_any().unbind(),
        HostValue::Uint(d, data) => array_of(py, &d, data)?.into_any().unbind(),
        HostValue::Table(t) => table_dict(py, &t)?.into_any().unbind(),
    })
}

/// Runs one operator (`spec`: its JSON) on arrays: float32 for images, uint32 for masks and labels.
#[pyfunction]
fn run_op(py: Python<'_>, spec: &str, inputs: Vec<Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let op = parse_op(spec)?;
    let ports = op.inputs();
    if ports.len() != inputs.len() {
        return Err(value_err(format!("{} takes {} input(s), {} given", op.title(), ports.len(), inputs.len())));
    }
    let hosts = ports.iter().zip(&inputs).map(|((p, _), a)| host_input(*p, a)).collect::<PyResult<Vec<_>>>()?;
    let client = client()?;
    let out = py
        .detach(move || -> Result<HostValue, String> {
            let values = hosts.into_iter().map(|h| upload(&client, h)).collect::<Result<Vec<_>, _>>()?;
            Ok(download(&client, op.run(&client, &values)?))
        })
        .map_err(py_err)?;
    to_python(py, out)
}

/// Runs a chain of one-input operators (`specs`: their JSON) on one array, staying on the device
/// between them: one upload, every operator, one download.
#[pyfunction]
fn run_chain(py: Python<'_>, specs: Vec<String>, input: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let ops = specs.iter().map(|s| parse_op(s)).collect::<PyResult<Vec<_>>>()?;
    let first = ops.first().ok_or_else(|| value_err("a chain needs at least one operator"))?;
    if let Some(bad) = ops.iter().find(|o| o.inputs().len() != 1) {
        return Err(value_err(format!("{} takes {} inputs: it cannot be a chain stage", bad.title(), bad.inputs().len())));
    }
    for w in ops.windows(2) {
        if w[0].output() != w[1].inputs()[0].0 {
            return Err(value_err(format!("{} gives {}, {} takes {}", w[0].title(), w[0].output().name(), w[1].title(), w[1].inputs()[0].0.name())));
        }
    }
    let host = host_input(first.inputs()[0].0, &input)?;
    let client = client()?;
    let out = py
        .detach(move || -> Result<HostValue, String> {
            let mut value = upload(&client, host)?;
            for op in &ops {
                value = op.run(&client, std::slice::from_ref(&value))?;
            }
            Ok(download(&client, value))
        })
        .map_err(py_err)?;
    to_python(py, out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(op_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(op_halo, m)?)?;
    m.add_function(wrap_pyfunction!(run_op, m)?)?;
    m.add_function(wrap_pyfunction!(run_chain, m)?)?;
    Ok(())
}

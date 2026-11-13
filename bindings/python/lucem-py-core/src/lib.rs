//! Python bindings of lucem's core, registered into the one `lucem._native` extension
//! (`bindings/python/lucem-py`): the runtime, images and OME-Zarr stores, the operator catalog, the
//! pieces `lucem.custom` walks images with, and the ML plugin client.
//!
//! Other binding crates (`lucem-py-hist`) build on [`client`] and [`io::Image`]: they are in the
//! same extension, so there is one device per runtime and one image type.

pub mod custom;
pub mod io;
pub mod ml;
pub mod ops;
pub mod runtime;

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub use runtime::client;

/// A failure inside lucem, as Python's `RuntimeError`.
pub fn py_err(e: impl ToString) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// A bad argument, as Python's `ValueError`.
pub fn value_err(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `[z, y, x]` from a list of three.
pub fn zyx(v: Vec<usize>, what: &str) -> PyResult<[usize; 3]> {
    <[usize; 3]>::try_from(v).map_err(|_| value_err(format!("{what} must have three values (z, y, x)")))
}

/// `data` as a NumPy array of shape `dims`.
pub fn array_of<'py, T: numpy::Element>(py: Python<'py>, dims: &[usize], data: Vec<T>) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    Ok(ArrayD::from_shape_vec(IxDyn(dims), data).map_err(py_err)?.into_pyarray(py))
}

/// A float32 array's shape and values (row-major).
pub fn float_input(a: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<f32>)> {
    let arr: PyReadonlyArrayDyn<'_, f32> = a.extract().map_err(|_| value_err("expected a float32 array"))?;
    Ok((arr.shape().to_vec(), arr.as_array().iter().copied().collect()))
}

/// A uint32 array's shape and values (row-major).
pub fn uint_input(a: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<u32>)> {
    let arr: PyReadonlyArrayDyn<'_, u32> = a.extract().map_err(|_| value_err("expected a uint32 array"))?;
    Ok((arr.shape().to_vec(), arr.as_array().iter().copied().collect()))
}

/// Adds the core's classes and functions to `m` (`lucem._native`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    runtime::register(m)?;
    io::register(m)?;
    ops::register(m)?;
    custom::register(m)?;
    ml::register(m)?;
    Ok(())
}

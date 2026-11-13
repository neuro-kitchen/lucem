//! `lucem._native`: the one extension module behind the `lucem` Python package. Binding crates add
//! their parts here; one extension keeps one CubeCL device per runtime and one image type. It is
//! built twice, without histology for the `lucem` wheel and with it for the `lucem-hist` wheel;
//! only one is ever loaded (`lucem/__init__.py`).

use pyo3::prelude::*;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    lucem_py_core::register(m)?;
    #[cfg(feature = "hist")]
    lucem_py_hist::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

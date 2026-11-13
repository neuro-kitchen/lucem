//! The pieces `lucem.custom` walks an image of any size with: the block schedule, and the
//! stitcher that joins objects cut by block borders.

use lucem_core::tiling::schedule;
use lucem_core::Block;
use lucem_pipeline::LabelStitcher;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::io::size_of;
use crate::ops::table_dict;
use crate::{array_of, float_input, uint_input, value_err, zyx};

type Scheduled = (usize, [usize; 3], [usize; 3], [usize; 3], [usize; 3]);

/// The blocks of a walk: `(index, origin, size, read_origin, read_size)`.
#[pyfunction]
fn schedule_blocks(dims: Vec<usize>, block: Vec<usize>, halo: Vec<usize>) -> PyResult<Vec<Scheduled>> {
    Ok(schedule(zyx(dims, "dims")?, zyx(block, "block")?, zyx(halo, "halo")?).into_iter().map(|b| (b.index, b.origin, b.size, b.read_origin, b.read_size)).collect())
}

/// Joins objects cut by block borders.
#[pyclass(module = "lucem._native")]
struct Stitcher {
    inner: Option<LabelStitcher>,
}

#[pymethods]
impl Stitcher {
    #[new]
    #[pyo3(signature = (grid, link_z=false))]
    fn new(grid: Vec<usize>, link_z: bool) -> PyResult<Self> {
        Ok(Self { inner: Some(LabelStitcher::new(zyx(grid, "grid")?, link_z)) })
    }

    /// Adds a block's owned labels (and optionally its intensity) and returns them with ids unique
    /// across blocks.
    #[pyo3(signature = (origin, labels, intensity=None))]
    fn add<'py>(&mut self, py: Python<'py>, origin: Vec<usize>, labels: Bound<'py, PyAny>, intensity: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyArrayDyn<u32>>> {
        let s = self.inner.as_mut().ok_or_else(|| value_err("the stitcher is finished"))?;
        let (d, l) = uint_input(&labels)?;
        let size = size_of(&d)?;
        let origin = zyx(origin, "origin")?;
        let values = intensity.map(|a| float_input(&a)).transpose()?;
        if values.as_ref().is_some_and(|(id, _)| *id != d) {
            return Err(value_err("labels and intensity differ in shape"));
        }
        let block = Block { index: 0, origin, size, read_origin: origin, read_size: size };
        let ids = s.add(&block, &l, values.as_ref().map(|(_, v)| v.as_slice()));
        array_of(py, &d, ids)
    }

    /// `(map, table)`: `map[id]` is each object's final label; one table row per object.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyArrayDyn<u32>>, Bound<'py, PyDict>)> {
        let s = self.inner.take().ok_or_else(|| value_err("the stitcher is finished"))?;
        let done = s.finish();
        let n = done.map.len();
        Ok((array_of(py, &[n], done.map)?, table_dict(py, &done.table)?))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(schedule_blocks, m)?)?;
    m.add_class::<Stitcher>()?;
    Ok(())
}

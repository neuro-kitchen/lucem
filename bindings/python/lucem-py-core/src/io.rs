//! Images opened for block reads (OME-Zarr, TIFF, image files; 2D or 3D) and OME-Zarr stores
//! written block by block.

use std::path::PathBuf;
use std::sync::Arc;

use lucem_io::{PlaneAxis, Pyramid, WriterOptions, ZarrWriter};
use numpy::PyArrayDyn;
use pyo3::prelude::*;

use crate::{array_of, float_input, py_err, uint_input, value_err, zyx};

/// An image opened for block reads, 2D or 3D. Nothing is decoded until it is read.
#[pyclass(frozen, module = "lucem._native")]
pub struct Image {
    pub inner: Arc<Pyramid>,
}

impl Image {
    /// The opened image, for other binding crates.
    pub fn pyramid(&self) -> &Arc<Pyramid> {
        &self.inner
    }
}

#[pymethods]
impl Image {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self { inner: Arc::new(Pyramid::open(&path).map_err(py_err)?) })
    }

    /// A synthetic two-channel volume.
    #[staticmethod]
    #[pyo3(signature = (size=128, depth=64))]
    fn sample_volume(size: usize, depth: usize) -> PyResult<Self> {
        Ok(Self { inner: Arc::new(lucem_io::sample_volume(size, depth).map_err(py_err)?) })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    /// The file or container (`None` for generated images).
    #[getter]
    fn path(&self) -> Option<PathBuf> {
        self.inner.path.clone()
    }
    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype.clone()
    }
    #[getter]
    fn channels(&self) -> Vec<String> {
        self.inner.channels.iter().map(|c| c.label.clone()).collect()
    }
    /// `(min, max)` of the values each channel's type can hold.
    #[getter]
    fn channel_ranges(&self) -> Vec<(f32, f32)> {
        self.inner.channels.iter().map(|c| c.range).collect()
    }
    #[getter]
    fn pixel_size_um(&self) -> Option<f64> {
        self.inner.pixel_size_um
    }
    #[getter]
    fn z_spacing_um(&self) -> Option<f64> {
        self.inner.z_spacing_um
    }
    /// Each level, finest first: `(depth, height, width, tile_d, tile_h, tile_w, downsample)`.
    #[getter]
    fn levels(&self) -> Vec<(usize, usize, usize, usize, usize, usize, f64)> {
        self.inner.levels.iter().map(|l| (l.depth, l.height, l.width, l.tile_d, l.tile_h, l.tile_w, l.downsample)).collect()
    }

    /// `size` values from `origin` (`z, y, x`) of `channel`, as float32 `[d, h, w]`.
    fn read_block<'py>(&self, py: Python<'py>, level: usize, channel: usize, origin: Vec<usize>, size: Vec<usize>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let (origin, size) = (zyx(origin, "origin")?, zyx(size, "size")?);
        let inner = self.inner.clone();
        let data = py.detach(move || inner.read_block(level, channel, origin, size)).map_err(py_err)?;
        array_of(py, &size, data)
    }

    /// A plane across `axis` (`"z"`, `"y"` or `"x"`) at `index`, as float32 `[h, w]`.
    fn read_plane<'py>(&self, py: Python<'py>, level: usize, channel: usize, axis: &str, index: usize) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let axis = match axis {
            "z" => PlaneAxis::Z,
            "y" => PlaneAxis::Y,
            "x" => PlaneAxis::X,
            other => return Err(value_err(format!("axis {other:?}: expected z, y or x"))),
        };
        let inner = self.inner.clone();
        let p = py.detach(move || inner.read_plane(level, channel, axis, index)).map_err(py_err)?;
        array_of(py, &[p.height, p.width], p.data)
    }

    /// A region `(x, y, width, height)` of plane `z` of `channel` at `level` (that level's pixels,
    /// clipped to it), as float32 `[h, w]`.
    #[pyo3(signature = (level, channel, x, y, width, height, z=0))]
    #[allow(clippy::too_many_arguments)]
    fn read_region<'py>(&self, py: Python<'py>, level: usize, channel: usize, x: usize, y: usize, width: usize, height: usize, z: usize) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let inner = self.inner.clone();
        let data = py.detach(move || inner.read_region_at(level, channel, z, x, y, width, height)).map_err(py_err)?;
        array_of(py, &[height, width], data)
    }

    fn __repr__(&self) -> String {
        let l = &self.inner.levels[0];
        format!("Image({:?}, {} channel(s), {} × {} × {}, {} level(s))", self.inner.name, self.inner.channels.len(), l.depth, l.height, l.width, self.inner.levels.len())
    }
}

/// An OME-Zarr store written block by block.
#[pyclass(module = "lucem._native")]
struct ZarrStore {
    inner: Option<ZarrWriter>,
    float: bool,
}

#[pymethods]
impl ZarrStore {
    #[new]
    #[pyo3(signature = (path, channels, shape, chunks, dtype="float32", name="image", pixel_size_um=None, z_spacing_um=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(path: PathBuf, channels: Vec<String>, shape: Vec<usize>, chunks: Vec<usize>, dtype: &str, name: &str, pixel_size_um: Option<f64>, z_spacing_um: Option<f64>) -> PyResult<Self> {
        let options = WriterOptions { name: name.into(), channels, shape: zyx(shape, "shape")?, chunks: zyx(chunks, "chunks")?, pixel_size_um, z_spacing_um };
        let (inner, float) = match dtype {
            "float32" => (ZarrWriter::create::<f32>(&path, options).map_err(py_err)?, true),
            "uint32" => (ZarrWriter::create::<u32>(&path, options).map_err(py_err)?, false),
            other => return Err(value_err(format!("dtype {other:?}: expected float32 or uint32"))),
        };
        Ok(Self { inner: Some(inner), float })
    }

    /// Stores a whole chunk at `origin` of `channel`.
    fn write(&self, channel: usize, origin: Vec<usize>, data: Bound<'_, PyAny>) -> PyResult<()> {
        let w = self.inner.as_ref().ok_or_else(|| value_err("the store is finished"))?;
        let origin = zyx(origin, "origin")?;
        if self.float {
            let (d, v) = float_input(&data)?;
            w.write(channel, origin, size_of(&d)?, &v).map_err(py_err)
        } else {
            let (d, v) = uint_input(&data)?;
            w.write(channel, origin, size_of(&d)?, &v).map_err(py_err)
        }
    }

    /// Replaces each label `l` by `map[l]`.
    fn remap_labels(&self, map: Bound<'_, PyAny>) -> PyResult<()> {
        let w = self.inner.as_ref().ok_or_else(|| value_err("the store is finished"))?;
        let (_, map) = uint_input(&map)?;
        w.remap_labels(&map).map_err(py_err)
    }

    /// Builds the coarser levels and writes the metadata; returns the path.
    fn finish(&mut self, py: Python<'_>) -> PyResult<PathBuf> {
        let w = self.inner.take().ok_or_else(|| value_err("the store is finished"))?;
        let float = self.float;
        py.detach(move || if float { w.finish::<f32>() } else { w.finish::<u32>() }).map_err(py_err)
    }
}

/// `[z, y, x]` of an array of two or three dimensions.
pub fn size_of(dims: &[usize]) -> PyResult<[usize; 3]> {
    match *dims {
        [h, w] => Ok([1, h, w]),
        [d, h, w] => Ok([d, h, w]),
        _ => Err(value_err(format!("expected a 2D or 3D array, got shape {dims:?}"))),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Image>()?;
    m.add_class::<ZarrStore>()?;
    Ok(())
}

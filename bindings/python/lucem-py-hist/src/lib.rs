//! Python bindings of `lucem-hist`, registered into the `lucem` extension as `lucem._native.hist`
//! (the `lucem-hist` wheel is that extension built with them; `lucem.hist` is their Python face).
//!
//! Settings, stains, steps and objects cross as JSON: the same serde forms the app's session
//! records, so a step written in Python is a step the app can run, and the other way round.

use lucem_hist::cells::{cells_from_nuclei, detect_cells, CellDetection, CellImage, DetectedCells};
use lucem_hist::export::{geojson, measurements_table, region_summary};
use lucem_hist::raster::Grid;
use lucem_hist::session::{self, Action, Session};
use lucem_hist::stains::{deconvolve, estimate_background, estimate_macenko, StainVectors};
use lucem_hist::tissue::{detect as detect_tissue_raster, TissueSettings};
use lucem_hist::{HistObject, Polygon};
use lucem_py_core::io::Image;
use lucem_py_core::{array_of, client, float_input, py_err, uint_input, value_err};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

fn from_json<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    serde_json::from_str(text).map_err(|e| value_err(format!("{what}: {e}")))
}

fn to_json<T: serde::Serialize>(v: &T) -> PyResult<String> {
    serde_json::to_string(v).map_err(py_err)
}

/// An RGB uint8 array `[h, w, 3]` as interleaved bytes, with its height and width.
fn rgb_input(a: &Bound<'_, PyAny>) -> PyResult<(usize, usize, Vec<u8>)> {
    let arr: PyReadonlyArrayDyn<'_, u8> = a.extract().map_err(|_| value_err("expected a uint8 RGB array [h, w, 3]"))?;
    match *arr.shape() {
        [h, w, 3] => Ok((h, w, arr.as_array().iter().copied().collect())),
        ref s => Err(value_err(format!("expected an RGB array [h, w, 3], got shape {s:?}"))),
    }
}

/// The stain presets as JSON: `{"h_dab": …, "h_e": …, "h_e_dab": …}`.
#[pyfunction]
fn stain_presets() -> PyResult<String> {
    to_json(&serde_json::json!({ "h_dab": StainVectors::h_dab(), "h_e": StainVectors::h_e(), "h_e_dab": StainVectors::h_e_dab() }))
}

/// Colour deconvolution of RGB `[h, w, 3]` (uint8) with `stains` (JSON): three float32 `[h, w]`
/// stain amounts (optical densities).
#[pyfunction]
fn deconvolve_rgb<'py>(py: Python<'py>, rgb: Bound<'py, PyAny>, stains: &str) -> PyResult<Vec<Bound<'py, PyArrayDyn<f32>>>> {
    let (h, w, bytes) = rgb_input(&rgb)?;
    let stains: StainVectors = from_json(stains, "stains")?;
    let planes = py.detach(move || deconvolve(&bytes, &stains)).map_err(value_err)?;
    planes.into_iter().map(|p| array_of(py, &[h, w], p)).collect()
}

/// The background (white) RGB of an RGB `[h, w, 3]` (uint8) region.
#[pyfunction]
fn background_rgb(rgb: Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
    let (_, _, bytes) = rgb_input(&rgb)?;
    Ok(estimate_background(&bytes))
}

/// Two stains fitted to an RGB `[h, w, 3]` (uint8) region (Macenko et al. 2009), as JSON.
#[pyfunction]
#[pyo3(signature = (rgb, background, first, second, min_od=0.15, alpha=1.0))]
fn estimate_stains(py: Python<'_>, rgb: Bound<'_, PyAny>, background: [f64; 3], first: &str, second: &str, min_od: f64, alpha: f64) -> PyResult<String> {
    let (_, _, bytes) = rgb_input(&rgb)?;
    let (a, b) = (first.to_string(), second.to_string());
    let est = py.detach(move || estimate_macenko(&bytes, background, (&a, &b), min_od, alpha)).map_err(value_err)?;
    to_json(&est)
}

/// The default settings, as JSON: `{"tissue": …, "cells": …}`.
#[pyfunction]
fn defaults() -> PyResult<String> {
    to_json(&serde_json::json!({ "tissue": TissueSettings::default(), "cells": CellDetection::default() }))
}

/// Tissue of a low-resolution image `[h, w]` or `[h, w, c]` (float32; brightfield: RGB 0–255) at
/// `downsample` level-0 pixels per pixel from `origin`: `(mask, threshold, regions)`, the mask
/// uint8 `[h, w]`, the regions as JSON objects (level-0 pixels).
#[pyfunction]
#[pyo3(signature = (image, pixel_um, settings, downsample=1.0, origin=(0.0, 0.0)))]
fn detect_tissue<'py>(py: Python<'py>, image: Bound<'py, PyAny>, pixel_um: f64, settings: &str, downsample: f64, origin: (f64, f64)) -> PyResult<(Bound<'py, PyArrayDyn<u8>>, f64, String)> {
    let (dims, values) = float_input(&image)?;
    let (h, w, c) = match *dims {
        [h, w] => (h, w, 1),
        [h, w, c] => (h, w, c),
        ref s => return Err(value_err(format!("expected [h, w] or [h, w, c], got shape {s:?}"))),
    };
    let s: TissueSettings = from_json(settings, "settings")?;
    let grid = Grid::new(w, h, [origin.0, origin.1], downsample);
    let t = py.detach(move || detect_tissue_raster(&values, c, grid, pixel_um, &s));
    Ok((array_of(py, &[h, w], t.mask)?, t.threshold, to_json(&t.regions)?))
}

type Cells<'py> = (String, Bound<'py, PyArrayDyn<u32>>, Bound<'py, PyArrayDyn<u32>>);

fn cells_out<'py>(py: Python<'py>, d: DetectedCells, h: usize, w: usize) -> PyResult<Cells<'py>> {
    Ok((to_json(&d.objects)?, array_of(py, &[h, w], d.nuclei)?, array_of(py, &[h, w], d.cells)?))
}

/// Cells of a region: nuclei found on `detection` (float32 `[h, w]`), every channel of
/// `channels` (`[(name, float32 [h, w])]`) measured per compartment. Returns `(objects, nuclei,
/// cells)`: the objects as JSON, the nucleus and cell labels uint32 `[h, w]` (label = object id).
/// `nuclei` (uint32 `[h, w]`): a model's nucleus labels to start from instead of the threshold.
#[pyfunction]
#[pyo3(signature = (detection, channels, pixel_um, settings, downsample=1.0, origin=(0.0, 0.0), nuclei=None))]
#[allow(clippy::too_many_arguments)]
fn detect_cells_in<'py>(py: Python<'py>, detection: Bound<'py, PyAny>, channels: Vec<(String, Bound<'py, PyAny>)>, pixel_um: f64, settings: &str, downsample: f64, origin: (f64, f64), nuclei: Option<Bound<'py, PyAny>>) -> PyResult<Cells<'py>> {
    let (dims, det) = float_input(&detection)?;
    let [h, w] = <[usize; 2]>::try_from(dims.clone()).map_err(|_| value_err(format!("detection: expected [h, w], got shape {dims:?}")))?;
    let mut planes = Vec::with_capacity(channels.len());
    for (name, a) in &channels {
        let (d, v) = float_input(a)?;
        if d != dims {
            return Err(value_err(format!("channel {name:?} has shape {d:?}, the detection {dims:?}")));
        }
        planes.push((name.clone(), v));
    }
    let labels = nuclei.map(|a| uint_input(&a)).transpose()?;
    if labels.as_ref().is_some_and(|(d, _)| *d != dims) {
        return Err(value_err("nuclei and detection differ in shape"));
    }
    let s: CellDetection = from_json(settings, "settings")?;
    let client = client()?;
    let found = py
        .detach(move || {
            let img = CellImage { width: w, height: h, detection: &det, channels: planes.iter().map(|(n, v)| (n.clone(), v.as_slice())).collect(), origin: [origin.0, origin.1], downsample, pixel_um };
            match &labels {
                Some((_, l)) => cells_from_nuclei(&client, &img, l, &s),
                None => detect_cells(&client, &img, &s),
            }
        })
        .map_err(value_err)?;
    cells_out(py, found, h, w)
}

/// The steps of a saved session (`session.json`) to run on another image, as JSON actions.
#[pyfunction]
fn session_recipe(path: std::path::PathBuf) -> PyResult<String> {
    let s = Session::load(&path).map_err(value_err)?;
    to_json(&s.recipe())
}

/// Validates a step (JSON) and returns it in its full form (every default filled in) and its title.
#[pyfunction]
fn check_step(step: &str) -> PyResult<(String, String)> {
    let a: Action = from_json(step, "step")?;
    Ok((to_json(&a)?, a.title().to_string()))
}

/// Runs steps (JSON actions, as `session_recipe` gives) on an image through the engine the app
/// runs. Returns JSON `{"tissue": [outlines], "cells": [objects], "tiles": [objects], "notes": [..]}`.
/// A step naming a model starts its plugin (`lucem[stardist]`, …).
#[pyfunction]
fn run_steps(py: Python<'_>, image: PyRef<'_, Image>, steps: Vec<String>) -> PyResult<String> {
    let actions = steps.iter().map(|s| from_json::<Action>(s, "step")).collect::<PyResult<Vec<_>>>()?;
    let pyramid = image.pyramid().clone();
    let client = client()?;
    let r = py
        .detach(move || {
            let recipe: Vec<&Action> = actions.iter().collect();
            session::run(&client, &pyramid, &recipe, &mut |_| {})
        })
        .map_err(value_err)?;
    to_json(&serde_json::json!({ "tissue": r.tissue, "cells": r.cells, "tiles": r.tiles, "notes": r.notes }))
}

/// One row per object (JSON objects): `(header, rows)` of text cells, centroids in µm.
#[pyfunction]
fn objects_table(objects: &str, image: &str, pixel_um: f64) -> PyResult<(Vec<String>, Vec<Vec<String>>)> {
    let objects: Vec<HistObject> = from_json(objects, "objects")?;
    let t = measurements_table(&objects, image, pixel_um);
    Ok((t.header, t.rows))
}

/// The objects (JSON) as a GeoJSON FeatureCollection (text).
#[pyfunction]
fn objects_geojson(objects: &str) -> PyResult<String> {
    let objects: Vec<HistObject> = from_json(objects, "objects")?;
    to_json(&geojson(&objects))
}

/// What a region (an outline, level-0 pixels) reports about the objects in it: counts per class,
/// intensity summary, area and densities; as JSON.
#[pyfunction]
fn summarize_region(region: Polygon, objects: &str, pixel_um: f64) -> PyResult<String> {
    let objects: Vec<HistObject> = from_json(objects, "objects")?;
    to_json(&region_summary(&region, &objects, pixel_um))
}

/// Adds the `hist` submodule to `m` (`lucem._native`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let h = PyModule::new(m.py(), "hist")?;
    h.add_function(wrap_pyfunction!(stain_presets, &h)?)?;
    h.add_function(wrap_pyfunction!(deconvolve_rgb, &h)?)?;
    h.add_function(wrap_pyfunction!(background_rgb, &h)?)?;
    h.add_function(wrap_pyfunction!(estimate_stains, &h)?)?;
    h.add_function(wrap_pyfunction!(defaults, &h)?)?;
    h.add_function(wrap_pyfunction!(detect_tissue, &h)?)?;
    h.add_function(wrap_pyfunction!(detect_cells_in, &h)?)?;
    h.add_function(wrap_pyfunction!(session_recipe, &h)?)?;
    h.add_function(wrap_pyfunction!(check_step, &h)?)?;
    h.add_function(wrap_pyfunction!(run_steps, &h)?)?;
    h.add_function(wrap_pyfunction!(objects_table, &h)?)?;
    h.add_function(wrap_pyfunction!(objects_geojson, &h)?)?;
    h.add_function(wrap_pyfunction!(summarize_region, &h)?)?;
    m.add_submodule(&h)?;
    Ok(())
}

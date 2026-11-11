//! Where an analysis is stored, and how: next to the pixels it was made from.
//!
//! - **Where**: inside the image's own OME-Zarr when it is one and can be written (labels
//!   where other viewers look for them), else `<fallback>/<image>.lumen.zarr`.
//! - **What**:
//!   - `labels/cells`, `labels/nuclei`: label images (OME-Zarr `labels`, multiscale, `u32`; each
//!     object its id), rasterised chunk by chunk in parallel;
//!   - `lumen/cells.parquet` (and `tiles.parquet`): one row per object — id, kind, class,
//!     intensity class, region, position (px, µm), then every measurement — Arrow / Parquet,
//!     read by Polars, PyArrow, pandas, DuckDB;
//!   - `lumen/shapes.parquet`: GeoParquet (WKB polygons with the `geo` metadata): the objects'
//!     outlines and nuclei, and the regions;
//!   - `lumen/provenance.json`: the session (every step and its parameters) and the versions.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use polars::prelude::*;

use crate::error::{invalid, HistResult};
use crate::objects::{HistObject, Polygon};

/// The store of an image: its own OME-Zarr when writable, else under `fallback`.
pub fn store_for(image: &Path, name: &str, fallback: &Path) -> PathBuf {
    let ome = image.is_dir() && (image.join(".zgroup").is_file() || image.join("zarr.json").is_file());
    if ome && std::fs::create_dir_all(image.join("lumen")).is_ok() && std::fs::write(image.join("lumen").join(".probe"), b"").is_ok() {
        let _ = std::fs::remove_file(image.join("lumen").join(".probe"));
        return image.to_path_buf();
    }
    let safe: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect();
    fallback.join(format!("{safe}.lumen.zarr"))
}

fn io(e: impl std::fmt::Display) -> crate::HistError {
    invalid(e.to_string())
}

/// A region of the image for the shapes table.
pub struct RegionShape<'a> {
    pub id: &'a str,
    pub class: &'a str,
    pub outline: &'a [[f64; 2]],
}

/// Writes `lumen/<name>.parquet`: the objects and their measurements (`region`: each object's
/// region, `None` the image).
pub fn write_objects(store: &Path, name: &str, objects: &[HistObject], region: &[Option<String>], pixel_um: f64) -> HistResult<PathBuf> {
    let dir = store.join("lumen");
    std::fs::create_dir_all(&dir).map_err(io)?;
    let n = objects.len();
    let keys: Vec<String> = objects.iter().flat_map(|o| o.measurements.keys().cloned()).collect::<BTreeSet<_>>().into_iter().collect();
    let mut cols = vec![
        Column::new("id".into(), objects.iter().map(|o| o.id).collect::<Vec<u32>>()),
        Column::new("kind".into(), objects.iter().map(|o| format!("{:?}", o.kind).to_lowercase()).collect::<Vec<_>>()),
        Column::new("class".into(), objects.iter().map(|o| o.class.clone()).collect::<Vec<Option<String>>>()),
        Column::new("intensity".into(), objects.iter().map(|o| o.intensity.map(|i| i.name().to_string())).collect::<Vec<Option<String>>>()),
        Column::new("region".into(), (0..n).map(|i| region.get(i).cloned().flatten()).collect::<Vec<Option<String>>>()),
        Column::new("x_px".into(), objects.iter().map(|o| o.centroid[0]).collect::<Vec<f64>>()),
        Column::new("y_px".into(), objects.iter().map(|o| o.centroid[1]).collect::<Vec<f64>>()),
        Column::new("x_um".into(), objects.iter().map(|o| o.centroid[0] * pixel_um).collect::<Vec<f64>>()),
        Column::new("y_um".into(), objects.iter().map(|o| o.centroid[1] * pixel_um).collect::<Vec<f64>>()),
    ];
    for k in &keys {
        cols.push(Column::new(k.as_str().into(), objects.iter().map(|o| o.measurements.get(k).copied()).collect::<Vec<Option<f64>>>()));
    }
    let mut df = DataFrame::new(n, cols).map_err(io)?;
    let path = dir.join(format!("{name}.parquet"));
    let file = std::fs::File::create(&path).map_err(io)?;
    ParquetWriter::new(file).with_compression(ParquetCompression::Zstd(None)).finish(&mut df).map_err(io)?;
    Ok(path)
}

/// A closed polygon as WKB (little-endian Polygon of one ring).
fn wkb(p: &Polygon) -> Vec<u8> {
    let mut ring: Vec<[f64; 2]> = p.clone();
    if ring.first() != ring.last()
        && let Some(f) = ring.first().copied()
    {
        ring.push(f);
    }
    let mut out = Vec::with_capacity(13 + ring.len() * 16);
    out.push(1u8);
    out.extend(3u32.to_le_bytes());
    out.extend(1u32.to_le_bytes());
    out.extend((ring.len() as u32).to_le_bytes());
    for [x, y] in ring {
        out.extend(x.to_le_bytes());
        out.extend(y.to_le_bytes());
    }
    out
}

/// Writes `lumen/shapes.parquet` (GeoParquet 1.1, WKB, level-0 pixel coordinates): every
/// object's outline (`cell`, `tile`) and nucleus, and the regions.
pub fn write_shapes(store: &Path, objects: &[HistObject], regions: &[RegionShape]) -> HistResult<PathBuf> {
    let dir = store.join("lumen");
    std::fs::create_dir_all(&dir).map_err(io)?;
    let (mut id, mut kind, mut class, mut geom): (Vec<String>, Vec<String>, Vec<Option<String>>, Vec<Vec<u8>>) = (vec![], vec![], vec![], vec![]);
    for o in objects {
        let k = format!("{:?}", o.kind).to_lowercase();
        id.push(format!("{}{}", &k[..1], o.id));
        kind.push(k.clone());
        class.push(o.class.clone());
        geom.push(wkb(&o.roi));
        if let Some(n) = &o.nucleus {
            id.push(format!("{}{}", &k[..1], o.id));
            kind.push("nucleus".into());
            class.push(o.class.clone());
            geom.push(wkb(n));
        }
    }
    for r in regions {
        id.push(r.id.to_string());
        kind.push("region".into());
        class.push(Some(r.class.to_string()));
        geom.push(wkb(&r.outline.to_vec()));
    }
    let n = id.len();
    let mut df = DataFrame::new(n, vec![Column::new("id".into(), id), Column::new("kind".into(), kind), Column::new("class".into(), class), Column::new("geometry".into(), geom.into_iter().map(Some).collect::<Vec<Option<Vec<u8>>>>())]).map_err(io)?;
    let geo = serde_json::json!({
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon"], "crs": null, "edges": "planar"}},
    });
    let path = dir.join("shapes.parquet");
    let file = std::fs::File::create(&path).map_err(io)?;
    ParquetWriter::new(file).with_compression(ParquetCompression::Zstd(None)).with_key_value_metadata(Some(KeyValueMetadata::from_static(vec![("geo".into(), geo.to_string())]))).finish(&mut df).map_err(io)?;
    Ok(path)
}

/// Writes `labels/<name>`: an OME-Zarr label image of the objects (`nucleus`: their nuclei,
/// else their outlines), `width × height` level-0 pixels, chunks of 1024 rasterised in parallel.
pub fn write_labels(store: &Path, name: &str, objects: &[HistObject], nucleus: bool, (width, height): (usize, usize), pixel_um: Option<f64>) -> HistResult<PathBuf> {
    let labels = store.join("labels");
    std::fs::create_dir_all(&labels).map_err(io)?;
    let root = labels.join(name);
    if root.exists() {
        std::fs::remove_dir_all(&root).map_err(io)?;
    }
    const CHUNK: usize = 1024;
    let options = lumen_io::WriterOptions { name: name.to_string(), channels: vec![name.to_string()], shape: [1, height, width], chunks: [1, CHUNK, CHUNK], pixel_size_um: pixel_um, z_spacing_um: None };
    let writer = lumen_io::ZarrWriter::create::<u32>(&root, options).map_err(io)?;
    // Each object once: its outline's box, for the chunks it touches
    let shapes: Vec<(u32, &Polygon, (f64, f64, f64, f64))> = objects
        .iter()
        .filter_map(|o| {
            let p = if nucleus { o.nucleus.as_ref()? } else { &o.roi };
            Some((o.id, p, crate::polygon::bounds(p)))
        })
        .collect();
    let (nx, ny) = (width.div_ceil(CHUNK), height.div_ceil(CHUNK));
    let blocks: Vec<(usize, [usize; 3], [usize; 3], Vec<u32>)> = {
        use rayon::prelude::*;
        (0..nx * ny)
            .into_par_iter()
            .filter_map(|k| {
                let (cx, cy) = (k % nx, k / nx);
                let (x0, y0) = (cx * CHUNK, cy * CHUNK);
                let (w, h) = (CHUNK.min(width - x0), CHUNK.min(height - y0));
                let (fx0, fy0, fx1, fy1) = (x0 as f64, y0 as f64, (x0 + w) as f64, (y0 + h) as f64);
                let mut raster = vec![0u32; w * h];
                let mut any = false;
                for (id, p, (bx0, by0, bx1, by1)) in &shapes {
                    if *bx1 < fx0 || *by1 < fy0 || *bx0 > fx1 || *by0 > fy1 {
                        continue;
                    }
                    crate::polygon::fill(p, &mut raster, w, h, [fx0, fy0], 1.0, *id);
                    any = true;
                }
                // Empty chunks are not stored (a missing chunk reads as zeros)
                any.then_some((0usize, [0usize, y0, x0], [1usize, h, w], raster))
            })
            .collect()
    };
    writer.write_many(&blocks).map_err(io)?;
    let root = writer.finish::<u32>().map_err(io)?;
    // An image-label (OME-Zarr), listed in `labels/`
    let attrs_path = root.join(".zattrs");
    let mut attrs: serde_json::Value = std::fs::read_to_string(&attrs_path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(serde_json::json!({}));
    attrs["image-label"] = serde_json::json!({"version": "0.4", "source": {"image": "../../"}});
    std::fs::write(&attrs_path, serde_json::to_string_pretty(&attrs).map_err(io)?).map_err(io)?;
    let list_path = labels.join(".zattrs");
    let mut list: serde_json::Value = std::fs::read_to_string(&list_path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(serde_json::json!({"labels": []}));
    let names = list["labels"].as_array_mut().map(|a| {
        if !a.iter().any(|v| v == name) {
            a.push(serde_json::json!(name));
        }
    });
    let _ = names;
    std::fs::write(&list_path, serde_json::to_string_pretty(&list).map_err(io)?).map_err(io)?;
    if !labels.join(".zgroup").exists() {
        std::fs::write(labels.join(".zgroup"), "{\n  \"zarr_format\": 2\n}\n").map_err(io)?;
    }
    Ok(root)
}

/// Writes `lumen/provenance.json`.
pub fn write_provenance(store: &Path, provenance: &serde_json::Value) -> HistResult<PathBuf> {
    let dir = store.join("lumen");
    std::fs::create_dir_all(&dir).map_err(io)?;
    let path = dir.join("provenance.json");
    std::fs::write(&path, serde_json::to_string_pretty(provenance).map_err(io)?).map_err(io)?;
    Ok(path)
}

/// Creates the fallback store's group when it is not the image's own.
pub fn ensure_group(store: &Path) -> HistResult<()> {
    std::fs::create_dir_all(store).map_err(io)?;
    if !store.join(".zgroup").exists() && !store.join("zarr.json").exists() {
        std::fs::write(store.join(".zgroup"), "{\n  \"zarr_format\": 2\n}\n").map_err(io)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_shapes_and_labels_are_written() {
        let dir = std::env::temp_dir().join(format!("lumen-output-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        ensure_group(&dir).unwrap();
        let mut a = HistObject::new(1, crate::ObjectKind::Cell, vec![[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]);
        a.nucleus = Some(vec![[15.0, 15.0], [25.0, 15.0], [25.0, 25.0], [15.0, 25.0]]);
        a.class = Some("Tumor".into());
        a.measurements.insert("Nucleus: Area µm^2".into(), 100.0);
        let b = HistObject::new(2, crate::ObjectKind::Cell, vec![[1500.0, 1500.0], [1520.0, 1500.0], [1520.0, 1520.0], [1500.0, 1520.0]]);
        let p = write_objects(&dir, "cells", &[a.clone(), b.clone()], &[Some("r1".into()), None], 0.5).unwrap();
        let df = ParquetReader::new(std::fs::File::open(&p).unwrap()).finish().unwrap();
        assert_eq!(df.height(), 2);
        assert!(df.column("Nucleus: Area µm^2").is_ok());
        let region = RegionShape { id: "r1", class: "Region", outline: &[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]] };
        let s = write_shapes(&dir, &[a.clone(), b.clone()], &[region]).unwrap();
        let shapes = ParquetReader::new(std::fs::File::open(&s).unwrap()).finish().unwrap();
        assert_eq!(shapes.height(), 4, "two cells, one nucleus, one region");
        let l = write_labels(&dir, "cells", &[a, b], false, (2048, 2048), Some(0.5)).unwrap();
        let py = lumen_io::Pyramid::open(&l).unwrap();
        let v = py.read_region_at(0, 0, 0, 0, 0, 2048, 2048).unwrap();
        assert_eq!(v[20 * 2048 + 20], 1.0);
        assert_eq!(v[1510 * 2048 + 1510], 2.0);
        assert_eq!(v[1000 * 2048 + 1000], 0.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}

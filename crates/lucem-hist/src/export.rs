//! Exporting measurements: one row per object (TSV or CSV), a summary per region (counts per
//! class, positive %, H-score, densities, area), and objects as GeoJSON.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::intensity;
use crate::objects::{HistObject, Measurements, Polygon};
use crate::polygon;

/// A table: its header and rows (text cells).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    /// The table as text, cells separated by `sep` (`'\t'`, `','`); a cell holding the separator,
    /// a quote or a line break is quoted.
    pub fn to_delimited(&self, sep: char) -> String {
        let cell = |s: &str| if s.contains(sep) || s.contains('"') || s.contains('\n') { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() };
        let mut out = String::new();
        for r in std::iter::once(&self.header).chain(&self.rows) {
            out.push_str(&r.iter().map(|c| cell(c)).collect::<Vec<_>>().join(&sep.to_string()));
            out.push('\n');
        }
        out
    }
}

fn number(v: f64) -> String {
    if v.is_finite() { format!("{v}") } else { String::new() }
}

/// One row per object: image, id, kind, class, centroid (µm), then every measurement any of
/// them has (sorted by name; empty where one lacks it).
pub fn measurements_table(objects: &[HistObject], image: &str, pixel_um: f64) -> Table {
    let names: BTreeSet<&String> = objects.iter().flat_map(|o| o.measurements.keys()).collect();
    let mut header: Vec<String> = ["Image", "Object ID", "Object type", "Classification", "Centroid X µm", "Centroid Y µm"].map(String::from).to_vec();
    header.extend(names.iter().map(|n| n.to_string()));
    let rows = objects
        .iter()
        .map(|o| {
            let mut r = vec![image.to_string(), o.id.to_string(), format!("{:?}", o.kind), o.full_class().unwrap_or_default(), number(o.centroid[0] * pixel_um), number(o.centroid[1] * pixel_um)];
            r.extend(names.iter().map(|n| o.measurements.get(*n).map_or(String::new(), |v| number(*v))));
            r
        })
        .collect();
    Table { header, rows }
}

/// What a region reports about the objects whose centroid is in it: their number, per class,
/// the intensity summary (when they have intensity classes), the region's area and the
/// densities per mm².
pub fn region_summary(region: &Polygon, objects: &[HistObject], pixel_um: f64) -> Measurements {
    let inside: Vec<&HistObject> = objects.iter().filter(|o| polygon::contains(region, o.centroid)).collect();
    let area_um2 = polygon::area(region) * pixel_um * pixel_um;
    let mm2 = area_um2 / 1e6;
    let mut m = Measurements::new();
    m.insert("Area µm^2".into(), area_um2);
    m.insert("Num Detections".into(), inside.len() as f64);
    if mm2 > 0.0 {
        m.insert("Num Detections per mm^2".into(), inside.len() as f64 / mm2);
    }
    let mut per_class: BTreeMap<String, usize> = BTreeMap::new();
    for o in &inside {
        if let Some(c) = &o.class {
            *per_class.entry(c.clone()).or_default() += 1;
        }
    }
    for (c, n) in per_class {
        m.insert(format!("Num {c}"), n as f64);
        if mm2 > 0.0 {
            m.insert(format!("Num {c} per mm^2"), n as f64 / mm2);
        }
    }
    let s = intensity::summarize(inside.iter().copied());
    if s.total > 0 {
        m.insert("Num Negative".into(), s.negative as f64);
        m.insert("Num Positive".into(), (s.total - s.negative) as f64);
        m.insert("Positive %".into(), s.positive_percent);
        if s.one_plus + s.two_plus + s.three_plus > 0 {
            m.insert("Num 1+".into(), s.one_plus as f64);
            m.insert("Num 2+".into(), s.two_plus as f64);
            m.insert("Num 3+".into(), s.three_plus as f64);
            m.insert("H-score".into(), s.h_score);
        }
    }
    m
}

fn ring(p: &Polygon) -> Value {
    let mut pts: Vec<[f64; 2]> = p.clone();
    if let Some(first) = p.first() {
        pts.push(*first);
    }
    json!([pts])
}

/// The objects as a GeoJSON FeatureCollection (level-0 pixels): the outline as the geometry,
/// a cell's nucleus as `nucleusGeometry`, the class and measurements as properties.
pub fn geojson(objects: &[HistObject]) -> Value {
    let features: Vec<Value> = objects
        .iter()
        .map(|o| {
            let mut props = json!({ "objectType": format!("{:?}", o.kind).to_lowercase(), "id": o.id, "measurements": o.measurements });
            if let Some(c) = o.full_class() {
                props["classification"] = json!({ "name": c });
            }
            let mut f = json!({ "type": "Feature", "geometry": { "type": "Polygon", "coordinates": ring(&o.roi) }, "properties": props });
            if let Some(n) = &o.nucleus {
                f["nucleusGeometry"] = json!({ "type": "Polygon", "coordinates": ring(n) });
            }
            f
        })
        .collect();
    json!({ "type": "FeatureCollection", "features": features })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::{IntensityClass, ObjectKind};

    #[test]
    fn tables_summaries_and_geojson() {
        let objs: Vec<HistObject> = (0..4)
            .map(|i| {
                let x = i as f64 * 10.0;
                let mut o = HistObject::new(i, ObjectKind::Cell, vec![[x, 0.0], [x + 4.0, 0.0], [x + 4.0, 4.0], [x, 4.0]]);
                o.class = Some(if i < 3 { "Tumor".into() } else { "Stroma".into() });
                o.intensity = Some([IntensityClass::Negative, IntensityClass::OnePlus, IntensityClass::ThreePlus, IntensityClass::Negative][i as usize]);
                o.measurements.insert("Nucleus: Area µm^2".into(), 10.0 + i as f64);
                if i == 0 {
                    o.measurements.insert("Only here".into(), 1.0);
                }
                o
            })
            .collect();
        let t = measurements_table(&objs, "slide, 1", 0.5);
        assert_eq!(t.header.len(), 8);
        let text = t.to_delimited(',');
        assert!(text.starts_with("Image,Object ID") && text.contains("\"slide, 1\"") && text.contains("Tumor: 3+"));
        let s = region_summary(&vec![[-1.0, -1.0], [100.0, -1.0], [100.0, 10.0], [-1.0, 10.0]], &objs, 0.5);
        assert_eq!(s["Num Detections"], 4.0);
        assert_eq!(s["Num Tumor"], 3.0);
        assert_eq!(s["Positive %"], 50.0);
        assert_eq!(s["H-score"], 100.0);
        let g = geojson(&objs);
        assert_eq!(g["features"].as_array().unwrap().len(), 4);
        assert_eq!(g["features"][2]["properties"]["classification"]["name"], "Tumor: 3+");
    }
}

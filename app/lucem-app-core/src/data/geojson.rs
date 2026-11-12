//! Annotations as GeoJSON (a FeatureCollection in level-0 pixels), the format histology tools
//! exchange: regions as polygons, points as points, rulers and brush strokes as lines (a stroke
//! keeps its radius). Properties: `class` (also read from `classification.name`), `note`, `z`.

use serde_json::{json, Value};

use crate::domain::annotation::{Annotation, Shape};

/// The annotations as a FeatureCollection.
pub fn export(annotations: &[&Annotation]) -> String {
    let features: Vec<Value> = annotations
        .iter()
        .map(|a| {
            let ring = |pts: Vec<[f64; 2]>| {
                let mut r: Vec<Value> = pts.iter().map(|p| json!([p[0], p[1]])).collect();
                if let Some(first) = r.first().cloned() {
                    r.push(first);
                }
                Value::Array(r)
            };
            let (geometry, extra) = match &a.shape {
                Shape::Point { x, y } => (json!({ "type": "Point", "coordinates": [x, y] }), json!({})),
                Shape::Line { x0, y0, x1, y1 } => (json!({ "type": "LineString", "coordinates": [[x0, y0], [x1, y1]] }), json!({ "measurement": true })),
                Shape::Stroke { points, radius, erase } => (json!({ "type": "LineString", "coordinates": points.iter().map(|p| json!([p[0], p[1]])).collect::<Vec<_>>() }), json!({ "radius": radius, "erase": erase })),
                s => (json!({ "type": "Polygon", "coordinates": [ring(s.outline())] }), json!({})),
            };
            let mut props = json!({ "id": a.id, "class": a.class, "classification": { "name": a.class }, "kind": a.shape.kind() });
            if !a.note.is_empty() {
                props["note"] = json!(a.note);
            }
            if let Some(z) = a.z {
                props["z"] = json!(z);
            }
            if let (Some(p), Some(e)) = (props.as_object_mut(), extra.as_object()) {
                p.extend(e.clone());
            }
            json!({ "type": "Feature", "geometry": geometry, "properties": props })
        })
        .collect();
    serde_json::to_string_pretty(&json!({ "type": "FeatureCollection", "features": features })).unwrap_or_default()
}

/// The features of a GeoJSON text as annotations of `image` (ids given when added); classes the
/// project lacks are listed in the second value.
pub fn import(text: &str, image: &str, default_class: &str) -> Result<Vec<Annotation>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let features: Vec<&Value> = match v["type"].as_str() {
        Some("FeatureCollection") => v["features"].as_array().map(|a| a.iter().collect()).unwrap_or_default(),
        Some("Feature") => vec![&v],
        _ => v.as_array().map(|a| a.iter().collect()).ok_or("not GeoJSON (no FeatureCollection)")?,
    };
    let pt = |c: &Value| Some([c.get(0)?.as_f64()?, c.get(1)?.as_f64()?]);
    let mut out = Vec::new();
    for f in features {
        let g = &f["geometry"];
        let p = &f["properties"];
        let class = p["class"].as_str().or_else(|| p["classification"]["name"].as_str()).unwrap_or(default_class).to_string();
        let coords = &g["coordinates"];
        let shapes: Vec<Shape> = match g["type"].as_str() {
            Some("Point") => pt(coords).map(|[x, y]| Shape::Point { x, y }).into_iter().collect(),
            Some("MultiPoint") => coords.as_array().map(|a| a.iter().filter_map(pt).map(|[x, y]| Shape::Point { x, y }).collect()).unwrap_or_default(),
            Some("LineString") => {
                let pts: Vec<[f64; 2]> = coords.as_array().map(|a| a.iter().filter_map(pt).collect()).unwrap_or_default();
                match (p["radius"].as_f64(), pts.as_slice()) {
                    (Some(radius), _) => vec![Shape::Stroke { points: pts.clone(), radius, erase: p["erase"].as_bool().unwrap_or(false) }],
                    (None, [a, b]) => vec![Shape::Line { x0: a[0], y0: a[1], x1: b[0], y1: b[1] }],
                    (None, _) if pts.len() > 2 => vec![Shape::Stroke { points: pts, radius: 1.0, erase: false }],
                    _ => Vec::new(),
                }
            }
            Some("Polygon") => coords.get(0).and_then(Value::as_array).map(|r| ring_shape(r.iter().filter_map(pt).collect())).into_iter().flatten().collect(),
            Some("MultiPolygon") => coords.as_array().map(|polys| polys.iter().filter_map(|poly| poly.get(0)?.as_array()).filter_map(|r| ring_shape(r.iter().filter_map(pt).collect())).collect()).unwrap_or_default(),
            _ => Vec::new(),
        };
        for shape in shapes {
            out.push(Annotation { id: String::new(), image: image.to_string(), shape, class: class.clone(), note: p["note"].as_str().unwrap_or("").to_string(), z: p["z"].as_u64().map(|z| z as usize) });
        }
    }
    Ok(out)
}

/// A closed ring (the last point repeating the first is dropped) as a polygon.
fn ring_shape(mut pts: Vec<[f64; 2]>) -> Option<Shape> {
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    (pts.len() >= 3).then_some(Shape::Polygon { points: pts })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_points_and_strokes_round_trip() {
        let ann = |shape: Shape| Annotation { id: "a1".into(), image: "img".into(), shape, class: "Tumor".into(), note: String::new(), z: Some(3) };
        let list = [ann(Shape::Polygon { points: vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]] }), ann(Shape::Point { x: 1.0, y: 2.0 }), ann(Shape::Stroke { points: vec![[0.0, 0.0], [5.0, 5.0], [9.0, 1.0]], radius: 3.0, erase: false })];
        let text = export(&list.iter().collect::<Vec<_>>());
        let back = import(&text, "img", "Region").unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back[0].shape, list[0].shape);
        assert_eq!(back[1].shape, list[1].shape);
        assert_eq!(back[2].shape, list[2].shape);
        assert!(back.iter().all(|a| a.class == "Tumor" && a.z == Some(3)));
    }
}

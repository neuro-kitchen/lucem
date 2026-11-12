//! Areas: of each class of a class map, and of positive staining, inside a region.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassArea {
    pub class: String,
    pub area_um2: f64,
    /// Of the classified area inside the region (unclassified pixels left out).
    pub percent: f64,
}

/// The area of each class of `class_map` (0: none, `k`: `classes[k − 1]`), only where `within`
/// is non-zero when given; `pixel_area_um2` is the area of one map pixel.
pub fn class_areas(class_map: &[u8], classes: &[String], within: Option<&[u8]>, pixel_area_um2: f64) -> Vec<ClassArea> {
    let mut counts = vec![0usize; classes.len() + 1];
    for (i, &c) in class_map.iter().enumerate() {
        if within.is_none_or(|m| m[i] != 0) && (c as usize) <= classes.len() {
            counts[c as usize] += 1;
        }
    }
    let total: usize = counts[1..].iter().sum();
    classes
        .iter()
        .enumerate()
        .map(|(k, name)| {
            let n = counts[k + 1];
            ClassArea { class: name.clone(), area_um2: n as f64 * pixel_area_um2, percent: if total == 0 { 0.0 } else { 100.0 * n as f64 / total as f64 } }
        })
        .collect()
}

/// Positive area: pixels of `values` (a stain's optical density, say) above `threshold` inside
/// `within`; `(positive µm², region µm², percent)`.
pub fn positive_area(values: &[f32], threshold: f32, within: Option<&[u8]>, pixel_area_um2: f64) -> (f64, f64, f64) {
    let (mut pos, mut all) = (0usize, 0usize);
    for (i, &v) in values.iter().enumerate() {
        if within.is_none_or(|m| m[i] != 0) {
            all += 1;
            if v > threshold {
                pos += 1;
            }
        }
    }
    let pct = if all == 0 { 0.0 } else { 100.0 * pos as f64 / all as f64 };
    (pos as f64 * pixel_area_um2, all as f64 * pixel_area_um2, pct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn areas_follow_the_map_and_the_region() {
        let map = [1u8, 1, 2, 0, 2, 2];
        let within = [1u8, 1, 1, 1, 0, 0];
        let a = class_areas(&map, &["Tumor".into(), "Stroma".into()], Some(&within), 4.0);
        assert_eq!((a[0].area_um2, a[1].area_um2), (8.0, 4.0));
        assert!((a[0].percent - 200.0 / 3.0).abs() < 1e-9);
        let (p, t, pct) = positive_area(&[0.1, 0.5, 0.9, 0.2], 0.3, None, 1.0);
        assert_eq!((p, t, pct), (2.0, 4.0, 50.0));
    }
}

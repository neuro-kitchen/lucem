//! Density maps: on a grid over a region, how many objects (of a class, or positive ones) are
//! within a radius of each grid pixel — or what percentage of them are positive; then the
//! hotspots (the highest maxima, apart from each other) and the regions above a value.

use serde::{Deserialize, Serialize};

use crate::objects::{HistObject, Point, Polygon};
use crate::{polygon, raster};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DensityValue {
    /// Objects within the radius.
    Count,
    /// Objects per mm².
    PerMm2,
    /// Positive objects (intensity class) as a percentage of those within the radius.
    PercentPositive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DensitySettings {
    /// The map's pixel (µm).
    pub pixel_um: f64,
    /// The radius counted around each map pixel (µm).
    pub radius_um: f64,
    /// Only objects of this class.
    pub class: Option<String>,
    pub value: DensityValue,
}

impl Default for DensitySettings {
    fn default() -> Self {
        Self { pixel_um: 20.0, radius_um: 50.0, class: None, value: DensityValue::Count }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DensityMap {
    pub width: usize,
    pub height: usize,
    /// Level-0 position of the first map pixel, and level-0 pixels per map pixel.
    pub origin: Point,
    pub ds: f64,
    pub values: Vec<f32>,
}

impl DensityMap {
    pub fn centre_of(&self, i: usize) -> Point {
        [self.origin[0] + ((i % self.width) as f64 + 0.5) * self.ds, self.origin[1] + ((i / self.width) as f64 + 0.5) * self.ds]
    }
}

/// The density of `objects` over `bounds` (level-0 `x0, y0, x1, y1`); `pixel_um` µm per level-0
/// pixel.
pub fn density_map(objects: &[HistObject], bounds: (f64, f64, f64, f64), pixel_um: f64, s: &DensitySettings) -> DensityMap {
    let ds = s.pixel_um / pixel_um;
    let width = (((bounds.2 - bounds.0) / ds).ceil() as usize).max(1);
    let height = (((bounds.3 - bounds.1) / ds).ceil() as usize).max(1);
    let n = width * height;
    let (mut all, mut pos) = (vec![0f32; n], vec![0f32; n]);
    for o in objects.iter().filter(|o| s.class.as_deref().is_none_or(|c| o.has_class(c))) {
        let (gx, gy) = (((o.centroid[0] - bounds.0) / ds).floor(), ((o.centroid[1] - bounds.1) / ds).floor());
        if gx < 0.0 || gy < 0.0 || gx >= width as f64 || gy >= height as f64 {
            continue;
        }
        let i = gy as usize * width + gx as usize;
        all[i] += 1.0;
        if o.intensity.is_some_and(|c| c.is_positive()) {
            pos[i] += 1.0;
        }
    }
    // Sum within the radius (a disk of map pixels)
    let r = (s.radius_um / s.pixel_um).max(0.0);
    let ri = r.ceil() as isize;
    let disk: Vec<(isize, isize)> = (-ri..=ri).flat_map(|dy| (-ri..=ri).map(move |dx| (dx, dy))).filter(|(dx, dy)| ((dx * dx + dy * dy) as f64) <= r * r).collect();
    let sum = |src: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; n];
        for y in 0..height as isize {
            for x in 0..width as isize {
                let mut acc = 0.0;
                for (dx, dy) in &disk {
                    let (sx, sy) = (x + dx, y + dy);
                    if sx >= 0 && sy >= 0 && sx < width as isize && sy < height as isize {
                        acc += src[sy as usize * width + sx as usize];
                    }
                }
                out[y as usize * width + x as usize] = acc;
            }
        }
        out
    };
    let counts = sum(&all);
    let values = match s.value {
        DensityValue::Count => counts,
        DensityValue::PerMm2 => {
            let area_mm2 = disk.len() as f64 * (s.pixel_um / 1000.0).powi(2);
            counts.iter().map(|c| (*c as f64 / area_mm2) as f32).collect()
        }
        DensityValue::PercentPositive => {
            let p = sum(&pos);
            counts.iter().zip(p).map(|(c, p)| if *c > 0.0 { 100.0 * p / c } else { 0.0 }).collect()
        }
    };
    DensityMap { width, height, origin: [bounds.0, bounds.1], ds, values }
}

/// The `count` highest local maxima of the map at least `min_value`, each at least
/// `min_distance_um` from the ones before it (level-0 points, highest first).
pub fn hotspots(map: &DensityMap, count: usize, min_distance_um: f64, min_value: f32, pixel_um: f64) -> Vec<(Point, f32)> {
    let mut order: Vec<usize> = (0..map.values.len()).filter(|&i| map.values[i] >= min_value).collect();
    order.sort_by(|&a, &b| map.values[b].total_cmp(&map.values[a]));
    let min_d = min_distance_um / pixel_um;
    let mut out: Vec<(Point, f32)> = Vec::new();
    for i in order {
        if out.len() >= count {
            break;
        }
        let p = map.centre_of(i);
        if out.iter().all(|(q, _)| ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt() >= min_d) {
            out.push((p, map.values[i]));
        }
    }
    out
}

/// The regions where the map is at least `value` (pieces under `min_area_um2` dropped), as
/// level-0 outlines.
pub fn regions_above(map: &DensityMap, value: f32, min_area_um2: f64, pixel_um: f64) -> Vec<Polygon> {
    let mut mask: Vec<u8> = map.values.iter().map(|&v| u8::from(v >= value)).collect();
    let px_area = (map.ds * pixel_um).powi(2);
    raster::clean(&mut mask, map.width, map.height, min_area_um2 / px_area, 0.0);
    let (labels, _) = raster::components(&mask, map.width, map.height);
    polygon::trace_labels(&labels, map.width, map.height, map.origin, map.ds).into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::{IntensityClass, ObjectKind};

    #[test]
    fn a_cluster_is_the_hotspot() {
        let mut objs = Vec::new();
        for i in 0..40 {
            let (x, y) = (500.0 + (i % 7) as f64 * 3.0, 500.0 + (i / 7) as f64 * 3.0);
            let mut o = HistObject::new(i, ObjectKind::Cell, vec![[x, y], [x + 1.0, y], [x + 1.0, y + 1.0]]);
            o.intensity = Some(if i % 4 == 0 { IntensityClass::Positive } else { IntensityClass::Negative });
            objs.push(o);
        }
        let mut lone = HistObject::new(99, ObjectKind::Cell, vec![[100.0, 100.0], [101.0, 100.0], [101.0, 101.0]]);
        lone.intensity = Some(IntensityClass::Positive);
        objs.push(lone);
        let s = DensitySettings { pixel_um: 10.0, radius_um: 30.0, ..DensitySettings::default() };
        let m = density_map(&objs, (0.0, 0.0, 1000.0, 1000.0), 1.0, &s);
        let h = hotspots(&m, 2, 200.0, 1.0, 1.0);
        assert_eq!(h.len(), 2);
        assert!((h[0].0[0] - 510.0).abs() < 30.0 && h[0].1 >= 40.0, "{h:?}");
        assert_eq!(regions_above(&m, 20.0, 0.0, 1.0).len(), 1);
        let pct = density_map(&objs, (0.0, 0.0, 1000.0, 1000.0), 1.0, &DensitySettings { value: DensityValue::PercentPositive, ..s });
        let at_cluster = pct.values[(510.0f64 / 10.0) as usize * pct.width + 51];
        assert!((at_cluster - 25.0).abs() < 5.0, "{at_cluster}");
    }
}

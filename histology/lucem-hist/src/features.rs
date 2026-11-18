//! Smoothed features: each object gets, for every measurement, the mean of its neighbours'
//! values weighted by a Gaussian of their centroid distance (FWHM in µm), and how many
//! neighbours are near. A cell then carries what its surroundings look like, which helps a
//! classifier tell regions apart (tumour vs stroma) where single cells are ambiguous.

use std::collections::HashMap;

use crate::objects::HistObject;

/// Adds `Smoothed: {fwhm} µm: {name}` for every measurement, and `Smoothed: {fwhm} µm: Nearby
/// detection counts` (the sum of the weights). `pixel_um`: µm per level-0 pixel.
pub fn add_smoothed(objects: &mut [HistObject], fwhm_um: f64, pixel_um: f64) {
    if objects.is_empty() || fwhm_um <= 0.0 || pixel_um <= 0.0 {
        return;
    }
    let sigma = fwhm_um / (2.0 * (2.0 * 2f64.ln()).sqrt()) / pixel_um;
    let reach = 3.0 * sigma;
    let cell = reach.max(1.0);
    let key = |p: [f64; 2]| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, o) in objects.iter().enumerate() {
        grid.entry(key(o.centroid)).or_default().push(i);
    }
    let prefix = format!("Smoothed: {} µm: ", trim(fwhm_um));
    let names: Vec<Vec<(String, f64)>> = objects.iter().map(|o| o.measurements.iter().filter(|(k, v)| v.is_finite() && !k.starts_with("Smoothed: ")).map(|(k, v)| (k.clone(), *v)).collect()).collect();
    let mut results: Vec<Vec<(String, f64)>> = Vec::with_capacity(objects.len());
    for o in objects.iter() {
        let (kx, ky) = key(o.centroid);
        let mut sums: HashMap<&str, (f64, f64)> = HashMap::new();
        let mut count = 0.0;
        for gx in kx - 1..=kx + 1 {
            for gy in ky - 1..=ky + 1 {
                for &j in grid.get(&(gx, gy)).map(Vec::as_slice).unwrap_or(&[]) {
                    let q = objects[j].centroid;
                    let d2 = (q[0] - o.centroid[0]).powi(2) + (q[1] - o.centroid[1]).powi(2);
                    if d2 > reach * reach {
                        continue;
                    }
                    let wgt = (-d2 / (2.0 * sigma * sigma)).exp();
                    count += wgt;
                    for (k, v) in &names[j] {
                        let e = sums.entry(k.as_str()).or_insert((0.0, 0.0));
                        e.0 += wgt * v;
                        e.1 += wgt;
                    }
                }
            }
        }
        let mut r: Vec<(String, f64)> = sums.into_iter().map(|(k, (s, w))| (format!("{prefix}{k}"), s / w)).collect();
        r.push((format!("{prefix}Nearby detection counts"), count));
        results.push(r);
    }
    for (o, r) in objects.iter_mut().zip(results) {
        o.measurements.extend(r);
    }
}

fn trim(v: f64) -> String {
    if v.fract() == 0.0 { format!("{v:.0}") } else { format!("{v}") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::ObjectKind;

    #[test]
    fn neighbours_average_and_count() {
        let mut objs: Vec<HistObject> = (0..3)
            .map(|i| {
                let x = [0.0, 4.0, 1000.0][i];
                let mut o = HistObject::new(i as u32, ObjectKind::Cell, vec![[x, 0.0], [x + 1.0, 0.0], [x + 1.0, 1.0], [x, 1.0]]);
                o.measurements.insert("Area".into(), [10.0, 30.0, 100.0][i]);
                o
            })
            .collect();
        add_smoothed(&mut objs, 25.0, 1.0);
        let m = objs[0].get("Smoothed: 25 µm: Area").unwrap();
        assert!(m > 10.0 && m < 30.0, "{m}");
        assert_eq!(objs[2].get("Smoothed: 25 µm: Area"), Some(100.0));
        assert!((objs[2].get("Smoothed: 25 µm: Nearby detection counts").unwrap() - 1.0).abs() < 1e-9);
    }
}

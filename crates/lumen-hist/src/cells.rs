//! Cell detection and measurement.
//!
//! Nuclei are found on one channel where they are bright (hematoxylin optical density from
//! [`crate::stains::deconvolve`], or a nuclear stain such as DAPI), on the GPU through
//! `lumen-ops`: an optional rolling-ball background is subtracted, a median and a Gaussian
//! smooth it, a threshold marks the nuclei, and touching nuclei are split by shape (distance
//! transform peaks as markers of a watershed). Nuclei out of the area range go; each grows by
//! `cell_expansion_um` (without crossing a neighbour) into its cell.
//!
//! Every cell is measured in its compartments (nucleus, cytoplasm, cell) for every channel
//! given (mean, std dev, min, max) with its shape (area, perimeter, circularity, eccentricity),
//! named as `Nucleus: Area µm^2`, `Cytoplasm: DAB OD mean`, `Nucleus/Cell area ratio`.
//!
//! A model's nucleus labels (a plugin of `plugins/ml`) become cells with [`cells_from_nuclei`]:
//! the same filtering, expansion and measurements.

use lumen_core::cubecl::prelude::Client;
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;
use lumen_ops::core::distance::euclidean_distance;
use lumen_ops::core::StructuringElement;
use lumen_ops::feature::{local_peaks, LocalPeaks};
use lumen_ops::filter::{gaussian_blur, median_filter, GaussianBlur, OrderFilter};
use lumen_ops::restore::{rolling_ball, RollingBall};
use lumen_ops::segment::{expand_labels, watershed, WatershedOptions};
use serde::{Deserialize, Serialize};

use crate::error::{invalid, HistResult};
use crate::objects::{HistObject, ObjectKind, Point, Polygon};
use crate::{polygon, raster};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellDetection {
    /// Rolling-ball radius of the background subtracted (µm; 0: none).
    pub background_radius_um: f64,
    /// Median filter radius (µm; 0: none).
    pub median_radius_um: f64,
    /// Gaussian smoothing (µm).
    pub sigma_um: f64,
    /// Nuclei kept (µm²).
    pub min_area_um2: f64,
    pub max_area_um2: f64,
    /// Of the smoothed detection channel (an optical density, or the channel's units).
    pub threshold: f64,
    /// Touching nuclei split by their shape.
    pub split_by_shape: bool,
    /// Split at the dark gaps between nuclei instead: seeds at each nucleus' brightest point
    /// (of the smoothed image), boundaries along the valleys between them. Better in dense
    /// tissue, where the threshold leaves nuclei joined in large pieces.
    #[serde(default = "yes")]
    pub split_by_intensity: bool,
    /// How far a nucleus grows into its cell (µm; 0: nuclei only).
    pub cell_expansion_um: f64,
}

impl Default for CellDetection {
    fn default() -> Self {
        Self { background_radius_um: 8.0, median_radius_um: 0.0, sigma_um: 1.5, min_area_um2: 10.0, max_area_um2: 400.0, threshold: 0.1, split_by_shape: true, split_by_intensity: true, cell_expansion_um: 5.0 }
    }
}

fn yes() -> bool {
    true
}

/// A region to detect cells in: the channel nuclei are bright in, and the channels measured
/// (name, values), all `width × height`, at `downsample` level-0 pixels per pixel from `origin`
/// (level-0 pixels), `pixel_um` µm per level-0 pixel.
pub struct CellImage<'a> {
    pub width: usize,
    pub height: usize,
    pub detection: &'a [f32],
    pub channels: Vec<(String, &'a [f32])>,
    pub origin: Point,
    pub downsample: f64,
    pub pixel_um: f64,
}

impl CellImage<'_> {
    fn um_per_px(&self) -> f64 {
        self.pixel_um * self.downsample
    }

    fn check(&self) -> HistResult<()> {
        let n = self.width * self.height;
        if n == 0 {
            return Err(invalid("an empty region"));
        }
        if self.detection.len() != n || self.channels.iter().any(|(_, c)| c.len() != n) {
            return Err(invalid(format!("every channel must hold {} × {} values", self.width, self.height)));
        }
        if !(self.pixel_um > 0.0 && self.downsample > 0.0) {
            return Err(invalid("the pixel size must be known (µm)"));
        }
        Ok(())
    }
}

/// Cells found: the objects, and the nucleus and cell labels (`1..=n`, object `id` = label).
#[derive(Debug, Clone)]
pub struct DetectedCells {
    pub objects: Vec<HistObject>,
    pub nuclei: Vec<u32>,
    pub cells: Vec<u32>,
}

/// Finds and measures the cells of `img`.
pub fn detect_cells(client: &Client, img: &CellImage, s: &CellDetection) -> HistResult<DetectedCells> {
    img.check()?;
    let (w, h, px) = (img.width, img.height, img.um_per_px());
    let shape = Shape::new([h, w]);
    let mut values = img.detection.to_vec();
    if s.background_radius_um > 0.0 {
        let dev = DeviceArray::from_host(client, shape.clone(), values.clone())?;
        let bg = rolling_ball(client, &dev, &RollingBall { radius: s.background_radius_um / px, ..RollingBall::default() })?.to_host(client);
        for (v, b) in values.iter_mut().zip(bg) {
            *v -= b;
        }
    }
    let mut dev = DeviceArray::from_host(client, shape.clone(), values)?;
    let median_px = (s.median_radius_um / px).round() as usize;
    if median_px >= 1 {
        dev = median_filter(client, &dev, &StructuringElement::disk(median_px), &OrderFilter::default())?;
    }
    if s.sigma_um > 0.0 {
        let sg = s.sigma_um / px;
        dev = gaussian_blur(client, &dev, &[sg, sg], &GaussianBlur::default())?;
    }
    let smooth = dev.to_host(client);
    let mask: Vec<u32> = smooth.iter().map(|&v| u32::from(v as f64 > s.threshold)).collect();
    let nuclei = if !mask.iter().any(|&m| m != 0) {
        vec![0u32; w * h]
    } else if s.split_by_intensity {
        // Seeds about a typical nucleus' radius apart (from the area range's geometric mean)
        let typical_radius_px = ((s.min_area_um2 * s.max_area_um2).sqrt() / std::f64::consts::PI).sqrt() / px;
        split_by_intensity(client, &smooth, &mask, w, h, typical_radius_px, s.threshold)?
    } else if s.split_by_shape {
        let min_radius_px = (s.min_area_um2 / std::f64::consts::PI).sqrt() / px;
        split_by_shape(client, &mask, w, h, min_radius_px)?
    } else {
        let m8: Vec<u8> = mask.iter().map(|&m| m as u8).collect();
        raster::components(&m8, w, h).0
    };
    cells_from_nuclei(client, img, &nuclei, s)
}

/// Touching nuclei apart at the dark gaps between them: the peaks of the smoothed image inside
/// the mask (at least `0.6 ×` a typical radius apart, above the threshold) seed a watershed of the
/// inverted image inside the mask.
fn split_by_intensity(client: &Client, smooth: &[f32], mask: &[u32], w: usize, h: usize, typical_radius_px: f64, threshold: f64) -> HistResult<Vec<u32>> {
    let shape = Shape::new([h, w]);
    let inside: Vec<f32> = smooth.iter().zip(mask).map(|(&v, &m)| if m != 0 { v } else { f32::MIN }).collect();
    let dev = DeviceArray::from_host(client, shape.clone(), inside)?;
    let peaks = local_peaks(client, &dev, &LocalPeaks { min_distance: (typical_radius_px * 0.6).round().max(1.0) as usize, threshold_abs: Some(threshold), exclude_border: Some(0), ..LocalPeaks::default() })?;
    let mut markers = vec![0u32; w * h];
    let mut n = 0u32;
    for p in &peaks {
        let i = p[0] * w + p[1];
        if mask[i] != 0 && markers[i] == 0 {
            n += 1;
            markers[i] = n;
        }
    }
    if n == 0 {
        let m8: Vec<u8> = mask.iter().map(|&m| m as u8).collect();
        return Ok(raster::components(&m8, w, h).0);
    }
    let relief: Vec<f32> = smooth.iter().map(|v| -v).collect();
    let relief = DeviceArray::from_host(client, shape.clone(), relief)?;
    let markers = DeviceArray::from_host(client, shape.clone(), markers)?;
    let mask_dev = DeviceArray::from_host(client, shape, mask.to_vec())?;
    let mut labels = watershed(client, &relief, &markers, None, Some(&mask_dev), &WatershedOptions::default())?.to_host(client);
    // Pieces of the mask without a seed keep their own label
    let m8: Vec<u8> = mask.iter().map(|&m| m as u8).collect();
    let (pieces, k) = raster::components(&m8, w, h);
    let mut covered = vec![false; k as usize + 1];
    for (i, &p) in pieces.iter().enumerate() {
        if p != 0 && labels[i] != 0 {
            covered[p as usize] = true;
        }
    }
    let mut next = labels.iter().copied().max().unwrap_or(0);
    let mut new_label = vec![0u32; k as usize + 1];
    for (i, &p) in pieces.iter().enumerate() {
        if p != 0 && !covered[p as usize] {
            if new_label[p as usize] == 0 {
                next += 1;
                new_label[p as usize] = next;
            }
            labels[i] = new_label[p as usize];
        }
    }
    Ok(labels)
}

/// Touching nuclei apart: the peaks of the distance to the background (at least the smallest
/// nucleus' radius apart) seed a watershed of the inverted distance inside the mask; pieces
/// without a peak keep their own label.
fn split_by_shape(client: &Client, mask: &[u32], w: usize, h: usize, min_radius_px: f64) -> HistResult<Vec<u32>> {
    let shape = Shape::new([h, w]);
    let mask_dev = DeviceArray::from_host(client, shape.clone(), mask.to_vec())?;
    let dist: DeviceArray<f32> = euclidean_distance::<u32, f32>(client, &mask_dev, None)?;
    let dist_host = raster::gaussian(&dist.to_host(client), w, h, 1.0);
    let dist_dev = DeviceArray::from_host(client, shape.clone(), dist_host.clone())?;
    let peaks = local_peaks(client, &dist_dev, &LocalPeaks { min_distance: (min_radius_px * 0.8).round().max(1.0) as usize, threshold_abs: Some(0.5), exclude_border: Some(0), ..LocalPeaks::default() })?;
    let mut markers = vec![0u32; w * h];
    let mut n = 0u32;
    for p in &peaks {
        let (y, x) = (p[0], p[1]);
        if mask[y * w + x] != 0 && markers[y * w + x] == 0 {
            n += 1;
            markers[y * w + x] = n;
        }
    }
    let mut labels = if n == 0 {
        vec![0u32; w * h]
    } else {
        let relief: Vec<f32> = dist_host.iter().map(|d| -d).collect();
        let relief = DeviceArray::from_host(client, shape.clone(), relief)?;
        let markers = DeviceArray::from_host(client, shape.clone(), markers)?;
        watershed(client, &relief, &markers, None, Some(&mask_dev), &WatershedOptions::default())?.to_host(client)
    };
    // Pieces of the mask the watershed left out (no peak in them)
    let m8: Vec<u8> = mask.iter().map(|&m| m as u8).collect();
    let (pieces, k) = raster::components(&m8, w, h);
    let mut covered = vec![false; k as usize + 1];
    for (i, &p) in pieces.iter().enumerate() {
        if p != 0 && labels[i] != 0 {
            covered[p as usize] = true;
        }
    }
    let mut next = labels.iter().copied().max().unwrap_or(0);
    let mut new_label = vec![0u32; k as usize + 1];
    for (i, &p) in pieces.iter().enumerate() {
        if p != 0 && !covered[p as usize] {
            if new_label[p as usize] == 0 {
                next += 1;
                new_label[p as usize] = next;
            }
            labels[i] = new_label[p as usize];
        }
    }
    Ok(labels)
}

/// Cells from nucleus labels (`0`: none; any ids), from [`detect_cells`] or from a model: out
/// of the area range they go, the rest is relabelled `1..=n`, expanded into cells and measured.
pub fn cells_from_nuclei(client: &Client, img: &CellImage, nuclei: &[u32], s: &CellDetection) -> HistResult<DetectedCells> {
    img.check()?;
    let (w, h, px) = (img.width, img.height, img.um_per_px());
    if nuclei.len() != w * h {
        return Err(invalid("the nucleus labels do not cover the region"));
    }
    // Areas, then the kept labels renumbered
    let max = nuclei.iter().copied().max().unwrap_or(0) as usize;
    let mut count = vec![0usize; max + 1];
    for &l in nuclei {
        count[l as usize] += 1;
    }
    let px_area = px * px;
    let mut map = vec![0u32; max + 1];
    let mut n = 0u32;
    for l in 1..=max {
        let a = count[l] as f64 * px_area;
        if count[l] > 0 && a >= s.min_area_um2 && a <= s.max_area_um2 {
            n += 1;
            map[l] = n;
        }
    }
    let nuclei: Vec<u32> = nuclei.iter().map(|&l| map[l as usize]).collect();
    let cells = if s.cell_expansion_um > 0.0 && n > 0 {
        let dev = DeviceArray::from_host(client, Shape::new([h, w]), nuclei.clone())?;
        expand_labels(client, &dev, s.cell_expansion_um / px, None)?.to_host(client)
    } else {
        nuclei.clone()
    };
    let objects = measure(img, &nuclei, &cells, n, s.cell_expansion_um > 0.0);
    Ok(DetectedCells { objects, nuclei, cells })
}

#[derive(Clone, Copy)]
struct Acc {
    n: f64,
    sum: f64,
    sq: f64,
    min: f64,
    max: f64,
}

impl Default for Acc {
    fn default() -> Self {
        Self { n: 0.0, sum: 0.0, sq: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY }
    }
}

impl Acc {
    fn add(&mut self, v: f64) {
        self.n += 1.0;
        self.sum += v;
        self.sq += v * v;
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }
}

/// The outline through the midpoints of its unit pixel edges (`ds` level-0 pixels each): every
/// step of the pixel staircase becomes a diagonal, so a disk's perimeter and circularity come
/// out right (the usual digital-perimeter estimate); straight runs stay straight.
fn smooth(p: &Polygon, ds: f64) -> Polygon {
    let n = p.len();
    let mut unit: Vec<Point> = Vec::new();
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        let len = ((b[0] - a[0]).abs() + (b[1] - a[1]).abs()) / ds;
        let steps = len.round().max(1.0) as usize;
        for k in 0..steps {
            let t = k as f64 / steps as f64;
            unit.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    }
    let m = unit.len();
    let mids: Polygon = (0..m).map(|i| [(unit[i][0] + unit[(i + 1) % m][0]) / 2.0, (unit[i][1] + unit[(i + 1) % m][1]) / 2.0]).collect();
    let k = mids.len();
    (0..k)
        .filter(|&i| {
            let (a, c, b) = (mids[(i + k - 1) % k], mids[i], mids[(i + 1) % k]);
            ((c[0] - a[0]) * (b[1] - c[1]) - (c[1] - a[1]) * (b[0] - c[0])).abs() > 1e-9
        })
        .map(|i| mids[i])
        .collect()
}

const COMPARTMENTS: [&str; 3] = ["Nucleus", "Cytoplasm", "Cell"];

fn measure(img: &CellImage, nuclei: &[u32], cells: &[u32], n: u32, expanded: bool) -> Vec<HistObject> {
    let (w, h) = (img.width, img.height);
    let nl = n as usize + 1;
    let nc = img.channels.len();
    // [label][channel][compartment]
    let mut acc = vec![[Acc::default(); 3]; nl * nc.max(1)];
    // Nucleus pixel moments: count, Σx, Σy, Σxx, Σyy, Σxy (pixels)
    let mut mom = vec![[0.0f64; 6]; nl];
    let mut cell_px = vec![0.0f64; nl];
    for i in 0..w * h {
        let (nu, ce) = (nuclei[i] as usize, cells[i] as usize);
        cell_px[ce] += 1.0;
        if nu != 0 {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let m = &mut mom[nu];
            m[0] += 1.0;
            m[1] += x;
            m[2] += y;
            m[3] += x * x;
            m[4] += y * y;
            m[5] += x * y;
        }
        for (c, (_, ch)) in img.channels.iter().enumerate() {
            let v = ch[i] as f64;
            if nu != 0 {
                acc[nu * nc + c][0].add(v);
            }
            if ce != 0 {
                if nu == 0 {
                    acc[ce * nc + c][1].add(v);
                }
                acc[ce * nc + c][2].add(v);
            }
        }
    }
    let nucleus_polys: std::collections::HashMap<u32, Polygon> = polygon::trace_labels(nuclei, w, h, img.origin, img.downsample).into_iter().map(|(l, p)| (l, smooth(&p, img.downsample))).collect();
    let cell_polys: std::collections::HashMap<u32, Polygon> = if expanded { polygon::trace_labels(cells, w, h, img.origin, img.downsample).into_iter().map(|(l, p)| (l, smooth(&p, img.downsample))).collect() } else { Default::default() };
    let um = img.pixel_um;
    let px_area = img.um_per_px().powi(2);
    let mut out = Vec::with_capacity(n as usize);
    for l in 1..=n {
        let Some(nuc) = nucleus_polys.get(&l).cloned() else { continue };
        let li = l as usize;
        let m = mom[li];
        let mut o = HistObject::new(l, if expanded { ObjectKind::Cell } else { ObjectKind::Detection }, cell_polys.get(&l).cloned().unwrap_or_else(|| nuc.clone()));
        // The centroid is the nucleus'
        o.centroid = [img.origin[0] + (m[1] / m[0] + 0.5) * img.downsample, img.origin[1] + (m[2] / m[0] + 0.5) * img.downsample];
        let nuc_area = m[0] * px_area;
        let nuc_perim = polygon::perimeter(&nuc) * um;
        let mm = &mut o.measurements;
        mm.insert("Nucleus: Area µm^2".into(), nuc_area);
        mm.insert("Nucleus: Perimeter µm".into(), nuc_perim);
        mm.insert("Nucleus: Circularity".into(), (4.0 * std::f64::consts::PI * nuc_area / (nuc_perim * nuc_perim)).min(1.0));
        let (mx, my) = (m[1] / m[0], m[2] / m[0]);
        let (sxx, syy, sxy) = (m[3] / m[0] - mx * mx, m[4] / m[0] - my * my, m[5] / m[0] - mx * my);
        let tr = sxx + syy;
        let det = (((sxx - syy) / 2.0).powi(2) + sxy * sxy).sqrt();
        let (l1, l2) = (tr / 2.0 + det, (tr / 2.0 - det).max(0.0));
        mm.insert("Nucleus: Eccentricity".into(), if l1 > 0.0 { (1.0 - l2 / l1).max(0.0).sqrt() } else { 0.0 });
        if expanded {
            let cell_area = cell_px[li] * px_area;
            mm.insert("Cell: Area µm^2".into(), cell_area);
            mm.insert("Cell: Perimeter µm".into(), polygon::perimeter(&o.roi) * um);
            mm.insert("Nucleus/Cell area ratio".into(), if cell_area > 0.0 { (nuc_area / cell_area).min(1.0) } else { 1.0 });
        }
        for (c, (name, _)) in img.channels.iter().enumerate() {
            for (k, comp) in COMPARTMENTS.iter().enumerate() {
                if !expanded && k > 0 {
                    continue;
                }
                let a = acc[li * nc + c][k];
                if a.n == 0.0 {
                    continue;
                }
                let mean = a.sum / a.n;
                mm.insert(format!("{comp}: {name} mean"), mean);
                mm.insert(format!("{comp}: {name} std dev"), (a.sq / a.n - mean * mean).max(0.0).sqrt());
                mm.insert(format!("{comp}: {name} min"), a.min);
                mm.insert(format!("{comp}: {name} max"), a.max);
            }
        }
        o.nucleus = Some(nuc);
        out.push(o);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Five disks (two of them touching) of a nuclear stain on a dark background.
    fn image() -> (usize, usize, Vec<f32>) {
        let (w, h) = (96, 64);
        let disks = [(16.0, 16.0, 6.0), (48.0, 16.0, 7.0), (78.0, 40.0, 6.0), (30.0, 46.0, 7.0), (41.0, 46.0, 7.0)];
        let mut v = vec![0.02f32; w * h];
        for y in 0..h {
            for x in 0..w {
                for (cx, cy, r) in disks {
                    if (x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2) <= r * r {
                        v[y * w + x] = 1.0;
                    }
                }
            }
        }
        (w, h, v)
    }

    fn finds_and_measures_cells(client: &Client) {
        let (w, h, v) = image();
        let img = CellImage { width: w, height: h, detection: &v, channels: vec![("DAPI".into(), &v)], origin: [100.0, 200.0], downsample: 1.0, pixel_um: 0.5 };
        let s = CellDetection { background_radius_um: 0.0, sigma_um: 0.5, threshold: 0.5, min_area_um2: 5.0, max_area_um2: 100.0, cell_expansion_um: 2.0, ..CellDetection::default() };
        let found = detect_cells(client, &img, &s).unwrap();
        assert_eq!(found.objects.len(), 5, "{}: {:?}", client.name(), found.objects.iter().map(|o| o.get("Nucleus: Area µm^2")).collect::<Vec<_>>());
        for o in &found.objects {
            let a = o.get("Nucleus: Area µm^2").unwrap();
            assert!(a > 25.0 && a < 45.0, "{}: area {a}", client.name());
            assert!(o.get("Nucleus: Circularity").unwrap() > 0.8, "{}: {:?}", client.name(), o.measurements);
            assert!(o.get("Cell: Area µm^2").unwrap() > a);
            assert!((o.get("Nucleus: DAPI mean").unwrap() - 1.0).abs() < 0.05);
            assert!(o.get("Cytoplasm: DAPI mean").unwrap() < 0.5);
            assert!(o.centroid[0] > 100.0 && o.centroid[1] > 200.0);
        }
        // Cells from labels given (a model's)
        let none = cells_from_nuclei(client, &img, &found.nuclei, &CellDetection { cell_expansion_um: 0.0, ..s }).unwrap();
        assert_eq!(none.objects.len(), 5);
        assert!(none.objects.iter().all(|o| o.kind == ObjectKind::Detection && o.get("Cell: Area µm^2").is_none()));
    }

    lumen_core::runtime_test!(cells_are_found_and_measured, finds_and_measures_cells);
}

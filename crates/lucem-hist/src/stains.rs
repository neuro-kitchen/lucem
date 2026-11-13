//! Brightfield stains: optical density (Beer–Lambert), stain vectors and colour deconvolution.
//!
//! A pixel's optical density per channel is `-log10(I / I₀)` with `I₀` the background (white)
//! value. Each stain has a unit OD vector (how much it absorbs in red, green and blue); three
//! vectors form a matrix whose inverse separates a pixel's OD into the amount of each stain.
//! Presets are Ruifrok & Johnston's (2001); [`estimate_macenko`] fits two stains to a region of
//! the slide itself (Macenko et al., 2009), which follows the actual staining better.

use serde::{Deserialize, Serialize};

use crate::error::{invalid, HistResult};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StainVector {
    pub name: String,
    /// Unit optical density in red, green, blue.
    pub od: [f64; 3],
}

impl StainVector {
    pub fn new(name: &str, od: [f64; 3]) -> Self {
        Self { name: name.into(), od: normalize(od) }
    }
}

/// Three stains (the third may be the residual: what the first two do not explain) and the
/// background RGB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StainVectors {
    pub name: String,
    pub stains: [StainVector; 3],
    pub background: [f64; 3],
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n <= 0.0 { v } else { [v[0] / n, v[1] / n, v[2] / n] }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

const HEMATOXYLIN: [f64; 3] = [0.650, 0.704, 0.286];
const EOSIN: [f64; 3] = [0.072, 0.990, 0.105];
const DAB: [f64; 3] = [0.268, 0.570, 0.776];

impl StainVectors {
    /// Two stains and the residual (their cross product).
    pub fn two(name: &str, a: StainVector, b: StainVector, background: [f64; 3]) -> Self {
        let r = StainVector::new("Residual", cross(a.od, b.od));
        Self { name: name.into(), stains: [a, b, r], background }
    }

    /// Hematoxylin and DAB (immunohistochemistry).
    pub fn h_dab() -> Self {
        Self::two("H-DAB", StainVector::new("Hematoxylin", HEMATOXYLIN), StainVector::new("DAB", DAB), [255.0; 3])
    }

    /// Hematoxylin and eosin.
    pub fn h_e() -> Self {
        Self::two("H&E", StainVector::new("Hematoxylin", HEMATOXYLIN), StainVector::new("Eosin", EOSIN), [255.0; 3])
    }

    /// Hematoxylin, eosin and DAB.
    pub fn h_e_dab() -> Self {
        Self { name: "H-E-DAB".into(), stains: [StainVector::new("Hematoxylin", HEMATOXYLIN), StainVector::new("Eosin", EOSIN), StainVector::new("DAB", DAB)], background: [255.0; 3] }
    }

    /// The matrix that turns a pixel's OD into stain amounts (rows: stains).
    fn unmixing(&self) -> HistResult<[[f64; 3]; 3]> {
        // Columns of M are the stain vectors: od = M · amounts
        let m = [[self.stains[0].od[0], self.stains[1].od[0], self.stains[2].od[0]], [self.stains[0].od[1], self.stains[1].od[1], self.stains[2].od[1]], [self.stains[0].od[2], self.stains[1].od[2], self.stains[2].od[2]]];
        invert(m).ok_or_else(|| invalid(format!("the stain vectors of {} are not independent", self.name)))
    }
}

fn invert(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    Some([
        [c(1, 1, 2, 2) / det, -c(0, 1, 2, 2) / det, c(0, 1, 1, 2) / det],
        [-c(1, 0, 2, 2) / det, c(0, 0, 2, 2) / det, -c(0, 0, 1, 2) / det],
        [c(1, 0, 2, 1) / det, -c(0, 0, 2, 1) / det, c(0, 0, 1, 1) / det],
    ])
}

/// The optical density of a value against its background (0 at the background, larger darker).
pub fn od(value: f64, background: f64) -> f64 {
    -(value.max(1.0) / background.max(1.0)).log10()
}

/// Separates interleaved RGB pixels (0–255) into the amount of each stain (optical density
/// units): three planes, in the order of `stains.stains`.
pub fn deconvolve(rgb: &[u8], stains: &StainVectors) -> HistResult<[Vec<f32>; 3]> {
    let inv = stains.unmixing()?;
    let n = rgb.len() / 3;
    let mut out = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    let bg = stains.background;
    for (i, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
        let o = [od(p[0] as f64, bg[0]), od(p[1] as f64, bg[1]), od(p[2] as f64, bg[2])];
        for (s, plane) in out.iter_mut().enumerate() {
            plane[i] = (inv[s][0] * o[0] + inv[s][1] * o[1] + inv[s][2] * o[2]) as f32;
        }
    }
    Ok(out)
}

/// The summed optical density of each pixel (how much any stain is there: tissue detection,
/// "OD sum" measurements).
pub fn od_sum(rgb: &[u8], background: [f64; 3]) -> Vec<f32> {
    rgb.as_chunks::<3>().0.iter().map(|p| (od(p[0] as f64, background[0]) + od(p[1] as f64, background[1]) + od(p[2] as f64, background[2])) as f32).collect()
}

/// The background (white) of a brightfield region: the mean of its brightest 1 % of pixels.
pub fn estimate_background(rgb: &[u8]) -> [f64; 3] {
    let mut px: Vec<[u8; 3]> = rgb.as_chunks::<3>().0.iter().map(|p| [p[0], p[1], p[2]]).collect();
    if px.is_empty() {
        return [255.0; 3];
    }
    px.sort_by_key(|p| std::cmp::Reverse(p[0] as u32 + p[1] as u32 + p[2] as u32));
    let k = (px.len() / 100).max(1);
    let s = px[..k].iter().fold([0.0; 3], |s, p| [s[0] + p[0] as f64, s[1] + p[1] as f64, s[2] + p[2] as f64]);
    [s[0] / k as f64, s[1] / k as f64, s[2] / k as f64]
}

/// Eigenvectors of a symmetric 3 × 3 matrix (Jacobi), largest eigenvalue first.
#[allow(clippy::needless_range_loop)] // matrix index arithmetic
fn eigen3(mut a: [[f64; 3]; 3]) -> [([f64; 3], f64); 3] {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        let (mut p, mut q, mut big) = (0, 1, 0.0);
        for i in 0..3 {
            for j in i + 1..3 {
                if a[i][j].abs() > big {
                    (p, q, big) = (i, j, a[i][j].abs());
                }
            }
        }
        if big < 1e-15 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for row in v.iter_mut() {
            let (vp, vq) = (row[p], row[q]);
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }
    let mut out: Vec<([f64; 3], f64)> = (0..3).map(|k| ([v[0][k], v[1][k], v[2][k]], a[k][k])).collect();
    out.sort_by(|x, y| y.1.total_cmp(&x.1));
    [out[0], out[1], out[2]]
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p / 100.0).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

/// Two stains fitted to a region (Macenko): the optical densities of its stained pixels (OD
/// above `min_od` in every channel) span a plane; the stains are its extreme directions (the
/// `alpha` and `100 − alpha` percentiles of their angles). The first stain is the one that
/// absorbs more red (hematoxylin before eosin or DAB). Fewer than 100 stained pixels: an error.
#[allow(clippy::needless_range_loop)] // the covariance's index arithmetic
pub fn estimate_macenko(rgb: &[u8], background: [f64; 3], names: (&str, &str), min_od: f64, alpha: f64) -> HistResult<StainVectors> {
    let ods: Vec<[f64; 3]> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| [od(p[0] as f64, background[0]), od(p[1] as f64, background[1]), od(p[2] as f64, background[2])])
        .filter(|o| o.iter().all(|&v| v > min_od && v.is_finite()))
        .collect();
    if ods.len() < 100 {
        return Err(invalid(format!("only {} stained pixels in the region: choose a region with more staining", ods.len())));
    }
    let n = ods.len() as f64;
    let mean = ods.iter().fold([0.0; 3], |s, o| [s[0] + o[0] / n, s[1] + o[1] / n, s[2] + o[2] / n]);
    let mut cov = [[0.0; 3]; 3];
    for o in &ods {
        let d = [o[0] - mean[0], o[1] - mean[1], o[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j] / n;
            }
        }
    }
    let e = eigen3(cov);
    let (mut e1, mut e2) = (e[0].0, e[1].0);
    // Point the plane's axes into positive OD
    if e1.iter().sum::<f64>() < 0.0 {
        e1 = [-e1[0], -e1[1], -e1[2]];
    }
    if e2.iter().sum::<f64>() < 0.0 {
        e2 = [-e2[0], -e2[1], -e2[2]];
    }
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut angles: Vec<f64> = ods.iter().map(|o| dot(*o, e2).atan2(dot(*o, e1))).collect();
    angles.sort_by(f64::total_cmp);
    let (lo, hi) = (percentile(&angles, alpha), percentile(&angles, 100.0 - alpha));
    let at = |a: f64| normalize([e1[0] * a.cos() + e2[0] * a.sin(), e1[1] * a.cos() + e2[1] * a.sin(), e1[2] * a.cos() + e2[2] * a.sin()]);
    let (mut s1, mut s2) = (at(lo), at(hi));
    if s1[0] < s2[0] {
        std::mem::swap(&mut s1, &mut s2);
    }
    Ok(StainVectors::two("Estimated", StainVector::new(names.0, s1), StainVector::new(names.1, s2), background))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pixel made of `a` of the first stain and `b` of the second, over a white background.
    fn pixel(s: &StainVectors, a: f64, b: f64) -> [u8; 3] {
        let o: Vec<f64> = (0..3).map(|c| a * s.stains[0].od[c] + b * s.stains[1].od[c]).collect();
        [0, 1, 2].map(|c| (255.0 * 10f64.powf(-o[c])).round().clamp(0.0, 255.0) as u8)
    }

    #[test]
    fn deconvolution_returns_the_stain_amounts() {
        let s = StainVectors::h_dab();
        let rgb: Vec<u8> = [(0.8, 0.0), (0.0, 0.6), (0.4, 0.3)].iter().flat_map(|&(a, b)| pixel(&s, a, b)).collect();
        let [h, d, r] = deconvolve(&rgb, &s).unwrap();
        for (i, (a, b)) in [(0.8, 0.0), (0.0, 0.6), (0.4, 0.3)].into_iter().enumerate() {
            assert!((h[i] as f64 - a).abs() < 0.03 && (d[i] as f64 - b).abs() < 0.03, "pixel {i}: H {} DAB {} residual {}", h[i], d[i], r[i]);
        }
    }

    #[test]
    fn macenko_finds_the_two_stains_of_a_region() {
        let truth = StainVectors::h_e();
        let mut rgb = Vec::new();
        for i in 0..400 {
            let (a, b) = ((i % 20) as f64 / 20.0 * 1.2 + 0.2, (i / 20) as f64 / 20.0 * 0.9 + 0.2);
            rgb.extend(pixel(&truth, a, 0.0));
            rgb.extend(pixel(&truth, 0.0, b));
            rgb.extend(pixel(&truth, a * 0.5, b * 0.5));
        }
        let est = estimate_macenko(&rgb, [255.0; 3], ("Hematoxylin", "Eosin"), 0.05, 1.0).unwrap();
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        assert!(dot(est.stains[0].od, truth.stains[0].od) > 0.98, "{:?}", est.stains[0].od);
        assert!(dot(est.stains[1].od, truth.stains[1].od) > 0.98, "{:?}", est.stains[1].od);
        let bg = estimate_background(&[250, 251, 252, 10, 10, 10, 249, 250, 251]);
        assert!(bg[0] > 240.0);
    }
}

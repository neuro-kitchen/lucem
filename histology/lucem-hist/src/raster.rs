//! Small raster helpers on the host, for images already reduced to an analysis resolution
//! (tissue at a coarse level, density maps, class maps): smoothing, Otsu, connected components,
//! small pieces and holes.

use std::collections::VecDeque;

use crate::objects::Point;

/// Where a raster lies on the slide: its size, the level-0 position of its first pixel and how
/// many level-0 pixels one of its pixels covers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub origin: Point,
    pub downsample: f64,
}

impl Grid {
    pub fn new(width: usize, height: usize, origin: Point, downsample: f64) -> Self {
        Self { width, height, origin, downsample }
    }

    pub fn len(&self) -> usize {
        self.width * self.height
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A separable Gaussian blur (σ in pixels; edges repeat the border).
pub fn gaussian(values: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 || w == 0 || h == 0 {
        return values.to_vec();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f64> = (-r..=r).map(|i| (-(i as f64).powi(2) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f64 = k.iter().sum();
    let k: Vec<f64> = k.iter().map(|v| v / sum).collect();
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0f32; src.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (t, kv) in k.iter().enumerate() {
                    let d = t as isize - r;
                    let (sx, sy) = if horizontal { ((x as isize + d).clamp(0, w as isize - 1) as usize, y) } else { (x, (y as isize + d).clamp(0, h as isize - 1) as usize) };
                    acc += kv * src[sy * w + sx] as f64;
                }
                out[y * w + x] = acc as f32;
            }
        }
        out
    };
    pass(&pass(values, true), false)
}

/// Otsu's threshold of finite values (256 bins between their min and max).
pub fn otsu(values: &[f32]) -> f32 {
    let (lo, hi) = values.iter().filter(|v| v.is_finite()).fold((f32::MAX, f32::MIN), |b, &v| (b.0.min(v), b.1.max(v)));
    if lo >= hi {
        return lo;
    }
    let mut hist = [0u64; 256];
    for &v in values.iter().filter(|v| v.is_finite()) {
        hist[(((v - lo) / (hi - lo)) * 255.0).round() as usize] += 1;
    }
    let total: u64 = hist.iter().sum();
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut w0, mut s0, mut best, mut best_t) = (0.0, 0.0, -1.0, 0usize);
    for (t, &c) in hist.iter().enumerate() {
        w0 += c as f64;
        s0 += t as f64 * c as f64;
        let w1 = total as f64 - w0;
        if w0 == 0.0 || w1 == 0.0 {
            continue;
        }
        let (m0, m1) = (s0 / w0, (sum_all - s0) / w1);
        let between = w0 * w1 * (m0 - m1).powi(2);
        if between > best {
            best = between;
            best_t = t;
        }
    }
    lo + (best_t as f32 + 0.5) / 255.0 * (hi - lo)
}

/// 8-connected components of the non-zero pixels: labels `1..=n` and `n`.
pub fn components(mask: &[u8], w: usize, h: usize) -> (Vec<u32>, u32) {
    let mut labels = vec![0u32; w * h];
    let mut n = 0u32;
    let mut queue = VecDeque::new();
    for start in 0..w * h {
        if mask[start] == 0 || labels[start] != 0 {
            continue;
        }
        n += 1;
        labels[start] = n;
        queue.push_back(start);
        while let Some(i) = queue.pop_front() {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] != 0 && labels[j] == 0 {
                        labels[j] = n;
                        queue.push_back(j);
                    }
                }
            }
        }
    }
    (labels, n)
}

/// Pixels per label (index 0: background).
pub fn label_areas(labels: &[u32], n: u32) -> Vec<usize> {
    let mut a = vec![0usize; n as usize + 1];
    for &l in labels {
        a[l as usize] += 1;
    }
    a
}

/// Drops pieces of `mask` smaller than `min_area` pixels and fills holes smaller than
/// `min_hole` pixels (background pieces not touching the border).
pub fn clean(mask: &mut [u8], w: usize, h: usize, min_area: f64, min_hole: f64) {
    if min_area > 0.0 {
        let (labels, n) = components(mask, w, h);
        let areas = label_areas(&labels, n);
        for (m, &l) in mask.iter_mut().zip(&labels) {
            if l != 0 && (areas[l as usize] as f64) < min_area {
                *m = 0;
            }
        }
    }
    if min_hole > 0.0 {
        let inverse: Vec<u8> = mask.iter().map(|&m| u8::from(m == 0)).collect();
        let (labels, n) = components(&inverse, w, h);
        let areas = label_areas(&labels, n);
        let mut border = vec![false; n as usize + 1];
        for x in 0..w {
            border[labels[x] as usize] = true;
            border[labels[(h - 1) * w + x] as usize] = true;
        }
        for y in 0..h {
            border[labels[y * w] as usize] = true;
            border[labels[y * w + w - 1] as usize] = true;
        }
        for (m, &l) in mask.iter_mut().zip(&labels) {
            if l != 0 && !border[l as usize] && (areas[l as usize] as f64) < min_hole {
                *m = 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_splits_two_levels() {
        let v: Vec<f32> = (0..100).map(|i| if i < 60 { 10.0 } else { 200.0 }).collect();
        let t = otsu(&v);
        assert!(t > 10.0 && t < 200.0, "{t}");
    }

    #[test]
    fn small_pieces_go_and_small_holes_fill() {
        let (w, h) = (12, 12);
        let mut m = vec![0u8; w * h];
        for y in 2..10 {
            for x in 2..10 {
                m[y * w + x] = 1;
            }
        }
        m[5 * w + 5] = 0; // a hole of 1 px
        m[0] = 1; // a speck
        clean(&mut m, w, h, 4.0, 4.0);
        assert_eq!(m[0], 0);
        assert_eq!(m[5 * w + 5], 1);
        assert_eq!(m.iter().filter(|&&v| v == 1).count(), 64);
        let s = gaussian(&[1.0; 25], 5, 5, 1.0);
        assert!(s.iter().all(|v| (v - 1.0).abs() < 1e-5));
    }
}

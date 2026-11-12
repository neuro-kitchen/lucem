//! Outlines: area, perimeter, centroid, point inside, and labels traced into outlines.

use std::collections::HashMap;

use crate::objects::{Point, Polygon};

/// Signed area (positive when counter-clockwise in a y-down image: clockwise on screen).
fn signed_area(p: &[Point]) -> f64 {
    let n = p.len();
    (0..n).map(|i| p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1]).sum::<f64>() / 2.0
}

pub fn area(p: &[Point]) -> f64 {
    signed_area(p).abs()
}

pub fn perimeter(p: &[Point]) -> f64 {
    let n = p.len();
    if n < 2 {
        return 0.0;
    }
    (0..n).map(|i| ((p[(i + 1) % n][0] - p[i][0]).powi(2) + (p[(i + 1) % n][1] - p[i][1]).powi(2)).sqrt()).sum()
}

/// The centroid of the enclosed area (the mean of the points for a degenerate outline).
pub fn centroid(p: &[Point]) -> Point {
    let a = signed_area(p);
    let n = p.len();
    if n == 0 {
        return [0.0, 0.0];
    }
    if a.abs() < 1e-12 {
        let (sx, sy) = p.iter().fold((0.0, 0.0), |s, q| (s.0 + q[0], s.1 + q[1]));
        return [sx / n as f64, sy / n as f64];
    }
    let (mut cx, mut cy) = (0.0, 0.0);
    for i in 0..n {
        let (a0, a1) = (p[i], p[(i + 1) % n]);
        let c = a0[0] * a1[1] - a1[0] * a0[1];
        cx += (a0[0] + a1[0]) * c;
        cy += (a0[1] + a1[1]) * c;
    }
    [cx / (6.0 * a), cy / (6.0 * a)]
}

/// Whether `q` is inside `p` (even-odd rule).
pub fn contains(p: &[Point], q: Point) -> bool {
    let n = p.len();
    let mut inside = false;
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (p[i], p[j]);
        if (a[1] > q[1]) != (b[1] > q[1]) && q[0] < (b[0] - a[0]) * (q[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// The bounding box `(x0, y0, x1, y1)`.
pub fn bounds(p: &[Point]) -> (f64, f64, f64, f64) {
    p.iter().fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, q| (b.0.min(q[0]), b.1.min(q[1]), b.2.max(q[0]), b.3.max(q[1])))
}

/// The outer outline of every label of a `w × h` raster (pixel corners, level-0 pixels:
/// `origin + corner · ds`), by label. Each pixel side between a label and anything else is an
/// edge; edges chain into loops and the largest loop of a label is its outline (holes are left
/// out). Points along straight runs are dropped.
pub fn trace_labels(labels: &[u32], w: usize, h: usize, origin: Point, ds: f64) -> Vec<(u32, Polygon)> {
    type Edges = HashMap<(i32, i32), Vec<(i32, i32)>>;
    let at = |x: i32, y: i32| if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 { 0 } else { labels[y as usize * w + x as usize] };
    let mut by_label: HashMap<u32, Edges> = HashMap::new();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let l = at(x, y);
            if l == 0 {
                continue;
            }
            let e = by_label.entry(l).or_default();
            // Clockwise around the pixel on screen (y down)
            if at(x, y - 1) != l {
                e.entry((x, y)).or_default().push((x + 1, y));
            }
            if at(x + 1, y) != l {
                e.entry((x + 1, y)).or_default().push((x + 1, y + 1));
            }
            if at(x, y + 1) != l {
                e.entry((x + 1, y + 1)).or_default().push((x, y + 1));
            }
            if at(x - 1, y) != l {
                e.entry((x, y + 1)).or_default().push((x, y));
            }
        }
    }
    let mut out: Vec<(u32, Polygon)> = Vec::with_capacity(by_label.len());
    for (label, mut edges) in by_label {
        let mut best: Vec<(i32, i32)> = Vec::new();
        let mut best_area = 0.0;
        while let Some(&start) = edges.keys().next() {
            let mut ring = vec![start];
            let mut cur = start;
            while let Some(nexts) = edges.get_mut(&cur) {
                let next = nexts.pop().expect("an edge list is never left empty");
                if nexts.is_empty() {
                    edges.remove(&cur);
                }
                if next == start {
                    break;
                }
                ring.push(next);
                cur = next;
            }
            let pts: Vec<Point> = ring.iter().map(|&(x, y)| [x as f64, y as f64]).collect();
            let a = area(&pts);
            if a > best_area {
                best_area = a;
                best = ring;
            }
        }
        // Only the corners
        let n = best.len();
        let corners: Polygon = (0..n)
            .filter(|&i| {
                let (p, c, q) = (best[(i + n - 1) % n], best[i], best[(i + 1) % n]);
                (c.0 - p.0) * (q.1 - c.1) != (c.1 - p.1) * (q.0 - c.0)
            })
            .map(|i| [origin[0] + best[i].0 as f64 * ds, origin[1] + best[i].1 as f64 * ds])
            .collect();
        if corners.len() >= 3 {
            out.push((label, corners));
        }
    }
    out.sort_by_key(|(l, _)| *l);
    out
}

/// Fills `p` (level-0 pixels) into a `w × h` raster of a region at `origin` with `ds` level-0
/// pixels per raster pixel: pixels whose centre is inside get `value`.
pub fn fill(p: &[Point], raster: &mut [u32], w: usize, h: usize, origin: Point, ds: f64, value: u32) {
    if p.len() < 3 {
        return;
    }
    let (x0, y0, x1, y1) = bounds(p);
    let to = |v: f64, o: f64| (v - o) / ds;
    let (ry0, ry1) = ((to(y0, origin[1]).floor().max(0.0)) as usize, (to(y1, origin[1]).ceil().max(0.0) as usize).min(h));
    let (rx0, rx1) = ((to(x0, origin[0]).floor().max(0.0)) as usize, (to(x1, origin[0]).ceil().max(0.0) as usize).min(w));
    for ry in ry0..ry1 {
        for rx in rx0..rx1 {
            let q = [origin[0] + (rx as f64 + 0.5) * ds, origin[1] + (ry as f64 + 0.5) * ds];
            if contains(p, q) {
                raster[ry * w + rx] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_label_traces_to_its_corners() {
        let (w, h) = (6, 5);
        let mut l = vec![0u32; w * h];
        for y in 1..4 {
            for x in 2..5 {
                l[y * w + x] = 7;
            }
        }
        let polys = trace_labels(&l, w, h, [10.0, 20.0], 2.0);
        assert_eq!(polys.len(), 1);
        let (label, p) = &polys[0];
        assert_eq!(*label, 7);
        assert_eq!(p.len(), 4, "{p:?}");
        assert!((area(p) - 9.0 * 4.0).abs() < 1e-9);
        assert!((perimeter(p) - 24.0).abs() < 1e-9);
        let c = centroid(p);
        assert!((c[0] - (10.0 + 3.5 * 2.0)).abs() < 1e-9 && (c[1] - (20.0 + 2.5 * 2.0)).abs() < 1e-9);
        assert!(contains(p, c) && !contains(p, [0.0, 0.0]));
        let mut r = vec![0u32; w * h];
        fill(p, &mut r, w, h, [10.0, 20.0], 2.0, 1);
        assert_eq!(r.iter().filter(|&&v| v == 1).count(), 9);
    }

    #[test]
    fn a_ring_keeps_its_outer_outline() {
        let (w, h) = (7, 7);
        let mut l = vec![1u32; w * h];
        l[3 * w + 3] = 0;
        let polys = trace_labels(&l, w, h, [0.0, 0.0], 1.0);
        assert_eq!(polys.len(), 1);
        assert!((area(&polys[0].1) - 49.0).abs() < 1e-9);
    }
}

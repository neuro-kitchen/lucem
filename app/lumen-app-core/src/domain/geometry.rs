//! Host geometry for the interactive tools: masks to polygons (the Threshold tool's "Create
//! annotations", the Wand brush), polygon simplification, a fast blur and a histogram. These run on
//! regions a person is looking at (≤ a few million px), so they stay on the CPU and answer at once.

/// Outlines of the objects of a mask (`width × height`, non-zero = in): one closed polygon per
/// 8-connected object (its outer boundary; holes are not traced), in pixel-corner coordinates.
/// Objects smaller than `min_area` px are skipped.
pub fn mask_to_polygons(mask: &[u8], width: usize, height: usize, min_area: usize) -> Vec<Vec<[f64; 2]>> {
    let (w, h) = (width as isize, height as isize);
    let at = |x: isize, y: isize| x >= 0 && y >= 0 && x < w && y < h && mask[(y * w + x) as usize] != 0;
    let mut seen = vec![false; width * height];
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            // Row-major order: an object's first unseen pixel has background above and left
            if mask[i] == 0 || seen[i] {
                continue;
            }
            // Flood the object (marks it, counts its area)
            let mut stack = vec![(x, y)];
            seen[i] = true;
            let mut area = 0usize;
            while let Some((cx, cy)) = stack.pop() {
                area += 1;
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)] {
                    let (nx, ny) = (cx + dx, cy + dy);
                    if at(nx, ny) {
                        let j = (ny * w + nx) as usize;
                        if !seen[j] {
                            seen[j] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            if area < min_area {
                continue;
            }
            let poly = trace(&at, x, y);
            if poly.len() >= 3 {
                out.push(poly);
            }
        }
    }
    out
}

/// The outer boundary of the object whose top-left-most pixel is `(x0, y0)`, walking pixel edges
/// with the object on the right (square tracing on the corner grid).
fn trace(at: &dyn Fn(isize, isize) -> bool, x0: isize, y0: isize) -> Vec<[f64; 2]> {
    // Corners: (x, y) is the top-left corner of pixel (x, y). Start at pixel (x0, y0)'s top-left
    // corner heading right (its top edge has background above).
    let (sx, sy) = (x0, y0);
    let (mut x, mut y) = (sx, sy);
    // Direction: 0 right, 1 down, 2 left, 3 up
    let mut d = 0usize;
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let limit = 4 * 1_000_000;
    for _ in 0..limit {
        pts.push([x as f64, y as f64]);
        // Pixels around the corner in the heading's frame: ahead-left and ahead-right
        let (al, ar) = match d {
            0 => (at(x, y - 1), at(x, y)),
            1 => (at(x, y), at(x - 1, y)),
            2 => (at(x - 1, y), at(x - 1, y - 1)),
            _ => (at(x - 1, y - 1), at(x, y - 1)),
        };
        // Keep the object on the right: turn left if ahead-left is in, go straight if only
        // ahead-right is in, else turn right
        d = if al { (d + 3) % 4 } else if ar { d } else { (d + 1) % 4 };
        match d {
            0 => x += 1,
            1 => y += 1,
            2 => x -= 1,
            _ => y -= 1,
        }
        if x == sx && y == sy {
            break;
        }
    }
    // Drop collinear corners
    let n = pts.len();
    if n < 3 {
        return pts;
    }
    (0..n)
        .filter(|&i| {
            let (a, b, c) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
            (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0.0
        })
        .map(|i| pts[i])
        .collect()
}

/// Ramer–Douglas–Peucker: drops points closer than `eps` to the simplified outline.
pub fn simplify(points: &[[f64; 2]], eps: f64) -> Vec<[f64; 2]> {
    if points.len() < 4 || eps <= 0.0 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0usize, points.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        let (pa, pb) = (points[a], points[b]);
        let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
        let len = dx.hypot(dy).max(1e-12);
        let (mut best, mut idx) = (0.0, 0);
        for (i, p) in points.iter().enumerate().take(b).skip(a + 1) {
            let d = ((p[0] - pa[0]) * dy - (p[1] - pa[1]) * dx).abs() / len;
            if d > best {
                best = d;
                idx = i;
            }
        }
        if best > eps {
            keep[idx] = true;
            stack.push((a, idx));
            stack.push((idx, b));
        }
    }
    points.iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| *p).collect()
}

/// Screen-space thinning: drops points closer than `min` to the last kept one (for drawing).
pub fn thin(points: &[(f32, f32)], min: f32) -> Vec<(f32, f32)> {
    let mut out: Vec<(f32, f32)> = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let last = i + 1 == points.len();
        match out.last() {
            Some(q) if !last && (p.0 - q.0).hypot(p.1 - q.1) < min => {}
            _ => out.push(*p),
        }
    }
    out
}

/// Approximate Gaussian blur (three box passes) of `v` (`w × h`), radius from `sigma`.
pub fn blur(v: &mut [f32], w: usize, h: usize, sigma: f64) {
    if sigma < 0.5 || w == 0 || h == 0 {
        return;
    }
    // Box width for 3 passes (Wells 1986)
    let r = ((((12.0 * sigma * sigma / 3.0) + 1.0).sqrt() - 1.0) / 2.0).round().max(1.0) as usize;
    let mut tmp = vec![0f32; v.len()];
    for _ in 0..3 {
        box_pass(v, &mut tmp, w, h, r, true);
        box_pass(&tmp, v, w, h, r, false);
    }
}

fn box_pass(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize, horizontal: bool) {
    let (n, m) = if horizontal { (h, w) } else { (w, h) };
    let idx = |line: usize, k: usize| if horizontal { line * w + k } else { k * w + line };
    let norm = 1.0 / (2 * r + 1) as f32;
    for line in 0..n {
        // Edge-clamped running sum
        let get = |k: isize| src[idx(line, k.clamp(0, m as isize - 1) as usize)];
        let mut s: f32 = (-(r as isize)..=r as isize).map(get).sum();
        for k in 0..m {
            dst[idx(line, k)] = s * norm;
            s += get(k as isize + r as isize + 1) - get(k as isize - r as isize);
        }
    }
}

/// A histogram of `v` over `[lo, hi]` in `bins` bins (values outside are clamped in).
pub fn histogram(v: &[f32], lo: f32, hi: f32, bins: usize) -> Vec<u32> {
    let mut out = vec![0u32; bins.max(1)];
    let span = (hi - lo).max(1e-12);
    for &x in v {
        if x.is_finite() {
            let i = (((x - lo) / span) * bins as f32).clamp(0.0, bins as f32 - 1.0) as usize;
            out[i] += 1;
        }
    }
    out
}

/// The Otsu threshold of a histogram over `[lo, hi]`.
pub fn otsu(hist: &[u32], lo: f32, hi: f32) -> f32 {
    let total: f64 = hist.iter().map(|&c| c as f64).sum();
    if total == 0.0 {
        return (lo + hi) / 2.0;
    }
    let n = hist.len();
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut w0, mut sum0, mut best, mut at) = (0.0, 0.0, -1.0, 0usize);
    for (i, &c) in hist.iter().enumerate() {
        w0 += c as f64;
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0.0 {
            break;
        }
        sum0 += i as f64 * c as f64;
        let (m0, m1) = (sum0 / w0, (sum_all - sum0) / w1);
        let between = w0 * w1 * (m0 - m1) * (m0 - m1);
        if between > best {
            best = between;
            at = i;
        }
    }
    lo + (at as f32 + 1.0) / n as f32 * (hi - lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_traces_to_four_corners() {
        let (w, h) = (6, 6);
        let mut m = vec![0u8; w * h];
        for y in 1..4 {
            for x in 2..5 {
                m[y * w + x] = 1;
            }
        }
        let polys = mask_to_polygons(&m, w, h, 1);
        assert_eq!(polys.len(), 1);
        let mut p = polys[0].clone();
        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(p, vec![[2.0, 1.0], [2.0, 4.0], [5.0, 1.0], [5.0, 4.0]]);
        assert_eq!(crate::domain::annotation::polygon_area(&polys[0]), 9.0);
    }

    #[test]
    fn two_objects_and_a_small_one() {
        let (w, h) = (10, 4);
        let mut m = vec![0u8; w * h];
        for x in 0..3 {
            m[w + x] = 1;
            m[2 * w + x] = 1;
        }
        m[w + 8] = 1;
        assert_eq!(mask_to_polygons(&m, w, h, 1).len(), 2);
        assert_eq!(mask_to_polygons(&m, w, h, 2).len(), 1);
    }

    #[test]
    fn otsu_splits_two_modes() {
        let mut v = vec![0.1f32; 500];
        v.extend(vec![0.8f32; 500]);
        let h = histogram(&v, 0.0, 1.0, 100);
        let t = otsu(&h, 0.0, 1.0);
        assert!(t > 0.1 && t <= 0.8, "{t}");
    }

    #[test]
    fn blur_keeps_the_mean_and_simplify_drops_points() {
        let mut v = vec![0.0f32; 100];
        v[50] = 100.0;
        blur(&mut v, 10, 10, 2.0);
        let s: f32 = v.iter().sum();
        assert!((s - 100.0).abs() < 25.0, "{s}");
        let line: Vec<[f64; 2]> = (0..10).map(|i| [i as f64, 0.0]).collect();
        assert_eq!(simplify(&line, 0.1).len(), 2);
    }
}

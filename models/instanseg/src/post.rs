//! From the network's output to instance labels, as InstanSeg's TorchScript `forward` does it
//! (`instanseg/utils/loss/instanseg_loss.py`, one image, nuclei only):
//!
//! 1. Embedding: `8 · (sigmoid(e) − 0.5)` plus a coordinate map (`linspace(0, side / 4)` along x
//!    and along y); sigma as the network gives it; the seed map `s / 15 + 0.5`.
//! 2. Seeds: local maxima of the seed map (a pixel that is the first maximum of the
//!    `(2d + 1)²` square around it, as max pooling with indices finds them) above
//!    `seed_threshold`.
//! 3. For each seed, a window of `2 · window` pixels around it (kept inside the image): for every
//!    pixel the embedding's difference from the seed's and its sigma go through the pixel
//!    classifier (4 → 5 → 5 → 1, ReLU, then sigmoid): how likely the pixel belongs to the seed.
//! 4. Fragments: only what is connected to the seed (8-connected flood fill of the pixels at
//!    `mask_threshold` or above) is kept, and enclosed holes are filled.
//! 5. Each pixel goes to the most probable seed that reaches `mask_threshold` there.
//! 6. Seeds whose pixels overlap by more than `overlap_threshold` of the smaller one are joined
//!    (connected components of the overlap graph; the smallest label names each).
//! 7. Instances under `min_size` pixels are dropped, and the rest numbered 1..=n.

use rayon::prelude::*;

/// The settings of the published models (`default_*` in the archive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub mask_threshold: f32,
    pub peak_distance: usize,
    pub seed_threshold: f32,
    pub overlap_threshold: f32,
    pub window_size: usize,
    pub cleanup_fragments: bool,
    /// Instances smaller than this (pixels) are dropped (`default_min_size`; InstanSeg applies it
    /// outside the archive's `forward`, and the published test outputs have it applied).
    pub min_size: usize,
}

/// The pixel classifier: three linear layers (`fc1`: 4 → 5, `fc2`: 5 → 5, `fc3`: 5 → 1).
#[derive(Debug, Clone)]
pub struct PixelClassifier {
    /// `(weight [out × in], bias [out])` per layer.
    pub layers: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
}

impl PixelClassifier {
    /// The logit (the value before the sigmoid) of each pixel of a row of up to [`ROW`] pixels,
    /// from its four features (`dx`, `dy`: the embedding less the seed's; `s0`, `s1`: sigma).
    /// One layer at a time over the whole row, so the inner loops run across pixels and vectorize;
    /// the values stay in arrays on the stack. The sigmoid is never taken: it is increasing, so
    /// thresholds and comparisons are done on logits ([`logit`]).
    fn logits_row(&self, dx: &[f32], dy: &[f32], s0: &[f32], s1: &[f32], out: &mut [f32]) {
        let n = dx.len();
        let mut a = [[0f32; ROW]; MAX_WIDTH];
        let mut b = [[0f32; ROW]; MAX_WIDTH];
        a[0][..n].copy_from_slice(dx);
        a[1][..n].copy_from_slice(dy);
        a[2][..n].copy_from_slice(s0);
        a[3][..n].copy_from_slice(s1);
        let last = self.layers.len() - 1;
        for (k, (w, bias, o, i)) in self.layers.iter().enumerate() {
            for r in 0..*o {
                let acc = &mut b[r];
                acc[..n].fill(bias[r]);
                for c in 0..*i {
                    let wc = w[r * i + c];
                    let src = &a[c];
                    for p in 0..n {
                        acc[p] += wc * src[p];
                    }
                }
                if k < last {
                    for v in &mut acc[..n] {
                        *v = v.max(0.0);
                    }
                }
            }
            std::mem::swap(&mut a, &mut b);
        }
        out[..n].copy_from_slice(&a[0][..n]);
    }

    /// Whether every layer fits [`Self::logits_row`]'s arrays.
    pub fn fits(&self) -> bool {
        self.layers.iter().all(|(_, _, o, i)| *o <= MAX_WIDTH && *i <= MAX_WIDTH)
    }
}

/// The widest pixel-classifier layer [`PixelClassifier::logits_row`] evaluates (InstanSeg's are 5).
pub const MAX_WIDTH: usize = 16;

/// Pixels per batch of [`PixelClassifier::logits_row`]: a window's row (`2 · window_size`) or a
/// part of it.
const ROW: usize = 64;

/// The logit of probability `p`: `prob ≥ p` exactly when `logit ≥ logit(p)`.
fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

/// The network's output for one image: `[channels, h, w]` (embedding 2, sigma 2, seeds 1).
pub struct Output<'a> {
    pub data: &'a [f32],
    pub h: usize,
    pub w: usize,
}

/// One instance found: its label and the seed map's value at its seed (the highest, for joined seeds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Found {
    pub label: u32,
    pub score: f32,
}

/// `linspace(0, end, n)` at `i`.
fn linspace(end: f32, n: usize, i: usize) -> f32 {
    if n <= 1 { 0.0 } else { end * i as f32 / (n - 1) as f32 }
}

/// Seeds: `(y, x)` of the local maxima above `min` in row-major order, as
/// `max_pool2d(kernel 2d + 1, stride 1, padding d, return_indices)` and `nonzero` find them.
pub fn peaks(map: &[f32], h: usize, w: usize, d: usize, min: f32) -> Vec<(usize, usize)> {
    let rows: Vec<Vec<(usize, usize)>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut out = Vec::new();
            for x in 0..w {
                let v = map[y * w + x];
                if v <= min {
                    continue;
                }
                // The first maximum (row-major) of the window must be this pixel
                let (y0, y1, x0, x1) = (y.saturating_sub(d), (y + d + 1).min(h), x.saturating_sub(d), (x + d + 1).min(w));
                let mut first = true;
                'scan: for yy in y0..y1 {
                    for xx in x0..x1 {
                        let u = map[yy * w + xx];
                        if (yy, xx) == (y, x) {
                            break 'scan;
                        }
                        if u >= v {
                            first = false;
                            break 'scan;
                        }
                    }
                }
                if first {
                    // …and nothing after it in the window larger
                    let later = (y..y1).any(|yy| {
                        let from = if yy == y { x + 1 } else { x0 };
                        (from..x1).any(|xx| map[yy * w + xx] > v)
                    });
                    if !later {
                        out.push((y, x));
                    }
                }
            }
            out
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// The 8-neighbours of `i` in a `w × h` grid.
fn neighbours(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let (y, x) = (i / w, i % w);
    (y.saturating_sub(1)..(y + 2).min(h)).flat_map(move |yy| (x.saturating_sub(1)..(x + 2).min(w)).map(move |xx| yy * w + xx)).filter(move |&j| j != i)
}

/// `mask` (`w × h`) grown from `seeds` within itself, 8-connected. As InstanSeg's (a 3 × 3 max
/// pool, then the mask, until nothing changes), the first step takes the mask's pixels next to a
/// seed, so a seed just under the threshold still grows.
fn flood_fill(mask: &[bool], seeds: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    let mut stack: Vec<usize> = (0..w * h).filter(|&i| mask[i] && (seeds[i] || neighbours(i, w, h).any(|j| seeds[j]))).collect();
    for &i in &stack {
        out[i] = true;
    }
    while let Some(i) = stack.pop() {
        for j in neighbours(i, w, h) {
            if mask[j] && !out[j] {
                out[j] = true;
                stack.push(j);
            }
        }
    }
    out
}

/// [`flood_fill`] from one seed pixel: only the pixels of its component are visited.
fn grow_from(mask: &[bool], seed: usize, w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    let mut stack: Vec<usize> = std::iter::once(seed).chain(neighbours(seed, w, h)).filter(|&i| mask[i]).collect();
    for &i in &stack {
        out[i] = true;
    }
    while let Some(i) = stack.pop() {
        for j in neighbours(i, w, h) {
            if mask[j] && !out[j] {
                out[j] = true;
                stack.push(j);
            }
        }
    }
    out
}

/// `mask` (`w × h`) with its holes filled: background not reached from the background beyond
/// three dilations of the mask (as InstanSeg's `fill_holes`).
fn fill_holes(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut grown = mask.to_vec();
    for _ in 0..3 {
        let prev = grown.clone();
        for i in 0..w * h {
            if !prev[i] && neighbours(i, w, h).any(|j| prev[j]) {
                grown[i] = true;
            }
        }
    }
    let background: Vec<bool> = mask.iter().map(|m| !m).collect();
    let far: Vec<bool> = grown.iter().map(|g| !g).collect();
    let outside = flood_fill(&background, &far, w, h);
    outside.iter().map(|o| !o).collect()
}

/// [`fill_holes`] of a `side × side` window, computed on the mask's bounding box and 4 pixels
/// around it: beyond that everything is more than three dilations away, so outside either way.
fn fill_holes_near(mask: &[bool], side: usize) -> Vec<bool> {
    let mut out = mask.to_vec();
    let (mut y0, mut y1, mut x0, mut x1) = (usize::MAX, 0, usize::MAX, 0);
    for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
        let (y, x) = (i / side, i % side);
        (y0, y1, x0, x1) = (y0.min(y), y1.max(y), x0.min(x), x1.max(x));
    }
    if y0 == usize::MAX {
        return out;
    }
    let (y0, x0) = (y0.saturating_sub(4), x0.saturating_sub(4));
    let (y1, x1) = ((y1 + 5).min(side), (x1 + 5).min(side));
    let (cw, ch) = (x1 - x0, y1 - y0);
    let crop: Vec<bool> = (0..ch * cw).map(|i| mask[(y0 + i / cw) * side + x0 + i % cw]).collect();
    let filled = fill_holes(&crop, cw, ch);
    for i in 0..ch * cw {
        out[(y0 + i / cw) * side + x0 + i % cw] = filled[i];
    }
    out
}

/// Labels (`h × w`, 0: background) and the instances found.
pub fn instances(out: &Output, classifier: &PixelClassifier, p: &Params) -> (Vec<u32>, Vec<Found>) {
    let (h, w) = (out.h, out.w);
    let n = h * w;
    let plane = |c: usize| &out.data[c * n..(c + 1) * n];
    let clock = std::time::Instant::now();
    let mut stages: Vec<(&str, f64)> = Vec::new();
    let lap = |name: &'static str, stages: &mut Vec<(&str, f64)>| stages.push((name, clock.elapsed().as_secs_f64() * 1e3));
    let seeds_map: Vec<f32> = plane(4).iter().map(|v| v / 15.0 + 0.5).collect();
    let (s0, s1) = (plane(2), plane(3));
    // The embedding, once per pixel (each pixel is in about ten windows)
    let (xend, yend) = (w as f32 * 64.0 / 256.0, h as f32 * 64.0 / 256.0);
    let embed = |e: f32| (1.0 / (1.0 + (-e).exp()) - 0.5) * 8.0;
    let fx: Vec<f32> = plane(0).par_iter().enumerate().map(|(i, &e)| embed(e) + linspace(xend, w, i % w)).collect();
    let fy: Vec<f32> = plane(1).par_iter().enumerate().map(|(i, &e)| embed(e) + linspace(yend, h, i / w)).collect();
    lap("embedding", &mut stages);
    let seeds = peaks(&seeds_map, h, w, p.peak_distance, p.seed_threshold);
    lap("peaks", &mut stages);
    if seeds.is_empty() {
        return (vec![0; n], Vec::new());
    }
    // The window: half-side `ws`, at most half the image (InstanSeg uses min(window, h, w))
    let ws = p.window_size.min(h / 2).min(w / 2).max(1);
    let side = 2 * ws;
    let threshold = logit(p.mask_threshold);
    // Each seed's window: its top-left and its pixels at `mask_threshold` or above (image index,
    // logit), by index
    let windows: Vec<(usize, usize, Vec<(u32, f32)>)> = seeds
        .par_iter()
        .map(
            |&(cy, cx)| {
                let (ty, tx) = (cy.clamp(ws, h - ws) - ws, cx.clamp(ws, w - ws) - ws);
                let (c0, c1) = (fx[cy * w + cx], fy[cy * w + cx]);
                // Logits, not probabilities: the same order and thresholds without an exponential
                let mut prob = vec![0f32; side * side];
                let (mut dx, mut dy) = ([0f32; ROW], [0f32; ROW]);
                for wy in 0..side {
                    let row = (ty + wy) * w + tx;
                    for start in (0..side).step_by(ROW) {
                        let len = ROW.min(side - start);
                        let at = row + start;
                        for p in 0..len {
                            dx[p] = fx[at + p] - c0;
                            dy[p] = fy[at + p] - c1;
                        }
                        let o = wy * side + start;
                        classifier.logits_row(&dx[..len], &dy[..len], &s0[at..at + len], &s1[at..at + len], &mut prob[o..o + len]);
                    }
                }
                if p.cleanup_fragments {
                    let mask: Vec<bool> = prob.iter().map(|&v| v >= threshold).collect();
                    let filled = grow_from(&mask, (cy - ty) * side + (cx - tx), side, side);
                    let closed = fill_holes_near(&filled, side);
                    for i in 0..side * side {
                        if !filled[i] {
                            prob[i] = if closed[i] { threshold } else { f32::NEG_INFINITY };
                        }
                    }
                }
                let fg: Vec<(u32, f32)> = (0..side * side).filter(|&i| prob[i] >= threshold).map(|i| (((ty + i / side) * w + tx + i % side) as u32, prob[i])).collect();
                (ty, tx, fg)
            },
        )
        .collect();
    lap("windows", &mut stages);
    // Each pixel to its most probable seed (ties: the lower label, as InstanSeg's stable sorts)
    let mut best = vec![(f32::NEG_INFINITY, 0u32); n];
    for (k, (_, _, fg)) in windows.iter().enumerate() {
        let label = k as u32 + 1;
        for &(i, v) in fg {
            if v > best[i as usize].0 {
                best[i as usize] = (v, label);
            }
        }
    }
    // Seeds joined where their pixels overlap by more than `overlap_threshold` of the smaller:
    // only windows that touch are compared (buckets of a window's side), by merging their sorted
    // pixel lists
    let mut buckets: std::collections::HashMap<(usize, usize), Vec<u32>> = std::collections::HashMap::new();
    for (k, (ty, tx, _)) in windows.iter().enumerate() {
        buckets.entry((ty / side, tx / side)).or_default().push(k as u32);
    }
    let joined: Vec<(u32, u32)> = (0..windows.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            let (ty, tx, a) = &windows[i];
            let (by, bx) = (ty / side, tx / side);
            let mut out = Vec::new();
            for gy in by.saturating_sub(1)..=by + 1 {
                for gx in bx.saturating_sub(1)..=bx + 1 {
                    let Some(list) = buckets.get(&(gy, gx)) else { continue };
                    for &j in list {
                        let (oy, ox, b) = &windows[j as usize];
                        if j as usize <= i || oy.abs_diff(*ty) >= side || ox.abs_diff(*tx) >= side {
                            continue;
                        }
                        let smaller = a.len().min(b.len());
                        if smaller == 0 {
                            continue;
                        }
                        // Pixels both windows hold (both lists sorted by index)
                        let (mut p1, mut p2, mut shared) = (0, 0, 0usize);
                        while p1 < a.len() && p2 < b.len() {
                            match a[p1].0.cmp(&b[p2].0) {
                                std::cmp::Ordering::Less => p1 += 1,
                                std::cmp::Ordering::Greater => p2 += 1,
                                std::cmp::Ordering::Equal => {
                                    shared += 1;
                                    p1 += 1;
                                    p2 += 1;
                                }
                            }
                        }
                        if shared as f32 / smaller as f32 > p.overlap_threshold {
                            out.push((i as u32, j));
                        }
                    }
                }
            }
            out
        })
        .collect();
    lap("best + overlaps", &mut stages);
    let mut parent: Vec<u32> = (0..windows.len() as u32).collect();
    fn find(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    for (i, j) in joined {
        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
        // The smaller index names the group (its label is the smallest)
        if a < b {
            parent[b as usize] = a;
        } else if b < a {
            parent[a as usize] = b;
        }
    }
    // Each group's area, then labels 1..=n for the groups of at least `min_size` pixels, in order
    // of their first pixel (row-major)
    let mut group_area = vec![0usize; windows.len()];
    let group_of: Vec<u32> = (0..n).map(|i| if best[i].1 == 0 { u32::MAX } else { find(&mut parent, best[i].1 - 1) }).collect();
    for &g in &group_of {
        if g != u32::MAX {
            group_area[g as usize] += 1;
        }
    }
    let mut number = vec![0u32; windows.len()];
    let mut found: Vec<Found> = Vec::new();
    let mut labels = vec![0u32; n];
    for i in 0..n {
        let g = group_of[i];
        if g == u32::MAX || group_area[g as usize] < p.min_size {
            continue;
        }
        let g = g as usize;
        if number[g] == 0 {
            found.push(Found { label: found.len() as u32 + 1, score: 0.0 });
            number[g] = found.len() as u32;
        }
        labels[i] = number[g];
    }
    for (k, &(y, x)) in seeds.iter().enumerate() {
        let g = find(&mut parent, k as u32) as usize;
        if number[g] > 0 {
            let f = &mut found[number[g] as usize - 1];
            f.score = f.score.max(seeds_map[y * w + x]);
        }
    }
    lap("labels", &mut stages);
    if std::env::var_os("LUCEM_INSTANSEG_TIMING").is_some() {
        let mut last = 0.0;
        eprintln!("instanseg post: {}", stages.iter().map(|(n, t)| { let d = t - last; last = *t; format!("{n} {d:.1} ms") }).collect::<Vec<_>>().join(", "));
    }
    (labels, found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_are_first_maxima_above_the_threshold() {
        let (h, w) = (5, 7);
        let mut m = vec![0f32; h * w];
        m[2 * w + 1] = 1.0;
        m[2 * w + 3] = 1.0; // a tie within distance 2: only the first (row-major) is a peak
        m[2 * w + 6] = 0.5;
        assert_eq!(peaks(&m, h, w, 2, 0.1), vec![(2, 1), (2, 6)]);
        assert!(peaks(&m, h, w, 2, 1.0).is_empty());
    }

    #[test]
    fn holes_fill_and_fragments_drop() {
        // Wide enough that background lies beyond three dilations of the ring (InstanSeg's
        // "outside"): in a window without any, everything counts as enclosed
        let s = 15;
        let ring: Vec<bool> = (0..s * s).map(|i| { let (y, x) = (i / s, i % s); (5..=9).contains(&y) && (5..=9).contains(&x) && !(y == 7 && x == 7) }).collect();
        let filled = fill_holes_near(&ring, s);
        assert_eq!(filled, fill_holes(&ring, s, s), "the cropped fill is the whole window's");
        assert!(filled[7 * s + 7] && !filled[0]);
        let mut mask = ring.clone();
        mask[0] = true; // a fragment not touching the ring
        let mut seed = vec![false; s * s];
        seed[5 * s + 5] = true;
        let kept = flood_fill(&mask, &seed, s, s);
        assert_eq!(kept, grow_from(&mask, 5 * s + 5, s, s));
        assert!(!kept[0] && kept[9 * s + 9]);
    }
}

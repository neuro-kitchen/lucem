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
    /// The probability (after the sigmoid) for one feature vector; `scratch` holds the layers'
    /// values between them (reused across pixels).
    fn prob(&self, x: &[f32; 4], scratch: &mut [Vec<f32>; 2]) -> f32 {
        let [a, b] = scratch;
        a.clear();
        a.extend_from_slice(x);
        let last = self.layers.len() - 1;
        for (k, (w, bias, o, i)) in self.layers.iter().enumerate() {
            b.clear();
            for r in 0..*o {
                let mut v = bias[r];
                for c in 0..*i {
                    v += w[r * i + c] * a[c];
                }
                b.push(if k < last { v.max(0.0) } else { v });
            }
            std::mem::swap(a, b);
        }
        1.0 / (1.0 + (-a[0]).exp())
    }
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

/// `mask` (side `s × s`) grown from `seed` within itself, 8-connected. As InstanSeg's (a 3 × 3
/// max pool, then the mask, until nothing changes), the first step takes the mask's pixels next
/// to a seed, so a seed just under the threshold still grows.
fn flood_fill(mask: &[bool], seed: &[bool], s: usize) -> Vec<bool> {
    let mut out = vec![false; s * s];
    let near_seed = |i: usize| {
        let (y, x) = (i / s, i % s);
        (y.saturating_sub(1)..(y + 2).min(s)).any(|yy| (x.saturating_sub(1)..(x + 2).min(s)).any(|xx| seed[yy * s + xx]))
    };
    let mut stack: Vec<usize> = (0..s * s).filter(|&i| mask[i] && near_seed(i)).collect();
    for &i in &stack {
        out[i] = true;
    }
    while let Some(i) = stack.pop() {
        let (y, x) = (i / s, i % s);
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let (ny, nx) = (y as i64 + dy, x as i64 + dx);
                if ny < 0 || nx < 0 || ny >= s as i64 || nx >= s as i64 {
                    continue;
                }
                let j = ny as usize * s + nx as usize;
                if mask[j] && !out[j] {
                    out[j] = true;
                    stack.push(j);
                }
            }
        }
    }
    out
}

/// `mask` with its holes filled: background not reached from the background beyond three
/// dilations of the mask (as InstanSeg's `fill_holes`).
fn fill_holes(mask: &[bool], s: usize) -> Vec<bool> {
    let mut grown = mask.to_vec();
    for _ in 0..3 {
        let prev = grown.clone();
        for y in 0..s {
            for x in 0..s {
                if prev[y * s + x] {
                    continue;
                }
                let near = (y.saturating_sub(1)..(y + 2).min(s)).any(|yy| (x.saturating_sub(1)..(x + 2).min(s)).any(|xx| prev[yy * s + xx]));
                grown[y * s + x] = near;
            }
        }
    }
    let background: Vec<bool> = mask.iter().map(|m| !m).collect();
    let far: Vec<bool> = grown.iter().map(|g| !g).collect();
    let outside = flood_fill(&background, &far, s);
    outside.iter().map(|o| !o).collect()
}

/// Labels (`h × w`, 0: background) and the instances found.
pub fn instances(out: &Output, classifier: &PixelClassifier, p: &Params) -> (Vec<u32>, Vec<Found>) {
    let (h, w) = (out.h, out.w);
    let n = h * w;
    let plane = |c: usize| &out.data[c * n..(c + 1) * n];
    let seeds_map: Vec<f32> = plane(4).iter().map(|v| v / 15.0 + 0.5).collect();
    let (ex, ey) = (plane(0), plane(1));
    let (s0, s1) = (plane(2), plane(3));
    let (xend, yend) = (w as f32 * 64.0 / 256.0, h as f32 * 64.0 / 256.0);
    let field = |c: usize, y: usize, x: usize| {
        let e = if c == 0 { ex[y * w + x] } else { ey[y * w + x] };
        let coord = if c == 0 { linspace(xend, w, x) } else { linspace(yend, h, y) };
        (1.0 / (1.0 + (-e).exp()) - 0.5) * 8.0 + coord
    };
    let seeds = peaks(&seeds_map, h, w, p.peak_distance, p.seed_threshold);
    if seeds.is_empty() {
        return (vec![0; n], Vec::new());
    }
    // The window: half-side `ws`, at most half the image (InstanSeg uses min(window, h, w))
    let ws = p.window_size.min(h / 2).min(w / 2).max(1);
    let side = 2 * ws;
    // Each seed's window: its top-left and the probability of each pixel
    let windows: Vec<(usize, usize, Vec<f32>)> = seeds
        .par_iter()
        .map_init(
            || [Vec::with_capacity(8), Vec::with_capacity(8)],
            |scratch, &(cy, cx)| {
                let (ty, tx) = (cy.clamp(ws, h - ws) - ws, cx.clamp(ws, w - ws) - ws);
                let (c0, c1) = (field(0, cy, cx), field(1, cy, cx));
                let mut prob = vec![0f32; side * side];
                for wy in 0..side {
                    for wx in 0..side {
                        let (y, x) = (ty + wy, tx + wx);
                        let f = [field(0, y, x) - c0, field(1, y, x) - c1, s0[y * w + x], s1[y * w + x]];
                        prob[wy * side + wx] = classifier.prob(&f, scratch);
                    }
                }
                if p.cleanup_fragments {
                    let mask: Vec<bool> = prob.iter().map(|&v| v >= p.mask_threshold).collect();
                    let mut seed = vec![false; side * side];
                    seed[(cy - ty) * side + (cx - tx)] = true;
                    let filled = flood_fill(&mask, &seed, side);
                    let closed = fill_holes(&filled, side);
                    for i in 0..side * side {
                        if !filled[i] {
                            prob[i] = 0.0;
                        }
                        if closed[i] && !filled[i] {
                            prob[i] = p.mask_threshold;
                        }
                    }
                }
                (ty, tx, prob)
            },
        )
        .collect();
    // Each pixel to its most probable seed (ties: the lower label, as InstanSeg's stable sorts)
    let mut best = vec![(f32::NEG_INFINITY, 0u32); n];
    let mut area = vec![0u32; windows.len()];
    let mut cover: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (k, (ty, tx, prob)) in windows.iter().enumerate() {
        let label = k as u32 + 1;
        for wy in 0..side {
            for wx in 0..side {
                let v = prob[wy * side + wx];
                if v < p.mask_threshold {
                    continue;
                }
                let i = (ty + wy) * w + tx + wx;
                area[k] += 1;
                cover[i].push(k as u32);
                if v > best[i].0 {
                    best[i] = (v, label);
                }
            }
        }
    }
    // Seeds joined where their pixels overlap: intersection over the smaller area
    let mut parent: Vec<u32> = (0..windows.len() as u32).collect();
    fn find(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    let mut inter: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for c in &cover {
        for (a, &i) in c.iter().enumerate() {
            for &j in &c[a + 1..] {
                *inter.entry((i.min(j), i.max(j))).or_insert(0) += 1;
            }
        }
    }
    for (&(i, j), &shared) in &inter {
        let smaller = area[i as usize].min(area[j as usize]).max(1);
        if shared as f32 / smaller as f32 > p.overlap_threshold {
            let (a, b) = (find(&mut parent, i), find(&mut parent, j));
            // The smaller index names the group (its label is the smallest)
            if a < b {
                parent[b as usize] = a;
            } else if b < a {
                parent[a as usize] = b;
            }
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
        let filled = fill_holes(&ring, s);
        assert!(filled[7 * s + 7] && !filled[0]);
        let mut mask = ring.clone();
        mask[0] = true; // a fragment not touching the ring
        let mut seed = vec![false; s * s];
        seed[5 * s + 5] = true;
        let kept = flood_fill(&mask, &seed, s);
        assert!(!kept[0] && kept[9 * s + 9]);
    }
}

//! From an RGB image to nucleus labels, as StarDist2D's `predict_instances`:
//!
//! 1. Normalization over the whole image (all channels): the 1st and 99.8th percentiles to 0 and 1.
//! 2. The network on tiles (a core of [`Tiling::tile`] pixels and a halo of [`Tiling::halo`] each
//!    side, padded to a multiple of 16 by repeating the edge); each tile's core goes into the
//!    whole image's probability and distances on the grid.
//! 3. Candidates: grid points above the probability threshold (not within 2 grid pixels of the
//!    image's edge), each a star-convex polygon of 32 rays.
//! 4. Non-maximum suppression, most probable first: a polygon is kept when at most
//!    `nms_threshold` of its area is already taken; it takes the free pixels of the label image.

use burn::tensor::{Tensor, TensorData};

use crate::net::{StarDistNet, DIVISOR, GRID, N_RAYS};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    pub prob: f32,
    pub nms: f32,
}

/// The published H&E model's thresholds (its `rdf.yaml`).
pub const HE_THRESHOLDS: Thresholds = Thresholds { prob: 0.692_478_25, nms: 0.3 };

/// Tile sides are rounded up to this (a multiple of the network's 16).
pub const SHAPE_STEP: usize = 256;
const _: () = assert!(SHAPE_STEP.is_multiple_of(DIVISOR));

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tiling {
    /// The core of a tile (input pixels; a multiple of 16).
    pub tile: usize,
    /// Context each side (input pixels; a multiple of 16; the model's field of view needs 96).
    pub halo: usize,
}

impl Default for Tiling {
    fn default() -> Self {
        // Large tiles: less of the halo computed twice (the largest activation, 128 features at
        // half of 2240², is ~640 MB in f32)
        Self { tile: 2048, halo: 96 }
    }
}

/// The network's output over a whole image, on the grid (`gh × gw`, `gh = ⌈h / 2⌉`).
pub struct GridOutput {
    pub gh: usize,
    pub gw: usize,
    pub prob: Vec<f32>,
    /// `[gh, gw, 32]`.
    pub dist: Vec<f32>,
}

/// `i` mirrored into `0..n` (`n - 2, n - 3, …` past the end, then back), as numpy's `reflect`.
fn reflect(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let k = i % period;
    if k < n { k } else { period - k }
}

/// The 1st and 99.8th percentiles of `pixels` (`c` channels interleaved, every channel
/// together) to 0 and 1, as `[c, h, w]` planes.
pub fn normalize(pixels: &[u8], w: usize, h: usize, c: usize) -> Vec<f32> {
    let mut hist = [0usize; 256];
    for &v in pixels {
        hist[v as usize] += 1;
    }
    let pct = |p: f64| -> f32 {
        let target = (p / 100.0 * (pixels.len().saturating_sub(1)) as f64).floor() as usize;
        let mut seen = 0;
        for (v, &c) in hist.iter().enumerate() {
            seen += c;
            if seen > target {
                return v as f32;
            }
        }
        255.0
    };
    let (lo, hi) = (pct(1.0), pct(99.8));
    let scale = 1.0 / (hi - lo + 1e-20);
    let mut out = vec![0f32; c * w * h];
    for i in 0..w * h {
        for k in 0..c {
            out[k * w * h + i] = (pixels[i * c + k] as f32 - lo) * scale;
        }
    }
    out
}

/// Runs the network over `planes` (`[channels, h, w]`, normalized) tile by tile;
/// `progress(done, total)`.
pub fn predict_grid(net: &StarDistNet, planes: &[f32], w: usize, h: usize, t: Tiling, progress: &mut dyn FnMut(usize, usize)) -> Result<GridOutput, String> {
    let (gh, gw) = (h.div_ceil(GRID), w.div_ceil(GRID));
    let mut prob = vec![0f32; gh * gw];
    let mut dist = vec![0f32; gh * gw * N_RAYS];
    let step = t.tile.max(DIVISOR);
    let (ny, nx) = (h.div_ceil(step), w.div_ceil(step));
    let total = ny * nx;
    for ty in 0..ny {
        for tx in 0..nx {
            // Core [y0, y1) × [x0, x1); the read window with its halo, clipped to the image
            let (y0, x0) = (ty * step, tx * step);
            let (y1, x1) = ((y0 + step).min(h), (x0 + step).min(w));
            let (ry0, rx0) = (y0.saturating_sub(t.halo), x0.saturating_sub(t.halo));
            let (ry1, rx1) = ((y1 + t.halo).min(h), (x1 + t.halo).min(w));
            let (rh, rw) = (ry1 - ry0, rx1 - rx0);
            // Padded to a multiple of 256 (the network needs 16): CubeCL tunes and compiles its
            // kernels per shape, so few distinct shapes means they come from the cache instead of
            // being tuned again for every region size
            let (ph, pw) = (rh.div_ceil(SHAPE_STEP) * SHAPE_STEP, rw.div_ceil(SHAPE_STEP) * SHAPE_STEP);
            let nc = net.channels;
            let mut input = vec![0f32; nc * ph * pw];
            for c in 0..nc {
                for y in 0..ph {
                    // Past the read window: mirrored, as StarDist pads (csbdeep's `reflect`)
                    let sy = ry0 + reflect(y, rh);
                    for x in 0..pw {
                        let sx = rx0 + reflect(x, rw);
                        input[(c * ph + y) * pw + x] = planes[c * w * h + sy * w + sx];
                    }
                }
            }
            let t0 = std::time::Instant::now();
            let attempt = || -> Result<(Vec<f32>, Vec<f32>), String> {
                // In the network's precision on the wire: half the bytes up and back in F16
                let data = TensorData::new(input.clone(), [1, nc, ph, pw]);
                let x = match net.precision {
                    crate::net::Precision::F32 => Tensor::<4>::from_data(data, &net.device),
                    crate::net::Precision::F16 => Tensor::<4>::from_data(data.convert::<burn::tensor::f16>(), (&net.device, burn::tensor::DType::F16)),
                };
                let (p, d) = net.forward(x);
                let p: Vec<f32> = p.try_into_data().map_err(|e| format!("reading the probability: {e:?}"))?.convert::<f32>().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
                let d: Vec<f32> = d.try_into_data().map_err(|e| format!("reading the distances: {e:?}"))?.convert::<f32>().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
                Ok((p, d))
            };
            // A matrix product this device cannot launch: Burn's own convolutions from now on
            let (p, d) = match attempt() {
                Ok(r) => r,
                Err(e) if crate::gemm_cannot_launch(&e) && crate::net::disable_gemm() => {
                    eprintln!("stardist: the matrix-product convolutions failed on this device, using Burn's own ({e})");
                    attempt().map_err(crate::explain)?
                }
                Err(e) => return Err(crate::explain(e)),
            };
            if crate::timing() {
                eprintln!("stardist: tile {ph}×{pw}: network and read back {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
            let (tgh, tgw) = (ph / GRID, pw / GRID);
            // The core's grid pixels
            for gy in y0.div_ceil(GRID)..y1.div_ceil(GRID) {
                let ly = gy - ry0 / GRID;
                for gx in x0.div_ceil(GRID)..x1.div_ceil(GRID) {
                    let lx = gx - rx0 / GRID;
                    if ly >= tgh || lx >= tgw {
                        continue;
                    }
                    prob[gy * gw + gx] = p[ly * tgw + lx];
                    for r in 0..N_RAYS {
                        dist[(gy * gw + gx) * N_RAYS + r] = d[(r * tgh + ly) * tgw + lx];
                    }
                }
            }
            progress(ty * nx + tx + 1, total);
        }
    }
    if prob.iter().all(|&p| p == 0.0) {
        // A device that failed quietly reads back zeros: never an answer
        return Err("the network gave no probability at all (the device failed?)".into());
    }
    Ok(GridOutput { gh, gw, prob, dist })
}

/// The star polygon of grid point `(gy, gx)` (input pixels, `(y, x)` vertices).
pub fn polygon(out: &GridOutput, gy: usize, gx: usize) -> Vec<[f32; 2]> {
    let (cy, cx) = ((gy * GRID) as f32, (gx * GRID) as f32);
    (0..N_RAYS)
        .map(|k| {
            let phi = 2.0 * std::f32::consts::PI * k as f32 / N_RAYS as f32;
            let d = out.dist[(gy * out.gw + gx) * N_RAYS + k].max(1e-3);
            [cy + d * phi.sin(), cx + d * phi.cos()]
        })
        .collect()
}

/// The pixels (row-major indices of a `w × h` image) whose centres are inside `poly`.
fn rasterize(poly: &[[f32; 2]], w: usize, h: usize, out: &mut Vec<usize>) {
    out.clear();
    let (mut y0, mut y1) = (f32::MAX, f32::MIN);
    for p in poly {
        y0 = y0.min(p[0]);
        y1 = y1.max(p[0]);
    }
    let ys = (y0.floor().max(0.0)) as usize;
    let ye = ((y1.ceil()) as isize).clamp(0, h as isize) as usize;
    let mut xs: Vec<f32> = Vec::new();
    for y in ys..ye {
        let fy = y as f32 + 0.5;
        xs.clear();
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            if (a[0] <= fy) != (b[0] <= fy) {
                xs.push(a[1] + (fy - a[0]) / (b[0] - a[0]) * (b[1] - a[1]));
            }
        }
        xs.sort_by(f32::total_cmp);
        for pair in xs.as_chunks::<2>().0 {
            let xa = (pair[0] - 0.5).ceil().max(0.0) as usize;
            let xb = ((pair[1] - 0.5).floor() as isize).min(w as isize - 1);
            if xb < 0 {
                continue;
            }
            for x in xa..=(xb as usize) {
                out.push(y * w + x);
            }
        }
    }
}

/// One nucleus found: its label, probability and grid point (input pixels).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nucleus {
    pub label: u32,
    pub prob: f32,
    pub y: usize,
    pub x: usize,
}

/// The label image (`w × h`, 0 background) of the network's output after thresholds and NMS.
pub fn instances(out: &GridOutput, w: usize, h: usize, t: Thresholds) -> (Vec<u32>, Vec<Nucleus>) {
    let border = 2;
    let mut cands: Vec<(f32, usize, usize)> = Vec::new();
    for gy in border..out.gh.saturating_sub(border) {
        for gx in border..out.gw.saturating_sub(border) {
            let p = out.prob[gy * out.gw + gx];
            if p > t.prob {
                cands.push((p, gy, gx));
            }
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut labels = vec![0u32; w * h];
    let mut kept = Vec::new();
    let mut px = Vec::new();
    for (p, gy, gx) in cands {
        // A candidate inside a kept nucleus is suppressed at once
        let (cy, cx) = (gy * GRID, gx * GRID);
        if cy < h && cx < w && labels[cy * w + cx] != 0 {
            continue;
        }
        rasterize(&polygon(out, gy, gx), w, h, &mut px);
        if px.is_empty() {
            continue;
        }
        let taken = px.iter().filter(|&&i| labels[i] != 0).count();
        if taken as f32 > t.nms * px.len() as f32 {
            continue;
        }
        let label = kept.len() as u32 + 1;
        for &i in &px {
            if labels[i] == 0 {
                labels[i] = label;
            }
        }
        kept.push(Nucleus { label, prob: p, y: cy, x: cx });
    }
    (labels, kept)
}

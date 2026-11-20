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
        // 384 + 2 × 64 = 512: canonical 512×512 tiles batched as a 4-tile volume [4, C, 512, 512]
        // (~5× less VRAM than a single 2304² tile, zero cold per-ROI autotune stalls, and parallel
        // GPU throughput across tiles).
        Self { tile: 384, halo: 64 }
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

/// Runs the network over `planes` (`[channels, h, w]`, normalized) in fixed-shape batched volumes
/// (`[B, C, canon, canon]`, `B <= 4`) so CubeCL autotunes once and GPU SMs run tiles in parallel
/// at a fraction of the VRAM of a monolithic tile; `progress(done, total)`.
pub fn predict_grid(net: &StarDistNet, planes: &[f32], w: usize, h: usize, t: Tiling, progress: &mut dyn FnMut(usize, usize)) -> Result<GridOutput, String> {
    const MAX_BATCH: usize = 4;
    let (gh, gw) = (h.div_ceil(GRID), w.div_ceil(GRID));
    let mut prob = vec![0f32; gh * gw];
    let mut dist = vec![0f32; gh * gw * N_RAYS];
    let step = t.tile.max(DIVISOR);
    let (ny, nx) = (h.div_ceil(step), w.div_ceil(step));
    let total = ny * nx;
    // One canonical square tile shape for every tile in the run (zero cold autotune per ROI crop!)
    let canon = (step + 2 * t.halo).div_ceil(SHAPE_STEP) * SHAPE_STEP;
    let (ph, pw) = (canon, canon);
    let nc = net.channels;
    let tile_elems = nc * ph * pw;

    struct TileSpec {
        y0: usize,
        x0: usize,
        y1: usize,
        x1: usize,
        ry0: usize,
        rx0: usize,
        rh: usize,
        rw: usize,
    }
    let mut specs = Vec::with_capacity(total);
    for ty in 0..ny {
        for tx in 0..nx {
            let (y0, x0) = (ty * step, tx * step);
            let (y1, x1) = ((y0 + step).min(h), (x0 + step).min(w));
            let (ry0, rx0) = (y0.saturating_sub(t.halo), x0.saturating_sub(t.halo));
            let (ry1, rx1) = ((y1 + t.halo).min(h), (x1 + t.halo).min(w));
            specs.push(TileSpec { y0, x0, y1, x1, ry0, rx0, rh: ry1 - ry0, rw: rx1 - rx0 });
        }
    }

    let mut batch_buf: Vec<f32> = Vec::new();
    let mut done = 0usize;
    let (tgh, tgw) = (ph / GRID, pw / GRID);
    let p_stride = tgh * tgw;
    let d_stride = N_RAYS * tgh * tgw;

    for chunk in specs.chunks(MAX_BATCH) {
        let actual = chunk.len();
        // Canonical batch size ∈ {1, MAX_BATCH}: pad tail batches (2 or 3 tiles) to MAX_BATCH so
        // CubeCL only ever autotunes 2 shapes total (`[1, C, 512, 512]` and `[4, C, 512, 512]`).
        let bsz = if actual == 1 { 1 } else { MAX_BATCH };
        batch_buf.clear();
        batch_buf.resize(bsz * tile_elems, 0.0);
        for (bi, sp) in chunk.iter().enumerate() {
            let base = bi * tile_elems;
            for c in 0..nc {
                let plane_off = c * w * h;
                let dst_c = base + c * ph * pw;
                for y in 0..ph {
                    let sy = sp.ry0 + reflect(y, sp.rh);
                    let src_row = plane_off + sy * w;
                    let dst_row = dst_c + y * pw;
                    for x in 0..pw {
                        let sx = sp.rx0 + reflect(x, sp.rw);
                        batch_buf[dst_row + x] = planes[src_row + sx];
                    }
                }
            }
        }
        let t0 = std::time::Instant::now();
        let attempt = |buf: &[f32]| -> Result<(Vec<f32>, Vec<f32>), String> {
            let x = match net.precision {
                crate::net::Precision::F32 => Tensor::<4>::from_data(TensorData::new(buf.to_vec(), [bsz, nc, ph, pw]), &net.device),
                crate::net::Precision::F16 => {
                    let halfs: Vec<burn::tensor::f16> = buf.iter().map(|&v| burn::tensor::f16::from_f32(v)).collect();
                    Tensor::<4>::from_data(TensorData::new(halfs, [bsz, nc, ph, pw]), (&net.device, burn::tensor::DType::F16))
                }
            };
            let (p, d) = net.forward(x);
            let p: Vec<f32> = p.try_into_data().map_err(|e| format!("reading the probability: {e:?}"))?.convert::<f32>().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            let d: Vec<f32> = d.try_into_data().map_err(|e| format!("reading the distances: {e:?}"))?.convert::<f32>().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            Ok((p, d))
        };
        let (p, d) = match attempt(&batch_buf) {
            Ok(r) => r,
            Err(e) if crate::gemm_cannot_launch(&e) && crate::net::disable_gemm() => {
                eprintln!("stardist: the matrix-product convolutions failed on this device, using Burn's own ({e})");
                attempt(&batch_buf).map_err(crate::explain)?
            }
            Err(e) => return Err(crate::explain(e)),
        };
        if crate::timing() {
            eprintln!("stardist: batch {bsz}×{ph}×{pw}: network and read back {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        for (bi, sp) in chunk.iter().enumerate() {
            let p_off = bi * p_stride;
            let d_off = bi * d_stride;
            for gy in sp.y0.div_ceil(GRID)..sp.y1.div_ceil(GRID) {
                let ly = gy - sp.ry0 / GRID;
                for gx in sp.x0.div_ceil(GRID)..sp.x1.div_ceil(GRID) {
                    let lx = gx - sp.rx0 / GRID;
                    if ly >= tgh || lx >= tgw {
                        continue;
                    }
                    prob[gy * gw + gx] = p[p_off + ly * tgw + lx];
                    for r in 0..N_RAYS {
                        dist[(gy * gw + gx) * N_RAYS + r] = d[d_off + (r * tgh + ly) * tgw + lx];
                    }
                }
            }
        }
        done += actual;
        progress(done, total);
    }
    if prob.iter().all(|&p| p == 0.0) {
        // A device that failed quietly reads back zeros: never an answer
        return Err("the network gave no probability at all (the device failed?)".into());
    }
    Ok(GridOutput { gh, gw, prob, dist })
}

fn ray_table() -> &'static [(f32, f32); N_RAYS] {
    static TABLE: std::sync::OnceLock<[(f32, f32); N_RAYS]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|k| {
            let phi = 2.0 * std::f32::consts::PI * k as f32 / N_RAYS as f32;
            (phi.sin(), phi.cos())
        })
    })
}

/// The star polygon of grid point `(gy, gx)` (input pixels, `(y, x)` vertices).
pub fn polygon(out: &GridOutput, gy: usize, gx: usize) -> Vec<[f32; 2]> {
    let mut poly = Vec::with_capacity(N_RAYS);
    polygon_into(out, gy, gx, &mut poly);
    poly
}

fn polygon_into(out: &GridOutput, gy: usize, gx: usize, poly: &mut Vec<[f32; 2]>) {
    poly.clear();
    let (cy, cx) = ((gy * GRID) as f32, (gx * GRID) as f32);
    let base = (gy * out.gw + gx) * N_RAYS;
    let rays = ray_table();
    for (k, &(s, c)) in rays.iter().enumerate() {
        let d = out.dist[base + k].max(1e-3);
        poly.push([cy + d * s, cx + d * c]);
    }
}

/// The pixels (row-major indices of a `w × h` image) whose centres are inside `poly`.
fn rasterize(poly: &[[f32; 2]], w: usize, h: usize, xs: &mut Vec<f32>, out: &mut Vec<usize>) {
    out.clear();
    let (mut y0, mut y1) = (f32::MAX, f32::MIN);
    for p in poly {
        y0 = y0.min(p[0]);
        y1 = y1.max(p[0]);
    }
    let ys = (y0.floor().max(0.0)) as usize;
    let ye = ((y1.ceil()) as isize).clamp(0, h as isize) as usize;
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
    let mut poly = Vec::with_capacity(N_RAYS);
    let mut xs = Vec::with_capacity(16);
    let mut px = Vec::new();
    for (p, gy, gx) in cands {
        // A candidate inside a kept nucleus is suppressed at once
        let (cy, cx) = (gy * GRID, gx * GRID);
        if cy < h && cx < w && labels[cy * w + cx] != 0 {
            continue;
        }
        polygon_into(out, gy, gx, &mut poly);
        rasterize(&poly, w, h, &mut xs, &mut px);
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

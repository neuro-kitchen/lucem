//! Timing and tuning InstanSeg on this machine.
//!
//! - [`tune`] runs the network once on every tile shape inference meets (sides in steps of
//!   [`SHAPE_STEP`] up to a full tile with its halo), so Burn's autotuned convolutions and the
//!   compiled kernels are chosen and cached (CubeCL `persistence` and the kernel cache, in lucem's
//!   cache folder) before a real image needs them. `lucem autotune` runs it for every model.
//! - [`bench`] times whole images of several sizes, phase by phase ([`Timings`]): the first run
//!   (cold: compiling and tuning when the caches are empty) and the median of the next ones.
//!
//! Both use a synthetic image ([`synthetic`]): nuclei-like blobs on a textured background, so the
//! seed count and the postprocessing are like a real slide's.

use std::time::Instant;

use crate::{InstanSeg, Tiling, Timings, SHAPE_STEP};

/// Image sides [`bench`] times by default (pixels at the model's 0.5 µm).
pub const DEFAULT_SIZES: [usize; 4] = [512, 1024, 2048, 4096];

/// A deterministic test image, `channels` interleaved bytes per pixel: round nuclei about 8–14 px
/// across (at 0.5 µm per pixel), dark purple on pink for 3 channels (brightfield), bright on dark
/// for 1 (fluorescence).
pub fn synthetic(w: usize, h: usize, channels: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    // Nucleus amount per pixel, 0..1: one blob per ~20 × 20 px cell, jittered
    let mut amount = vec![0f32; w * h];
    let cell = 20usize;
    for cy in (0..h).step_by(cell) {
        for cx in (0..w).step_by(cell) {
            if next() < 0.15 {
                continue;
            }
            let (y0, x0) = (cy as f64 + next() * cell as f64, cx as f64 + next() * cell as f64);
            let r = 4.0 + next() * 3.0;
            let (ya, yb) = ((y0 - r - 1.0).max(0.0) as usize, ((y0 + r + 2.0) as usize).min(h));
            let (xa, xb) = ((x0 - r - 1.0).max(0.0) as usize, ((x0 + r + 2.0) as usize).min(w));
            for y in ya..yb {
                for x in xa..xb {
                    let d = (((y as f64 - y0).powi(2) + (x as f64 - x0).powi(2)).sqrt() - r).clamp(-1.0, 1.0);
                    let v = (0.5 - d * 0.5) as f32;
                    let a = &mut amount[y * w + x];
                    *a = a.max(v);
                }
            }
        }
    }
    let mut out = Vec::with_capacity(w * h * channels);
    for a in amount {
        let noise = (next() - 0.5) as f32 * 0.08;
        let a = (a + noise).clamp(0.0, 1.0);
        if channels == 1 {
            out.push((20.0 + 200.0 * a) as u8);
        } else {
            // Background pink (230, 190, 210) to nucleus purple (90, 60, 140)
            out.extend([230.0 - 140.0 * a, 190.0 - 130.0 * a, 210.0 - 70.0 * a].map(|v| v.clamp(0.0, 255.0) as u8));
            out.truncate(out.len() - 3 + channels.min(3));
        }
    }
    out
}

/// What [`tune`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuneSummary {
    pub shapes: usize,
    pub seconds: f64,
}

/// The tile shapes inference meets with `tiling`: each side a multiple of [`SHAPE_STEP`] up to a
/// tile with its halo on both sides.
pub fn tile_shapes(tiling: Tiling) -> Vec<(usize, usize)> {
    let max = (tiling.tile + 2 * tiling.halo).div_ceil(SHAPE_STEP) * SHAPE_STEP;
    let sides: Vec<usize> = (1..=max / SHAPE_STEP).map(|k| k * SHAPE_STEP).collect();
    sides.iter().flat_map(|&a| sides.iter().map(move |&b| (a, b))).collect()
}

/// Runs the network once on every tile shape (largest first, so a shape that does not fit in the
/// device's memory shows at once); `progress(done, total)` after each.
pub fn tune(model: &InstanSeg, tiling: Tiling, mut progress: impl FnMut(usize, usize)) -> Result<TuneSummary, String> {
    let t0 = Instant::now();
    let mut shapes = tile_shapes(tiling);
    shapes.sort_by_key(|&(a, b)| std::cmp::Reverse(a * b));
    let c = model.spec.channels;
    let total = shapes.len();
    for (k, &(h, w)) in shapes.iter().enumerate() {
        let planes = crate::normalize(&synthetic(w, h, c), w, h, c);
        // One tile exactly this shape: a tile as large as the image, no halo
        model.network(&planes, w, h, Tiling { tile: w.max(h), halo: 0 }, &mut |_, _| {})?;
        progress(k + 1, total);
    }
    Ok(TuneSummary { shapes: total, seconds: t0.elapsed().as_secs_f64() })
}

/// One image size timed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchRow {
    pub side: usize,
    pub nuclei: usize,
    /// The first run in this process.
    pub cold: Timings,
    /// The median run of the others (by total time).
    pub warm: Timings,
    pub runs: usize,
}

/// Times square synthetic images of each side, `runs` times each after the first.
pub fn bench(model: &InstanSeg, sides: &[usize], runs: usize, tiling: Tiling, mut progress: impl FnMut(&BenchRow)) -> Result<Vec<BenchRow>, String> {
    let c = model.spec.channels;
    let mut rows = Vec::new();
    for &side in sides {
        let pixels = synthetic(side, side, c);
        let (_, found, cold) = model.segment_timed(&pixels, side, side, tiling, &mut |_, _| {})?;
        let mut warm: Vec<Timings> = (0..runs.max(1)).map(|_| model.segment_timed(&pixels, side, side, tiling, &mut |_, _| {}).map(|r| r.2)).collect::<Result<_, _>>()?;
        warm.sort_by(|a, b| a.total().total_cmp(&b.total()));
        let row = BenchRow { side, nuclei: found.len(), cold, warm: warm[warm.len() / 2], runs: warm.len() };
        progress(&row);
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_synthetic_image_has_its_channels_and_nuclei() {
        let px = synthetic(64, 48, 3);
        assert_eq!(px.len(), 64 * 48 * 3);
        assert!(px.chunks(3).any(|p| p[0] < 150) && px.chunks(3).any(|p| p[0] > 200), "nuclei and background");
        assert_eq!(synthetic(64, 48, 1).len(), 64 * 48);
    }

    #[test]
    fn tile_shapes_cover_every_side_up_to_a_full_tile() {
        let s = tile_shapes(Tiling { tile: 1024, halo: 96 });
        assert_eq!(s.len(), 25, "256..=1280 in steps of 256, both sides");
        assert!(s.contains(&(1280, 1280)) && s.contains(&(256, 1280)));
    }
}

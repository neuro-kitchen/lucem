//! Canonical batched-volume tile primitives for GPU ops and ML models.
//!
//! CubeCL autotunes and caches compiled kernels per problem shape. Running variable edge-tile
//! sizes or variable tail batch sizes (`bsz = 2, 3`) causes cold autotune stalls on new ROI crops.
//! Packing tiles into canonical square volumes `[B, H_canon, W_canon]` with `B ∈ {1, max_batch}`
//! keeps VRAM bounded while executing multiple tiles in parallel across GPU SMs with zero
//! per-ROI re-autotuning.

/// Rounds a tile side (core + 2×halo) up to a canonical multiple of `step` (e.g. `256`).
#[inline]
pub const fn canonical_tile_side(core: usize, halo: usize, step: usize) -> usize {
    let raw = core + 2 * halo;
    let step = if step == 0 { 1 } else { step };
    raw.div_ceil(step) * step
}

/// Maps an actual chunk length `actual` (`1..=max_batch`) to a canonical batch size in
/// `{1, max_batch}` so partial tail batches never introduce extra autotune shapes.
#[inline]
pub const fn canonical_batch_size(actual: usize, max_batch: usize) -> usize {
    if actual <= 1 || max_batch <= 1 {
        1
    } else {
        max_batch
    }
}

/// One tile's core and halo coordinates inside a 2D plane `(width, height)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileSlice {
    pub y0: usize,
    pub x0: usize,
    pub y1: usize,
    pub x1: usize,
    pub ry0: usize,
    pub rx0: usize,
    pub rh: usize,
    pub rw: usize,
}

/// Partitions a `(width, height)` plane into uniform tiles with halo for batched volume execution.
pub fn plan_tile_grid(width: usize, height: usize, core: usize, halo: usize) -> Vec<TileSlice> {
    let step = core.max(1);
    let (ny, nx) = (height.div_ceil(step), width.div_ceil(step));
    let mut out = Vec::with_capacity(ny * nx);
    for ty in 0..ny {
        for tx in 0..nx {
            let (y0, x0) = (ty * step, tx * step);
            let (y1, x1) = ((y0 + step).min(height), (x0 + step).min(width));
            let (ry0, rx0) = (y0.saturating_sub(halo), x0.saturating_sub(halo));
            let (ry1, rx1) = ((y1 + halo).min(height), (x1 + halo).min(width));
            out.push(TileSlice { y0, x0, y1, x1, ry0, rx0, rh: ry1 - ry0, rw: rx1 - rx0 });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_batch_only_emits_one_or_max() {
        assert_eq!(canonical_batch_size(1, 4), 1);
        assert_eq!(canonical_batch_size(2, 4), 4);
        assert_eq!(canonical_batch_size(3, 4), 4);
        assert_eq!(canonical_batch_size(4, 4), 4);
    }

    #[test]
    fn canonical_side_and_grid_cover_plane() {
        assert_eq!(canonical_tile_side(384, 64, 256), 512);
        let grid = plan_tile_grid(700, 500, 384, 64);
        assert_eq!(grid.len(), 4);
        assert_eq!(grid[3].x1, 700);
        assert_eq!(grid[3].y1, 500);
    }
}

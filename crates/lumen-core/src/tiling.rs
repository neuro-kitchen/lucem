//! Walking an image in blocks: which part of it each step reads and which part it keeps.
//!
//! Sizes and positions are `[z, y, x]`; a 2D image has depth 1. Each [`Block`] says what it owns
//! (`origin`, `size`: the blocks cover the image exactly once) and what is read for it (`read_origin`,
//! `read_size`: the owned part plus a halo of context, clipped to the image). An operation runs on
//! the read part and only the owned part of its result is kept, so the outcome does not depend on
//! the block size for any operation that looks at most `halo` pixels away.

/// `[z, y, x]`.
pub type Zyx = [usize; 3];

/// One step of a walk over an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Block {
    /// Position in the walk.
    pub index: usize,
    /// The part this step owns.
    pub origin: Zyx,
    pub size: Zyx,
    /// The part read for it: the owned part plus the halo, clipped to the image.
    pub read_origin: Zyx,
    pub read_size: Zyx,
}

impl Block {
    /// Where the owned part starts inside the read part.
    pub fn inner_offset(&self) -> Zyx {
        [self.origin[0] - self.read_origin[0], self.origin[1] - self.read_origin[1], self.origin[2] - self.read_origin[2]]
    }

    /// Pixels the step reads.
    pub fn read_len(&self) -> usize {
        self.read_size[0] * self.read_size[1] * self.read_size[2]
    }

    /// Pixels the step owns.
    pub fn len(&self) -> usize {
        self.size[0] * self.size[1] * self.size[2]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The array dimensions of the read part: `[h, w]` when one plane deep, else `[d, h, w]`.
    pub fn read_dims(&self) -> Vec<usize> {
        if self.read_size[0] == 1 { vec![self.read_size[1], self.read_size[2]] } else { self.read_size.to_vec() }
    }
}

/// The blocks covering `dims`, in raster order (z, then y, then x), each owning `block` pixels
/// (less at the far edges) and reading `halo` more on each side. An axis of `block` at 0 means the
/// whole axis.
pub fn schedule(dims: Zyx, block: Zyx, halo: Zyx) -> Vec<Block> {
    if dims.contains(&0) {
        return Vec::new();
    }
    let step = |a: usize| if block[a] == 0 { dims[a] } else { block[a].min(dims[a]) };
    let (sz, sy, sx) = (step(0), step(1), step(2));
    let mut out = Vec::with_capacity(dims[0].div_ceil(sz) * dims[1].div_ceil(sy) * dims[2].div_ceil(sx));
    let span = |o: usize, s: usize, a: usize| -> (usize, usize) {
        let start = o.saturating_sub(halo[a]);
        let end = (o + s + halo[a]).min(dims[a]);
        (start, end - start)
    };
    for z in (0..dims[0]).step_by(sz) {
        for y in (0..dims[1]).step_by(sy) {
            for x in (0..dims[2]).step_by(sx) {
                let size = [sz.min(dims[0] - z), sy.min(dims[1] - y), sx.min(dims[2] - x)];
                let (rz, dz) = span(z, size[0], 0);
                let (ry, dy) = span(y, size[1], 1);
                let (rx, dx) = span(x, size[2], 2);
                out.push(Block { index: out.len(), origin: [z, y, x], size, read_origin: [rz, ry, rx], read_size: [dz, dy, dx] });
            }
        }
    }
    out
}

/// The `size` part at `at` of `data`, a C-ordered array of `from`.
///
/// # Panics
/// If the part is outside `from` or `data` is shorter than `from`.
pub fn crop<T: Copy>(data: &[T], from: Zyx, at: Zyx, size: Zyx) -> Vec<T> {
    assert!((0..3).all(|a| at[a] + size[a] <= from[a]), "crop {size:?} at {at:?} is outside {from:?}");
    assert!(data.len() >= from[0] * from[1] * from[2], "crop: data shorter than its shape");
    let mut out = Vec::with_capacity(size[0] * size[1] * size[2]);
    for z in at[0]..at[0] + size[0] {
        for y in at[1]..at[1] + size[1] {
            let start = (z * from[1] + y) * from[2] + at[2];
            out.extend_from_slice(&data[start..start + size[2]]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_cover_the_image_exactly_once() {
        let dims = [5, 70, 130];
        let blocks = schedule(dims, [2, 32, 64], [1, 4, 4]);
        let mut hits = vec![0u32; dims[0] * dims[1] * dims[2]];
        for b in &blocks {
            for z in b.origin[0]..b.origin[0] + b.size[0] {
                for y in b.origin[1]..b.origin[1] + b.size[1] {
                    for x in b.origin[2]..b.origin[2] + b.size[2] {
                        hits[(z * dims[1] + y) * dims[2] + x] += 1;
                    }
                }
            }
            for a in 0..3 {
                assert!(b.read_origin[a] + b.read_size[a] <= dims[a]);
                assert!(b.read_origin[a] <= b.origin[a]);
            }
        }
        assert!(hits.iter().all(|&h| h == 1));
        assert_eq!(blocks.iter().map(|b| b.index).collect::<Vec<_>>(), (0..blocks.len()).collect::<Vec<_>>());
    }

    #[test]
    fn crop_returns_the_owned_part() {
        let data: Vec<u32> = (0..2 * 3 * 4).collect();
        assert_eq!(crop(&data, [2, 3, 4], [1, 1, 2], [1, 1, 2]), vec![18, 19]);
    }

    #[test]
    fn a_plane_deep_block_has_two_dimensions() {
        let b = schedule([1, 10, 10], [1, 4, 4], [0, 2, 2])[0];
        assert_eq!(b.read_dims(), vec![6, 6]);
        assert_eq!(b.inner_offset(), [0, 0, 0]);
    }
}

//! Joining objects cut by block borders.
//!
//! Labelling a block numbers only what is in it, so one object crossing a border gets one number in
//! each block. [`LabelStitcher`] takes the owned part of each block's labels in walk order, gives
//! every object a provisional id, joins ids that touch across a border, and accumulates each
//! object's measurements. Pixels are owned by exactly one block, so nothing is counted twice and an
//! object's measurements are the same whatever the block size.

use std::collections::HashMap;

use lucem_core::{Block, Zyx};

use crate::table::Table;

/// Measurements of one object, added up as its pixels arrive.
#[derive(Debug, Clone)]
struct Stats {
    area: u64,
    sum: [f64; 3],
    low: Zyx,
    high: Zyx,
    intensity_sum: f64,
    intensity_min: f64,
    intensity_max: f64,
}

impl Stats {
    fn new() -> Self {
        Self { area: 0, sum: [0.0; 3], low: [usize::MAX; 3], high: [0; 3], intensity_sum: 0.0, intensity_min: f64::INFINITY, intensity_max: f64::NEG_INFINITY }
    }

    fn add(&mut self, at: Zyx, value: Option<f32>) {
        self.area += 1;
        for (a, &v) in at.iter().enumerate() {
            self.sum[a] += v as f64;
            self.low[a] = self.low[a].min(v);
            self.high[a] = self.high[a].max(v + 1);
        }
        if let Some(v) = value {
            self.intensity_sum += v as f64;
            self.intensity_min = self.intensity_min.min(v as f64);
            self.intensity_max = self.intensity_max.max(v as f64);
        }
    }

    fn merge(&mut self, o: &Stats) {
        self.area += o.area;
        for a in 0..3 {
            self.sum[a] += o.sum[a];
            self.low[a] = self.low[a].min(o.low[a]);
            self.high[a] = self.high[a].max(o.high[a]);
        }
        self.intensity_sum += o.intensity_sum;
        self.intensity_min = self.intensity_min.min(o.intensity_min);
        self.intensity_max = self.intensity_max.max(o.intensity_max);
    }
}

/// One face of a finished block, kept until the block after it along that axis arrives.
#[derive(Debug)]
struct Face {
    /// The two sizes across the face.
    size: [usize; 2],
    ids: Vec<u32>,
}

/// Provisional ids, their joins and measurements. Add blocks in raster order.
#[derive(Debug)]
pub struct LabelStitcher {
    /// Index 0 is the background.
    parent: Vec<u32>,
    stats: Vec<Stats>,
    /// By the origin of the block that will read them, and axis.
    faces: HashMap<(Zyx, usize), Face>,
    grid: Zyx,
    link_z: bool,
}

/// The final labels and a table of the objects.
#[derive(Debug, Clone)]
pub struct Stitched {
    /// `map[provisional]` is the final label (1, 2, …; 0 stays background).
    pub map: Vec<u32>,
    pub table: Table,
}

impl LabelStitcher {
    /// For blocks on a grid of `grid` (`[z, y, x]`). With `link_z` off, objects are joined within
    /// each plane only (a walk plane by plane, where each plane was segmented by itself).
    pub fn new(grid: Zyx, link_z: bool) -> Self {
        Self { parent: vec![0], stats: vec![Stats::new()], faces: HashMap::new(), grid, link_z }
    }

    fn find(&mut self, mut a: u32) -> u32 {
        while self.parent[a as usize] != a {
            let up = self.parent[self.parent[a as usize] as usize];
            self.parent[a as usize] = up;
            a = up;
        }
        a
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra.max(rb) as usize] = ra.min(rb);
        }
    }

    /// Adds the owned part of a block. `labels` holds its local labels (0: background) and
    /// `intensity`, when given, the matching image, both C-ordered of `block.size`. Returns the
    /// block with provisional ids.
    ///
    /// # Panics
    /// If the slices are not of the block's size.
    pub fn add(&mut self, block: &Block, labels: &[u32], intensity: Option<&[f32]>) -> Vec<u32> {
        let [d, h, w] = block.size;
        assert_eq!(labels.len(), d * h * w, "labels are not of the block's size");
        if let Some(v) = intensity {
            assert_eq!(v.len(), labels.len(), "intensity is not of the block's size");
        }
        let mut local: HashMap<u32, u32> = HashMap::new();
        let mut out = Vec::with_capacity(labels.len());
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let i = (z * h + y) * w + x;
                    let l = labels[i];
                    if l == 0 {
                        out.push(0);
                        continue;
                    }
                    let id = *local.entry(l).or_insert_with(|| {
                        let id = self.parent.len() as u32;
                        self.parent.push(id);
                        self.stats.push(Stats::new());
                        id
                    });
                    self.stats[id as usize].add([block.origin[0] + z, block.origin[1] + y, block.origin[2] + x], intensity.map(|v| v[i]));
                    out.push(id);
                }
            }
        }
        self.join_faces(block, &out);
        self.keep_faces(block, &out);
        out
    }

    /// The ids of the face of a block at index `at` along `axis`.
    fn face_of(size: Zyx, ids: &[u32], axis: usize, at: usize) -> Face {
        let [d, h, w] = size;
        let (across, mut out): ([usize; 2], Vec<u32>) = match axis {
            0 => ([h, w], Vec::with_capacity(h * w)),
            1 => ([d, w], Vec::with_capacity(d * w)),
            _ => ([d, h], Vec::with_capacity(d * h)),
        };
        for a in 0..across[0] {
            for b in 0..across[1] {
                let (z, y, x) = match axis {
                    0 => (at, a, b),
                    1 => (a, at, b),
                    _ => (a, b, at),
                };
                out.push(ids[(z * h + y) * w + x]);
            }
        }
        Face { size: across, ids: out }
    }

    fn join_faces(&mut self, block: &Block, ids: &[u32]) {
        for axis in 0..3 {
            if axis == 0 && !self.link_z {
                continue;
            }
            let Some(before) = self.faces.remove(&(block.origin, axis)) else { continue };
            let here = Self::face_of(block.size, ids, axis, 0);
            if before.size != here.size {
                continue;
            }
            for (&a, &b) in before.ids.iter().zip(here.ids.iter()) {
                if a != 0 && b != 0 {
                    self.union(a, b);
                }
            }
        }
    }

    fn keep_faces(&mut self, block: &Block, ids: &[u32]) {
        for axis in 0..3 {
            if axis == 0 && !self.link_z {
                continue;
            }
            let mut next = block.origin;
            next[axis] += self.grid[axis].max(1);
            let face = Self::face_of(block.size, ids, axis, block.size[axis] - 1);
            self.faces.insert((next, axis), face);
        }
    }

    /// Joins everything, numbers the objects 1, 2, … in order of first appearance and tabulates
    /// them. Positions are in the pixels of the level walked.
    pub fn finish(mut self) -> Stitched {
        let n = self.parent.len();
        let mut merged: Vec<Option<Stats>> = (0..n).map(|_| None).collect();
        for id in 1..n as u32 {
            let root = self.find(id) as usize;
            let s = self.stats[id as usize].clone();
            match &mut merged[root] {
                Some(m) => m.merge(&s),
                slot => *slot = Some(s),
            }
        }
        let mut map = vec![0u32; n];
        let mut next = 0u32;
        let mut table = Table::new();
        for id in 1..n as u32 {
            let root = self.find(id) as usize;
            if map[root] == 0 {
                next += 1;
                map[root] = next;
                let s = merged[root].as_ref().expect("a root has its stats");
                let a = s.area as f64;
                table.push_row(&[
                    ("label".into(), next as f64),
                    ("area".into(), a),
                    ("centroid_z".into(), s.sum[0] / a),
                    ("centroid_y".into(), s.sum[1] / a),
                    ("centroid_x".into(), s.sum[2] / a),
                    ("bbox_z0".into(), s.low[0] as f64),
                    ("bbox_y0".into(), s.low[1] as f64),
                    ("bbox_x0".into(), s.low[2] as f64),
                    ("bbox_z1".into(), s.high[0] as f64),
                    ("bbox_y1".into(), s.high[1] as f64),
                    ("bbox_x1".into(), s.high[2] as f64),
                    ("mean_intensity".into(), s.intensity_sum / a),
                    ("min_intensity".into(), if s.intensity_min.is_finite() { s.intensity_min } else { f64::NAN }),
                    ("max_intensity".into(), if s.intensity_max.is_finite() { s.intensity_max } else { f64::NAN }),
                ]);
            }
            map[id as usize] = map[root];
        }
        Stitched { map, table }
    }
}

#[cfg(test)]
mod tests {
    use lucem_core::tiling::{crop, schedule};

    use super::*;

    /// Labels connected pixels of one block, each local.
    fn label_block(mask: &[u8], [d, h, w]: Zyx) -> Vec<u32> {
        let mut out = vec![0u32; mask.len()];
        let mut next = 0;
        for start in 0..mask.len() {
            if mask[start] == 0 || out[start] != 0 {
                continue;
            }
            next += 1;
            out[start] = next;
            let mut stack = vec![start];
            while let Some(i) = stack.pop() {
                let (z, y, x) = (i / (h * w), (i / w) % h, i % w);
                let mut visit = |nz: isize, ny: isize, nx: isize| {
                    if nz < 0 || ny < 0 || nx < 0 || nz >= d as isize || ny >= h as isize || nx >= w as isize {
                        return;
                    }
                    let j = (nz as usize * h + ny as usize) * w + nx as usize;
                    if mask[j] != 0 && out[j] == 0 {
                        out[j] = next;
                        stack.push(j);
                    }
                };
                let (z, y, x) = (z as isize, y as isize, x as isize);
                visit(z + 1, y, x);
                visit(z - 1, y, x);
                visit(z, y + 1, x);
                visit(z, y - 1, x);
                visit(z, y, x + 1);
                visit(z, y, x - 1);
            }
        }
        out
    }

    fn check(dims: Zyx, grid: Zyx) {
        // A cross through the whole image and a separate speck: two objects however it is cut
        let n = dims[0] * dims[1] * dims[2];
        let mut mask = vec![0u8; n];
        let at = |z: usize, y: usize, x: usize| (z * dims[1] + y) * dims[2] + x;
        for x in 0..dims[2] {
            mask[at(dims[0] / 2, dims[1] / 2, x)] = 1;
        }
        for y in 0..dims[1] {
            mask[at(dims[0] / 2, y, dims[2] / 2)] = 1;
        }
        for z in 0..dims[0] {
            mask[at(z, dims[1] / 2, dims[2] / 2)] = 1;
        }
        mask[at(0, 0, 0)] = 1;
        let total = mask.iter().map(|&m| m as usize).sum::<usize>();
        let mut s = LabelStitcher::new(grid, true);
        for b in schedule(dims, grid, [0; 3]) {
            let part = crop(&mask, dims, b.origin, b.size);
            s.add(&b, &label_block(&part, b.size), None);
        }
        let done = s.finish();
        assert_eq!(done.table.rows(), 2, "dims {dims:?} grid {grid:?}");
        assert_eq!(done.table.column("area").unwrap().iter().sum::<f64>() as usize, total);
        let mut areas = done.table.column("area").unwrap().to_vec();
        areas.sort_by(f64::total_cmp);
        assert_eq!(areas[0], 1.0);
    }

    #[test]
    fn an_object_across_many_tiles_is_one_object() {
        check([1, 37, 53], [1, 8, 11]);
    }

    #[test]
    fn an_object_across_blocks_of_a_volume_is_one_object() {
        check([17, 23, 29], [5, 7, 6]);
    }

    #[test]
    fn objects_in_different_planes_stay_apart_without_z_links() {
        let mut s = LabelStitcher::new([1, 4, 4], false);
        for b in schedule([2, 4, 4], [1, 4, 4], [0; 3]) {
            s.add(&b, &[1; 16], None);
        }
        assert_eq!(s.finish().table.rows(), 2);
    }
}

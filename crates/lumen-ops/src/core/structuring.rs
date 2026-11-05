//! N-D structuring elements (footprints) for morphological, rank, and peak operations.
//!
//! Stores a boolean mask of active neighborhood positions, detects when a footprint is a filled
//! hyper-rectangle (allowing $O(1)$-axis separable decomposition), and caches flat coordinate
//! offsets on the device through [`DeviceCache`].

use cubecl::prelude::*;
use lumen_core::memory::{DeviceArray, DeviceCache};
use lumen_core::Shape;

use crate::error::{OpsError, OpsResult};

static OFFSET_CACHE: DeviceCache = DeviceCache::new();

/// An N-D boolean neighborhood footprint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructuringElement {
    shape: Shape,
    mask: Vec<bool>,
}

impl StructuringElement {
    /// Creates a structuring element from `shape` and a C-order boolean `mask`.
    pub fn from_mask(shape: impl Into<Shape>, mask: Vec<bool>) -> OpsResult<Self> {
        let shape = shape.into();
        if shape.ndim() == 0 || shape.is_empty() {
            return Err(OpsError::InvalidWeights {
                what: "structuring element",
                reason: "structuring element must have at least one dimension and non-zero size",
            });
        }
        if mask.len() != shape.len() {
            return Err(OpsError::ShapeMismatch {
                first_name: "footprint shape",
                first: shape.dims().to_vec(),
                second_name: "footprint mask",
                second: vec![mask.len()],
            });
        }
        if !mask.iter().any(|&b| b) {
            return Err(OpsError::InvalidWeights {
                what: "structuring element",
                reason: "structuring element must contain at least one active element",
            });
        }
        Ok(Self { shape, mask })
    }

    /// Hyper-rectangular footprint with every element active (`dims` lengths per axis).
    pub fn rectangle(dims: impl Into<Shape>) -> OpsResult<Self> {
        let shape = dims.into();
        let len = shape.len();
        Self::from_mask(shape, vec![true; len])
    }

    /// Square (2-D) or hyper-cubic (N-D) footprint of side `width` along `ndim` dimensions.
    pub fn hypercube(ndim: usize, width: usize) -> OpsResult<Self> {
        Self::rectangle(vec![width; ndim])
    }

    /// N-D connectivity structure: a $3^D$ element where a neighbor at offset $\Delta$ is active iff
    /// $\sum_d |\Delta_d|^2 \le \text{connectivity}$ (`connectivity` in `1..=ndim`).
    pub fn connectivity(ndim: usize, connectivity: usize) -> OpsResult<Self> {
        if ndim == 0 {
            return Err(OpsError::InvalidParameter {
                name: "ndim",
                reason: "ndim must be at least 1".to_string(),
            });
        }
        let conn = connectivity.clamp(1, ndim);
        let dims = vec![3usize; ndim];
        let shape = Shape::new(dims);
        let strides = shape.strides();
        let mut mask = vec![false; shape.len()];
        for (idx, active) in mask.iter_mut().enumerate() {
            let mut sq_dist = 0usize;
            for d in 0..ndim {
                let coord = (idx / strides[d]) % 3;
                let delta = coord.abs_diff(1);
                sq_dist += delta * delta;
            }
            *active = sq_dist <= conn;
        }
        Self::from_mask(shape, mask)
    }

    /// 2-D circular disk footprint of radius `radius` (`(2*radius + 1) x (2*radius + 1)`), active where
    /// $\Delta y^2 + \Delta x^2 \le r^2$.
    pub fn disk(radius: usize) -> Self {
        let side = 2 * radius + 1;
        let r = radius as isize;
        let mut mask = Vec::with_capacity(side * side);
        for y in -r..=r {
            for x in -r..=r {
                mask.push(y * y + x * x <= r * r);
            }
        }
        Self { shape: Shape::new([side, side]), mask }
    }

    /// 3-D spherical ball footprint of radius `radius` (`(2*radius + 1)^3`), active where
    /// $\Delta z^2 + \Delta y^2 + \Delta x^2 \le r^2$.
    pub fn ball(radius: usize) -> Self {
        let side = 2 * radius + 1;
        let r = radius as isize;
        let mut mask = Vec::with_capacity(side * side * side);
        for z in -r..=r {
            for y in -r..=r {
                for x in -r..=r {
                    mask.push(z * z + y * y + x * x <= r * r);
                }
            }
        }
        Self { shape: Shape::new([side, side, side]), mask }
    }

    /// 2-D L1 diamond of radius `radius`, active where $|\Delta y| + |\Delta x| \le r$.
    pub fn diamond(radius: usize) -> Self {
        let side = 2 * radius + 1;
        let r = radius as isize;
        let mut mask = Vec::with_capacity(side * side);
        for y in -r..=r {
            for x in -r..=r {
                mask.push(y.abs() + x.abs() <= r);
            }
        }
        Self { shape: Shape::new([side, side]), mask }
    }

    /// 3-D L1 octahedron of radius `radius`, active where $|\Delta z| + |\Delta y| + |\Delta x| \le r$.
    pub fn octahedron(radius: usize) -> Self {
        let side = 2 * radius + 1;
        let r = radius as isize;
        let mut mask = Vec::with_capacity(side * side * side);
        for z in -r..=r {
            for y in -r..=r {
                for x in -r..=r {
                    mask.push(z.abs() + y.abs() + x.abs() <= r);
                }
            }
        }
        Self { shape: Shape::new([side, side, side]), mask }
    }

    /// 2-D axis-aligned ellipse with semi-axes `(radius_y, radius_x)`.
    pub fn ellipse(radius_y: usize, radius_x: usize) -> Self {
        let (h, w) = (2 * radius_y + 1, 2 * radius_x + 1);
        let (ry, rx) = (radius_y as f64, radius_x as f64);
        let mut mask = Vec::with_capacity(h * w);
        for y in -(radius_y as isize)..=(radius_y as isize) {
            for x in -(radius_x as isize)..=(radius_x as isize) {
                let ny = if ry > 0.0 { y as f64 / ry } else { 0.0 };
                let nx = if rx > 0.0 { x as f64 / rx } else { 0.0 };
                mask.push(ny * ny + nx * nx <= 1.0 + 1e-12);
            }
        }
        Self { shape: Shape::new([h, w]), mask }
    }

    /// Returns `true` if every position in the bounding box is active (so min/max filters can use
    /// separable 1-D passes along each axis).
    pub fn is_rectangle(&self) -> bool {
        self.mask.iter().all(|&b| b)
    }

    /// Reverses the footprint along every axis (turning correlation into convolution or dilation's
    /// reflected structuring element).
    pub fn mirrored(&self) -> Self {
        let mut mask = self.mask.clone();
        mask.reverse();
        Self { shape: self.shape.clone(), mask }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Number of active (`true`) positions in the footprint.
    pub fn count_active(&self) -> usize {
        self.mask.iter().filter(|&&b| b).count()
    }

    /// Halo `(before, after)` along each axis for given `origins` (defaults to `0` per axis).
    pub fn halo(&self, origins: Option<&[i32]>) -> Vec<(usize, usize)> {
        self.dims()
            .iter()
            .enumerate()
            .map(|(d, &size)| {
                let orig = origins.and_then(|o| o.get(d).copied()).unwrap_or(0);
                let left = (size / 2) as i32 + orig;
                let right = size as i32 - 1 - left;
                (left.max(0) as usize, right.max(0) as usize)
            })
            .collect()
    }

    /// Coordinate offsets `(delta_0, ..., delta_{D-1})` of every active element relative to the
    /// footprint center `(dims[d] / 2 + origin[d])`.
    pub fn coordinate_offsets(&self, origins: Option<&[i32]>) -> Vec<Vec<i32>> {
        let ndim = self.ndim();
        let strides = self.shape.strides();
        let centers: Vec<i32> = self
            .dims()
            .iter()
            .enumerate()
            .map(|(d, &s)| (s / 2) as i32 + origins.and_then(|o| o.get(d).copied()).unwrap_or(0))
            .collect();
        let mut coords = Vec::with_capacity(self.count_active());
        for (idx, &active) in self.mask.iter().enumerate() {
            if active {
                let mut offset = Vec::with_capacity(ndim);
                for d in 0..ndim {
                    let c = ((idx / strides[d]) % self.dims()[d]) as i32;
                    offset.push(c - centers[d]);
                }
                coords.push(offset);
            }
        }
        coords
    }

    /// Device-cached flattened `i32` coordinate offsets `[num_active, ndim]` for `client`.
    pub fn device_offsets(&self, client: &Client, origins: Option<&[i32]>) -> DeviceArray<i32> {
        let coords = self.coordinate_offsets(origins);
        let n_active = coords.len();
        let ndim = self.ndim();
        let key = format!("se-{:?}-{:?}-{n_active}", self.dims(), origins);
        OFFSET_CACHE.get_or_upload(client, &key, Shape::new([n_active, ndim]), || {
            coords.into_iter().flatten().collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_and_detects_rectangles_and_disks() {
        let rect = StructuringElement::rectangle([3, 5]).unwrap();
        assert!(rect.is_rectangle());
        assert_eq!(rect.count_active(), 15);
        assert_eq!(rect.halo(None), vec![(1, 1), (2, 2)]);

        let disk = StructuringElement::disk(1);
        assert!(!disk.is_rectangle());
        assert_eq!(disk.count_active(), 5);
        let conn = StructuringElement::connectivity(2, 1).unwrap();
        assert_eq!(conn.mask(), disk.mask());
    }
}

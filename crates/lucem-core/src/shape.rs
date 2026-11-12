//! Shapes of C-contiguous arrays (last axis contiguous), as numpy lays them out by default.

use crate::error::{CoreError, CoreResult};

/// Shape of a C-contiguous array. A shape with no dimensions is a scalar (one element).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(Vec<usize>);

/// One axis of a C-contiguous array seen as `[outer, len, inner]`: element `(o, i, k)` sits at
/// `(o · len + i) · inner + k`. Lets one 1-D kernel work along any axis of any rank: neighbouring
/// units differ in `k` (or in `i` when `inner == 1`), so reads coalesce on every axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisSplit {
    /// Product of the dimensions before the axis.
    pub outer: usize,
    /// Length of the axis.
    pub len: usize,
    /// Product of the dimensions after the axis (the axis's stride in elements).
    pub inner: usize,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    /// A 1-D shape of `len` elements.
    pub fn vector(len: usize) -> Self {
        Self(vec![len])
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.0.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `axis` normalized as numpy does (negative counts from the end).
    pub fn axis(&self, axis: isize) -> CoreResult<usize> {
        let ndim = self.ndim() as isize;
        let normalized = if axis < 0 { axis + ndim } else { axis };
        if (0..ndim).contains(&normalized) {
            Ok(normalized as usize)
        } else {
            Err(CoreError::AxisOutOfBounds { axis, ndim: self.ndim() })
        }
    }

    /// `axes` normalized (numpy rules), every axis at most once; `None` means every axis.
    pub fn axes(&self, axes: Option<&[isize]>) -> CoreResult<Vec<usize>> {
        let Some(axes) = axes else {
            return Ok((0..self.ndim()).collect());
        };
        let mut out = Vec::with_capacity(axes.len());
        for &axis in axes {
            let a = self.axis(axis)?;
            if out.contains(&a) {
                return Err(CoreError::RepeatedAxis { axis });
            }
            out.push(a);
        }
        Ok(out)
    }

    /// The `[outer, len, inner]` view of `axis` (already normalized).
    ///
    /// # Panics
    /// If `axis >= ndim`.
    pub fn axis_split(&self, axis: usize) -> AxisSplit {
        AxisSplit {
            outer: self.0[..axis].iter().product(),
            len: self.0[axis],
            inner: self.0[axis + 1..].iter().product(),
        }
    }

    /// C-order strides in elements.
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.ndim()];
        for i in (0..self.ndim().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.0[i + 1];
        }
        strides
    }

    /// `Ok` if `self` equals `other`, else [`CoreError::ShapeMismatch`].
    pub fn expect(&self, other: &Shape) -> CoreResult<()> {
        if self == other {
            Ok(())
        } else {
            Err(CoreError::ShapeMismatch { expected: self.0.clone(), actual: other.0.clone() })
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self(dims.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_every_axis_and_normalizes_like_numpy() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.len(), 24);
        assert_eq!(s.axis_split(0), AxisSplit { outer: 1, len: 2, inner: 12 });
        assert_eq!(s.axis_split(1), AxisSplit { outer: 2, len: 3, inner: 4 });
        assert_eq!(s.axis_split(2), AxisSplit { outer: 6, len: 4, inner: 1 });
        assert_eq!(s.axis(-1).unwrap(), 2);
        assert!(s.axis(3).is_err() && s.axis(-4).is_err());
        assert_eq!(s.strides(), vec![12, 4, 1]);
        assert_eq!(s.axes(None).unwrap(), vec![0, 1, 2]);
        assert!(s.axes(Some(&[0, -3])).is_err(), "repeated axis");
        assert_eq!(Shape::new(Vec::<usize>::new()).len(), 1, "scalar");
    }
}

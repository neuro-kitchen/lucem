//! Shared argument validation for image operations.

use cubecl::prelude::CubeElement;
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;

use crate::error::{OpsError, OpsResult};

/// Verifies that `a` and `b` have identical shapes.
pub fn same_shape<A: CubeElement, B: CubeElement>(
    first_name: &'static str,
    first: &DeviceArray<A>,
    second_name: &'static str,
    second: &DeviceArray<B>,
) -> OpsResult<()> {
    if first.shape() == second.shape() {
        Ok(())
    } else {
        Err(OpsError::ShapeMismatch {
            first_name,
            first: first.dims().to_vec(),
            second_name,
            second: second.dims().to_vec(),
        })
    }
}

/// Verifies that `shape_a` and `shape_b` are equal.
pub fn expect_shape(
    first_name: &'static str,
    first: &Shape,
    second_name: &'static str,
    second: &Shape,
) -> OpsResult<()> {
    if first == second {
        Ok(())
    } else {
        Err(OpsError::ShapeMismatch {
            first_name,
            first: first.dims().to_vec(),
            second_name,
            second: second.dims().to_vec(),
        })
    }
}

/// Verifies that `input` and `output` do not share the same underlying device buffer.
pub fn not_aliased<A: CubeElement, B: CubeElement>(
    input: &DeviceArray<A>,
    output: &DeviceArray<B>,
) -> OpsResult<()> {
    if input.shares_buffer_with(output) {
        Err(OpsError::AliasedBuffers)
    } else {
        Ok(())
    }
}

/// Verifies that `origin` lies in `-(size as i32 / 2) ..= ((size - 1) as i32 / 2)`.
pub fn origin(origin: i32, size: usize) -> OpsResult<()> {
    if size == 0 {
        return Err(OpsError::InvalidWeights { what: "window", reason: "size must be at least 1" });
    }
    let min = -((size / 2) as i32);
    let max = ((size - 1) / 2) as i32;
    if (min..=max).contains(&origin) {
        Ok(())
    } else {
        Err(OpsError::InvalidOrigin { origin, size, min, max })
    }
}

/// Verifies that a host slice of weights is non-empty and every weight is finite.
pub fn weights(w: &[f64]) -> OpsResult<()> {
    if w.is_empty() {
        return Err(OpsError::InvalidWeights { what: "weights", reason: "filter weights cannot be empty" });
    }
    if w.iter().any(|x| !x.is_finite()) {
        return Err(OpsError::InvalidWeights { what: "weights", reason: "filter weights must be finite" });
    }
    Ok(())
}

/// Verifies that `ndim` matches one of the allowed ranks.
pub fn ndim_in(what: &'static str, shape: &Shape, allowed: &[usize], expected: &'static str) -> OpsResult<()> {
    if allowed.contains(&shape.ndim()) {
        Ok(())
    } else {
        Err(OpsError::InvalidDimension { what, expected, actual: shape.ndim(), shape: shape.dims().to_vec() })
    }
}

/// Broadcasts a per-axis parameter slice `values` to `axes.len()`: a length-1 slice is repeated for
/// every axis; otherwise its length must equal `axes.len()`.
pub fn broadcast_per_axis<T: Copy>(name: &'static str, values: &[T], num_axes: usize) -> OpsResult<Vec<T>> {
    if num_axes == 0 {
        return Ok(Vec::new());
    }
    if values.len() == 1 {
        Ok(vec![values[0]; num_axes])
    } else if values.len() == num_axes {
        Ok(values.to_vec())
    } else {
        Err(OpsError::InvalidParameter {
            name,
            reason: format!("expected 1 or {num_axes} values, got {}", values.len()),
        })
    }
}

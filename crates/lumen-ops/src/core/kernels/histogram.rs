//! Histogram kernels using exact `u32` atomic increments.

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Bins float values in `[min_val, max_val]` into `nbins` equal-width buckets (clamping the upper
/// endpoint `max_val` into `nbins - 1`, matching NumPy / scikit-image `histogram` edge rules).
#[cube(launch)]
pub fn histogram_float_kernel<F: Float>(
    input: &[F],
    bins: &mut [Atomic<u32>],
    min_val: F,
    inv_span: F,
    nbins: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let v = input[i];
        if v >= min_val {
            let norm = (v - min_val) * inv_span;
            let raw = i32::cast_from(F::floor(norm * F::cast_from(nbins)));
            if raw >= 0i32 && raw <= nbins as i32 {
                let b = i32::min(raw, nbins as i32 - 1i32) as u32;
                Atomic::fetch_add(&bins[b as usize], 1u32);
            }
        }
    }
}

/// Direct integer histogram for unsigned integer values in `0..nbins`.
#[cube(launch)]
pub fn histogram_uint_kernel<I: Int>(
    input: &[I],
    bins: &mut [Atomic<u32>],
    nbins: u32,
    len: u32,
) {
    let i = element_position();
    if (i as u32) < len {
        let v = u32::cast_from(input[i]);
        if v < nbins {
            Atomic::fetch_add(&bins[v as usize], 1u32);
        }
    }
}

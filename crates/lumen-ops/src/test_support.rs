//! Test helpers for `lumen-ops`: re-exports `lumen_core::runtime_test!` and `assert_close`, plus
//! deterministic host data generators and `f64` reference boundary indexing.

pub use lumen_core::runtime_test;
pub use lumen_core::test_support::assert_close;

use crate::core::PadMode;

/// Deterministic pseudo-random `f64` test signal in `[-50.0, 51.2]` with no external RNG dependency.
pub fn sample_f64(len: usize) -> Vec<f64> {
    (0..len).map(|i| ((i.wrapping_mul(7919).wrapping_add(104_729)) % 1013) as f64 * 0.1 - 50.0).collect()
}

/// Deterministic positive `f64` test signal in `[0.1, 101.3]`.
pub fn sample_positive_f64(len: usize) -> Vec<f64> {
    (0..len).map(|i| ((i.wrapping_mul(6151).wrapping_add(49_999)) % 1013) as f64 * 0.1 + 0.1).collect()
}

/// Host `f64` reference for [`PadMode`] indexing along a 1-D axis of length `len`: returns `Some(idx)`
/// in `0..len` or `None` when `PadMode::Constant` reads outside `0..len`.
pub fn host_pad_index(p: isize, len: usize, mode: PadMode) -> Option<usize> {
    let n = len as isize;
    if p >= 0 && p < n {
        return Some(p as usize);
    }
    match mode {
        PadMode::Constant => None,
        PadMode::Nearest => Some(p.clamp(0, n - 1) as usize),
        PadMode::Wrap => Some(p.rem_euclid(n) as usize),
        PadMode::Reflect => {
            let period = 2 * n;
            let r = p.rem_euclid(period);
            Some((if r < n { r } else { period - 1 - r }) as usize)
        }
        PadMode::Mirror => {
            if n <= 1 {
                Some(0)
            } else {
                let period = 2 * n - 2;
                let r = p.rem_euclid(period);
                Some((if r < n { r } else { period - r }) as usize)
            }
        }
    }
}

/// Reads sample `p` along an axis starting at `base` with stride `stride` and length `len` using `mode`.
pub fn host_read_padded(data: &[f64], base: usize, stride: usize, p: isize, len: usize, cval: f64, mode: PadMode) -> f64 {
    match host_pad_index(p, len, mode) {
        Some(idx) => data[base + idx * stride],
        None => cval,
    }
}

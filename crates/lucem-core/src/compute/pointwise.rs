//! A run of pointwise ops applied to one pixel: the one definition shared by the fused pointwise
//! kernel (`lucem-ops`) and any kernel that folds trailing pointwise ops into a crop or reduction,
//! so a fused chain computes what its ops do one kernel each.
//!
//! A run is two device arrays: `codes[k]` (one of the `POINTWISE_*` codes) and `params[2k]`,
//! `params[2k + 1]` (that op's two numbers). The formulas match the single-op kernels in
//! `lucem-ops` (`affine_kernel`, `clip_kernel`, `adjust_gamma_kernel`, `rescale_clip_kernel`).

use cubecl::prelude::*;

/// `x * a + b` (affine scale and offset).
pub const POINTWISE_AFFINE: u32 = 0;
/// `clamp(x, a, b)`.
pub const POINTWISE_CLAMP: u32 = 1;
/// `b * max(x, 0)^a` (`a`: gamma, `b`: gain; matches `adjust_gamma_kernel`).
pub const POINTWISE_GAMMA: u32 = 2;
/// `clamp(x * a + b, 0, 1)` (`a`: scale `1 / (max - min)`, `b`: `-min * scale`; matches `Op::Rescale`).
pub const POINTWISE_RESCALE: u32 = 3;

/// `x` through the first `count` ops of a run.
#[cube]
pub fn apply_pointwise<F: Float>(x: F, codes: &[u32], params: &[F], count: u32) -> F {
    let mut v = x;
    for k in 0..count {
        let code = codes[k as usize];
        let a = params[(2u32 * k) as usize];
        let b = params[(2u32 * k + 1u32) as usize];
        if code == 0u32 {
            v = v * a + b;
        } else if code == 1u32 {
            v = F::clamp(v, a, b);
        } else if code == 2u32 {
            let pos = F::max(v, F::new(0.0f32));
            v = b * F::powf(pos, a);
        } else {
            v = F::clamp(v * a + b, F::new(0.0f32), F::new(1.0f32));
        }
    }
    v
}

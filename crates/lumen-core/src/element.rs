//! Element types kernels are generic over.
//!
//! Kernels take `F: Float` (or `I: Int`); host functions take [`ImageFloat`] / [`ImageInt`] so they
//! can size, upload and read buffers of the type and pass scalars of it to kernels. Nothing in
//! library code hard-codes `f32`: `f32` runs on every runtime, `f64` and `f16` only where
//! `client.properties().supports_type(..)` says so.

use cubecl::prelude::*;
use num_traits::{NumCast, ToPrimitive};

/// A CubeCL float that also crosses to and from the host and can be a scalar kernel argument
/// (`LaunchArg<RuntimeArg = Self>`; `Float` alone is not enough for a scalar argument).
pub trait ImageFloat: Float + CubeElement + LaunchArg<RuntimeArg = Self> {}

impl<F: Float + CubeElement + LaunchArg<RuntimeArg = F>> ImageFloat for F {}

/// A CubeCL integer that crosses to and from the host and can be a scalar kernel argument (labels,
/// counts, stored pixel types such as `u8` / `u16` / `u32`).
pub trait ImageInt: Int + CubeElement + LaunchArg<RuntimeArg = Self> {}

impl<I: Int + CubeElement + LaunchArg<RuntimeArg = I>> ImageInt for I {}

/// `value` as `F` (rounded to its precision). Host constants enter as `f64` and are cast once.
///
/// # Panics
/// If `value` is not representable in `F` (e.g. NaN in a type without it, or out of range).
pub fn cast<F: ImageFloat>(value: f64) -> F {
    <F as NumCast>::from(value).expect("finite value representable in the kernel float type")
}

/// `values` as `F`.
pub fn cast_all<F: ImageFloat>(values: &[f64]) -> Vec<F> {
    values.iter().map(|&v| cast(v)).collect()
}

/// `value` as `f64`.
pub fn to_f64<F: ImageFloat>(value: F) -> f64 {
    value.to_f64().expect("float converts to f64")
}

/// Whether the runtime behind `client` supports `E` (e.g. `f64` is missing on most wgpu adapters).
pub fn supports<E: CubeElement>(client: &Client) -> bool {
    client.properties().supports_type(E::elem_type_native())
}

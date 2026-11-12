//! Perceptual color difference metrics (CIE76, CIE94, CIEDE2000, CMC) in CIE L*a*b* space.

use cubecl::prelude::*;
use lucem_core::{
    compute::LaunchGeometry,
    element::{cast, ImageFloat},
    memory::DeviceArray,
    Shape,
};

use crate::{core::check, error::OpsResult};

use super::{
    kernels::delta::{cie76_kernel, ciede2000_kernel, ciede94_kernel, cmc_kernel},
    space::check_rgb_shape,
};

/// Compute pixel-wise CIE76 Euclidean color distance between two CIE L*a*b* images `[H, W, 3]`.
pub fn cie76<F: ImageFloat>(
    client: &Client,
    lab1: &DeviceArray<F>,
    lab2: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>> {
    let (h, w, num_pixels) = check_rgb_shape(lab1.shape(), "cie76")?;
    check::same_shape("lab1", lab1, "lab2", lab2)?;
    let out = DeviceArray::<F>::empty(client, Shape::new([h, w]))?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        cie76_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            lab1.binding(),
            lab2.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Compute pixel-wise CIE94 perceptual color difference between two CIE L*a*b* images `[H, W, 3]`.
#[allow(clippy::too_many_arguments)]
pub fn ciede94<F: ImageFloat>(
    client: &Client,
    lab1: &DeviceArray<F>,
    lab2: &DeviceArray<F>,
    k_l: f64,
    k_c: f64,
    k_h: f64,
    k_1: f64,
    k_2: f64,
) -> OpsResult<DeviceArray<F>> {
    let (h, w, num_pixels) = check_rgb_shape(lab1.shape(), "ciede94")?;
    check::same_shape("lab1", lab1, "lab2", lab2)?;
    let out = DeviceArray::<F>::empty(client, Shape::new([h, w]))?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        ciede94_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            lab1.binding(),
            lab2.binding(),
            out.binding(),
            cast::<F>(k_l),
            cast::<F>(k_c),
            cast::<F>(k_h),
            cast::<F>(k_1),
            cast::<F>(k_2),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Compute pixel-wise CIEDE2000 perceptual color difference between two CIE L*a*b* images `[H, W, 3]`.
pub fn ciede2000<F: ImageFloat>(
    client: &Client,
    lab1: &DeviceArray<F>,
    lab2: &DeviceArray<F>,
    k_l: f64,
    k_c: f64,
    k_h: f64,
) -> OpsResult<DeviceArray<F>> {
    let (h, w, num_pixels) = check_rgb_shape(lab1.shape(), "ciede2000")?;
    check::same_shape("lab1", lab1, "lab2", lab2)?;
    let out = DeviceArray::<F>::empty(client, Shape::new([h, w]))?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        ciede2000_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            lab1.binding(),
            lab2.binding(),
            out.binding(),
            k_l as f32,
            k_c as f32,
            k_h as f32,
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Compute pixel-wise CMC(l:c) color difference between two CIE L*a*b* images `[H, W, 3]`.
pub fn cmc<F: ImageFloat>(
    client: &Client,
    lab1: &DeviceArray<F>,
    lab2: &DeviceArray<F>,
    k_l: f64,
    k_c: f64,
) -> OpsResult<DeviceArray<F>> {
    let (h, w, num_pixels) = check_rgb_shape(lab1.shape(), "cmc")?;
    check::same_shape("lab1", lab1, "lab2", lab2)?;
    let out = DeviceArray::<F>::empty(client, Shape::new([h, w]))?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        cmc_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            lab1.binding(),
            lab2.binding(),
            out.binding(),
            k_l as f32,
            k_c as f32,
            num_pixels as u32,
        );
    }
    Ok(out)
}

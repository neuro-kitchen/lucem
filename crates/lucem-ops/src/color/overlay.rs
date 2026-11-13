//! False-color rendering of segmentation label maps (`colorize_labels`) using a device-cached palette.

use cubecl::prelude::*;
use lucem_core::{
    compute::LaunchGeometry,
    element::{cast, cast_all, ImageFloat},
    memory::{DeviceArray, DeviceCache},
    Shape,
};

use crate::{
    core::check,
    error::{OpsError, OpsResult},
};

use super::kernels::overlay::colorize_labels_gray_kernel;

static PALETTE_CACHE: DeviceCache = DeviceCache::new();

const DEFAULT_PALETTE_RGB: [f64; 30] = [
    1.0, 0.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 0.8, 0.0,
    1.0, 0.84, 0.0,
    1.0, 0.0, 1.0,
    0.0, 0.8, 0.8,
    1.0, 0.5, 0.0,
    0.58, 0.0, 0.83,
    0.6, 0.8, 0.2,
    1.0, 0.41, 0.71,
];

/// Render a 2-D label map (`u32`) into an RGB overlay `[H, W, 3]`, optionally blended with a
/// grayscale background image `[H, W]` using opacity `alpha`.
pub fn colorize_labels<F: ImageFloat>(
    client: &Client,
    labels: &DeviceArray<u32>,
    background_gray: Option<&DeviceArray<F>>,
    bg_label: u32,
    alpha: f64,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("colorize_labels", labels.shape(), &[2], "2")?;
    if labels.is_empty() {
        return Err(OpsError::EmptyInput("colorize_labels"));
    }
    let (h, w) = (labels.dims()[0], labels.dims()[1]);
    let num_pixels = h * w;

    let owned_bg;
    let bg_ref = if let Some(bg) = background_gray {
        check::same_shape("labels", labels, "background_gray", bg)?;
        bg
    } else {
        owned_bg = DeviceArray::<F>::zeros(client, labels.shape().clone())?;
        &owned_bg
    };

    let num_colors = DEFAULT_PALETTE_RGB.len() / 3;
    let pal = PALETTE_CACHE.get_or_upload::<F>(
        client,
        "default_label_palette_10",
        Shape::new([num_colors, 3]),
        || cast_all::<F>(&DEFAULT_PALETTE_RGB),
    );

    let effective_alpha = if background_gray.is_some() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0
    };

    let out = DeviceArray::<F>::empty(client, Shape::new([h, w, 3]))?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        colorize_labels_gray_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            labels.binding(),
            bg_ref.binding(),
            pal.binding(),
            out.binding(),
            num_colors as u32,
            bg_label,
            cast::<F>(effective_alpha),
            num_pixels as u32,
        );
    }
    Ok(out)
}

//! Histological Beer-Lambert stain unmixing (deconvolution) and remixing on the device.

use cubecl::prelude::*;
use lucem_core::{
    compute::LaunchGeometry,
    element::{cast_all, ImageFloat},
    memory::{DeviceArray, DeviceCache},
    Shape,
};

use crate::error::OpsResult;

use super::{
    kernels::stain::{mix_stains_kernel, unmix_stains_kernel},
    space::check_rgb_shape,
};

static STAIN_CACHE: DeviceCache = DeviceCache::new();

/// Standard histological stain presets based on Ruifrok & Johnston (2001) optical density vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StainPreset {
    /// Hematoxylin + Eosin + DAB (`hed_from_rgb` / `rgb_from_hed`).
    HematoxylinEosinDab,
    /// Hematoxylin + DAB + Eosin (`hdx`).
    HematoxylinDab,
    /// Feulgen + Light Green (`fgx`).
    FeulgenLightGreen,
    /// Giemsa (Methyl Blue + Eosin).
    Giemsa,
}

impl StainPreset {
    fn key(self, inverse: bool) -> &'static str {
        match (self, inverse) {
            (Self::HematoxylinEosinDab, false) => "stain_hed_fwd",
            (Self::HematoxylinEosinDab, true) => "stain_hed_inv",
            (Self::HematoxylinDab, false) => "stain_hdx_fwd",
            (Self::HematoxylinDab, true) => "stain_hdx_inv",
            (Self::FeulgenLightGreen, false) => "stain_fgx_fwd",
            (Self::FeulgenLightGreen, true) => "stain_fgx_inv",
            (Self::Giemsa, false) => "stain_giemsa_fwd",
            (Self::Giemsa, true) => "stain_giemsa_inv",
        }
    }
}

fn invert_3x3(m: &[f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let inv_det = if det.abs() > 1e-12 { 1.0 / det } else { 1.0 };
    [
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ]
}

fn preset_matrix(preset: StainPreset) -> [f64; 9] {
    match preset {
        StainPreset::HematoxylinEosinDab => [
            0.65, 0.70, 0.29,
            0.07, 0.99, 0.11,
            0.27, 0.57, 0.78,
        ],
        StainPreset::HematoxylinDab => [
            0.650, 0.704, 0.286,
            0.268, 0.570, 0.776,
            0.711, -0.423, 0.562,
        ],
        StainPreset::FeulgenLightGreen => [
            0.46420921, 0.83008335, 0.30827187,
            0.94705542, 0.25373821, 0.19650764,
            0.0, 0.0, 1.0,
        ],
        StainPreset::Giemsa => [
            0.834750233, 0.513556283, 0.196330403,
            0.092789, 0.954111, 0.283111,
            0.542588, -0.667389, 0.510096,
        ],
    }
}

fn get_stain_matrix<F: ImageFloat>(
    client: &Client,
    preset: StainPreset,
    inverse: bool,
) -> DeviceArray<F> {
    let fwd = preset_matrix(preset);
    let mat = if inverse { invert_3x3(&fwd) } else { fwd };
    STAIN_CACHE.get_or_upload::<F>(client, preset.key(inverse), Shape::new([3, 3]), || {
        cast_all::<F>(&mat)
    })
}

/// Separate an RGB image `[H, W, 3]` into optical density stain channels `[H, W, 3]` using `preset`.
pub fn unmix_stains<F: ImageFloat>(
    client: &Client,
    rgb: &DeviceArray<F>,
    preset: StainPreset,
) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(rgb.shape(), "unmix_stains")?;
    let inv_mat = get_stain_matrix::<F>(client, preset, true);
    let out = DeviceArray::<F>::empty(client, rgb.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        unmix_stains_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            rgb.binding(),
            inv_mat.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

/// Recombine optical density stain channels `[H, W, 3]` back into RGB `[H, W, 3]` using `preset`.
pub fn mix_stains<F: ImageFloat>(
    client: &Client,
    stains: &DeviceArray<F>,
    preset: StainPreset,
) -> OpsResult<DeviceArray<F>> {
    let (_, _, num_pixels) = check_rgb_shape(stains.shape(), "mix_stains")?;
    let fwd_mat = get_stain_matrix::<F>(client, preset, false);
    let out = DeviceArray::<F>::empty(client, stains.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    unsafe {
        mix_stains_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            stains.binding(),
            fwd_mat.binding(),
            out.binding(),
            num_pixels as u32,
        );
    }
    Ok(out)
}

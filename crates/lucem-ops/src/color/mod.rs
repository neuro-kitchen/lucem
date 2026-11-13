//! Color space conversions, Beer-Lambert histological stain unmixing, perceptual color
//! difference metrics (CIE76, CIE94, CIEDE2000, CMC), and label map false-color overlays.

pub mod delta;
pub(crate) mod kernels;
pub mod overlay;
pub mod space;
pub mod stain;

pub use delta::{cie76, ciede2000, ciede94, cmc};
pub use overlay::colorize_labels;
pub use space::{
    gray_to_rgb, hsv_to_rgb, lab_to_lch, lab_to_rgb, lab_to_xyz, lch_to_lab, luv_to_rgb,
    rgb_to_gray, rgb_to_hsv, rgb_to_lab, rgb_to_luv, rgb_to_xyz, rgb_to_ycbcr, rgb_to_yuv,
    rgba_to_rgb, xyz_to_lab, xyz_to_rgb, ycbcr_to_rgb, yuv_to_rgb,
};
pub use stain::{mix_stains, unmix_stains, StainPreset};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use cubecl::prelude::Client;
    use lucem_core::{memory::DeviceArray, Shape};

    fn color_roundtrips_work(client: &Client) {
        let rgb_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5,
        ];
        let rgb = DeviceArray::from_host(client, Shape::new([2, 2, 3]), rgb_data.clone()).unwrap();

        let hsv = rgb_to_hsv(client, &rgb).unwrap();
        let rgb_rt = hsv_to_rgb(client, &hsv).unwrap();
        let rt_host = rgb_rt.to_host(client);
        for (a, b) in rgb_data.iter().zip(rt_host.iter()) {
            assert!((a - b).abs() < 1e-4);
        }

        let lab = rgb_to_lab(client, &rgb).unwrap();
        let rgb_lab_rt = lab_to_rgb(client, &lab).unwrap();
        let lab_rt_host = rgb_lab_rt.to_host(client);
        for (a, b) in rgb_data.iter().zip(lab_rt_host.iter()) {
            assert!((a - b).abs() < 1e-3);
        }

        let de = ciede2000(client, &lab, &lab, 1.0, 1.0, 1.0).unwrap();
        let de_host = de.to_host(client);
        assert!(de_host.iter().all(|&v| v.abs() < 1e-5));
    }

    runtime_test!(test_color, color_roundtrips_work);
}

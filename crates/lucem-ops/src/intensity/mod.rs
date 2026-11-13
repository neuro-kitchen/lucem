//! Intensity conversions, contrast stretching, equalization, and stochastic noise synthesis.

pub mod contrast;
pub mod convert;
pub mod equalize;
pub(crate) mod kernels;
pub mod noise;

pub use contrast::{
    adjust_gamma, adjust_log, adjust_sigmoid, is_low_contrast, rescale_range, IntensityRange,
};
pub use convert::{crop, invert, pad, to_bool, to_float, to_u16, to_u32, to_u8};
pub use equalize::{cdf, cdf_device, clahe, equalize_hist, match_histograms, ClaheSettings};
pub use noise::{add_noise, NoiseModel};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use cubecl::prelude::Client;
    use lucem_core::{memory::DeviceArray, Shape};

    fn intensity_pipeline_works(client: &Client) {
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let img = DeviceArray::from_host(client, Shape::new([8, 8]), data.clone()).unwrap();

        let rescaled = rescale_range(client, &img, IntensityRange::Image, 0.2, 0.8).unwrap();
        let r_host = rescaled.to_host(client);
        assert!((r_host[0] - 0.2).abs() < 1e-5);
        assert!((r_host[63] - 0.8).abs() < 1e-5);

        let gamma = adjust_gamma(client, &img, 2.0, 1.0).unwrap();
        let g_host = gamma.to_host(client);
        assert!((g_host[32] - data[32].powi(2)).abs() < 1e-5);

        let eq = equalize_hist(client, &img, 64).unwrap();
        let eq_host = eq.to_host(client);
        assert!(eq_host[63] > eq_host[0]);

        let cl = clahe(
            client,
            &img,
            ClaheSettings {
                grid_size: (2, 2),
                clip_limit: 0.05,
                num_bins: 32,
            },
        )
        .unwrap();
        assert_eq!(cl.dims(), &[8, 8]);

        let noisy = add_noise(
            client,
            &img,
            NoiseModel::Gaussian {
                mean: 0.0,
                std: 0.05,
            },
            42,
            true,
        )
        .unwrap();
        let n_host = noisy.to_host(client);
        assert!(n_host.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    runtime_test!(test_intensity, intensity_pipeline_works);
}

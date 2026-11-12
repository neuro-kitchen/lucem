//! Kernel vs. end-to-end benchmarks for `lucem_ops::intensity` and `lucem_ops::color`.

use lucem_core::{
    compute::bench::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lucem_ops::{
    color::{ciede2000, rgb_to_hsv, rgb_to_lab, unmix_stains, StainPreset},
    intensity::{
        add_noise, clahe, equalize_hist, rescale_range, ClaheSettings, IntensityRange, NoiseModel,
    },
};

fn bench_intensity_suite(client: &Client) {
    let iters = 3usize;
    let h = 128usize;
    let w = 128usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 23) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();

    let rescale_t = time_device(client, iters, || {
        let _ = rescale_range(client, &arr, IntensityRange::Image, 0.0, 1.0).unwrap();
    });
    let eq_t = time_device(client, iters, || {
        let _ = equalize_hist(client, &arr, 256).unwrap();
    });
    let clahe_t = time_device(client, iters, || {
        let _ = clahe(client, &arr, ClaheSettings::default()).unwrap();
    });
    let noise_t = time_device(client, iters, || {
        let _ = add_noise(
            client,
            &arr,
            NoiseModel::Gaussian {
                mean: 0.0,
                std: 0.05,
            },
            42,
            true,
        )
        .unwrap();
    });

    let rgb_data: Vec<f32> = (0..(h * w * 3))
        .map(|i| ((i * 13) % 256) as f32 / 255.0)
        .collect();
    let rgb = DeviceArray::from_host(client, Shape::new([h, w, 3]), rgb_data).unwrap();
    let hsv_t = time_device(client, iters, || {
        let _ = rgb_to_hsv(client, &rgb).unwrap();
    });
    let lab = rgb_to_lab(client, &rgb).unwrap();
    let de_t = time_device(client, iters, || {
        let _ = ciede2000(client, &lab, &lab, 1.0, 1.0, 1.0).unwrap();
    });
    let stain_t = time_device(client, iters, || {
        let _ = unmix_stains(client, &rgb, StainPreset::HematoxylinEosinDab).unwrap();
    });

    eprintln!(
        "[bench_intensity_color] rescale={rescale_t:?} eq={eq_t:?} clahe={clahe_t:?} noise={noise_t:?} hsv={hsv_t:?} ciede2000={de_t:?} stain={stain_t:?}"
    );
}

#[test]
#[ignore = "benchmark: intensity and color ops"]
fn bench_intensity() {
    struct Task;
    impl lucem_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_intensity_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

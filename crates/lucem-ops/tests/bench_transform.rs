//! Kernel vs. end-to-end benchmarks for `lucem_ops::transform`.

use lucem_core::{
    compute::bench::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lucem_ops::transform::{
    gaussian_pyramid, hough_lines, integral_image, radon, resize, rotate, swirl, Pyramid,
    WarpSettings,
};

fn bench_transform_suite(client: &Client) {
    let iters = 2usize;
    let h = 64usize;
    let w = 64usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 47) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).unwrap();
    let mask_bits: Vec<u32> = data.iter().map(|&v| u32::from(v > 0.8)).collect();
    let mask = DeviceArray::from_host(client, Shape::new([h, w]), mask_bits).unwrap();

    let rot_t = time_device(client, iters, || {
        let _ = rotate(client, &arr, 30.0, false, None, &WarpSettings::default()).unwrap();
    });
    let res_t = time_device(client, iters, || {
        let _ = resize(client, &arr, [32, 32], true, &WarpSettings::default()).unwrap();
    });
    let swirl_t = time_device(client, iters, || {
        let _ = swirl(client, &arr, None, 2.0, 25.0, 0.0, &WarpSettings::default()).unwrap();
    });
    let int_t = time_device(client, iters, || {
        let _ = integral_image(client, &arr).unwrap();
    });
    let pyr_t = time_device(client, iters, || {
        let _ = gaussian_pyramid(
            client,
            &arr,
            2,
            &Pyramid::default(),
        )
        .unwrap();
    });
    let hough_t = time_device(client, iters, || {
        let thetas: Vec<f64> = (0..45).map(|i| (i as f64).to_radians()).collect();
        let _ = hough_lines(client, &mask, Some(&thetas)).unwrap();
    });
    let radon_t = time_device(client, iters, || {
        let thetas: Vec<f64> = (0..30).map(|i| i as f64 * 6.0).collect();
        let _ = radon(client, &arr, Some(&thetas)).unwrap();
    });

    eprintln!(
        "[bench_transform] rotate={rot_t:?} resize={res_t:?} swirl={swirl_t:?} integral={int_t:?} pyramid={pyr_t:?} hough={hough_t:?} radon={radon_t:?}"
    );
}

#[test]
#[ignore = "benchmark: transform ops"]
fn bench_transform() {
    struct Task;
    impl lucem_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_transform_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

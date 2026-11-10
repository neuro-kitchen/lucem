//! Kernel vs. end-to-end benchmarks for `lumen_ops::measure`.

use lumen_core::{
    compute::bench::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lumen_ops::measure::{
    blur_score, euler_number, hu_moments, label_components, marching_squares, moments, pearson_r,
    perimeter, shannon_entropy,
};

fn bench_measure_suite(client: &Client) {
    let iters = 3usize;
    let h = 128usize;
    let w = 128usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 37) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).unwrap();
    let mask_bits: Vec<u32> = data.iter().map(|&v| u32::from(v > 0.5)).collect();
    let mask = DeviceArray::from_host(client, Shape::new([h, w]), mask_bits).unwrap();

    let lbl_t = time_device(client, iters, || {
        let _ = label_components(client, &mask, None).unwrap();
    });
    let mom_t = time_device(client, iters, || {
        let _ = moments(client, &arr, 3, None).unwrap();
        let _ = hu_moments(client, &arr).unwrap();
    });
    let perim_t = time_device(client, iters, || {
        let _ = perimeter(client, &mask, 4).unwrap();
        let _ = euler_number(client, &mask, 2).unwrap();
    });
    let ent_t = time_device(client, iters, || {
        let _ = shannon_entropy(client, &arr, 256, 2.0).unwrap();
        let _ = blur_score(client, &arr).unwrap();
    });
    let coloc_t = time_device(client, iters, || {
        let _ = pearson_r(client, &arr, &arr).unwrap();
    });
    let ms_t = time_device(client, iters, || {
        let _ = marching_squares(client, &arr, 0.5).unwrap();
    });

    eprintln!(
        "[bench_measure] label={lbl_t:?} moments={mom_t:?} perim_euler={perim_t:?} entropy_blur={ent_t:?} pearson={coloc_t:?} marching_sq={ms_t:?}"
    );
}

#[test]
#[ignore = "benchmark: measure ops"]
fn bench_measure() {
    struct Task;
    impl lumen_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_measure_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

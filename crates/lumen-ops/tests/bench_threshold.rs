//! Kernel vs. end-to-end benchmarks for `lumen_ops::threshold`.

use lumen_core::{
    compute::bench::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lumen_ops::threshold::{
    hysteresis_threshold, niblack, otsu_device, sauvola, yen_device, WindowThreshold,
};

fn bench_threshold_suite(client: &Client) {
    let iters = 3usize;
    let h = 128usize;
    let w = 128usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 29) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();

    let otsu_t = time_device(client, iters, || {
        let _ = otsu_device(client, &arr, 256).unwrap();
    });
    let yen_t = time_device(client, iters, || {
        let _ = yen_device(client, &arr, 256).unwrap();
    });
    let nib_t = time_device(client, iters, || {
        let _ = niblack(client, &arr, &WindowThreshold::default()).unwrap();
    });
    let sauv_t = time_device(client, iters, || {
        let _ = sauvola(client, &arr, &WindowThreshold::default()).unwrap();
    });
    let hyst_t = time_device(client, iters, || {
        let _ = hysteresis_threshold(client, &arr, 0.3, 0.7).unwrap();
    });

    eprintln!(
        "[bench_threshold] otsu={otsu_t:?} yen={yen_t:?} niblack={nib_t:?} sauvola={sauv_t:?} hysteresis={hyst_t:?}"
    );
}

#[test]
#[ignore = "benchmark: threshold ops"]
fn bench_threshold() {
    struct Task;
    impl lumen_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_threshold_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

//! Kernel vs. end-to-end benchmarks for `lucem_ops::core` primitives.

use lucem_core::{
    compute::bench::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lucem_ops::core::{components, distance, histogram, reduce, scan, select};

fn bench_core_suite(client: &Client) {
    let iters = 3usize;
    let h = 128usize;
    let w = 128usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 31) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).unwrap();

    let red = time_device(client, iters, || {
        let _ = reduce::sum_min_max_device(client, &arr).unwrap();
    });
    let hist = time_device(client, iters, || {
        let _ = histogram::histogram_ranged_device(client, &arr, 256, (0.0, 1.0)).unwrap();
    });
    let scan_t = time_device(client, iters, || {
        let _ = scan::integral_scan(client, &arr).unwrap();
    });
    let sel = time_device(client, iters, || {
        let _ = select::kth_f32(client, &arr, (h * w) / 2).unwrap();
    });

    let mask_bits: Vec<u32> = data.iter().map(|&v| u32::from(v > 0.5)).collect();
    let mask = DeviceArray::from_host(client, Shape::new([h, w]), mask_bits).unwrap();
    let ccl = time_device(client, iters, || {
        let _ = components::label_components(client, &mask, None).unwrap();
    });
    let edt = time_device(client, iters, || {
        let _ = distance::euclidean_distance::<u32, f32>(client, &mask, None).unwrap();
    });

    eprintln!(
        "[bench_core] sum={red:?} hist={hist:?} integral={scan_t:?} kth={sel:?} ccl={ccl:?} edt={edt:?}"
    );
}

#[test]
#[ignore = "benchmark: core primitives"]
fn bench_core() {
    struct Task;
    impl lucem_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_core_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

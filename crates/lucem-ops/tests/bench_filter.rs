//! Kernel vs. end-to-end benchmarks for `lucem_ops::filter`.

use lucem_core::{
    compute::bench::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lucem_ops::core::StructuringElement;
use lucem_ops::filter::{
    box_blur, frangi, gaussian_blur, median_filter, sobel, BoxBlur, GaussianBlur, GradientFilter,
    OrderFilter, RidgeFilter,
};

fn bench_filter_suite(client: &Client) {
    let iters = 3usize;
    let h = 128usize;
    let w = 128usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 17) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();

    let gauss = time_device(client, iters, || {
        let _ = gaussian_blur(client, &arr, &[2.0, 2.0], &GaussianBlur::default()).unwrap();
    });
    let box_t = time_device(client, iters, || {
        let _ = box_blur(client, &arr, &[5, 5], &BoxBlur::default()).unwrap();
    });
    let sobel_t = time_device(client, iters, || {
        let _ = sobel(client, &arr, &GradientFilter::default()).unwrap();
    });
    let med = time_device(client, iters, || {
        let _ = median_filter(client, &arr, &StructuringElement::rectangle([3usize, 3]).unwrap(), &OrderFilter::default()).unwrap();
    });
    let frangi_t = time_device(client, iters, || {
        let _ = frangi(
            client,
            &arr,
            &RidgeFilter {
                sigmas: vec![1.0, 2.0],
                ..RidgeFilter::default()
            },
        )
        .unwrap();
    });

    eprintln!(
        "[bench_filter] gaussian={gauss:?} box={box_t:?} sobel={sobel_t:?} median={med:?} frangi={frangi_t:?}"
    );
}

#[test]
#[ignore = "benchmark: filter ops"]
fn bench_filter() {
    struct Task;
    impl lucem_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_filter_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

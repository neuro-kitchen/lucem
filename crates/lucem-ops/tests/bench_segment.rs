//! Kernel vs. end-to-end benchmarks for `lucem_ops::segment`.

use lucem_core::{
    compute::bench::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lucem_ops::segment::{
    chan_vese, clear_border, expand_labels, find_boundaries, slic, watershed, BoundaryMode,
    ChanVeseOptions, SlicOptions, WatershedOptions,
};

fn bench_segment_suite(client: &Client) {
    let iters = 2usize;
    let h = 64usize;
    let w = 64usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 41) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();
    let mut markers_host = vec![0u32; h * w];
    markers_host[16 * w + 16] = 1;
    markers_host[48 * w + 48] = 2;
    let markers = DeviceArray::from_host(client, Shape::new([h, w]), markers_host).unwrap();

    let ws_t = time_device(client, iters, || {
        let _ = watershed(client, &arr, &markers, None, None, &WatershedOptions::default()).unwrap();
    });
    let exp_t = time_device(client, iters, || {
        let _ = expand_labels(client, &markers, 8.0, None).unwrap();
    });
    let bnd_t = time_device(client, iters, || {
        let _ = find_boundaries(client, &markers, 1, BoundaryMode::Thick, 0).unwrap();
        let _ = clear_border(client, &markers, 0, 0).unwrap();
    });
    let slic_t = time_device(client, iters, || {
        let _ = slic(
            client,
            &arr,
            &SlicOptions {
                n_segments: 16,
                max_iter: 3,
                ..SlicOptions::default()
            },
        )
        .unwrap();
    });
    let cv_t = time_device(client, iters, || {
        let _ = chan_vese(
            client,
            &arr,
            None,
            &ChanVeseOptions {
                max_iter: 5,
                ..ChanVeseOptions::default()
            },
        )
        .unwrap();
    });

    eprintln!(
        "[bench_segment] watershed={ws_t:?} expand={exp_t:?} boundaries={bnd_t:?} slic={slic_t:?} chan_vese={cv_t:?}"
    );
}

#[test]
#[ignore = "benchmark: segment ops"]
fn bench_segment() {
    struct Task;
    impl lucem_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_segment_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

//! Kernel vs. end-to-end benchmarks for `lumen_ops::feature`.

use lumen_core::{
    compute::bench::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lumen_ops::feature::{
    blobs_dog, canny, harris, hog, lbp, local_peaks, structure_tensor, BlobDetector, Canny,
    Harris, Hog, LbpMethod, LocalPeaks,
};

fn bench_feature_suite(client: &Client) {
    let iters = 2usize;
    let h = 64usize;
    let w = 64usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 43) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();

    let peaks_t = time_device(client, iters, || {
        let _ = local_peaks(client, &arr, &LocalPeaks::default()).unwrap();
    });
    let canny_t = time_device(client, iters, || {
        let _ = canny(client, &arr, &Canny::default()).unwrap();
    });
    let harris_t = time_device(client, iters, || {
        let _ = harris(client, &arr, &Harris::default()).unwrap();
    });
    let st_t = time_device(client, iters, || {
        let _ = structure_tensor(client, &arr, 1.5, lumen_ops::core::PadMode::Reflect, 0.0).unwrap();
    });
    let lbp_t = time_device(client, iters, || {
        let _ = lbp(client, &arr, 8, 1.0, LbpMethod::Uniform).unwrap();
    });
    let hog_t = time_device(client, iters, || {
        let _ = hog(client, &arr, &Hog::default()).unwrap();
    });
    let dog_t = time_device(client, iters, || {
        let _ = blobs_dog(client, &arr, &BlobDetector::default()).unwrap();
    });

    eprintln!(
        "[bench_feature] peaks={peaks_t:?} canny={canny_t:?} harris={harris_t:?} struct_tensor={st_t:?} lbp={lbp_t:?} hog={hog_t:?} dog={dog_t:?}"
    );
}

#[test]
#[ignore = "benchmark: feature ops"]
fn bench_feature() {
    struct Task;
    impl lumen_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_feature_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

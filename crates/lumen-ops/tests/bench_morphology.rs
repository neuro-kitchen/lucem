//! Kernel vs. end-to-end benchmarks for `lumen_ops::morphology`.

use lumen_core::{
    compute::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lumen_ops::{
    core::StructuringElement,
    morphology::{
        binary_close, binary_open, close, isotropic_open, open, skeletonize, BinaryMorphology,
        GrayMorphology,
    },
};

fn bench_morphology_suite(client: &Client) {
    let iters = 3usize;
    let h = 128usize;
    let w = 128usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 19) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).unwrap();
    let mask_u8: Vec<u8> = data.iter().map(|&v| u8::from(v > 0.5)).collect();
    let mask = DeviceArray::from_host(client, Shape::new([h, w]), mask_u8).unwrap();
    let fp = StructuringElement::rectangle(&[5, 5]).unwrap();

    let gray_open = time_device(client, iters, || {
        let _ = open(client, &arr, Some(&fp), None, &GrayMorphology::default()).unwrap();
    });
    let gray_close = time_device(client, iters, || {
        let _ = close(client, &arr, Some(&fp), None, &GrayMorphology::default()).unwrap();
    });
    let bin_open = time_device(client, iters, || {
        let _ = binary_open(client, &mask, Some(&fp), &BinaryMorphology::default()).unwrap();
    });
    let bin_close = time_device(client, iters, || {
        let _ = binary_close(client, &mask, Some(&fp), &BinaryMorphology::default()).unwrap();
    });
    let iso_open = time_device(client, iters, || {
        let _ = isotropic_open(client, &mask, 4.0, None).unwrap();
    });
    let skel = time_device(client, iters, || {
        let _ = skeletonize(client, &mask).unwrap();
    });

    eprintln!(
        "[bench_morphology] gray_open={gray_open:?} gray_close={gray_close:?} bin_open={bin_open:?} bin_close={bin_close:?} iso_open={iso_open:?} skel={skel:?}"
    );
}

#[test]
#[ignore = "benchmark: morphology ops"]
fn bench_morphology() {
    struct Task;
    impl lumen_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_morphology_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

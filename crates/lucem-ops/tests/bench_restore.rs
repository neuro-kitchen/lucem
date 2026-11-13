//! Kernel vs. end-to-end benchmarks for `lucem_ops::restore`, `lucem_ops::register`, and `lucem_ops::metric`.

use lucem_core::{
    compute::bench::time_device,
    cubecl::prelude::Client,
    memory::DeviceArray,
    ComputeTarget, Shape,
};
use lucem_ops::{
    metric::{psnr, ssim, Ssim},
    register::{ilk_flow, phase_correlation, IlkFlow, PhaseCorrelation},
    restore::{
        bilateral, non_local_means, richardson_lucy, tv_chambolle, wiener_deconvolve, Bilateral,
        NonLocalMeans, RichardsonLucy, TvChambolle, WienerDeconvolve,
    },
};

fn bench_restore_suite(client: &Client) {
    let iters = 2usize;
    let h = 64usize;
    let w = 64usize;
    let data: Vec<f32> = (0..(h * w)).map(|i| ((i * 53) % 256) as f32 / 255.0).collect();
    let arr = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();
    let psf_data = vec![1.0f32 / 9.0; 9];
    let psf = DeviceArray::from_host(client, Shape::new([3, 3]), psf_data).unwrap();

    let tv_t = time_device(client, iters, || {
        let _ = tv_chambolle(
            client,
            &arr,
            &TvChambolle {
                max_num_iter: 5,
                ..TvChambolle::default()
            },
        )
        .unwrap();
    });
    let bil_t = time_device(client, iters, || {
        let _ = bilateral(client, &arr, &Bilateral::default()).unwrap();
    });
    let nlm_t = time_device(client, iters, || {
        let _ = non_local_means(
            client,
            &arr,
            &NonLocalMeans {
                patch_size: 3,
                patch_distance: 3,
                ..NonLocalMeans::default()
            },
        )
        .unwrap();
    });
    let rl_t = time_device(client, iters, || {
        let _ = richardson_lucy(
            client,
            &arr,
            &psf,
            &RichardsonLucy {
                num_iter: 3,
                ..RichardsonLucy::default()
            },
        )
        .unwrap();
    });
    let wiener_t = time_device(client, iters, || {
        let _ = wiener_deconvolve(client, &arr, &psf, &WienerDeconvolve::default()).unwrap();
    });
    let pc_t = time_device(client, iters, || {
        let _ = phase_correlation(client, &arr, &arr, &PhaseCorrelation::default()).unwrap();
    });
    let ilk_t = time_device(client, iters, || {
        let _ = ilk_flow(
            client,
            &arr,
            &arr,
            &IlkFlow {
                radius: 3,
                num_warp: 2,
                ..IlkFlow::default()
            },
        )
        .unwrap();
    });
    let ssim_t = time_device(client, iters, || {
        let _ = psnr(client, &arr, &arr, Some(1.0)).unwrap();
        let _ = ssim(client, &arr, &arr, &Ssim::default()).unwrap();
    });

    eprintln!(
        "[bench_restore_register_metric] tv={tv_t:?} bilateral={bil_t:?} nlm={nlm_t:?} rl={rl_t:?} wiener={wiener_t:?} phase_corr={pc_t:?} ilk={ilk_t:?} ssim={ssim_t:?}"
    );
}

#[test]
#[ignore = "benchmark: restore, register, and metric ops"]
fn bench_restore() {
    struct Task;
    impl lucem_core::compute::ComputeTask for Task {
        type Output = ();
        fn run(self, client: Client) {
            bench_restore_suite(&client);
        }
    }
    for target in ComputeTarget::for_tests() {
        target.run(Task).expect("runtime");
    }
}

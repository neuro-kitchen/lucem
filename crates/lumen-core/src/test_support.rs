//! Test helpers shared by every lumen crate (not part of the public API).

/// Device tests share the devices (read); a test that shuts them down takes them alone (write),
/// so it neither stops the others' devices nor counts their threads.
pub static DEVICE_TESTS: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Defines a `#[test]` that runs `$body(&client)` on every CubeCL runtime of
/// [`ComputeTarget::for_tests`](crate::ComputeTarget::for_tests) (compiled in, limited by
/// `LUMEN_TEST_RUNTIMES`), so tests never name a runtime. Several bugs appear on one runtime only
/// (WGSL has no infinity literal, SPIR-V rejects some loop conditions, the CPU runtime rejects short
/// writes), so every device test runs everywhere.
///
/// ```ignore
/// fn round_trips(client: &Client) { /* … */ }
/// runtime_test!(test_round_trips, round_trips);
/// ```
#[macro_export]
macro_rules! runtime_test {
    ($name:ident, $body:ident) => {
        #[test]
        fn $name() {
            struct Task;
            impl $crate::compute::ComputeTask for Task {
                type Output = ();
                fn run(self, client: $crate::cubecl::prelude::Client) {
                    $body(&client)
                }
            }
            let _shared = $crate::test_support::DEVICE_TESTS.read().unwrap_or_else(|e| e.into_inner());
            let targets = $crate::ComputeTarget::for_tests();
            assert!(!targets.is_empty(), "no CubeCL runtime to test (compiled in and allowed by LUMEN_TEST_RUNTIMES)");
            for target in targets {
                target.run(Task).expect("compiled-in runtime");
            }
        }
    };
}

/// Asserts that `got` matches the `f64` reference `want` element by element: `|got − want| ≤
/// rtol · |want| + atol`, with `atol = rtol · max|want|` (so values near zero in a large-valued array
/// are not held to a relative tolerance). Panics naming the first failing index and `context`.
pub fn assert_close<F: Into<f64> + Copy>(got: &[F], want: &[f64], rtol: f64, context: &str) {
    assert_eq!(got.len(), want.len(), "{context}: {} values, expected {}", got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let atol = rtol * scale;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let g: f64 = g.into();
        assert!((g - w).abs() <= rtol * w.abs() + atol, "{context}: index {i}: got {g}, expected {w} (rtol {rtol})");
    }
}

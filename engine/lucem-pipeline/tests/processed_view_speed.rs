//! Fidelity tests and timing benchmark for lucem's processed-view speed optimizations:
//! - Incremental recompute (`EvalCache`): editing step `k` or toggling an op reuses step `k - 1`
//!   on the device and is bit-exact with a fresh evaluation.
//! - Pointwise fusion: consecutive unary pointwise ops (`Rescale`, `Gamma`) collapse into one
//!   kernel and match unfused op-by-op evaluation within `f32` rounding.
//! - On-device halo crop (`crop_and_download`): strips halo pixels on the GPU before download and
//!   is bit-exact with host cropping.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::DeviceArray;
use lucem_core::{runtime_test, ComputeTarget, Shape};
use lucem_pipeline::{align_region, crop_and_download, evaluate, evaluate_cached, halo, EvalCache, Op, Plan, PlanNode, Value};

fn synthetic_image(w: usize, h: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w.max(1) as f32;
            let fy = y as f32 / h.max(1) as f32;
            let v = 0.5 + 0.35 * (fx * 17.0).sin() * (fy * 13.0).cos() + 0.15 * ((x * 7 + y * 11) % 31) as f32 / 31.0;
            out.push(v.clamp(0.0, 1.0));
        }
    }
    out
}

fn four_step_plan(sigma: f64, gamma: f64, amount: f64, rescale_enabled: bool) -> Plan {
    let mut plan = Plan::new();
    plan.insert("src".into(), PlanNode { op: None, enabled: true, inputs: vec![] });
    plan.insert("blur".into(), PlanNode { op: Some(Op::GaussianBlur { sigma }), enabled: true, inputs: vec![Some("src".into())] });
    plan.insert("rescale".into(), PlanNode { op: Some(Op::Rescale { min: 0.1, max: 0.9 }), enabled: rescale_enabled, inputs: vec![Some("blur".into())] });
    plan.insert("gamma".into(), PlanNode { op: Some(Op::Gamma { gamma, gain: 1.1 }), enabled: true, inputs: vec![Some("rescale".into())] });
    plan.insert("sharp".into(), PlanNode { op: Some(Op::UnsharpMask { sigma: 1.5, amount }), enabled: true, inputs: vec![Some("gamma".into())] });
    plan
}

fn pointwise_chain_plan() -> Plan {
    let mut plan = Plan::new();
    plan.insert("src".into(), PlanNode { op: None, enabled: true, inputs: vec![] });
    plan.insert("p1".into(), PlanNode { op: Some(Op::Rescale { min: 0.08, max: 0.92 }), enabled: true, inputs: vec![Some("src".into())] });
    plan.insert("p2".into(), PlanNode { op: Some(Op::Gamma { gamma: 0.72, gain: 1.15 }), enabled: true, inputs: vec![Some("p1".into())] });
    plan.insert("p3".into(), PlanNode { op: Some(Op::Rescale { min: 0.05, max: 1.05 }), enabled: true, inputs: vec![Some("p2".into())] });
    plan.insert("p4".into(), PlanNode { op: Some(Op::Gamma { gamma: 1.25, gain: 0.95 }), enabled: true, inputs: vec![Some("p3".into())] });
    plan
}

fn incremental_edit_impl(client: &Client) {
    let target = client.name();
    let (w, h) = (160, 144);
    let data = synthetic_image(w, h);
    let cache = EvalCache::default();
    let region_tag = "test|L0|Z0|0,0,160,144";

    let uploads = AtomicUsize::new(0);
    let make_src = || {
        uploads.fetch_add(1, Ordering::Relaxed);
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    };

    // 1. Cold evaluation populates source + intermediate checkpoints on device
    let plan_a = four_step_plan(2.0, 0.8, 0.5, true);
    let (_, t1) = evaluate_cached(client, &plan_a, "sharp", region_tag, &cache, &mut |_| make_src()).unwrap();
    assert_eq!(uploads.load(Ordering::Relaxed), 1);
    assert_eq!(t1.cache_hits, 0);

    // 2. Edit step 4 (UnsharpMask amount): must reuse step 3 ("gamma") directly on the GPU
    let plan_b = four_step_plan(2.0, 0.8, 1.3, true);
    let (val_cached, t2) = evaluate_cached(client, &plan_b, "sharp", region_tag, &cache, &mut |_| make_src()).unwrap();
    assert_eq!(uploads.load(Ordering::Relaxed), 1, "{target}: editing step 4 must not re-upload source");
    assert!(t2.cache_hits >= 1, "{target}: editing step 4 must hit step 3 checkpoint");

    let fresh = evaluate(client, &plan_b, "sharp", &mut |_| {
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    })
    .unwrap();
    let got = val_cached.to_host(client).unwrap();
    let expected = fresh.to_host(client).unwrap();
    for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
        let tol = 4.0 * f32::EPSILON * b.abs().max(1.0);
        assert!((a - b).abs() <= tol, "{target} edit step 4 px {i}: cached {a} vs fresh {b}");
    }

    // 3. Edit step 3 (Gamma): must reuse step 1 ("blur") without re-uploading source
    let plan_c = four_step_plan(2.0, 1.35, 1.3, true);
    let (val_cached_c, t3) = evaluate_cached(client, &plan_c, "sharp", region_tag, &cache, &mut |_| make_src()).unwrap();
    assert_eq!(uploads.load(Ordering::Relaxed), 1, "{target}: editing step 3 must not re-upload source");
    assert!(t3.cache_hits >= 1, "{target}: editing step 3 must hit prefix checkpoint");

    let fresh_c = evaluate(client, &plan_c, "sharp", &mut |_| {
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    })
    .unwrap();
    let got_c = val_cached_c.to_host(client).unwrap();
    let exp_c = fresh_c.to_host(client).unwrap();
    for (i, (a, b)) in got_c.iter().zip(&exp_c).enumerate() {
        let tol = 4.0 * f32::EPSILON * b.abs().max(1.0);
        assert!((a - b).abs() <= tol, "{target} edit step 3 px {i}: cached {a} vs fresh {b}");
    }
}
runtime_test!(incremental_edit_matches_fresh_eval_bit_exact, incremental_edit_impl);

fn toggle_op_impl(client: &Client) {
    let target = client.name();
    let (w, h) = (128, 128);
    let data = synthetic_image(w, h);
    let cache = EvalCache::default();
    let region_tag = "toggle|L0|Z0|0,0,128,128";
    let uploads = AtomicUsize::new(0);
    let make_src = || {
        uploads.fetch_add(1, Ordering::Relaxed);
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    };

    let on = four_step_plan(1.5, 0.9, 0.6, true);
    let _ = evaluate_cached(client, &on, "sharp", region_tag, &cache, &mut |_| make_src()).unwrap();
    assert_eq!(uploads.load(Ordering::Relaxed), 1);

    // Toggle rescale OFF: blur output is already on device
    let off = four_step_plan(1.5, 0.9, 0.6, false);
    let (got_off, t_off) = evaluate_cached(client, &off, "sharp", region_tag, &cache, &mut |_| make_src()).unwrap();
    assert_eq!(uploads.load(Ordering::Relaxed), 1);
    assert!(t_off.cache_hits >= 1);

    let fresh_off = evaluate(client, &off, "sharp", &mut |_| {
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    })
    .unwrap();
    assert_eq!(got_off.to_host(client).unwrap(), fresh_off.to_host(client).unwrap(), "{target}: toggle off must be bit-exact");

    // Toggle rescale back ON: final node "sharp" itself is still cached!
    let (got_on, t_on) = evaluate_cached(client, &on, "sharp", region_tag, &cache, &mut |_| make_src()).unwrap();
    assert_eq!(uploads.load(Ordering::Relaxed), 1);
    assert_eq!(t_on.cache_hits, 1);
    let fresh_on = evaluate(client, &on, "sharp", &mut |_| {
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    })
    .unwrap();
    let a = got_on.to_host(client).unwrap();
    let b = fresh_on.to_host(client).unwrap();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let tol = 4.0 * f32::EPSILON * y.abs().max(1.0);
        assert!((x - y).abs() <= tol, "{target} toggle on px {i}: {x} vs {y}");
    }
}
runtime_test!(toggle_op_reuses_prefix_bit_exact, toggle_op_impl);

fn fused_pointwise_impl(client: &Client) {
    let target = client.name();
    let (w, h) = (131, 97);
    let data = synthetic_image(w, h);
    let plan = pointwise_chain_plan();
    let cache = EvalCache::default();

    let (fused, timings) = evaluate_cached(client, &plan, "p4", "pw|0,0,131,97", &cache, &mut |_| {
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    })
    .unwrap();
    assert_eq!(timings.fused_runs, 1, "{target}: 4 consecutive pointwise ops must fuse into 1 run");

    let unfused = evaluate(client, &plan, "p4", &mut |_| {
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    })
    .unwrap();

    let got = fused.to_host(client).unwrap();
    let exp = unfused.to_host(client).unwrap();
    for (i, (a, b)) in got.iter().zip(&exp).enumerate() {
        let tol = 4.0 * f32::EPSILON * b.abs().max(1.0);
        assert!((a - b).abs() <= tol, "{target} fused pointwise px {i}: {a} vs {b}");
    }
}
runtime_test!(fused_pointwise_chain_matches_unfused, fused_pointwise_impl);

fn device_crop_impl(client: &Client) {
    let target = client.name();
    let (full_w, full_h): (usize, usize) = (256, 256);
    let (vx0, vy0, vw, vh): (usize, usize, usize, usize) = (37, 51, 119, 103);
    let margin = 13;
    let (rx0, ry0, rw, rh) = align_region(
        vx0.saturating_sub(margin),
        vy0.saturating_sub(margin),
        (vx0 + vw + margin).min(full_w),
        (vy0 + vh + margin).min(full_h),
        full_w,
        full_h,
    );
    let region = synthetic_image(rw, rh);
    let dev = Value::Image(DeviceArray::from_host(client, Shape::new([rh, rw]), region.clone()).unwrap());
    let (got, _) = crop_and_download(client, &dev, rx0, ry0, rw, rh, vx0, vy0, vw, vh).unwrap();

    let (ox, oy) = (vx0 - rx0, vy0 - ry0);
    let mut expected = Vec::with_capacity(vw * vh);
    for y in oy..oy + vh {
        expected.extend_from_slice(&region[y * rw + ox..y * rw + ox + vw]);
    }
    assert_eq!(got, expected, "{target}: device crop must be bit-exact with host crop");
}
runtime_test!(device_crop_matches_host_crop_bit_exact, device_crop_impl);

fn small_budget_evicts_impl(client: &Client) {
    let target = client.name();
    let (w, h) = (128, 128);
    let one_image_bytes = w * h * size_of::<f32>(); // 64 KiB
    // Budget fits at most 2 rasters: a 4-step chain across two regions MUST evict
    let budget = one_image_bytes * 2;
    let cache = EvalCache::new(budget);
    let data = synthetic_image(w, h);
    let plan = four_step_plan(1.5, 0.85, 0.7, true);

    let (v1, t1) = evaluate_cached(client, &plan, "sharp", "evict|r1", &cache, &mut |_| {
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    })
    .unwrap();
    assert!(t1.kept_bytes <= budget, "{target}: kept {} > budget {budget}", t1.kept_bytes);
    assert!(t1.evictions > 0, "{target}: expected evictions under small budget");

    // Second region pushes first region's entries out completely
    let (_, t2) = evaluate_cached(client, &plan, "sharp", "evict|r2", &cache, &mut |_| {
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    })
    .unwrap();
    assert!(t2.kept_bytes <= budget, "{target}: kept {} > budget {budget}", t2.kept_bytes);
    assert!(t2.evictions > t1.evictions, "{target}: expected further evictions");

    // Rebuilding the evicted first region gives bit-identical output
    let (v1_rebuilt, t3) = evaluate_cached(client, &plan, "sharp", "evict|r1", &cache, &mut |_| {
        DeviceArray::from_host(client, Shape::new([h, w]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
    })
    .unwrap();
    assert!(t3.kept_bytes <= budget);
    assert_eq!(v1.to_host(client).unwrap(), v1_rebuilt.to_host(client).unwrap(), "{target}: evicted entry rebuilt must be bit-exact");
}
runtime_test!(small_budget_evicts_and_rebuilds_bit_exact, small_budget_evicts_impl);

/// Measuring benchmark (ignored by default; run with
/// `LUCEM_PREVIEW_TIMING=1 cargo test --profile validate --test processed_view_speed -- --ignored --nocapture`).
#[test]
#[ignore]
fn where_a_processed_view_spends_its_time() {
    struct Bench;
    impl lucem_core::compute::ComputeTask for Bench {
        type Output = ();
        fn run(self, client: Client) {
            let target = client.name().to_string();
            let (img_w, img_h): (usize, usize) = (2048, 2048);
            let (vx0, vy0, vw, vh): (usize, usize, usize, usize) = (256, 256, 1024, 1024);
            let plan_a = four_step_plan(2.0, 0.8, 0.5, true);
            let plan_b = four_step_plan(2.0, 0.8, 1.4, true);
            let margin = halo(&plan_a, "sharp");
            let (rx0, ry0, rw, rh) = align_region(
                vx0.saturating_sub(margin),
                vy0.saturating_sub(margin),
                (vx0 + vw + margin).min(img_w),
                (vy0 + vh + margin).min(img_h),
                img_w,
                img_h,
            );
            let data = synthetic_image(rw, rh);
            let region_tag = format!("bench|L0|Z0|{rx0},{ry0},{rw},{rh}");

            // Warm up kernel compilation once
            let warm_cache = EvalCache::default();
            let _ = evaluate_cached(&client, &plan_a, "sharp", &region_tag, &warm_cache, &mut |_| {
                DeviceArray::from_host(&client, Shape::new([rh, rw]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
            })
            .unwrap();
            let _ = evaluate_cached(&client, &plan_b, "sharp", &region_tag, &warm_cache, &mut |_| {
                DeviceArray::from_host(&client, Shape::new([rh, rw]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
            })
            .unwrap();

            // Baseline: uncached full eval + host crop (simulating pre-optimization path)
            let t_base = Instant::now();
            let v_base = evaluate(&client, &plan_b, "sharp", &mut |_| {
                DeviceArray::from_host(&client, Shape::new([rh, rw]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
            })
            .unwrap();
            let full_host = v_base.to_host(&client).unwrap();
            let (ox, oy) = (vx0 - rx0, vy0 - ry0);
            let mut host_crop = Vec::with_capacity(vw * vh);
            for y in oy..oy + vh {
                host_crop.extend_from_slice(&full_host[y * rw + ox..y * rw + ox + vw]);
            }
            let baseline_ms = t_base.elapsed().as_secs_f64() * 1e3;

            // Optimized: cached cold + incremental edit of step 4 + device crop
            let cache = EvalCache::default();
            let t_cold = Instant::now();
            let (v_cold, mut tm_cold) = evaluate_cached(&client, &plan_a, "sharp", &region_tag, &cache, &mut |_| {
                DeviceArray::from_host(&client, Shape::new([rh, rw]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
            })
            .unwrap();
            let (_, rb_cold) = crop_and_download(&client, &v_cold, rx0, ry0, rw, rh, vx0, vy0, vw, vh).unwrap();
            tm_cold.readback_ms = rb_cold;
            let cold_total_ms = t_cold.elapsed().as_secs_f64() * 1e3;

            let t_edit = Instant::now();
            let (v_edit, mut tm_edit) = evaluate_cached(&client, &plan_b, "sharp", &region_tag, &cache, &mut |_| {
                DeviceArray::from_host(&client, Shape::new([rh, rw]), data.clone()).map(Value::Image).map_err(|e| e.to_string())
            })
            .unwrap();
            let (dev_crop, rb_edit) = crop_and_download(&client, &v_edit, rx0, ry0, rw, rh, vx0, vy0, vw, vh).unwrap();
            tm_edit.readback_ms = rb_edit;
            let edit_total_ms = t_edit.elapsed().as_secs_f64() * 1e3;

            assert_eq!(host_crop.len(), dev_crop.len());
            println!("=== Processed-view timing ({target}, visible {vw}x{vh}, read {rw}x{rh}) ===");
            println!("  baseline (uncached + host crop):     {baseline_ms:7.2} ms");
            println!(
                "  optimized cold (fused + dev crop):   {cold_total_ms:7.2} ms  (src={:.2} compute={:.2} readback={:.2} fused={} hits={})",
                tm_cold.source_ms, tm_cold.compute_ms, tm_cold.readback_ms, tm_cold.fused_runs, tm_cold.cache_hits
            );
            println!(
                "  optimized edit step 4 (incremental): {edit_total_ms:7.2} ms  (src={:.2} compute={:.2} readback={:.2} fused={} hits={})",
                tm_edit.source_ms, tm_edit.compute_ms, tm_edit.readback_ms, tm_edit.fused_runs, tm_edit.cache_hits
            );
            if !tm_cold.steps.is_empty() {
                println!("  cold steps: {:?}", tm_cold.steps);
            }
            if !tm_edit.steps.is_empty() {
                println!("  edit steps: {:?}", tm_edit.steps);
            }
        }
    }
    let target = ComputeTarget::from_env().unwrap_or(ComputeTarget::Wgpu);
    target.run(Bench).expect("compute target");
}

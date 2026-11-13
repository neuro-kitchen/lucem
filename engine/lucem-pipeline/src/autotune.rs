//! Tunes lucem's operators on this device ahead of time, as dsp-kitchen's `autotune`.
//!
//! Every operator of the [`catalog`] runs once with its default settings on synthetic 2D images
//! over a grid of sizes, so its kernels are compiled, and any autotuned choice made, then saved
//! (CubeCL's `persistence` and the compiled-kernel cache, in lucem's cache folder). A first real run
//! then neither stops to compile nor to tune; a size off the grid still tunes when first met.
//! `lucem autotune` runs this, then every model plugin's own tuning ([`lucem_ml`]'s
//! `tune_plugins`).

use std::time::Instant;

use lucem_core::compute::bench::sync;
use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::DeviceArray;
use lucem_core::Shape;

use crate::ops::{catalog, Op, PortType, Value};

/// Image sides tuned by default.
pub const DEFAULT_SIDES: [usize; 3] = [256, 1024, 2048];

/// What a run of [`autotune`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct AutotuneSummary {
    /// Operator × size runs done.
    pub runs: usize,
    /// Operators that failed at a size: `(tag, side, why)`. The rest of the grid still ran.
    pub failed: Vec<(String, usize, String)>,
    pub seconds: f64,
}

/// A textured image in 0..1: smooth blobs and fine noise, so thresholds, labels and
/// iterative operators meet real-looking structure.
fn image(side: usize) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..side * side)
        .map(|i| {
            let (y, x) = ((i / side) as f32, (i % side) as f32);
            let blobs = ((x * 0.09).sin() * (y * 0.07).cos() * 0.5 + 0.5).powi(3);
            (blobs * 0.85 + noise() * 0.15).clamp(0.0, 1.0)
        })
        .collect()
}

/// An input for `port` on the device: the image, its mask (above 0.5) or labels (one per mask
/// cell of a 16 px grid).
fn input(client: &Client, port: PortType, side: usize, pixels: &[f32]) -> Result<Value, String> {
    let shape = || Shape::new([side, side]);
    let e = |e: lucem_core::CoreError| e.to_string();
    Ok(match port {
        PortType::Image => Value::Image(DeviceArray::from_host(client, shape(), pixels.to_vec()).map_err(e)?),
        PortType::Mask => Value::Mask(DeviceArray::from_host(client, shape(), pixels.iter().map(|&v| u32::from(v > 0.5)).collect()).map_err(e)?),
        PortType::Labels => {
            let cells = side.div_ceil(16);
            let labels = pixels.iter().enumerate().map(|(i, &v)| if v > 0.5 { ((i / side / 16) * cells + (i % side / 16) + 1) as u32 } else { 0 }).collect();
            Value::Labels(DeviceArray::from_host(client, shape(), labels).map_err(e)?)
        }
        PortType::Table => return Err("a table cannot be an input".into()),
    })
}

fn run_one(client: &Client, op: &Op, side: usize, pixels: &[f32]) -> Result<(), String> {
    let inputs = op.inputs().iter().map(|(port, _)| input(client, *port, side, pixels)).collect::<Result<Vec<_>, _>>()?;
    op.run(client, &inputs)?;
    sync(client);
    Ok(())
}

/// Runs every operator on every side on `client`'s device; `progress(done, total, tag)` after each.
pub fn autotune(client: &Client, sides: &[usize], mut progress: impl FnMut(usize, usize, &str)) -> AutotuneSummary {
    let t0 = Instant::now();
    let ops = catalog();
    let total = ops.len() * sides.len();
    let mut failed = Vec::new();
    let mut done = 0;
    for &side in sides {
        let pixels = image(side);
        for op in &ops {
            // A panic inside a kernel launch (an unsupported size on this device) is a failure of
            // this operator here, not of the whole tuning
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_one(client, op, side, &pixels))).unwrap_or_else(|_| Err("panicked".into()));
            if let Err(why) = r {
                failed.push((op.tag().to_string(), side, why));
            }
            done += 1;
            progress(done, total, op.tag());
        }
    }
    AutotuneSummary { runs: total, failed, seconds: t0.elapsed().as_secs_f64() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_operator_runs_on_a_small_grid() {
        let client = lucem_core::ComputeTarget::Cpu.client().unwrap();
        let mut last = (0, 0);
        let s = autotune(&client, &[64], |d, t, _| last = (d, t));
        assert_eq!(last.0, last.1, "every step reported");
        assert_eq!(s.runs, catalog().len());
        assert!(s.failed.is_empty(), "{:?}", s.failed);
    }
}

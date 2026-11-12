//! Multi-axis separable pass driver using [`Workspace`] ping-pong buffers.
//!
//! Runs a sequence of 1-D passes across axes without allocating intermediate buffers once the
//! workspace has warmed up.

use cubecl::prelude::*;
use lucem_core::element::ImageFloat;
use lucem_core::memory::{DeviceArray, Workspace};

use crate::core::{check, elementwise};
use crate::error::OpsResult;

/// Runs `passes` sequentially from `input` into `output`, alternating `"sep_ping"` and `"sep_pong"`
/// slots from `ws` for intermediate passes.
///
/// - When `passes` is empty, copies `input` into `output` on the device.
/// - When `passes.len() == 1`, runs directly from `input` into `output` without touching `ws`.
/// - When `passes.len() > 1`, alternates workspace buffers and writes the final pass directly into
///   `output`.
pub fn run<F, P, Step>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    ws: &mut Workspace,
    passes: &[P],
    mut step: Step,
) -> OpsResult<()>
where
    F: ImageFloat,
    Step: FnMut(&Client, &DeviceArray<F>, &DeviceArray<F>, &P) -> OpsResult<()>,
{
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    match passes.len() {
        0 => elementwise::copy_into(client, input, output),
        1 => step(client, input, output, &passes[0]),
        n => {
            let shape = input.shape();
            let ping = ws.array::<F>("sep_ping", shape)?;
            step(client, input, &ping, &passes[0])?;
            if n == 2 {
                return step(client, &ping, output, &passes[1]);
            }
            let pong = ws.array::<F>("sep_pong", shape)?;
            let mut src_is_ping = true;
            for pass in &passes[1..n - 1] {
                if src_is_ping {
                    step(client, &ping, &pong, pass)?;
                } else {
                    step(client, &pong, &ping, pass)?;
                }
                src_is_ping = !src_is_ping;
            }
            let last_src = if src_is_ping { &ping } else { &pong };
            step(client, last_src, output, &passes[n - 1])
        }
    }
}

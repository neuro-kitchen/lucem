//! Biharmonic image inpainting (`biharmonic_inpaint`) and reliability-guided / Poisson phase
//! unwrapping (`unwrap_phase`).

use std::f64::consts::PI;

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast, cast_all, to_f64, ImageFloat};
use lucem_core::memory::{DeviceArray, Workspace};

use super::kernels::{biharmonic_2d_step_kernel, unwrap_phase_2d_poisson_rhs_kernel};
use crate::core::{check, compact, elementwise};
use crate::error::{OpsError, OpsResult};

/// Settings for 2-D biharmonic PDE inpainting ([`biharmonic_inpaint`]).
#[derive(Debug, Clone, PartialEq)]
pub struct BiharmonicInpaint {
    /// Number of harmonic ($\nabla^2 u = 0$) warm-start iterations used to propagate boundary
    /// values across missing regions before fourth-order refinement.
    pub harmonic_iter: usize,
    /// Number of fourth-order biharmonic ($\nabla^4 u = 0$) relaxation iterations.
    pub biharmonic_iter: usize,
}

impl Default for BiharmonicInpaint {
    fn default() -> Self {
        Self {
            harmonic_iter: 200,
            biharmonic_iter: 400,
        }
    }
}

/// Inpaints missing/damaged regions of 2-D `input` (where `mask != 0`) into `output` by solving the
/// discrete biharmonic equation $\nabla^4 u = 0$ with reflective Neumann boundary conditions.
pub fn biharmonic_inpaint_into<F: ImageFloat, M: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    mask: &DeviceArray<M>,
    output: &DeviceArray<F>,
    settings: &BiharmonicInpaint,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "mask", mask)?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::ndim_in("biharmonic_inpaint", input.shape(), &[2], "2")?;
    if input.is_empty() {
        return Ok(());
    }

    let mask_u32 = compact::flag_nonzero(client, mask)?;
    let shape = input.shape();
    let h = shape.dims()[0] as i32;
    let w = shape.dims()[1] as i32;
    let geom = LaunchGeometry::elementwise(client, input.len());

    elementwise::copy_into(client, input, output)?;
    let buf = ws.array::<F>("biharm_buf", shape)?;

    let omega_harm = cast::<F>(0.95);
    let mut ping = true;
    for _ in 0..settings.harmonic_iter {
        let (src, dst) = if ping { (output, &buf) } else { (&buf, output) };
        unsafe {
            biharmonic_2d_step_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                src.binding(),
                mask_u32.binding(),
                dst.binding(),
                omega_harm,
                h,
                w,
                false,
            );
        }
        ping = !ping;
    }

    let omega_biharm = cast::<F>(0.65);
    for _ in 0..settings.biharmonic_iter {
        let (src, dst) = if ping { (output, &buf) } else { (&buf, output) };
        unsafe {
            biharmonic_2d_step_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                src.binding(),
                mask_u32.binding(),
                dst.binding(),
                omega_biharm,
                h,
                w,
                true,
            );
        }
        ping = !ping;
    }

    if !ping {
        elementwise::copy_into(client, &buf, output)?;
    }
    Ok(())
}

/// Allocates and returns biharmonic-inpainted `input` over pixels where `mask != 0`.
pub fn biharmonic_inpaint<F: ImageFloat, M: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<F>,
    mask: &DeviceArray<M>,
    settings: &BiharmonicInpaint,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    biharmonic_inpaint_into(client, input, mask, &output, settings, &mut ws)?;
    Ok(output)
}

/// Wraps a phase difference $\Delta \phi$ into $(-\pi, \pi]$.
#[inline]
fn wrap_diff(delta: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let shifted = (delta + PI).rem_euclid(two_pi) - PI;
    if shifted <= -PI { shifted + two_pi } else { shifted }
}

/// Computes the 2-D wrapped-gradient divergence $\nabla \cdot W(\nabla \phi)$ on the device (the
/// right-hand side of the least-squares phase-unwrapping Poisson equation).
pub fn unwrap_phase_poisson_rhs_2d<F: ImageFloat>(
    client: &Client,
    phase: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("unwrap_phase_poisson_rhs_2d", phase.shape(), &[2], "2")?;
    let rhs = DeviceArray::<F>::empty(client, phase.shape().clone())?;
    if phase.is_empty() {
        return Ok(rhs);
    }
    let dims = phase.dims();
    let geom = LaunchGeometry::elementwise(client, phase.len());
    unsafe {
        unwrap_phase_2d_poisson_rhs_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            phase.binding(),
            rhs.binding(),
            cast::<F>(PI),
            cast::<F>(2.0 * PI),
            dims[0] as u32,
            dims[1] as u32,
        );
    }
    Ok(rhs)
}

/// Unwraps a 1-D, 2-D, or 3-D phase array measured in radians modulo $2\pi$ into a continuous phase
/// field `output` using Herraez, Burton, Lalor & Gdeisat's (2002) reliability-guided non-continuous
/// path algorithm.
///
/// `wrap_around` specifies per-axis periodic boundary wrapping (or `false` on every axis when `None`).
#[allow(clippy::needless_range_loop)] // neighbourhood / DP index arithmetic reads clearer with indices
pub fn unwrap_phase_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    wrap_around: Option<&[bool]>,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::ndim_in("unwrap_phase", input.shape(), &[1, 2, 3], "1, 2, or 3")?;
    let ndim = input.ndim();
    let wrap_axes = match wrap_around {
        Some(w) => {
            if w.len() != ndim {
                return Err(OpsError::InvalidParameter {
                    name: "wrap_around",
                    reason: format!("expected {ndim} booleans, got {}", w.len()),
                });
            }
            w.to_vec()
        }
        None => vec![false; ndim],
    };
    if input.is_empty() {
        return Ok(());
    }

    let phi: Vec<f64> = input.to_host(client).into_iter().map(to_f64).collect();
    let n = phi.len();
    if ndim == 1 {
        let mut out = vec![0.0f64; n];
        out[0] = phi[0];
        for i in 1..n {
            out[i] = out[i - 1] + wrap_diff(phi[i] - phi[i - 1]);
        }
        let dev = DeviceArray::from_host(client, input.shape().clone(), cast_all::<F>(&out))?;
        return elementwise::copy_into(client, &dev, output);
    }

    // Touch the 2-D device Poisson RHS kernel when 2-D so device phase gradient paths stay warm
    if ndim == 2 {
        let _ = unwrap_phase_poisson_rhs_2d(client, input)?;
    }

    let dims = input.dims();
    let strides = input.shape().strides();

    // Compute per-pixel second-difference unreliability D(p)
    let mut unreliability = vec![0.0f64; n];
    for idx in 0..n {
        let mut sum_sq = 0.0;
        for d in 0..ndim {
            let c = (idx / strides[d]) % dims[d];
            let len = dims[d];
            if len < 3 {
                continue;
            }
            let (prev_c, next_c) = if c > 0 && c + 1 < len {
                (c - 1, c + 1)
            } else if wrap_axes[d] {
                ((c + len - 1) % len, (c + 1) % len)
            } else {
                let clamped = c.clamp(1, len - 2);
                (clamped - 1, clamped + 1)
            };
            let base = idx - c * strides[d];
            let p_prev = phi[base + prev_c * strides[d]];
            let p_cur = phi[base + c * strides[d]];
            let p_next = phi[base + next_c * strides[d]];
            let d1 = wrap_diff(p_prev - p_cur);
            let d2 = wrap_diff(p_cur - p_next);
            let sec = d1 - d2;
            sum_sq += sec * sec;
        }
        unreliability[idx] = sum_sq.sqrt();
    }

    // Collect edges (u, v, cost = D(u) + D(v)) and sort ascending by unreliability cost
    let mut edges: Vec<(u32, u32, f64)> = Vec::with_capacity(n * ndim);
    for idx in 0..n {
        for d in 0..ndim {
            let c = (idx / strides[d]) % dims[d];
            let next_c = if c + 1 < dims[d] {
                Some(c + 1)
            } else if wrap_axes[d] && dims[d] > 1 {
                Some(0)
            } else {
                None
            };
            if let Some(nc) = next_c {
                let n_idx = idx - c * strides[d] + nc * strides[d];
                let cost = unreliability[idx] + unreliability[n_idx];
                edges.push((idx as u32, n_idx as u32, cost));
            }
        }
    }
    edges.sort_by(|a, b| a.2.total_cmp(&b.2));

    // Weighted Union-Find maintaining relative integer 2*pi wraps `offset[x]` to parent[x]
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut size: Vec<u32> = vec![1; n];
    let mut rel_k: Vec<i64> = vec![0; n];

    fn find_with_offset(parent: &mut [u32], rel_k: &mut [i64], x: u32) -> (u32, i64) {
        let mut cur = x;
        let mut total_k = 0i64;
        while parent[cur as usize] != cur {
            total_k += rel_k[cur as usize];
            cur = parent[cur as usize];
        }
        let root = cur;
        // Path compression updating rel_k
        cur = x;
        let mut rem_k = total_k;
        while parent[cur as usize] != cur {
            let next = parent[cur as usize];
            let step_k = rel_k[cur as usize];
            parent[cur as usize] = root;
            rel_k[cur as usize] = rem_k;
            rem_k -= step_k;
            cur = next;
        }
        (root, total_k)
    }

    let two_pi = 2.0 * PI;
    for (u, v, _) in edges {
        let (ru, ku) = find_with_offset(&mut parent, &mut rel_k, u);
        let (rv, kv) = find_with_offset(&mut parent, &mut rel_k, v);
        if ru == rv {
            continue;
        }
        let pu = phi[u as usize] + (ku as f64) * two_pi;
        let pv = phi[v as usize] + (kv as f64) * two_pi;
        // Desired unwrapped value at v is pu + wrap_diff(phi[v] - phi[u])
        let target_pv = pu + wrap_diff(phi[v as usize] - phi[u as usize]);
        let delta_k = ((target_pv - pv) / two_pi).round() as i64;

        if size[ru as usize] >= size[rv as usize] {
            parent[rv as usize] = ru;
            rel_k[rv as usize] = delta_k;
            size[ru as usize] += size[rv as usize];
        } else {
            parent[ru as usize] = rv;
            rel_k[ru as usize] = -delta_k;
            size[rv as usize] += size[ru as usize];
        }
    }

    let (_, k0) = find_with_offset(&mut parent, &mut rel_k, 0);
    let mut unwrapped = Vec::with_capacity(n);
    for (idx, &p) in phi.iter().enumerate().take(n) {
        let (_, ki) = find_with_offset(&mut parent, &mut rel_k, idx as u32);
        unwrapped.push(p + ((ki - k0) as f64) * two_pi);
    }

    let dev = DeviceArray::from_host(client, input.shape().clone(), cast_all::<F>(&unwrapped))?;
    elementwise::copy_into(client, &dev, output)
}

/// Allocates and returns the unwrapped continuous phase of `input`.
pub fn unwrap_phase<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    wrap_around: Option<&[bool]>,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    unwrap_phase_into(client, input, &output, wrap_around)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lucem_core::Shape;

    fn biharmonic_inpaint_and_unwrap_phase_recover_smooth_fields(client: &Client) {
        // Linear ramp f(y, x) = 0.05 * y + 0.03 * x + 0.1 has zero Laplacian and biharmonic everywhere
        let h = 16usize;
        let w = 16usize;
        let mut ramp = Vec::with_capacity(h * w);
        let mut mask = vec![0u32; h * w];
        for y in 0..h {
            for x in 0..w {
                ramp.push(0.05 * (y as f64) + 0.03 * (x as f64) + 0.1);
                if (6..=9).contains(&y) && (6..=9).contains(&x) {
                    mask[y * w + x] = 1;
                }
            }
        }
        let mut damaged = ramp.clone();
        for (idx, &m) in mask.iter().enumerate() {
            if m != 0 {
                damaged[idx] = 0.0;
            }
        }
        let dev_in = DeviceArray::from_host(client, Shape::new([h, w]), cast_all::<f32>(&damaged)).unwrap();
        let dev_mask = DeviceArray::from_host(client, Shape::new([h, w]), mask).unwrap();
        let restored = biharmonic_inpaint(client, &dev_in, &dev_mask, &BiharmonicInpaint::default())
            .unwrap()
            .to_host(client);
        assert_close(&restored, &ramp, 5e-3, client.name());

        // 2-D phase ramp spanning [-4*pi, +4*pi] wrapped into (-pi, pi]
        let mut true_phase = Vec::with_capacity(h * w);
        let mut wrapped_phase = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                let p = 0.45 * (y as f64) + 0.35 * (x as f64) - 3.0;
                true_phase.push(p);
                wrapped_phase.push(wrap_diff(p));
            }
        }
        let dev_wrapped =
            DeviceArray::from_host(client, Shape::new([h, w]), cast_all::<f32>(&wrapped_phase)).unwrap();
        let unwrapped = unwrap_phase(client, &dev_wrapped, None).unwrap().to_host(client);
        let shift = ((true_phase[0] - unwrapped[0] as f64) / (2.0 * PI)).round() * 2.0 * PI;
        let shifted: Vec<f32> = unwrapped.into_iter().map(|v| v + (shift as f32)).collect();
        assert_close(&shifted, &true_phase, 1e-4, client.name());
    }
    runtime_test!(test_restore_inpaint_and_unwrap, biharmonic_inpaint_and_unwrap_phase_recover_smooth_fields);
}

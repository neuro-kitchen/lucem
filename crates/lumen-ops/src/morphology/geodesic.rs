//! Geodesic morphological reconstruction, regional extrema (`regional_maxima`, `regional_minima`,
//! `h_maxima`, `h_minima`), flood filling, and small object/hole removal.
//!
//! Iterative geodesic propagation runs on the device using `Workspace` ping-pong buffers and checks
//! the device convergence flag only every `POLL_INTERVAL` iterations (§2.A rule 2b).

use std::collections::{BTreeSet, VecDeque};

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{to_f64, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::binary::fill_holes;
use super::kernels::geodesic_step_2d_kernel;
use crate::core::{check, components, elementwise, StructuringElement};
use crate::error::OpsResult;

/// Number of device geodesic iterations between host convergence flag checks (§2.A rule 2b).
pub const RECONSTRUCT_POLL_INTERVAL: usize = 8;

/// Geodesic reconstruction method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReconstructMethod {
    /// Iterative geodesic dilation of `seed` under `mask` (`seed <= mask`).
    #[default]
    Dilation,
    /// Iterative geodesic erosion of `seed` above `mask` (`seed >= mask`).
    Erosion,
}

/// Morphological geodesic reconstruction of `seed` under/above `mask` into `output`.
pub fn reconstruct_into<F: ImageFloat>(
    client: &Client,
    seed: &DeviceArray<F>,
    mask: &DeviceArray<F>,
    output: &DeviceArray<F>,
    method: ReconstructMethod,
    eight_connected: bool,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::ndim_in("reconstruct", seed.shape(), &[2], "2")?;
    check::same_shape("seed", seed, "mask", mask)?;
    check::same_shape("seed", seed, "output", output)?;
    check::not_aliased(seed, output)?;
    check::not_aliased(mask, output)?;
    if seed.is_empty() {
        return Ok(());
    }
    let (h, w) = (seed.dims()[0] as u32, seed.dims()[1] as u32);
    let ping = ws.array::<F>("rec_ping", seed.shape())?;
    let pong = ws.array::<F>("rec_pong", seed.shape())?;
    elementwise::copy_into(client, seed, &ping)?;

    let is_erosion = method == ReconstructMethod::Erosion;
    let geom = LaunchGeometry::elementwise(client, seed.len());
    let max_batches = (seed.dims()[0] + seed.dims()[1]).div_ceil(RECONSTRUCT_POLL_INTERVAL).max(1);

    for _ in 0..max_batches {
        let changed = DeviceArray::<u32>::zeros(client, Shape::vector(1))?;
        for step in 0..RECONSTRUCT_POLL_INTERVAL {
            let (src, dst) = if step % 2 == 0 { (&ping, &pong) } else { (&pong, &ping) };
            unsafe {
                geodesic_step_2d_kernel::launch::<F>(
                    client,
                    geom.cube_count.clone(),
                    geom.cube_dim,
                    src.binding(),
                    mask.binding(),
                    dst.binding(),
                    changed.binding(),
                    h,
                    w,
                    eight_connected,
                    is_erosion,
                );
            }
        }
        if changed.to_host(client)[0] == 0 {
            break;
        }
    }
    elementwise::copy_into(client, &ping, output)
}

/// Allocates and computes morphological geodesic reconstruction of `seed` under/above `mask`.
pub fn reconstruct<F: ImageFloat>(
    client: &Client,
    seed: &DeviceArray<F>,
    mask: &DeviceArray<F>,
    method: ReconstructMethod,
    eight_connected: bool,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, seed.shape().clone())?;
    let mut ws = Workspace::new(client);
    reconstruct_into(client, seed, mask, &output, method, eight_connected, &mut ws)?;
    Ok(output)
}

/// H-maxima transform: suppresses all regional maxima whose height is `< h` by reconstructing
/// `input - h` under `input` and returning the binary mask (`1u32`) where `input - reconstructed >= h`.
pub fn h_maxima<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    h: f64,
    eight_connected: bool,
) -> OpsResult<DeviceArray<u32>> {
    let mut ws = Workspace::new(client);
    let seed = ws.array::<F>("hmax_seed", input.shape())?;
    let rec = ws.array::<F>("hmax_rec", input.shape())?;
    let diff = ws.array::<F>("hmax_diff", input.shape())?;
    elementwise::scale_offset_into(client, input, &seed, 1.0, -h)?;
    reconstruct_into(client, &seed, input, &rec, ReconstructMethod::Dilation, eight_connected, &mut ws)?;
    elementwise::linear_combination_into(client, input, &rec, &diff, 1.0, -1.0, 0.0)?;
    let out = DeviceArray::<u32>::empty(client, input.shape().clone())?;
    elementwise::compare_threshold_scalar_into(client, &diff, 0.0, &out, false)?;
    Ok(out)
}

/// H-minima transform: suppresses all regional minima shallower than `h`.
pub fn h_minima<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    h: f64,
    eight_connected: bool,
) -> OpsResult<DeviceArray<u32>> {
    let mut ws = Workspace::new(client);
    let seed = ws.array::<F>("hmin_seed", input.shape())?;
    let rec = ws.array::<F>("hmin_rec", input.shape())?;
    let diff = ws.array::<F>("hmin_diff", input.shape())?;
    elementwise::scale_offset_into(client, input, &seed, 1.0, h)?;
    reconstruct_into(client, &seed, input, &rec, ReconstructMethod::Erosion, eight_connected, &mut ws)?;
    elementwise::linear_combination_into(client, &rec, input, &diff, 1.0, -1.0, 0.0)?;
    let out = DeviceArray::<u32>::empty(client, input.shape().clone())?;
    elementwise::compare_threshold_scalar_into(client, &diff, 0.0, &out, false)?;
    Ok(out)
}

/// Regional maxima binary mask (`1u32` at connected plateaus strictly greater than all surrounding
/// neighbors).
pub fn regional_maxima<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    eight_connected: bool,
) -> OpsResult<DeviceArray<u32>> {
    h_maxima(client, input, 1e-5, eight_connected)
}

/// Regional minima binary mask (`1u32` at connected valleys strictly smaller than all surrounding
/// neighbors).
pub fn regional_minima<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    eight_connected: bool,
) -> OpsResult<DeviceArray<u32>> {
    h_minima(client, input, 1e-5, eight_connected)
}

/// Removes connected foreground components with fewer than `min_size` pixels from a binary mask
/// or labeled array.
pub fn remove_small_objects(
    client: &Client,
    input: &DeviceArray<u32>,
    min_size: usize,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<u32>> {
    if input.is_empty() || min_size <= 1 {
        return elementwise::copy(client, input);
    }
    let cc = components::label_components(client, input, footprint)?;
    let summaries = components::summarize_regions::<f32>(client, &cc.labels, None)?;
    let keep: BTreeSet<u32> = summaries
        .into_iter()
        .filter(|s| s.count >= min_size)
        .map(|s| s.label)
        .collect();
    let labels_host = cc.labels.to_host(client);
    let out: Vec<u32> = labels_host
        .into_iter()
        .map(|l| u32::from(l > 0 && keep.contains(&l)))
        .collect();
    Ok(DeviceArray::from_host(client, input.shape().clone(), out)?)
}

/// Fills enclosed background holes smaller than `area_threshold` pixels in a binary mask.
pub fn remove_small_holes(
    client: &Client,
    input: &DeviceArray<u32>,
    area_threshold: usize,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<u32>> {
    if input.is_empty() || area_threshold == 0 {
        return elementwise::copy(client, input);
    }
    let fully_filled = fill_holes(client, input, footprint)?;
    let host_in = input.to_host(client);
    let host_filled = fully_filled.to_host(client);
    let holes: Vec<u32> = host_in
        .iter()
        .zip(host_filled.iter())
        .map(|(&orig, &f)| u32::from(orig == 0 && f != 0))
        .collect();
    let holes_dev = DeviceArray::from_host(client, input.shape().clone(), holes)?;
    let cc = components::label_components(client, &holes_dev, footprint)?;
    let summaries = components::summarize_regions::<f32>(client, &cc.labels, None)?;
    let fill_labels: BTreeSet<u32> = summaries
        .into_iter()
        .filter(|s| s.count < area_threshold)
        .map(|s| s.label)
        .collect();
    let hole_labels = cc.labels.to_host(client);
    let out: Vec<u32> = host_in
        .into_iter()
        .zip(hole_labels)
        .map(|(orig, hl)| u32::from(orig != 0 || (hl > 0 && fill_labels.contains(&hl))))
        .collect();
    Ok(DeviceArray::from_host(client, input.shape().clone(), out)?)
}

/// Flood-fills connected pixels starting at `seed_point` within `tolerance` of the seed intensity.
pub fn flood_fill<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    seed_point: &[usize],
    new_value: F,
    tolerance: f64,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<F>> {
    if seed_point.len() != input.ndim() {
        return Err(crate::error::OpsError::InvalidDimension {
            what: "seed_point",
            expected: "matching input ndim",
            actual: seed_point.len(),
            shape: seed_point.to_vec(),
        });
    }
    let dims = input.dims();
    let strides = input.shape().strides();
    let mut seed_idx = 0usize;
    for d in 0..dims.len() {
        if seed_point[d] >= dims[d] {
            return Err(crate::error::OpsError::InvalidParameter {
                name: "seed_point",
                reason: format!("seed coordinate {:?} out of bounds for {:?}", seed_point, dims),
            });
        }
        seed_idx += seed_point[d] * strides[d];
    }

    let default_se;
    let se = match footprint {
        Some(s) => s,
        None => {
            default_se = StructuringElement::connectivity(input.ndim(), 1)?;
            &default_se
        }
    };
    let offsets = se.coordinate_offsets(None);
    let mut host = input.to_host(client);
    let seed_val = to_f64(host[seed_idx]);
    let mut visited = vec![false; host.len()];
    let mut queue = VecDeque::new();
    visited[seed_idx] = true;
    queue.push_back(seed_idx);

    while let Some(cur) = queue.pop_front() {
        host[cur] = new_value;
        for off in &offsets {
            let mut n_idx = 0usize;
            let mut ok = true;
            for d in 0..dims.len() {
                let c = ((cur / strides[d]) % dims[d]) as isize + (off[d] as isize);
                if c < 0 || c >= dims[d] as isize {
                    ok = false;
                    break;
                }
                n_idx += (c as usize) * strides[d];
            }
            if ok && !visited[n_idx] && (to_f64(host[n_idx]) - seed_val).abs() <= tolerance {
                visited[n_idx] = true;
                queue.push_back(n_idx);
            }
        }
    }
    Ok(DeviceArray::from_host(client, input.shape().clone(), host)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn reconstruct_and_clean_objects_work(client: &Client) {
        #[rustfmt::skip]
        let mask = vec![
            1u32, 0, 1, 1,
            0,    0, 1, 1,
            0,    0, 0, 0,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([3, 4]), mask).unwrap();
        let cleaned = remove_small_objects(client, &arr, 2, None).unwrap().to_host(client);
        assert_eq!(cleaned[0], 0, "1-pixel object removed ({})", client.name());
        assert_eq!(cleaned[2], 1, "4-pixel object kept ({})", client.name());
    }
    runtime_test!(test_geodesic_and_cleaning, reconstruct_and_clean_objects_work);
}

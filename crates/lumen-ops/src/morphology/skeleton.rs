//! Binary skeletonization (Zhang-Suen parallel thinning), medial axis with exact Euclidean distance
//! radius, and 2-D convex hull rasterization.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::geodesic::RECONSTRUCT_POLL_INTERVAL;
use super::kernels::zhang_suen_subpass_kernel;
use crate::core::{check, compact, distance, elementwise};
use crate::error::OpsResult;

/// 2-D parallel binary skeletonization (Zhang-Suen thinning) into `output`, polling the device
/// convergence flag every `RECONSTRUCT_POLL_INTERVAL` iterations.
pub fn skeletonize_into(
    client: &Client,
    input: &DeviceArray<u32>,
    output: &DeviceArray<u32>,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::ndim_in("skeletonize", input.shape(), &[2], "2")?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let (h, w) = (input.dims()[0] as u32, input.dims()[1] as u32);
    let ping = ws.array::<u32>("skel_ping", input.shape())?;
    let pong = ws.array::<u32>("skel_pong", input.shape())?;
    elementwise::copy_into(client, input, &ping)?;

    let geom = LaunchGeometry::elementwise(client, input.len());
    let max_batches = (input.dims()[0] + input.dims()[1]).div_ceil(RECONSTRUCT_POLL_INTERVAL).max(1);

    for _ in 0..max_batches {
        let changed = DeviceArray::<u32>::zeros(client, Shape::vector(1))?;
        for _ in 0..RECONSTRUCT_POLL_INTERVAL {
            unsafe {
                zhang_suen_subpass_kernel::launch(
                    client,
                    geom.cube_count.clone(),
                    geom.cube_dim,
                    ping.binding(),
                    pong.binding(),
                    changed.binding(),
                    h,
                    w,
                    false,
                );
                zhang_suen_subpass_kernel::launch(
                    client,
                    geom.cube_count.clone(),
                    geom.cube_dim,
                    pong.binding(),
                    ping.binding(),
                    changed.binding(),
                    h,
                    w,
                    true,
                );
            }
        }
        if changed.to_host(client)[0] == 0 {
            break;
        }
    }
    elementwise::copy_into(client, &ping, output)
}

/// Allocates and computes the 2-D binary skeleton of `input`.
pub fn skeletonize(client: &Client, input: &DeviceArray<u32>) -> OpsResult<DeviceArray<u32>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    skeletonize_into(client, input, &output, &mut ws)?;
    Ok(output)
}

/// Computes the 2-D medial axis skeleton and the exact Euclidean distance transform at every pixel:
/// returns `(skeleton_mask, distance_map)`.
pub fn medial_axis(
    client: &Client,
    input: &DeviceArray<u32>,
) -> OpsResult<(DeviceArray<u32>, DeviceArray<f32>)> {
    let skel = skeletonize(client, input)?;
    let dist = distance::euclidean_distance::<u32, f32>(client, input, None)?;
    Ok((skel, dist))
}

/// Computes the 2-D binary convex hull mask enclosing all foreground (`!= 0`) pixels of `input`
/// using Andrew's monotone chain on compacted foreground coordinates and half-plane rasterization.
pub fn convex_hull_image(client: &Client, input: &DeviceArray<u32>) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("convex_hull_image", input.shape(), &[2], "2")?;
    if input.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, input.shape().clone())?);
    }
    let coords = compact::nonzero_coordinates(client, input)?;
    if coords.len() <= 2 {
        return elementwise::copy(client, input);
    }
    let mut pts: Vec<(isize, isize)> = coords.into_iter().map(|c| (c[1] as isize, c[0] as isize)).collect();
    pts.sort();
    pts.dedup();

    let cross = |o: (isize, isize), a: (isize, isize), b: (isize, isize)| -> isize {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut lower = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    let hull = lower;

    let (h, w) = (input.dims()[0], input.dims()[1]);
    let mut mask = vec![0u32; h * w];
    for y in 0..h {
        for x in 0..w {
            let p = (x as isize, y as isize);
            let inside = (0..hull.len()).all(|i| {
                let a = hull[i];
                let b = hull[(i + 1) % hull.len()];
                cross(a, b, p) >= 0
            });
            if inside {
                mask[y * w + x] = 1;
            }
        }
    }
    Ok(DeviceArray::from_host(client, input.shape().clone(), mask)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn convex_hull_fills_l_shape(client: &Client) {
        #[rustfmt::skip]
        let l_shape = vec![
            1u32, 0, 0,
            1,    0, 0,
            1,    1, 1,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([3, 3]), l_shape).unwrap();
        let hull = convex_hull_image(client, &arr).unwrap().to_host(client);
        assert_eq!(hull[4], 1, "center (1,1) is inside convex hull of L ({})", client.name());
        assert_eq!(hull[2], 0, "top-right (0,2) is outside convex hull of L ({})", client.name());
    }
    runtime_test!(test_skeleton_and_hull, convex_hull_fills_l_shape);
}

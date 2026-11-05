//! Exact-Euclidean isotropic binary morphology (`isotropic_erode`, `isotropic_dilate`,
//! `isotropic_open`, `isotropic_close`) via thresholding the exact Euclidean distance transform.

use cubecl::prelude::*;
use lumen_core::memory::{DeviceArray, Workspace};

use crate::core::{distance, elementwise};
use crate::error::OpsResult;

/// Isotropic binary dilation of `input` by Euclidean ball of radius `radius`: foreground where the
/// Euclidean distance to the nearest foreground pixel is `<= radius`.
pub fn isotropic_dilate(
    client: &Client,
    input: &DeviceArray<u32>,
    radius: f64,
) -> OpsResult<DeviceArray<u32>> {
    if input.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, input.shape().clone())?);
    }
    let host = input.to_host(client);
    // EDT measures distance from foreground (!=0) to nearest background (==0), so invert `input`
    // to measure distance from background to nearest foreground.
    let inv: Vec<u32> = host.into_iter().map(|v| u32::from(v == 0)).collect();
    let inv_dev = DeviceArray::from_host(client, input.shape().clone(), inv)?;
    let dist = distance::euclidean_distance::<u32, f32>(client, &inv_dev, None)?;
    let dist_host = dist.to_host(client);
    let r = radius as f32;
    let out: Vec<u32> = dist_host.into_iter().map(|d| u32::from(d <= r + 1e-6)).collect();
    Ok(DeviceArray::from_host(client, input.shape().clone(), out)?)
}

/// Isotropic binary erosion of `input` by Euclidean ball of radius `radius`: foreground where the
/// Euclidean distance to the nearest background pixel (with outside treated as background) is `> radius`.
pub fn isotropic_erode(
    client: &Client,
    input: &DeviceArray<u32>,
    radius: f64,
) -> OpsResult<DeviceArray<u32>> {
    if input.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, input.shape().clone())?);
    }
    let dist = distance::euclidean_distance::<u32, f32>(client, input, None)?;
    let output = DeviceArray::<u32>::empty(client, input.shape().clone())?;
    elementwise::compare_threshold_scalar_into(client, &dist, radius, &output, false)?;
    Ok(output)
}

/// Isotropic binary opening (isotropic erosion followed by isotropic dilation).
pub fn isotropic_open(
    client: &Client,
    input: &DeviceArray<u32>,
    radius: f64,
) -> OpsResult<DeviceArray<u32>> {
    let eroded = isotropic_erode(client, input, radius)?;
    isotropic_dilate(client, &eroded, radius)
}

/// Isotropic binary closing (isotropic dilation followed by isotropic erosion).
pub fn isotropic_close(
    client: &Client,
    input: &DeviceArray<u32>,
    radius: f64,
) -> OpsResult<DeviceArray<u32>> {
    let _ws = Workspace::new(client);
    let dilated = isotropic_dilate(client, input, radius)?;
    isotropic_erode(client, &dilated, radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lumen_core::Shape;

    fn isotropic_dilation_creates_disk(client: &Client) {
        let mut seed = vec![0u32; 25];
        seed[12] = 1;
        let arr = DeviceArray::from_host(client, Shape::new([5, 5]), seed).unwrap();
        let dil = isotropic_dilate(client, &arr, 1.0).unwrap().to_host(client);
        let active: usize = dil.iter().map(|&v| v as usize).sum();
        assert_eq!(active, 5, "radius 1 disk has 5 pixels ({})", client.name());
    }
    runtime_test!(test_isotropic_morphology, isotropic_dilation_creates_disk);
}

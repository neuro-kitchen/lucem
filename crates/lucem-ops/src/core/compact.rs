//! Stream compaction: extracts flat indices and N-D coordinates of non-zero elements in canonical
//! C-contiguous raster order.

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::memory::DeviceArray;
use lucem_core::Shape;

use super::kernels::flag_nonzero_kernel;
use crate::error::OpsResult;

/// Flags non-zero elements of `input` into a `u32` device array (`1` for non-zero, `0` for zero).
pub fn flag_nonzero<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
) -> OpsResult<DeviceArray<u32>> {
    let flags = DeviceArray::<u32>::zeros(client, input.shape().clone())?;
    if input.is_empty() {
        return Ok(flags);
    }
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        flag_nonzero_kernel::launch::<E>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            flags.binding(),
            input.len() as u32,
        );
    }
    Ok(flags)
}

/// Returns the N-D coordinates `[num_nonzero, ndim]` in C raster order of all non-zero elements in
/// `input`.
pub fn nonzero_coordinates<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
) -> OpsResult<Vec<Vec<usize>>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let flags = flag_nonzero(client, input)?.to_host(client);
    let strides = input.shape().strides();
    let dims = input.dims();
    let mut coords = Vec::new();
    for (idx, &f) in flags.iter().enumerate() {
        if f != 0 {
            let pt: Vec<usize> = (0..dims.len()).map(|d| (idx / strides[d]) % dims[d]).collect();
            coords.push(pt);
        }
    }
    Ok(coords)
}

/// Returns a device array `[num_nonzero, ndim]` (`u32`) of non-zero coordinates in raster order.
pub fn nonzero_coordinates_device<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
) -> OpsResult<DeviceArray<u32>> {
    let coords = nonzero_coordinates(client, input)?;
    let n = coords.len();
    let ndim = input.ndim();
    let flat: Vec<u32> = coords.into_iter().flatten().map(|c| c as u32).collect();
    if flat.is_empty() {
        Ok(DeviceArray::<u32>::zeros(client, Shape::new([0, ndim]))?)
    } else {
        Ok(DeviceArray::from_host(client, Shape::new([n, ndim]), flat)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn compacts_nonzero_in_raster_order(client: &Client) {
        let data = vec![0u32, 5, 0, 0, 2, 9];
        let arr = DeviceArray::from_host(client, Shape::new([2, 3]), data).unwrap();
        let coords = nonzero_coordinates(client, &arr).unwrap();
        assert_eq!(coords, vec![vec![0, 1], vec![1, 1], vec![1, 2]], "{}", client.name());
    }
    runtime_test!(test_compact_nonzero, compacts_nonzero_in_raster_order);
}

//! Summed-area tables (`integral_image`) and $O(1)$ rectangular box integration (`integrate_box`)
//! built on [`crate::core::scan::integral_scan`].

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::ImageFloat;
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::kernels::integrate_box_2d_kernel;
use crate::core::check;
use crate::core::scan::{integral_scan, integral_scan_into};
use crate::error::{OpsError, OpsResult};

/// Computes the N-D inclusive summed-area table (integral image) of `input` into `output`.
pub fn integral_image_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    ws: &mut Workspace,
) -> OpsResult<()> {
    integral_scan_into(client, input, output, ws)
}

/// Allocates and returns the N-D inclusive summed-area table (integral image) of `input`.
pub fn integral_image<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>> {
    integral_scan(client, input)
}

/// Evaluates $O(1)$ rectangular sums $\sum_{r=r_0}^{r_1} \sum_{c=c_0}^{c_1} I[r, c]$ for a batch of
/// inclusive 2-D bounding boxes `&[[r0, c0, r1, c1]]` from a 2-D [`integral_image`] `sat`.
pub fn integrate_box<F: ImageFloat>(
    client: &Client,
    sat: &DeviceArray<F>,
    boxes: &[[usize; 4]],
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("integrate_box", sat.shape(), &[2], "2")?;
    let (h, w) = (sat.dims()[0], sat.dims()[1]);
    let n = boxes.len();
    if n == 0 {
        return Ok(DeviceArray::<F>::zeros(client, Shape::vector(0))?);
    }

    let mut r0 = Vec::with_capacity(n);
    let mut c0 = Vec::with_capacity(n);
    let mut r1 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);
    for &[top, left, bot, right] in boxes {
        if top > bot || left > right || bot >= h || right >= w {
            return Err(OpsError::InvalidParameter {
                name: "boxes",
                reason: format!("box [{top}, {left}, {bot}, {right}] out of bounds for shape [{h}, {w}]"),
            });
        }
        r0.push(top as u32);
        c0.push(left as u32);
        r1.push(bot as u32);
        c1.push(right as u32);
    }

    let vec_shape = Shape::vector(n);
    let r0_dev = DeviceArray::from_host(client, vec_shape.clone(), r0)?;
    let c0_dev = DeviceArray::from_host(client, vec_shape.clone(), c0)?;
    let r1_dev = DeviceArray::from_host(client, vec_shape.clone(), r1)?;
    let c1_dev = DeviceArray::from_host(client, vec_shape.clone(), c1)?;
    let output = DeviceArray::<F>::empty(client, vec_shape)?;

    let geom = LaunchGeometry::elementwise(client, n);
    unsafe {
        integrate_box_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            sat.binding(),
            r0_dev.binding(),
            c0_dev.binding(),
            r1_dev.binding(),
            c1_dev.binding(),
            output.binding(),
            w as u32,
            n as u32,
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};

    fn integral_image_and_box_sums_match_direct_summation(client: &Client) {
        let host: Vec<f32> = (1..=20).map(|v| v as f32).collect();
        let img = DeviceArray::from_host(client, Shape::new([4, 5]), host).unwrap();
        let sat = integral_image(client, &img).unwrap();

        let boxes = [[0, 0, 3, 4], [1, 1, 2, 3], [2, 2, 2, 2]];
        let got = integrate_box(client, &sat, &boxes).unwrap().to_host(client);
        // Full 4x5 sum = 20*21/2 = 210
        // Sub-box [1..=2, 1..=3]: row 1 (7+8+9=24) + row 2 (12+13+14=39) = 63
        // Single cell [2, 2] = 2*5 + 2 + 1 = 13
        assert_close(&got, &[210.0, 63.0, 13.0], 1e-5, client.name());
    }
    runtime_test!(test_integral_box, integral_image_and_box_sums_match_direct_summation);
}

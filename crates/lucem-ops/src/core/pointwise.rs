//! Runs of consecutive pointwise image ops (affine, clamp, gamma, rescale) fused into one kernel:
//! each pixel is read once and written once for the whole run instead of once per stage.
//!
//! Also provides [`PointwiseRun::apply_cropped_2d`], which fuses trailing pointwise ops with a
//! 2-D halo crop in one launch so halo pixels are never written or downloaded.

use cubecl::prelude::*;
use cubecl::server::Handle;
use lucem_core::compute::{
    apply_pointwise, element_position, LaunchGeometry, POINTWISE_AFFINE, POINTWISE_CLAMP, POINTWISE_GAMMA, POINTWISE_RESCALE,
};
use lucem_core::element::{cast as scalar_cast, ImageFloat};
use lucem_core::memory::{buffer, DeviceArray};
use lucem_core::Shape;

use super::check;
use crate::error::{OpsError, OpsResult};

/// A single pointwise operation encoded as `(code, a, b)` for [`PointwiseRun`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointwiseOp {
    pub code: u32,
    pub a: f64,
    pub b: f64,
}

impl PointwiseOp {
    pub const fn affine(scale: f64, offset: f64) -> Self {
        Self { code: POINTWISE_AFFINE, a: scale, b: offset }
    }

    pub const fn clamp(lo: f64, hi: f64) -> Self {
        Self { code: POINTWISE_CLAMP, a: lo, b: hi }
    }

    pub const fn gamma(gamma: f64, gain: f64) -> Self {
        Self { code: POINTWISE_GAMMA, a: gamma, b: gain }
    }

    pub fn rescale(min: f64, max: f64) -> Self {
        let span = max - min;
        let (scale, offset) = if span.abs() <= f64::EPSILON { (0.0, 0.0) } else { (1.0 / span, -min / span) };
        Self { code: POINTWISE_RESCALE, a: scale, b: offset }
    }
}

/// A run of pointwise ops on the device: its codes and parameters, uploaded once.
pub struct PointwiseRun {
    codes: Handle,
    params: Handle,
    count: usize,
}

impl PointwiseRun {
    /// Uploads `ops` to `client` for element type `F`.
    pub fn upload<F: ImageFloat>(client: &Client, ops: &[PointwiseOp]) -> Self {
        let mut codes: Vec<u32> = ops.iter().map(|o| o.code).collect();
        let mut params: Vec<F> = ops.iter().flat_map(|o| [scalar_cast::<F>(o.a), scalar_cast::<F>(o.b)]).collect();
        // A binding may not be empty: an empty run still binds one identity op (never applied)
        if codes.is_empty() {
            codes.push(POINTWISE_AFFINE);
            params.extend([scalar_cast::<F>(1.0), scalar_cast::<F>(0.0)]);
        }
        Self {
            codes: client.create_from_slice(u32::as_bytes(&codes)),
            params: buffer::upload_owned(client, params),
            count: ops.len(),
        }
    }

    /// Number of ops in the run.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Applies the run from `input` into `output` (same shape, not aliased).
    pub fn apply_into<F: ImageFloat>(&self, client: &Client, input: &DeviceArray<F>, output: &DeviceArray<F>) -> OpsResult<()> {
        check::same_shape("input", input, "output", output)?;
        check::not_aliased(input, output)?;
        if input.is_empty() {
            return Ok(());
        }
        let geom = LaunchGeometry::elementwise(client, input.len());
        // SAFETY: input and output hold `input.len()` values of `F`; codes holds `count.max(1)` u32,
        // params holds `2 * count.max(1)` values of `F`.
        unsafe {
            pointwise_run_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                BufferArg::from_raw_parts(self.codes.clone(), self.count.max(1)),
                BufferArg::from_raw_parts(self.params.clone(), 2 * self.count.max(1)),
                self.count as u32,
                input.len() as u32,
            );
        }
        Ok(())
    }

    /// Allocates and returns `run(input)` on the device.
    pub fn apply<F: ImageFloat>(&self, client: &Client, input: &DeviceArray<F>) -> OpsResult<DeviceArray<F>> {
        let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
        self.apply_into(client, input, &output)?;
        Ok(output)
    }

    /// Applies the run while cropping a 2-D `[in_h, in_w]` array to `[y0..y0+out_h, x0..x0+out_w]`
    /// in a single kernel launch (no intermediate full-halo buffer).
    pub fn apply_cropped_2d<F: ImageFloat>(
        &self,
        client: &Client,
        input: &DeviceArray<F>,
        x0: usize,
        y0: usize,
        out_w: usize,
        out_h: usize,
    ) -> OpsResult<DeviceArray<F>> {
        check::ndim_in("apply_cropped_2d", input.shape(), &[2], "2")?;
        let dims = input.dims();
        let (in_h, in_w) = (dims[0], dims[1]);
        if x0 + out_w > in_w || y0 + out_h > in_h || out_w == 0 || out_h == 0 {
            return Err(OpsError::InvalidParameter {
                name: "crop",
                reason: format!("crop ({x0}..{}, {y0}..{}) out of bounds for {in_w}×{in_h}", x0 + out_w, y0 + out_h),
            });
        }
        let output = DeviceArray::<F>::empty(client, Shape::new([out_h, out_w]))?;
        let geom = LaunchGeometry::elementwise(client, output.len());
        // SAFETY: input holds `in_h * in_w` elements, output holds `out_h * out_w` elements,
        // and the crop bounds were checked above.
        unsafe {
            pointwise_crop_2d_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                BufferArg::from_raw_parts(self.codes.clone(), self.count.max(1)),
                BufferArg::from_raw_parts(self.params.clone(), 2 * self.count.max(1)),
                self.count as u32,
                in_w as u32,
                x0 as u32,
                y0 as u32,
                out_w as u32,
                out_h as u32,
            );
        }
        Ok(output)
    }
}

#[cube(launch)]
fn pointwise_run_kernel<F: ImageFloat>(
    input: &[F],
    output: &mut [F],
    codes: &[u32],
    params: &[F],
    count: u32,
    total: u32,
) {
    let idx = element_position();
    if (idx as u32) < total {
        output[idx] = apply_pointwise::<F>(input[idx], codes, params, count);
    }
}

#[cube(launch)]
fn pointwise_crop_2d_kernel<F: ImageFloat>(
    input: &[F],
    output: &mut [F],
    codes: &[u32],
    params: &[F],
    count: u32,
    in_w: u32,
    x0: u32,
    y0: u32,
    out_w: u32,
    out_h: u32,
) {
    let idx = element_position() as u32;
    let total = out_w * out_h;
    if idx < total {
        let ox = idx % out_w;
        let oy = idx / out_w;
        let in_idx = (y0 + oy) * in_w + (x0 + ox);
        output[idx as usize] = apply_pointwise::<F>(input[in_idx as usize], codes, params, count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::elementwise;
    use crate::intensity::contrast::adjust_gamma;
    use crate::intensity::convert::crop;
    use lucem_core::compute::ComputeTarget;

    #[test]
    fn a_fused_pointwise_run_equals_its_ops_one_by_one() {
        for target in ComputeTarget::for_tests() {
            let client = target.client().expect("compiled-in runtime");
            let (h, w) = (67, 113);
            let n = h * w;
            let data: Vec<f32> = (0..n).map(|i| ((i * 7919) % 2003) as f32 / 2003.0 * 1.4 - 0.1).collect();
            let input = DeviceArray::from_host(&client, Shape::new([h, w]), data).unwrap();

            // One kernel per op
            let s1 = DeviceArray::<f32>::empty(&client, input.shape().clone()).unwrap();
            elementwise::rescale_clip_into(&client, &input, &s1, (0.05, 0.95), (0.0, 1.0)).unwrap();
            let s2 = adjust_gamma(&client, &s1, 0.75, 1.2).unwrap();
            let s3 = DeviceArray::<f32>::empty(&client, input.shape().clone()).unwrap();
            elementwise::clip_into(&client, &s2, &s3, 0.0, 1.0).unwrap();
            let s4 = DeviceArray::<f32>::empty(&client, input.shape().clone()).unwrap();
            elementwise::scale_offset_into(&client, &s3, &s4, 0.8, 0.1).unwrap();
            let one_by_one = s4.to_host(&client);

            // Fused in one kernel
            let ops = [
                PointwiseOp::rescale(0.05, 0.95),
                PointwiseOp::gamma(0.75, 1.2),
                PointwiseOp::clamp(0.0, 1.0),
                PointwiseOp::affine(0.8, 0.1),
            ];
            let run = PointwiseRun::upload::<f32>(&client, &ops);
            let fused = run.apply::<f32>(&client, &input).unwrap().to_host(&client);

            for (i, (f, o)) in fused.iter().zip(&one_by_one).enumerate() {
                let tol = 4.0 * f32::EPSILON * o.abs().max(1.0);
                assert!((f - o).abs() <= tol, "{} pixel {i}: fused {f}, one by one {o}", client.name());
            }
        }
    }

    #[test]
    fn fused_pointwise_crop_matches_separate_ops_and_crop() {
        for target in ComputeTarget::for_tests() {
            let client = target.client().expect("compiled-in runtime");
            let (h, w) = (73, 97);
            let data: Vec<f32> = (0..h * w).map(|i| ((i * 1009) % 503) as f32 / 503.0).collect();
            let input = DeviceArray::from_host(&client, Shape::new([h, w]), data).unwrap();
            let ops = [PointwiseOp::rescale(0.1, 0.8), PointwiseOp::gamma(1.4, 0.9)];
            let run = PointwiseRun::upload::<f32>(&client, &ops);

            let full = run.apply::<f32>(&client, &input).unwrap();
            let expected = crop(&client, &full, &[(11, 13), (7, 19)]).unwrap().to_host(&client);
            let got = run.apply_cropped_2d::<f32>(&client, &input, 7, 11, w - 7 - 19, h - 11 - 13).unwrap().to_host(&client);
            assert_eq!(got, expected, "{} fused crop must be bit-exact with separate apply + crop", client.name());
        }
    }
}

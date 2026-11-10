//! 1-D and N-D spatial correlation and convolution.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, cast_all, ImageFloat};
use lumen_core::memory::{DeviceArray, DeviceCache};
use lumen_core::Shape;

use super::kernels::{correlate_1d_kernel, correlate_2d_kernel};
use crate::core::{check, PadMode};
use crate::error::OpsResult;

static WEIGHTS_CACHE: DeviceCache = DeviceCache::new();

/// Settings for 1-D correlation/convolution along an axis.
#[derive(Debug, Clone, PartialEq)]
pub struct Correlate1d {
    pub axis: isize,
    pub mode: PadMode,
    pub cval: f64,
    pub origin: i32,
}

impl Default for Correlate1d {
    fn default() -> Self {
        Self { axis: -1, mode: PadMode::Reflect, cval: 0.0, origin: 0 }
    }
}

impl Correlate1d {
    /// Samples read past each side `(before, after)` along the filtered axis.
    pub fn halo(&self, size: usize) -> (usize, usize) {
        let left = (size / 2) as i32 + self.origin;
        let right = (size as i32) - 1 - left;
        (left.max(0) as usize, right.max(0) as usize)
    }
}

/// 1-D correlation into `output` using device-resident `weights`.
pub fn correlate_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &Correlate1d,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let size = weights.len();
    check::origin(settings.origin, size)?;
    let axis = input.shape().axis(settings.axis)?;
    if input.is_empty() {
        return Ok(());
    }
    let split = input.shape().axis_split(axis);
    let geom = LaunchGeometry::elementwise(client, input.len());
    let left = (size / 2) as i32 + settings.origin;
    unsafe {
        correlate_1d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            weights.binding(),
            split.len as u32,
            split.inner as u32,
            left,
            cast::<F>(settings.cval),
            size as u32,
            settings.mode.id(),
        );
    }
    Ok(())
}

/// 1-D correlation allocating `output` and caching `weights` on the device.
pub fn correlate_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &[f64],
    settings: &Correlate1d,
) -> OpsResult<DeviceArray<F>> {
    check::weights(weights)?;
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let key = format!("corr1d-{weights:?}");
    let dev_weights = WEIGHTS_CACHE.get_or_upload::<F>(client, &key, Shape::vector(weights.len()), || {
        cast_all::<F>(weights)
    });
    correlate_1d_into(client, input, &dev_weights, &output, settings)?;
    Ok(output)
}

/// 1-D convolution into `output` (correlation with reversed weights and mirrored origin).
pub fn convolve_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &[f64],
    settings: &Correlate1d,
) -> OpsResult<DeviceArray<F>> {
    check::weights(weights)?;
    let size = weights.len();
    check::origin(settings.origin, size)?;
    let mut rev = weights.to_vec();
    rev.reverse();
    let mirrored_origin = -settings.origin - i32::from(size.is_multiple_of(2));
    let corr_settings = Correlate1d { origin: mirrored_origin, ..settings.clone() };
    correlate_1d(client, input, &rev, &corr_settings)
}

/// Settings for 2-D / N-D correlation and convolution.
#[derive(Debug, Clone, PartialEq)]
pub struct Correlate2d {
    pub mode: PadMode,
    pub cval: f64,
    pub origin: (i32, i32),
}

impl Default for Correlate2d {
    fn default() -> Self {
        Self { mode: PadMode::Reflect, cval: 0.0, origin: (0, 0) }
    }
}

/// 2-D correlation into `output` with device-resident `[k_h, k_w]` `weights`.
pub fn correlate_2d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &DeviceArray<F>,
    output: &DeviceArray<F>,
    settings: &Correlate2d,
) -> OpsResult<()> {
    check::ndim_in("correlate_2d", input.shape(), &[2], "2")?;
    check::ndim_in("weights", weights.shape(), &[2], "2")?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let (kh, kw) = (weights.dims()[0], weights.dims()[1]);
    check::origin(settings.origin.0, kh)?;
    check::origin(settings.origin.1, kw)?;
    if input.is_empty() {
        return Ok(());
    }
    let (h, w) = (input.dims()[0], input.dims()[1]);
    let left_y = (kh / 2) as i32 + settings.origin.0;
    let left_x = (kw / 2) as i32 + settings.origin.1;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        correlate_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            weights.binding(),
            h as i32,
            w as i32,
            left_y,
            left_x,
            cast::<F>(settings.cval),
            kh as u32,
            kw as u32,
            settings.mode.id(),
        );
    }
    Ok(())
}

/// 2-D correlation allocating `output`.
pub fn correlate_2d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    weights: &DeviceArray<F>,
    settings: &Correlate2d,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    correlate_2d_into(client, input, weights, &output, settings)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, host_read_padded, runtime_test, sample_f64};

    fn host_correlate_1d(
        x: &[f64],
        dims: &[usize],
        w: &[f64],
        axis: usize,
        mode: PadMode,
        cval: f64,
        origin: i32,
    ) -> Vec<f64> {
        let shape = Shape::new(dims);
        let split = shape.axis_split(axis);
        let left = (w.len() / 2) as isize + (origin as isize);
        let mut out = vec![0.0; x.len()];
        for (idx, o) in out.iter_mut().enumerate() {
            let i = (idx / split.inner) % split.len;
            let base = idx - i * split.inner;
            let start = (i as isize) - left;
            let mut sum = 0.0;
            for (t, &wt) in w.iter().enumerate() {
                sum += wt * host_read_padded(x, base, split.inner, start + (t as isize), split.len, cval, mode);
            }
            *o = sum;
        }
        out
    }

    fn correlate_1d_matches_f64_reference(client: &Client) {
        let modes = [PadMode::Reflect, PadMode::Constant, PadMode::Nearest, PadMode::Mirror, PadMode::Wrap];
        for dims in [vec![1usize], vec![7], vec![32, 33], vec![5, 1, 9]] {
            let x = sample_f64(dims.iter().product());
            let input = DeviceArray::from_host(client, Shape::new(dims.clone()), cast_all::<f32>(&x)).unwrap();
            for axis in 0..dims.len() {
                for (w, origin) in [(vec![1.0, -2.0, 3.0], 0), (vec![0.5; 4], -1)] {
                    for mode in modes {
                        let settings = Correlate1d { axis: axis as isize, mode, cval: 1.5, origin };
                        let got = correlate_1d::<f32>(client, &input, &w, &settings).unwrap().to_host(client);
                        let want = host_correlate_1d(&x, &dims, &w, axis, mode, 1.5, origin);
                        assert_close(&got, &want, 1e-5, client.name());
                    }
                }
            }
        }
    }
    runtime_test!(test_correlate_1d, correlate_1d_matches_f64_reference);
}

//! N-D forward and inverse Fourier transforms on split complex [`DeviceArray`] pairs.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::ImageFloat;
use lumen_core::memory::{DeviceArray, Workspace};

use super::check;
use super::elementwise;
use super::kernels::{complex_mul_kernel, dft_axis_kernel};
use crate::error::OpsResult;

/// 1-D forward (`inverse = false`) or inverse (`inverse = true`) DFT along `axis`.
#[allow(clippy::too_many_arguments)]
pub fn fft_1d_into<F: ImageFloat>(
    client: &Client,
    re_in: &DeviceArray<F>,
    im_in: &DeviceArray<F>,
    re_out: &DeviceArray<F>,
    im_out: &DeviceArray<F>,
    axis: isize,
    inverse: bool,
) -> OpsResult<()> {
    check::same_shape("re_in", re_in, "im_in", im_in)?;
    check::same_shape("re_in", re_in, "re_out", re_out)?;
    check::same_shape("re_in", re_in, "im_out", im_out)?;
    check::not_aliased(re_in, re_out)?;
    check::not_aliased(im_in, im_out)?;
    let ax = re_in.shape().axis(axis)?;
    if re_in.is_empty() {
        return Ok(());
    }
    let split = re_in.shape().axis_split(ax);
    let geom = LaunchGeometry::elementwise(client, re_in.len());
    unsafe {
        dft_axis_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            re_in.binding(),
            im_in.binding(),
            re_out.binding(),
            im_out.binding(),
            split.len as u32,
            split.inner as u32,
            re_in.len() as u32,
            inverse,
        );
    }
    Ok(())
}

/// N-D forward (`inverse = false`) or inverse (`inverse = true`) FFT across all axes of `(re_in, im_in)`
/// into `(re_out, im_out)` using `ws` ping-pong slots.
pub fn fft_nd_into<F: ImageFloat>(
    client: &Client,
    re_in: &DeviceArray<F>,
    im_in: &DeviceArray<F>,
    re_out: &DeviceArray<F>,
    im_out: &DeviceArray<F>,
    inverse: bool,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("re_in", re_in, "im_in", im_in)?;
    check::same_shape("re_in", re_in, "re_out", re_out)?;
    check::same_shape("re_in", re_in, "im_out", im_out)?;
    let ndim = re_in.ndim();
    if ndim == 0 || re_in.is_empty() {
        elementwise::copy_into(client, re_in, re_out)?;
        return elementwise::copy_into(client, im_in, im_out);
    }
    if ndim == 1 {
        return fft_1d_into(client, re_in, im_in, re_out, im_out, 0, inverse);
    }

    let shape = re_in.shape();
    let ping_re = ws.array::<F>("fft_ping_re", shape)?;
    let ping_im = ws.array::<F>("fft_ping_im", shape)?;
    fft_1d_into(client, re_in, im_in, &ping_re, &ping_im, 0, inverse)?;
    if ndim == 2 {
        return fft_1d_into(client, &ping_re, &ping_im, re_out, im_out, 1, inverse);
    }

    let pong_re = ws.array::<F>("fft_pong_re", shape)?;
    let pong_im = ws.array::<F>("fft_pong_im", shape)?;
    let mut src_is_ping = true;
    for ax in 1..ndim - 1 {
        if src_is_ping {
            fft_1d_into(client, &ping_re, &ping_im, &pong_re, &pong_im, ax as isize, inverse)?;
        } else {
            fft_1d_into(client, &pong_re, &pong_im, &ping_re, &ping_im, ax as isize, inverse)?;
        }
        src_is_ping = !src_is_ping;
    }
    if src_is_ping {
        fft_1d_into(client, &ping_re, &ping_im, re_out, im_out, (ndim - 1) as isize, inverse)
    } else {
        fft_1d_into(client, &pong_re, &pong_im, re_out, im_out, (ndim - 1) as isize, inverse)
    }
}

/// Multiplies two split-complex arrays pointwise (`conj_b = true` multiplies by the complex conjugate
/// of `b`, for cross-correlation).
#[allow(clippy::too_many_arguments)]
pub fn complex_mul_into<F: ImageFloat>(
    client: &Client,
    a_re: &DeviceArray<F>,
    a_im: &DeviceArray<F>,
    b_re: &DeviceArray<F>,
    b_im: &DeviceArray<F>,
    out_re: &DeviceArray<F>,
    out_im: &DeviceArray<F>,
    conj_b: bool,
) -> OpsResult<()> {
    check::same_shape("a_re", a_re, "b_re", b_re)?;
    check::same_shape("a_re", a_re, "out_re", out_re)?;
    if a_re.is_empty() {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, a_re.len());
    unsafe {
        complex_mul_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            a_re.binding(),
            a_im.binding(),
            b_re.binding(),
            b_im.binding(),
            out_re.binding(),
            out_im.binding(),
            a_re.len() as u32,
            conj_b,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test, sample_f64};
    use lumen_core::element::cast_all;
    use lumen_core::Shape;

    fn fft_ifft_round_trips(client: &Client) {
        let shape = Shape::new([7, 9]);
        let x = sample_f64(shape.len());
        let re_in = DeviceArray::from_host(client, shape.clone(), cast_all::<f32>(&x)).unwrap();
        let im_in = DeviceArray::<f32>::zeros(client, shape.clone()).unwrap();
        let freq_re = DeviceArray::<f32>::empty(client, shape.clone()).unwrap();
        let freq_im = DeviceArray::<f32>::empty(client, shape.clone()).unwrap();
        let rec_re = DeviceArray::<f32>::empty(client, shape.clone()).unwrap();
        let rec_im = DeviceArray::<f32>::empty(client, shape).unwrap();
        let mut ws = Workspace::new(client);

        fft_nd_into(client, &re_in, &im_in, &freq_re, &freq_im, false, &mut ws).unwrap();
        fft_nd_into(client, &freq_re, &freq_im, &rec_re, &rec_im, true, &mut ws).unwrap();
        assert_close(&rec_re.to_host(client), &x, 1e-4, client.name());
    }
    runtime_test!(test_fft_round_trip, fft_ifft_round_trips);
}

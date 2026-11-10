//! Rolling-ball background estimation and subtraction (`rolling_ball`, `ball_kernel`,
//! `ellipsoid_kernel`).
//!
//! Implements Sternberg's (1983) rolling-ball algorithm as a non-flat grayscale opening with a
//! spherical or ellipsoidal height profile.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast_all, to_f64, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::kernels::{rolling_ball_2d_dilate_kernel, rolling_ball_2d_erode_kernel};
use crate::core::{check, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::test_support::host_pad_index;

/// Generates an N-D spherical rolling-ball kernel of radius `radius`.
///
/// At offset $z \in [-r, r]^d$ ($r = \lfloor \text{radius} \rfloor$), the kernel height difference
/// from the ball apex is $R - \sqrt{R^2 - \|z\|_2^2}$ inside the ball ($\|z\|_2 \le R$) and
/// $+\infty$ outside.
pub fn ball_kernel(radius: f64, ndim: usize) -> (Shape, Vec<f64>) {
    let r = radius.max(0.0).floor() as isize;
    let side = (2 * r + 1) as usize;
    let dims = vec![side; ndim.max(1)];
    let shape = Shape::new(dims.clone());
    let strides = shape.strides();
    let r2 = radius * radius;
    let mut data = vec![f64::INFINITY; shape.len()];
    for (idx, val) in data.iter_mut().enumerate() {
        let mut d2 = 0.0;
        for d in 0..dims.len() {
            let c = ((idx / strides[d]) % dims[d]) as isize - r;
            d2 += (c * c) as f64;
        }
        if d2 <= r2 {
            *val = radius - (r2 - d2).max(0.0).sqrt();
        }
    }
    (shape, data)
}

/// Generates an N-D ellipsoidal rolling-ball kernel of bounding box `shape` and apex height
/// `intensity`.
///
/// Each dimension $d$ has semi-axis $a_d = \lfloor \text{shape}[d] / 2 \rfloor$. Inside the
/// ellipsoid $\sum_d (z_d / a_d)^2 \le 1$, the height difference from the apex is
/// $\text{intensity} \cdot \bigl(1 - \sqrt{1 - \sum_d (z_d / a_d)^2}\bigr)$, and $+\infty$ outside.
pub fn ellipsoid_kernel(shape: &[usize], intensity: f64) -> (Shape, Vec<f64>) {
    let odd_dims: Vec<usize> = shape
        .iter()
        .map(|&s| {
            let half = s / 2;
            (2 * half + 1).max(1)
        })
        .collect();
    let kshape = Shape::new(odd_dims.clone());
    let strides = kshape.strides();
    let radii: Vec<isize> = odd_dims.iter().map(|&s| (s / 2) as isize).collect();
    let mut data = vec![f64::INFINITY; kshape.len()];
    for (idx, val) in data.iter_mut().enumerate() {
        let mut norm2 = 0.0;
        for d in 0..odd_dims.len() {
            let c = ((idx / strides[d]) % odd_dims[d]) as isize - radii[d];
            let r = (radii[d] as f64).max(1.0);
            let u = (c as f64) / r;
            norm2 += u * u;
        }
        if norm2 <= 1.0 {
            *val = intensity * (1.0 - (1.0 - norm2).max(0.0).sqrt());
        }
    }
    (kshape, data)
}

/// Settings for rolling-ball background estimation ([`rolling_ball`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RollingBall {
    /// Ball radius in pixels (used when `kernel` is `None`).
    pub radius: f64,
    /// Optional explicit non-flat kernel `(shape, values)` from [`ball_kernel`] or [`ellipsoid_kernel`].
    pub kernel: Option<(Shape, Vec<f64>)>,
    /// Boundary padding mode.
    pub mode: PadMode,
}

impl Default for RollingBall {
    fn default() -> Self {
        Self {
            radius: 25.0,
            kernel: None,
            mode: PadMode::Nearest,
        }
    }
}

/// Estimates the smooth background intensity surface of `input` into `background` by rolling a
/// non-flat ball or ellipsoid kernel underneath the intensity profile.
#[allow(clippy::needless_range_loop)] // neighbourhood / DP index arithmetic reads clearer with indices
pub fn rolling_ball_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    background: &DeviceArray<F>,
    settings: &RollingBall,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "background", background)?;
    check::not_aliased(input, background)?;
    if input.is_empty() {
        return Ok(());
    }
    let ndim = input.ndim();
    let (kshape, kvals) = match &settings.kernel {
        Some((s, v)) => {
            if s.ndim() != ndim {
                return Err(OpsError::InvalidDimension {
                    what: "rolling_ball kernel",
                    expected: "matching input ndim",
                    actual: s.ndim(),
                    shape: s.dims().to_vec(),
                });
            }
            if s.len() != v.len() || v.is_empty() {
                return Err(OpsError::InvalidWeights {
                    what: "rolling_ball kernel",
                    reason: "kernel shape and buffer length must match and be non-empty",
                });
            }
            (s.clone(), v.clone())
        }
        None => {
            if !settings.radius.is_finite() || settings.radius <= 0.0 {
                return Err(OpsError::InvalidParameter {
                    name: "radius",
                    reason: format!("radius must be positive and finite, got {}", settings.radius),
                });
            }
            ball_kernel(settings.radius, ndim)
        }
    };

    if ndim == 2 {
        let safe_k: Vec<f64> = kvals
            .iter()
            .map(|&v| if v.is_finite() { v } else { 1.0e30 })
            .collect();
        let k_dev = DeviceArray::from_host(client, kshape.clone(), cast_all::<F>(&safe_k))?;
        let eroded = ws.array::<F>("rb_eroded", input.shape())?;
        let dims = input.dims();
        let kdims = kshape.dims();
        let ry = (kdims[0] / 2) as i32;
        let rx = (kdims[1] / 2) as i32;
        let geom = LaunchGeometry::elementwise(client, input.len());
        unsafe {
            rolling_ball_2d_erode_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                input.binding(),
                k_dev.binding(),
                eroded.binding(),
                dims[0] as i32,
                dims[1] as i32,
                ry,
                rx,
                kdims[0] as u32,
                kdims[1] as u32,
                settings.mode.id(),
            );
            rolling_ball_2d_dilate_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                eroded.binding(),
                k_dev.binding(),
                background.binding(),
                dims[0] as i32,
                dims[1] as i32,
                ry,
                rx,
                kdims[0] as u32,
                kdims[1] as u32,
                settings.mode.id(),
            );
        }
        return Ok(());
    }

    // General N-D fallback on host for 1-D or >=3-D volumes
    let host_in: Vec<f64> = input.to_host(client).into_iter().map(to_f64).collect();
    let dims = input.dims();
    let strides = input.shape().strides();
    let kdims = kshape.dims();
    let kstrides = kshape.strides();
    let kradii: Vec<isize> = kdims.iter().map(|&s| (s / 2) as isize).collect();

    let mut active_offsets: Vec<(Vec<isize>, f64)> = Vec::new();
    for (kidx, &kv) in kvals.iter().enumerate() {
        if kv.is_finite() {
            let off: Vec<isize> = (0..ndim)
                .map(|d| ((kidx / kstrides[d]) % kdims[d]) as isize - kradii[d])
                .collect();
            active_offsets.push((off, kv));
        }
    }

    let n = host_in.len();
    let mut eroded = vec![f64::INFINITY; n];
    for idx in 0..n {
        let mut best = f64::INFINITY;
        for (off, kv) in &active_offsets {
            let mut n_idx = 0usize;
            for d in 0..ndim {
                let c = ((idx / strides[d]) % dims[d]) as isize + off[d];
                let pc = host_pad_index(c, dims[d], settings.mode).unwrap_or(0);
                n_idx += pc * strides[d];
            }
            best = best.min(host_in[n_idx] + kv);
        }
        eroded[idx] = best;
    }

    let mut bg = vec![f64::NEG_INFINITY; n];
    for idx in 0..n {
        let mut best = f64::NEG_INFINITY;
        for (off, kv) in &active_offsets {
            let mut n_idx = 0usize;
            for d in 0..ndim {
                let c = ((idx / strides[d]) % dims[d]) as isize - off[d];
                let pc = host_pad_index(c, dims[d], settings.mode).unwrap_or(0);
                n_idx += pc * strides[d];
            }
            best = best.max(eroded[n_idx] - kv);
        }
        bg[idx] = best;
    }

    let dev_bg = DeviceArray::from_host(client, input.shape().clone(), cast_all::<F>(&bg))?;
    crate::core::elementwise::copy_into(client, &dev_bg, background)
}

/// Allocates and returns the rolling-ball estimated background of `input`.
pub fn rolling_ball<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &RollingBall,
) -> OpsResult<DeviceArray<F>> {
    let background = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    rolling_ball_into(client, input, &background, settings, &mut ws)?;
    Ok(background)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};

    fn rolling_ball_extracts_smooth_background_under_peak(client: &Client) {
        let (kshape, kvals) = ball_kernel(3.0, 2);
        assert_eq!(kshape.dims(), &[7, 7]);
        assert!((kvals[3 * 7 + 3]).abs() < 1e-12);
        assert!(kvals[0].is_infinite());

        let (eshape, evals) = ellipsoid_kernel(&[5, 7], 10.0);
        assert_eq!(eshape.dims(), &[5, 7]);
        assert!((evals[2 * 7 + 3]).abs() < 1e-12);

        // Constant background + narrow bright spike at center
        let h = 15usize;
        let w = 15usize;
        let mut img = vec![2.5f64; h * w];
        img[7 * w + 7] = 50.0;
        let dev_in = DeviceArray::from_host(client, Shape::new([h, w]), cast_all::<f32>(&img)).unwrap();
        let bg = rolling_ball(
            client,
            &dev_in,
            &RollingBall { radius: 4.0, kernel: None, mode: PadMode::Nearest },
        )
        .unwrap()
        .to_host(client);
        // Flat background everywhere; under the one-pixel spike the discrete ball rises to what its
        // neighbours at distance 1 allow: 2.5 + (r − √(r² − 1)) (as scikit-image)
        let mut want = vec![2.5; h * w];
        want[7 * w + 7] = 2.5 + (4.0 - (16.0f64 - 1.0).sqrt());
        assert_close(&bg, &want, 1e-5, client.name());
    }
    runtime_test!(test_rolling_ball, rolling_ball_extracts_smooth_background_under_peak);
}

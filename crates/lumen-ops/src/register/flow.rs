//! Dense 2-D optical flow estimation: Iterative Lucas-Kanade (`ilk_flow`) and Dual TV-$L_1$
//! (`tvl1_flow`).

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::kernels::{
    ilk_solve_2x2_kernel, ilk_tensor_rhs_2d_kernel, tvl1_dual_update_2d_kernel,
    tvl1_primal_threshold_2d_kernel, warp_bilinear_grad_2d_kernel,
};
use crate::core::{check, elementwise, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::filter::box_blur::{box_blur_into, BoxBlur};
use crate::filter::gaussian::{gaussian_blur_into, GaussianBlur};

/// Dense 2-D displacement vector field `(vy, vx)` mapping coordinates `(y, x)` in `reference` to
/// `(y + vy[y, x], x + vx[y, x])` in `moving`.
#[derive(Debug, Clone)]
pub struct OpticalFlow2d<F: ImageFloat> {
    /// Vertical (row / axis 0) displacement component.
    pub vy: DeviceArray<F>,
    /// Horizontal (column / axis 1) displacement component.
    pub vx: DeviceArray<F>,
}

/// Settings for Iterative Lucas-Kanade dense optical flow ([`ilk_flow`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IlkFlow {
    /// Neighborhood integration radius (`window_size = 2 * radius + 1`).
    pub radius: usize,
    /// Number of iterative warping refinements.
    pub num_warp: usize,
    /// When `true`, uses a Gaussian integration window ($\sigma = \text{radius} / 2$); when `false`,
    /// uses a uniform `(2 * radius + 1)` box window.
    pub gaussian: bool,
}

impl Default for IlkFlow {
    fn default() -> Self {
        Self {
            radius: 7,
            num_warp: 8,
            gaussian: false,
        }
    }
}

fn smooth_window_into<F: ImageFloat>(
    client: &Client,
    src: &DeviceArray<F>,
    dst: &DeviceArray<F>,
    radius: usize,
    gaussian: bool,
    ws: &mut Workspace,
) -> OpsResult<()> {
    if radius == 0 {
        return elementwise::copy_into(client, src, dst);
    }
    if gaussian {
        let sigma = (radius as f64) * 0.5;
        gaussian_blur_into(
            client,
            src,
            &[sigma],
            dst,
            &GaussianBlur {
                mode: PadMode::Nearest,
                radii: Some(vec![radius]),
                ..Default::default()
            },
            ws,
        )
    } else {
        let win = 2 * radius + 1;
        box_blur_into(
            client,
            src,
            &[win],
            dst,
            &BoxBlur {
                mode: PadMode::Nearest,
                ..Default::default()
            },
            ws,
        )
    }
}

/// Estimates dense 2-D optical flow `(vy, vx)` from `reference` (`image0`) to `moving` (`image1`)
/// using the Iterative Lucas-Kanade (ILK) solver on the device.
pub fn ilk_flow_into<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    vy_out: &DeviceArray<F>,
    vx_out: &DeviceArray<F>,
    settings: &IlkFlow,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("reference", reference, "moving", moving)?;
    check::same_shape("reference", reference, "vy_out", vy_out)?;
    check::same_shape("reference", reference, "vx_out", vx_out)?;
    check::not_aliased(vy_out, vx_out)?;
    check::ndim_in("ilk_flow", reference.shape(), &[2], "2")?;
    if settings.num_warp == 0 {
        return Err(OpsError::InvalidParameter {
            name: "num_warp",
            reason: "num_warp must be >= 1".to_string(),
        });
    }
    if reference.is_empty() {
        return Ok(());
    }

    let shape = reference.shape();
    let h = shape.dims()[0] as i32;
    let w = shape.dims()[1] as i32;
    let n = reference.len() as u32;
    let geom = LaunchGeometry::elementwise(client, reference.len());

    // Initialize displacement field to zero
    let zero = DeviceArray::<F>::zeros(client, shape.clone())?;
    elementwise::copy_into(client, &zero, vy_out)?;
    elementwise::copy_into(client, &zero, vx_out)?;

    let iy = ws.array::<F>("ilk_iy", shape)?;
    let ix = ws.array::<F>("ilk_ix", shape)?;
    let it = ws.array::<F>("ilk_it", shape)?;
    let raw_jyy = ws.array::<F>("ilk_rjyy", shape)?;
    let raw_jyx = ws.array::<F>("ilk_rjyx", shape)?;
    let raw_jxx = ws.array::<F>("ilk_rjxx", shape)?;
    let raw_by = ws.array::<F>("ilk_rby", shape)?;
    let raw_bx = ws.array::<F>("ilk_rbx", shape)?;
    let jyy = ws.array::<F>("ilk_jyy", shape)?;
    let jyx = ws.array::<F>("ilk_jyx", shape)?;
    let jxx = ws.array::<F>("ilk_jxx", shape)?;
    let by = ws.array::<F>("ilk_by", shape)?;
    let bx = ws.array::<F>("ilk_bx", shape)?;

    let reg_eps = cast::<F>(1.0e-5);
    for _ in 0..settings.num_warp {
        unsafe {
            warp_bilinear_grad_2d_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                reference.binding(),
                moving.binding(),
                vy_out.binding(),
                vx_out.binding(),
                iy.binding(),
                ix.binding(),
                it.binding(),
                h,
                w,
            );
            ilk_tensor_rhs_2d_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                iy.binding(),
                ix.binding(),
                it.binding(),
                raw_jyy.binding(),
                raw_jyx.binding(),
                raw_jxx.binding(),
                raw_by.binding(),
                raw_bx.binding(),
                n,
            );
        }
        smooth_window_into(client, &raw_jyy, &jyy, settings.radius, settings.gaussian, ws)?;
        smooth_window_into(client, &raw_jyx, &jyx, settings.radius, settings.gaussian, ws)?;
        smooth_window_into(client, &raw_jxx, &jxx, settings.radius, settings.gaussian, ws)?;
        smooth_window_into(client, &raw_by, &by, settings.radius, settings.gaussian, ws)?;
        smooth_window_into(client, &raw_bx, &bx, settings.radius, settings.gaussian, ws)?;

        unsafe {
            ilk_solve_2x2_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                jyy.binding(),
                jyx.binding(),
                jxx.binding(),
                by.binding(),
                bx.binding(),
                vy_out.binding(),
                vx_out.binding(),
                reg_eps,
                n,
            );
        }
    }
    Ok(())
}

/// Allocates and returns dense 2-D Iterative Lucas-Kanade optical flow `(vy, vx)`.
pub fn ilk_flow<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    settings: &IlkFlow,
) -> OpsResult<OpticalFlow2d<F>> {
    let vy = DeviceArray::<F>::empty(client, reference.shape().clone())?;
    let vx = DeviceArray::<F>::empty(client, reference.shape().clone())?;
    let mut ws = Workspace::new(client);
    ilk_flow_into(client, reference, moving, &vy, &vx, settings, &mut ws)?;
    Ok(OpticalFlow2d { vy, vx })
}

/// Settings for Zach, Pock & Bischof's (2007) Dual TV-$L_1$ dense optical flow ([`tvl1_flow`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Tvl1Flow {
    /// Data attachment weight $\lambda > 0$ (larger values follow image gradients more closely).
    pub attachment: f64,
    /// Coupling tightness $\theta > 0$ between primal flow $u$ and auxiliary flow $v$.
    pub tightness: f64,
    /// Number of outer warping linearizations.
    pub num_warp: usize,
    /// Number of inner primal-dual TV-$L_1$ iterations per warp.
    pub num_iter: usize,
}

impl Default for Tvl1Flow {
    fn default() -> Self {
        Self {
            attachment: 15.0,
            tightness: 0.3,
            num_warp: 5,
            num_iter: 30,
        }
    }
}

/// Estimates dense 2-D optical flow `(vy, vx)` from `reference` to `moving` using the Dual TV-$L_1$
/// variational solver on the device.
pub fn tvl1_flow_into<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    vy_out: &DeviceArray<F>,
    vx_out: &DeviceArray<F>,
    settings: &Tvl1Flow,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("reference", reference, "moving", moving)?;
    check::same_shape("reference", reference, "vy_out", vy_out)?;
    check::same_shape("reference", reference, "vx_out", vx_out)?;
    check::not_aliased(vy_out, vx_out)?;
    check::ndim_in("tvl1_flow", reference.shape(), &[2], "2")?;
    if !settings.attachment.is_finite() || settings.attachment <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "attachment",
            reason: format!("attachment must be positive and finite, got {}", settings.attachment),
        });
    }
    if !settings.tightness.is_finite() || settings.tightness <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "tightness",
            reason: format!("tightness must be positive and finite, got {}", settings.tightness),
        });
    }
    if settings.num_warp == 0 || settings.num_iter == 0 {
        return Err(OpsError::InvalidParameter {
            name: "num_warp/num_iter",
            reason: "num_warp and num_iter must be >= 1".to_string(),
        });
    }
    if reference.is_empty() {
        return Ok(());
    }

    let shape = reference.shape();
    let h = shape.dims()[0] as u32;
    let w = shape.dims()[1] as u32;
    let geom = LaunchGeometry::elementwise(client, reference.len());

    let zero = DeviceArray::<F>::zeros(client, shape.clone())?;
    elementwise::copy_into(client, &zero, vy_out)?;
    elementwise::copy_into(client, &zero, vx_out)?;

    let u0y = ws.array::<F>("tvl1_u0y", shape)?;
    let u0x = ws.array::<F>("tvl1_u0x", shape)?;
    let vy_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let vx_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let vy_b = ws.array::<F>("tvl1_vy_b", shape)?;
    let vx_b = ws.array::<F>("tvl1_vx_b", shape)?;

    let py_y_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let py_x_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let px_y_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let px_x_a = DeviceArray::<F>::zeros(client, shape.clone())?;
    let py_y_b = ws.array::<F>("tvl1_py_yb", shape)?;
    let py_x_b = ws.array::<F>("tvl1_py_xb", shape)?;
    let px_y_b = ws.array::<F>("tvl1_px_yb", shape)?;
    let px_x_b = ws.array::<F>("tvl1_px_xb", shape)?;

    let iy = ws.array::<F>("tvl1_iy", shape)?;
    let ix = ws.array::<F>("tvl1_ix", shape)?;
    let it = ws.array::<F>("tvl1_it", shape)?;

    let theta_f = cast::<F>(settings.tightness);
    let lam_theta_f = cast::<F>(settings.attachment * settings.tightness);
    let tau_over_theta_f = cast::<F>(0.248 / settings.tightness);
    let eps_grad = cast::<F>(1.0e-8);

    let mut ping = true;
    for _ in 0..settings.num_warp {
        elementwise::copy_into(client, vy_out, &u0y)?;
        elementwise::copy_into(client, vx_out, &u0x)?;
        unsafe {
            warp_bilinear_grad_2d_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                reference.binding(),
                moving.binding(),
                u0y.binding(),
                u0x.binding(),
                iy.binding(),
                ix.binding(),
                it.binding(),
                h as i32,
                w as i32,
            );
        }

        for _ in 0..settings.num_iter {
            let (vy_in, vx_in, vy_nxt, vx_nxt, py_yi, py_xi, px_yi, px_xi, py_yo, py_xo, px_yo, px_xo) =
                if ping {
                    (
                        &vy_a, &vx_a, &vy_b, &vx_b, &py_y_a, &py_x_a, &px_y_a, &px_x_a, &py_y_b,
                        &py_x_b, &px_y_b, &px_x_b,
                    )
                } else {
                    (
                        &vy_b, &vx_b, &vy_a, &vx_a, &py_y_b, &py_x_b, &px_y_b, &px_x_b, &py_y_a,
                        &py_x_a, &px_y_a, &px_x_a,
                    )
                };
            unsafe {
                tvl1_primal_threshold_2d_kernel::launch::<F>(
                    client,
                    geom.cube_count.clone(),
                    geom.cube_dim,
                    u0y.binding(),
                    u0x.binding(),
                    iy.binding(),
                    ix.binding(),
                    it.binding(),
                    py_yi.binding(),
                    py_xi.binding(),
                    px_yi.binding(),
                    px_xi.binding(),
                    vy_in.binding(),
                    vx_in.binding(),
                    vy_out.binding(),
                    vx_out.binding(),
                    vy_nxt.binding(),
                    vx_nxt.binding(),
                    theta_f,
                    lam_theta_f,
                    eps_grad,
                    h,
                    w,
                );
                tvl1_dual_update_2d_kernel::launch::<F>(
                    client,
                    geom.cube_count.clone(),
                    geom.cube_dim,
                    vy_out.binding(),
                    vx_out.binding(),
                    py_yi.binding(),
                    py_xi.binding(),
                    px_yi.binding(),
                    px_xi.binding(),
                    py_yo.binding(),
                    py_xo.binding(),
                    px_yo.binding(),
                    px_xo.binding(),
                    tau_over_theta_f,
                    h,
                    w,
                );
            }
            ping = !ping;
        }
    }
    Ok(())
}

/// Allocates and returns dense 2-D Dual TV-$L_1$ optical flow `(vy, vx)`.
pub fn tvl1_flow<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    settings: &Tvl1Flow,
) -> OpsResult<OpticalFlow2d<F>> {
    let vy = DeviceArray::<F>::empty(client, reference.shape().clone())?;
    let vx = DeviceArray::<F>::empty(client, reference.shape().clone())?;
    let mut ws = Workspace::new(client);
    tvl1_flow_into(client, reference, moving, &vy, &vx, settings, &mut ws)?;
    Ok(OpticalFlow2d { vy, vx })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lumen_core::element::cast_all;
    use lumen_core::Shape;

    fn ilk_and_tvl1_track_gaussian_blob_translation(client: &Client) {
        let h = 21usize;
        let w = 21usize;
        let shape = Shape::new([h, w]);
        // Smooth 2-D Gaussian blob shifted from (10.0, 10.0) in ref to (10.8, 11.2) in mov
        let mut img0 = Vec::with_capacity(h * w);
        let mut img1 = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                let dy0 = (y as f64) - 10.0;
                let dx0 = (x as f64) - 10.0;
                let dy1 = (y as f64) - 10.8;
                let dx1 = (x as f64) - 11.2;
                img0.push((-0.5 * (dy0 * dy0 + dx0 * dx0) / 9.0).exp());
                img1.push((-0.5 * (dy1 * dy1 + dx1 * dx1) / 9.0).exp());
            }
        }
        let ref_dev = DeviceArray::from_host(client, shape.clone(), cast_all::<f32>(&img0)).unwrap();
        let mov_dev = DeviceArray::from_host(client, shape, cast_all::<f32>(&img1)).unwrap();

        // Notice img1(y, x) = img0(y - 0.8, x - 1.2), so img1(y + (-0.8), x + (-1.2)) = img0(y, x):
        // displacement (vy, vx) at center is (-0.8, -1.2)
        let ilk = ilk_flow(
            client,
            &ref_dev,
            &mov_dev,
            &IlkFlow {
                radius: 5,
                num_warp: 8,
                gaussian: true,
            },
        )
        .unwrap();
        let center = 10 * w + 10;
        let ilk_vy = ilk.vy.to_host(client)[center];
        let ilk_vx = ilk.vx.to_host(client)[center];
        assert_close(&[ilk_vy, ilk_vx], &[-0.8, -1.2], 0.15, client.name());

        let tvl1 = tvl1_flow(
            client,
            &ref_dev,
            &mov_dev,
            &Tvl1Flow {
                attachment: 20.0,
                tightness: 0.3,
                num_warp: 6,
                num_iter: 40,
            },
        )
        .unwrap();
        let tv_vy = tvl1.vy.to_host(client)[center];
        let tv_vx = tvl1.vx.to_host(client)[center];
        assert_close(&[tv_vy, tv_vx], &[-0.8, -1.2], 0.25, client.name());
    }
    runtime_test!(test_optical_flow, ilk_and_tvl1_track_gaussian_blob_translation);
}

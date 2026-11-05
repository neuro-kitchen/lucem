//! Edge and gradient filters: Sobel, Prewitt, Scharr, Farid, Roberts cross, and discrete Laplacian.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, cast_all, ImageFloat};
use lumen_core::memory::{DeviceArray, DeviceCache, Workspace};
use lumen_core::Shape;

use super::convolve::{correlate_1d_into, correlate_2d_into, Correlate1d, Correlate2d};
use super::kernels::{add_sq_kernel, laplace_axis_add_kernel, sqrt_scale_kernel};
use crate::core::{check, separable, PadMode};
use crate::error::OpsResult;

static STENCIL_CACHE: DeviceCache = DeviceCache::new();

/// Settings for directional or isotropic edge stencils.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientFilter {
    /// Specific axis for a directional derivative, or `None` for the isotropic Euclidean magnitude
    /// $\sqrt{\frac{1}{D}\sum_{d=0}^{D-1} G_d^2}$ across all axes.
    pub axis: Option<isize>,
    pub mode: PadMode,
    pub cval: f64,
}

impl Default for GradientFilter {
    fn default() -> Self {
        Self { axis: None, mode: PadMode::Reflect, cval: 0.0 }
    }
}

/// Separable derivative stencil family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StencilKind {
    /// Central difference `[-1, 0, 1]` along derivative axis, `[1, 2, 1]` smoothing on orthogonal axes.
    Sobel,
    /// Central difference `[-1, 0, 1]` along derivative axis, `[1, 1, 1]` smoothing on orthogonal axes.
    Prewitt,
    /// Rotational-symmetry optimized 3-tap stencil (`[-1, 0, 1]` and `[3, 10, 3]`).
    Scharr,
    /// 5-tap 1st-order optimal equiripple Farid stencil.
    Farid,
}

impl StencilKind {
    fn taps(self) -> (&'static [f64], &'static [f64]) {
        match self {
            // Normalized 1-D taps so 2-D isotropic magnitude matches standard edge operators:
            Self::Sobel => (&[-1.0, 0.0, 1.0], &[1.0, 2.0, 1.0]),
            Self::Prewitt => (&[-1.0, 0.0, 1.0], &[1.0, 1.0, 1.0]),
            Self::Scharr => (&[-1.0, 0.0, 1.0], &[3.0, 10.0, 3.0]),
            Self::Farid => (
                &[-0.109_603_762_960_254, -0.276_690_988_455_557, 0.0, 0.276_690_988_455_557, 0.109_603_762_960_254],
                &[0.037_659_317_195_812_6, 0.249_153_396_177_344, 0.426_374_573_253_687, 0.249_153_396_177_344, 0.037_659_317_195_812_6],
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Sobel => "sobel",
            Self::Prewitt => "prewitt",
            Self::Scharr => "scharr",
            Self::Farid => "farid",
        }
    }
}

/// Computes a single directional derivative along `deriv_axis` for `kind` into `output`.
pub fn directional_stencil_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    deriv_axis: usize,
    kind: StencilKind,
    mode: PadMode,
    cval: f64,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let ndim = input.ndim();
    let (d_taps, s_taps) = kind.taps();
    let d_weights = STENCIL_CACHE.get_or_upload::<F>(
        client,
        &format!("{}-deriv", kind.name()),
        Shape::vector(d_taps.len()),
        || cast_all::<F>(d_taps),
    );
    let s_weights = STENCIL_CACHE.get_or_upload::<F>(
        client,
        &format!("{}-smooth", kind.name()),
        Shape::vector(s_taps.len()),
        || cast_all::<F>(s_taps),
    );

    let axes: Vec<usize> = (0..ndim).collect();
    separable::run(client, input, output, ws, &axes, |client, src, dst, &ax| {
        let w = if ax == deriv_axis { &d_weights } else { &s_weights };
        correlate_1d_into(
            client,
            src,
            w,
            dst,
            &Correlate1d { axis: ax as isize, mode, cval, origin: 0 },
        )
    })
}

/// Applies a separable gradient filter (`Sobel`, `Prewitt`, `Scharr`, or `Farid`) into `output`.
pub fn stencil_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    kind: StencilKind,
    settings: &GradientFilter,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    if let Some(axis) = settings.axis {
        let ax = input.shape().axis(axis)?;
        return directional_stencil_into(client, input, output, ax, kind, settings.mode, settings.cval, ws);
    }

    let ndim = input.ndim();
    let accum_sq = DeviceArray::<F>::zeros(client, input.shape().clone())?;
    let grad = ws.array::<F>("stencil_grad", input.shape())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    for ax in 0..ndim {
        directional_stencil_into(client, input, &grad, ax, kind, settings.mode, settings.cval, ws)?;
        unsafe {
            add_sq_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                grad.binding(),
                accum_sq.binding(),
                input.len() as u32,
            );
        }
    }
    unsafe {
        sqrt_scale_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            accum_sq.binding(),
            output.binding(),
            cast::<F>(1.0),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Sobel gradient filter (directional when `settings.axis = Some(ax)`, magnitude when `None`).
pub fn sobel<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &GradientFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    stencil_filter_into(client, input, &output, StencilKind::Sobel, settings, &mut ws)?;
    Ok(output)
}

/// Prewitt gradient filter.
pub fn prewitt<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &GradientFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    stencil_filter_into(client, input, &output, StencilKind::Prewitt, settings, &mut ws)?;
    Ok(output)
}

/// Scharr gradient filter.
pub fn scharr<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &GradientFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    stencil_filter_into(client, input, &output, StencilKind::Scharr, settings, &mut ws)?;
    Ok(output)
}

/// Farid 5-tap equiripple gradient filter.
pub fn farid<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: &GradientFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    stencil_filter_into(client, input, &output, StencilKind::Farid, settings, &mut ws)?;
    Ok(output)
}

/// 2-D Roberts cross diagonal edge magnitude into `output`.
pub fn roberts_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::ndim_in("roberts", input.shape(), &[2], "2")?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let pos_w = STENCIL_CACHE.get_or_upload::<F>(client, "roberts-pos", Shape::new([2, 2]), || {
        cast_all::<F>(&[1.0, 0.0, 0.0, -1.0])
    });
    let neg_w = STENCIL_CACHE.get_or_upload::<F>(client, "roberts-neg", Shape::new([2, 2]), || {
        cast_all::<F>(&[0.0, 1.0, -1.0, 0.0])
    });
    let g1 = ws.array::<F>("rob_g1", input.shape())?;
    let g2 = ws.array::<F>("rob_g2", input.shape())?;
    let corr = Correlate2d { mode, cval, origin: (0, 0) };
    correlate_2d_into(client, input, &pos_w, &g1, &corr)?;
    correlate_2d_into(client, input, &neg_w, &g2, &corr)?;
    crate::core::elementwise::hypot_into(client, &g1, &g2, output)
}

/// 2-D Roberts cross diagonal edge magnitude allocating `output`.
pub fn roberts<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    roberts_into(client, input, &output, mode, cval, &mut ws)?;
    Ok(output)
}

/// N-D discrete Laplacian ($\sum_{d=0}^{D-1} [1, -2, 1]_d * I$) into `output`.
pub fn laplace_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    output: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let zero = DeviceArray::<F>::zeros(client, output.shape().clone())?;
    crate::core::elementwise::copy_into(client, &zero, output)?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    for ax in 0..input.ndim() {
        let split = input.shape().axis_split(ax);
        unsafe {
            laplace_axis_add_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                input.binding(),
                output.binding(),
                split.len as u32,
                split.inner as u32,
                cast::<F>(cval),
                mode.id(),
            );
        }
    }
    Ok(())
}

/// N-D discrete Laplacian allocating `output`.
pub fn laplace<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    laplace_into(client, input, &output, mode, cval)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};

    fn sobel_and_laplace_zero_on_constant_image(client: &Client) {
        let shape = Shape::new([12, 15]);
        let img = DeviceArray::<f32>::filled(client, shape.clone(), 4.0).unwrap();
        let s = sobel(client, &img, &GradientFilter::default()).unwrap().to_host(client);
        let l = laplace(client, &img, PadMode::Reflect, 0.0).unwrap().to_host(client);
        assert_close(&s, &vec![0.0; shape.len()], 1e-6, client.name());
        assert_close(&l, &vec![0.0; shape.len()], 1e-6, client.name());
    }
    runtime_test!(test_sobel_and_laplace, sobel_and_laplace_zero_on_constant_image);
}

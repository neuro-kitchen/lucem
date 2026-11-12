//! Grayscale mathematical morphology: erosion, dilation, opening, closing, white/black top-hats,
//! morphological gradient, and morphological Laplacian.
//!
//! Flat footprints delegate directly to `filter::order::{min_filter_into, max_filter_into}` (using
//! separable 1-D passes when the footprint is rectangular); non-flat additive structuring element
//! weights use `grey_weighted_2d_kernel`.

use cubecl::prelude::*;
use lucem_core::element::ImageFloat;
use lucem_core::memory::{DeviceArray, Workspace};

use crate::core::{check, elementwise, PadMode, StructuringElement};
use crate::error::OpsResult;
use crate::filter::{max_filter_into, min_filter_into, OrderFilter};

/// Settings for grayscale morphological operations.
#[derive(Debug, Clone, PartialEq)]
pub struct GrayMorphology {
    pub mode: PadMode,
    pub cval: f64,
    pub origins: Option<Vec<i32>>,
}

impl Default for GrayMorphology {
    fn default() -> Self {
        Self { mode: PadMode::Reflect, cval: 0.0, origins: None }
    }
}

impl GrayMorphology {
    fn as_order_filter(&self) -> OrderFilter {
        OrderFilter { mode: self.mode, cval: self.cval, origins: self.origins.clone() }
    }

    fn mirrored_order_filter(&self, footprint: &StructuringElement) -> OrderFilter {
        let origins = self.origins.as_ref().map(|origs| {
            origs
                .iter()
                .zip(footprint.dims())
                .map(|(&o, &sz)| -o - i32::from(sz.is_multiple_of(2)))
                .collect()
        });
        OrderFilter { mode: self.mode, cval: self.cval, origins }
    }
}

/// Grayscale erosion ($I \ominus B$) into `output`.
pub fn erode_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()> {
    min_filter_into(client, input, footprint, output, &settings.as_order_filter(), ws)
}

/// Grayscale erosion allocating `output`.
pub fn erode<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    erode_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// Grayscale dilation ($I \oplus B$) into `output` (maximum filter over the reflected structuring
/// element).
pub fn dilate_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let reflected = footprint.mirrored();
    max_filter_into(
        client,
        input,
        &reflected,
        output,
        &settings.mirrored_order_filter(footprint),
        ws,
    )
}

/// Grayscale dilation allocating `output`.
pub fn dilate<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    dilate_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// Grayscale opening ($(I \ominus B) \oplus B$) into `output`.
pub fn open_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let tmp = ws.array::<F>("morph_open_tmp", input.shape())?;
    erode_into(client, input, footprint, &tmp, settings, ws)?;
    dilate_into(client, &tmp, footprint, output, settings, ws)
}

/// Grayscale opening allocating `output`.
pub fn open<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    open_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// Grayscale closing ($(I \oplus B) \ominus B$) into `output`.
pub fn close_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let tmp = ws.array::<F>("morph_close_tmp", input.shape())?;
    dilate_into(client, input, footprint, &tmp, settings, ws)?;
    erode_into(client, &tmp, footprint, output, settings, ws)
}

/// Grayscale closing allocating `output`.
pub fn close<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    close_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// White top-hat transform ($I - (I \ominus B) \oplus B$) into `output`, isolating bright features
/// smaller than `footprint`.
pub fn white_tophat_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let opened = ws.array::<F>("wth_open", input.shape())?;
    open_into(client, input, footprint, &opened, settings, ws)?;
    elementwise::linear_combination_into(client, input, &opened, output, 1.0, -1.0, 0.0)
}

/// White top-hat transform allocating `output`.
pub fn white_tophat<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    white_tophat_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// Black top-hat transform ($(I \oplus B) \ominus B - I$) into `output`, isolating dark features
/// smaller than `footprint`.
pub fn black_tophat_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let closed = ws.array::<F>("bth_close", input.shape())?;
    close_into(client, input, footprint, &closed, settings, ws)?;
    elementwise::linear_combination_into(client, &closed, input, output, 1.0, -1.0, 0.0)
}

/// Black top-hat transform allocating `output`.
pub fn black_tophat<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    black_tophat_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// Beucher morphological gradient ($(I \oplus B) - (I \ominus B)$) into `output`.
pub fn morph_gradient_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let dil = ws.array::<F>("mgrad_dil", input.shape())?;
    let ero = ws.array::<F>("mgrad_ero", input.shape())?;
    dilate_into(client, input, footprint, &dil, settings, ws)?;
    erode_into(client, input, footprint, &ero, settings, ws)?;
    elementwise::linear_combination_into(client, &dil, &ero, output, 1.0, -1.0, 0.0)
}

/// Beucher morphological gradient allocating `output`.
pub fn morph_gradient<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    morph_gradient_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// Morphological Laplacian ($(I \oplus B) + (I \ominus B) - 2 I$) into `output`.
pub fn morph_laplace_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &GrayMorphology,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let dil = ws.array::<F>("mlap_dil", input.shape())?;
    let ero = ws.array::<F>("mlap_ero", input.shape())?;
    let sum = ws.array::<F>("mlap_sum", input.shape())?;
    dilate_into(client, input, footprint, &dil, settings, ws)?;
    erode_into(client, input, footprint, &ero, settings, ws)?;
    elementwise::linear_combination_into(client, &dil, &ero, &sum, 1.0, 1.0, 0.0)?;
    elementwise::linear_combination_into(client, &sum, input, output, 1.0, -2.0, 0.0)
}

/// Morphological Laplacian allocating `output`.
pub fn morph_laplace<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &GrayMorphology,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    morph_laplace_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lucem_core::Shape;

    fn white_tophat_extracts_bright_puncta_from_flat_background(client: &Client) {
        let mut data = vec![10.0f32; 25];
        data[12] = 50.0;
        let img = DeviceArray::from_host(client, Shape::new([5, 5]), data).unwrap();
        let se = StructuringElement::rectangle([3, 3]).unwrap();
        let wth = white_tophat(client, &img, &se, &GrayMorphology::default()).unwrap().to_host(client);
        let mut want = vec![0.0f64; 25];
        want[12] = 40.0;
        assert_close(&wth, &want, 1e-6, client.name());
    }
    runtime_test!(test_grayscale_morphology, white_tophat_extracts_bright_puncta_from_flat_background);
}

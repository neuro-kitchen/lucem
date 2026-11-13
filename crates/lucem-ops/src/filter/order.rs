//! Order-statistic filters: minimum, maximum, median, rank, percentile, and global rank-order transform.

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast, to_f64, ImageFloat};
use lucem_core::memory::{DeviceArray, Workspace};

use super::kernels::{
    extrema_1d_kernel, extrema_2d_footprint_kernel, median_3x3_kernel, rank_2d_footprint_kernel,
};
use crate::core::{check, elementwise, separable, PadMode, StructuringElement};
use crate::error::{OpsError, OpsResult};

/// Settings for 1-D min/max filters ([`min_filter_1d`], [`max_filter_1d`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Extrema1d {
    pub axis: isize,
    pub mode: PadMode,
    pub cval: f64,
    pub origin: i32,
}

impl Default for Extrema1d {
    fn default() -> Self {
        Self { axis: -1, mode: PadMode::Reflect, cval: 0.0, origin: 0 }
    }
}

/// 1-D minimum (`is_max = false`) or maximum (`is_max = true`) filter of window `size`.
pub fn extrema_1d_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    output: &DeviceArray<F>,
    settings: &Extrema1d,
    is_max: bool,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    check::origin(settings.origin, size)?;
    let axis = input.shape().axis(settings.axis)?;
    if input.is_empty() {
        return Ok(());
    }
    let split = input.shape().axis_split(axis);
    let left = (size / 2) as i32 + settings.origin;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        extrema_1d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            split.len as u32,
            split.inner as u32,
            left,
            cast::<F>(settings.cval),
            size as u32,
            settings.mode.id(),
            is_max,
        );
    }
    Ok(())
}

/// 1-D minimum filter allocating `output`.
pub fn min_filter_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    settings: &Extrema1d,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    extrema_1d_into(client, input, size, &output, settings, false)?;
    Ok(output)
}

/// 1-D maximum filter allocating `output`.
pub fn max_filter_1d<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    size: usize,
    settings: &Extrema1d,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    extrema_1d_into(client, input, size, &output, settings, true)?;
    Ok(output)
}

/// Settings for N-D order-statistic filters.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderFilter {
    pub mode: PadMode,
    pub cval: f64,
    pub origins: Option<Vec<i32>>,
}

impl Default for OrderFilter {
    fn default() -> Self {
        Self { mode: PadMode::Reflect, cval: 0.0, origins: None }
    }
}

/// N-D minimum (`is_max = false`) or maximum (`is_max = true`) filter over `footprint`.
///
/// Automatically decomposes rectangular footprints into separable 1-D passes across axes via
/// `core::separable::run`.
pub fn extrema_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
    is_max: bool,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if footprint.ndim() != input.ndim() {
        return Err(OpsError::InvalidDimension {
            what: "footprint",
            expected: "matching input ndim",
            actual: footprint.ndim(),
            shape: footprint.dims().to_vec(),
        });
    }
    if input.is_empty() {
        return Ok(());
    }
    let origins = match &settings.origins {
        Some(o) => check::broadcast_per_axis("origins", o, input.ndim())?,
        None => vec![0; input.ndim()],
    };

    if footprint.is_rectangle() {
        let passes: Vec<(usize, usize, i32)> = (0..input.ndim())
            .map(|ax| (ax, footprint.dims()[ax], origins[ax]))
            .filter(|&(_, sz, _)| sz > 1)
            .collect();
        return separable::run(client, input, output, ws, &passes, |client, src, dst, &(ax, sz, orig)| {
            extrema_1d_into(
                client,
                src,
                sz,
                dst,
                &Extrema1d { axis: ax as isize, mode: settings.mode, cval: settings.cval, origin: orig },
                is_max,
            )
        });
    }

    check::ndim_in("non-rectangular extrema_filter", input.shape(), &[2], "2")?;
    let offsets = footprint.device_offsets(client, Some(&origins));
    let n_active = footprint.count_active();
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        extrema_2d_footprint_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            offsets.binding(),
            input.dims()[0] as i32,
            input.dims()[1] as i32,
            n_active as u32,
            cast::<F>(settings.cval),
            settings.mode.id(),
            is_max,
        );
    }
    Ok(())
}

/// N-D minimum filter over `footprint` into `output`.
pub fn min_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
    ws: &mut Workspace,
) -> OpsResult<()> {
    extrema_filter_into(client, input, footprint, output, settings, false, ws)
}

/// N-D minimum filter allocating `output`.
pub fn min_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    min_filter_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// N-D maximum filter over `footprint` into `output`.
pub fn max_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
    ws: &mut Workspace,
) -> OpsResult<()> {
    extrema_filter_into(client, input, footprint, output, settings, true, ws)
}

/// N-D maximum filter allocating `output`.
pub fn max_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    max_filter_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// 2-D rank filter selecting the `rank`-th smallest element (`0 <= rank < footprint.count_active()`;
/// negative `rank` counts from the end).
pub fn rank_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    rank: isize,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
) -> OpsResult<()> {
    check::ndim_in("rank_filter", input.shape(), &[2], "2")?;
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    let n_active = footprint.count_active();
    let norm_rank = if rank < 0 { rank + (n_active as isize) } else { rank };
    if norm_rank < 0 || (norm_rank as usize) >= n_active {
        return Err(OpsError::InvalidParameter {
            name: "rank",
            reason: format!("rank {rank} out of bounds for footprint with {n_active} active elements"),
        });
    }
    if input.is_empty() {
        return Ok(());
    }
    let origins = match &settings.origins {
        Some(o) => check::broadcast_per_axis("origins", o, 2)?,
        None => vec![0, 0],
    };

    // Fast path: 3x3 full square median with zero origin uses the 19-step `med9` network
    if footprint.is_rectangle()
        && footprint.dims() == [3, 3]
        && norm_rank == 4
        && origins == [0, 0]
    {
        let geom = LaunchGeometry::elementwise(client, input.len());
        unsafe {
            median_3x3_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                input.binding(),
                output.binding(),
                input.dims()[0] as i32,
                input.dims()[1] as i32,
                cast::<F>(settings.cval),
                settings.mode.id(),
            );
        }
        return Ok(());
    }

    let offsets = footprint.device_offsets(client, Some(&origins));
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        rank_2d_footprint_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            offsets.binding(),
            input.dims()[0] as i32,
            input.dims()[1] as i32,
            n_active as u32,
            norm_rank as u32,
            cast::<F>(settings.cval),
            settings.mode.id(),
        );
    }
    Ok(())
}

/// 2-D rank filter allocating `output`.
pub fn rank_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    rank: isize,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    rank_filter_into(client, input, rank, footprint, &output, settings)?;
    Ok(output)
}

/// 2-D median filter into `output`: uses `med9` for $3 \times 3$ windows and exact rank selection
/// for general footprints (averaging the two middle ranks when `count_active` is even).
pub fn median_filter_into<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    output: &DeviceArray<F>,
    settings: &OrderFilter,
    ws: &mut Workspace,
) -> OpsResult<()> {
    let n = footprint.count_active();
    if n % 2 == 1 {
        rank_filter_into(client, input, (n / 2) as isize, footprint, output, settings)
    } else {
        let lo = ws.array::<F>("med_lo", input.shape())?;
        let hi = ws.array::<F>("med_hi", input.shape())?;
        rank_filter_into(client, input, (n / 2 - 1) as isize, footprint, &lo, settings)?;
        rank_filter_into(client, input, (n / 2) as isize, footprint, &hi, settings)?;
        elementwise::linear_combination_into(client, &lo, &hi, output, 0.5, 0.5, 0.0)
    }
}

/// 2-D median filter allocating `output`.
pub fn median_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    median_filter_into(client, input, footprint, &output, settings, &mut ws)?;
    Ok(output)
}

/// 2-D percentile filter (`percentile` in `0.0..=100.0`) over `footprint`.
pub fn percentile_filter<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    percentile: f64,
    footprint: &StructuringElement,
    settings: &OrderFilter,
) -> OpsResult<DeviceArray<F>> {
    if !(0.0..=100.0).contains(&percentile) || !percentile.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "percentile",
            reason: format!("percentile must be in 0.0..=100.0, got {percentile}"),
        });
    }
    let n = footprint.count_active();
    let rank = ((percentile / 100.0) * (n as f64)).floor().clamp(0.0, (n - 1) as f64) as isize;
    rank_filter(client, input, rank, footprint, settings)
}

/// Replaces every pixel by its non-negative integer rank `0..num_unique` among unique sorted values
/// of the image, returning `(ranked_image, sorted_unique_values)`.
pub fn rank_order<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<(DeviceArray<u32>, Vec<f64>)> {
    if input.is_empty() {
        return Ok((DeviceArray::<u32>::zeros(client, input.shape().clone())?, Vec::new()));
    }
    let host: Vec<f64> = input.to_host(client).into_iter().map(to_f64).collect();
    let mut unique = host.clone();
    unique.sort_by(f64::total_cmp);
    unique.dedup_by(|a, b| (*a - *b).abs() == 0.0);
    let ranks: Vec<u32> = host
        .iter()
        .map(|v| unique.binary_search_by(|u| u.total_cmp(v)).unwrap_or(0) as u32)
        .collect();
    Ok((DeviceArray::from_host(client, input.shape().clone(), ranks)?, unique))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lucem_core::Shape;

    fn median_and_extrema_remove_impulse_spike(client: &Client) {
        let mut data = vec![2.0f32; 25];
        data[12] = 100.0; // single salt spike at center of 5x5
        let img = DeviceArray::from_host(client, Shape::new([5, 5]), data).unwrap();
        let se = StructuringElement::rectangle([3, 3]).unwrap();

        let med = median_filter(client, &img, &se, &OrderFilter::default()).unwrap().to_host(client);
        assert!(med.iter().all(|&v| v == 2.0), "3x3 median removes single spike ({})", client.name());

        let mx = max_filter(client, &img, &se, &OrderFilter::default()).unwrap().to_host(client);
        assert_eq!(mx[12], 100.0, "{}", client.name());
    }
    runtime_test!(test_order_filters, median_and_extrema_remove_impulse_spike);
}


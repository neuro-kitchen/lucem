//! Histogram equalization, CLAHE (Contrast Limited Adaptive Histogram Equalization),
//! cumulative distribution functions, and histogram matching on the device.

use cubecl::prelude::*;
use lucem_core::{
    compute::LaunchGeometry,
    element::{cast, ImageFloat},
    memory::DeviceArray,
    Shape,
};

use crate::{
    core::{check, elementwise, histogram, reduce, scan},
    error::{OpsError, OpsResult},
};

use super::kernels::equalize::{
    apply_cdf_lut_kernel, build_match_lut_kernel, clahe_interpolate_kernel, clahe_tile_cdfs_kernel,
    normalize_cdf_kernel,
};

/// Settings for [`clahe`] (Contrast Limited Adaptive Histogram Equalization).
#[derive(Debug, Clone, Copy)]
pub struct ClaheSettings {
    /// Number of contextual tiles `(tiles_y, tiles_x)`. Default `(8, 8)`.
    pub grid_size: (usize, usize),
    /// Normalized clip limit in `[0, 1]`. Default `0.01`.
    pub clip_limit: f64,
    /// Number of histogram bins per tile. Default `256`.
    pub num_bins: usize,
}

impl Default for ClaheSettings {
    fn default() -> Self {
        Self {
            grid_size: (8, 8),
            clip_limit: 0.01,
            num_bins: 256,
        }
    }
}

/// Compute the normalized cumulative distribution function (CDF) and device min/max bounds.
pub fn cdf_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    num_bins: usize,
) -> OpsResult<(DeviceArray<F>, DeviceArray<F>, DeviceArray<F>)> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("cdf"));
    }
    if num_bins == 0 {
        return Err(OpsError::InvalidParameter {
            name: "num_bins",
            reason: "num_bins must be >= 1".to_string(),
        });
    }
    let stats = reduce::sum_min_max(client, input)?;
    let v_min_dev = DeviceArray::from_host(client, Shape::vector(1), vec![cast::<F>(stats.min)])?;
    let v_max_dev = DeviceArray::from_host(client, Shape::vector(1), vec![cast::<F>(stats.max)])?;
    let hist_u32 = histogram::histogram_ranged_device(client, input, num_bins, (stats.min, stats.max))?;
    let hist_f = DeviceArray::<F>::empty(client, Shape::vector(num_bins))?;
    elementwise::cast_into(client, &hist_u32, &hist_f)?;
    let cum_hist = DeviceArray::<F>::empty(client, Shape::vector(num_bins))?;
    scan::scan_axis_into(client, &hist_f, &cum_hist, 0, true)?;

    let cdf_out = DeviceArray::<F>::empty(client, Shape::vector(num_bins))?;
    let geom = LaunchGeometry::elementwise(client, num_bins);
    unsafe {
        normalize_cdf_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            cum_hist.binding(),
            cdf_out.binding(),
            num_bins as u32,
        );
    }
    Ok((cdf_out, v_min_dev, v_max_dev))
}

/// Compute the normalized cumulative distribution function (CDF) of an image (`[num_bins]`).
pub fn cdf<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    num_bins: usize,
) -> OpsResult<DeviceArray<F>> {
    let (cdf_out, _, _) = cdf_device(client, input, num_bins)?;
    Ok(cdf_out)
}

/// Global histogram equalization mapping intensities to `[0, 1]` via the device-computed CDF.
pub fn equalize_hist<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    num_bins: usize,
) -> OpsResult<DeviceArray<F>> {
    let (cdf_lut, v_min_dev, v_max_dev) = cdf_device(client, input, num_bins)?;
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        apply_cdf_lut_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            cdf_lut.binding(),
            v_min_dev.binding(),
            v_max_dev.binding(),
            output.binding(),
            num_bins as u32,
            input.len() as u32,
        );
    }
    Ok(output)
}

/// Contrast Limited Adaptive Histogram Equalization (CLAHE) for 2-D images on the device.
pub fn clahe<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    settings: ClaheSettings,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("clahe", input.shape(), &[2], "2")?;
    if input.is_empty() {
        return Err(OpsError::EmptyInput("clahe"));
    }
    let (height, width) = (input.dims()[0], input.dims()[1]);
    let (tiles_y, tiles_x) = settings.grid_size;
    if tiles_y == 0 || tiles_x == 0 || settings.num_bins == 0 {
        return Err(OpsError::InvalidParameter {
            name: "clahe",
            reason: "grid_size and num_bins must be positive".to_string(),
        });
    }

    let (_sum, v_min_dev, v_max_dev) = reduce::sum_min_max_device(client, input)?;
    let total_tiles = tiles_y * tiles_x;
    let tile_cdfs = DeviceArray::<F>::empty(client, Shape::vector(total_tiles * settings.num_bins))?;
    let geom_tiles = LaunchGeometry::elementwise(client, total_tiles);
    unsafe {
        clahe_tile_cdfs_kernel::launch::<F>(
            client,
            geom_tiles.cube_count,
            geom_tiles.cube_dim,
            input.binding(),
            v_min_dev.binding(),
            v_max_dev.binding(),
            tile_cdfs.binding(),
            height as u32,
            width as u32,
            tiles_y as u32,
            tiles_x as u32,
            settings.num_bins as u32,
            cast::<F>(settings.clip_limit),
        );
    }

    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let geom_px = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        clahe_interpolate_kernel::launch::<F>(
            client,
            geom_px.cube_count,
            geom_px.cube_dim,
            input.binding(),
            v_min_dev.binding(),
            v_max_dev.binding(),
            tile_cdfs.binding(),
            output.binding(),
            height as u32,
            width as u32,
            tiles_y as u32,
            tiles_x as u32,
            settings.num_bins as u32,
        );
    }
    Ok(output)
}

/// Match the histogram of `source` to that of `reference` on the device.
pub fn match_histograms<F: ImageFloat>(
    client: &Client,
    source: &DeviceArray<F>,
    reference: &DeviceArray<F>,
    num_bins: usize,
) -> OpsResult<DeviceArray<F>> {
    let (src_cdf, src_min, src_max) = cdf_device(client, source, num_bins)?;
    let (ref_cdf, ref_min, ref_max) = cdf_device(client, reference, num_bins)?;

    let lut = DeviceArray::<F>::empty(client, Shape::vector(num_bins))?;
    let geom_lut = LaunchGeometry::elementwise(client, num_bins);
    unsafe {
        build_match_lut_kernel::launch::<F>(
            client,
            geom_lut.cube_count,
            geom_lut.cube_dim,
            src_cdf.binding(),
            ref_cdf.binding(),
            ref_min.binding(),
            ref_max.binding(),
            lut.binding(),
            num_bins as u32,
        );
    }

    let output = DeviceArray::<F>::empty(client, source.shape().clone())?;
    let geom_px = LaunchGeometry::elementwise(client, source.len());
    unsafe {
        apply_cdf_lut_kernel::launch::<F>(
            client,
            geom_px.cube_count,
            geom_px.cube_dim,
            source.binding(),
            lut.binding(),
            src_min.binding(),
            src_max.binding(),
            output.binding(),
            num_bins as u32,
            source.len() as u32,
        );
    }
    Ok(output)
}

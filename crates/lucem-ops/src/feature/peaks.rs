//! Local maxima (peak) detection with minimum-distance non-maximum suppression.

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast, to_f64, ImageFloat};
use lucem_core::memory::{DeviceArray, Workspace};

use super::kernels::peak_candidates_2d_kernel;
use crate::core::{check, reduce, PadMode, StructuringElement};
use crate::error::{OpsError, OpsResult};
use crate::filter::order::{max_filter_into, OrderFilter};

/// Settings for [`local_peaks`].
#[derive(Debug, Clone, PartialEq)]
pub struct LocalPeaks {
    /// Minimum Chebyshev/Euclidean distance between returned peaks (`>= 1`).
    pub min_distance: usize,
    /// Optional absolute minimum intensity threshold; defaults to the image minimum.
    pub threshold_abs: Option<f64>,
    /// Optional relative threshold in `0.0..=1.0` scaled by the image maximum.
    pub threshold_rel: Option<f64>,
    /// Number of border pixels to exclude (`None` defaults to `min_distance`; `Some(0)` keeps borders).
    pub exclude_border: Option<usize>,
    /// Maximum number of peaks to return, ordered by descending intensity then raster order.
    pub num_peaks: Option<usize>,
}

impl Default for LocalPeaks {
    fn default() -> Self {
        Self {
            min_distance: 1,
            threshold_abs: None,
            threshold_rel: None,
            exclude_border: None,
            num_peaks: None,
        }
    }
}

/// Finds local maxima in a 2-D or N-D image separated by at least `settings.min_distance`, returning
/// coordinates `[row, col, ...]` sorted by descending peak intensity (ties broken in raster order).
pub fn local_peaks<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &LocalPeaks,
) -> OpsResult<Vec<Vec<usize>>> {
    if settings.min_distance == 0 {
        return Err(OpsError::InvalidParameter {
            name: "min_distance",
            reason: "min_distance must be at least 1".to_string(),
        });
    }
    if image.is_empty() {
        return Ok(Vec::new());
    }

    let stats = reduce::sum_min_max(client, image)?;
    let mut threshold = settings.threshold_abs.unwrap_or(stats.min);
    if let Some(rel) = settings.threshold_rel {
        if !(0.0..=1.0).contains(&rel) || !rel.is_finite() {
            return Err(OpsError::InvalidParameter {
                name: "threshold_rel",
                reason: format!("threshold_rel must be in 0.0..=1.0, got {rel}"),
            });
        }
        threshold = threshold.max(stats.max * rel);
    }

    let size = 2 * settings.min_distance + 1;
    let se = StructuringElement::hypercube(image.ndim(), size)?;
    let max_filtered = DeviceArray::<F>::empty(client, image.shape().clone())?;
    let mut ws = Workspace::new(client);
    max_filter_into(
        client,
        image,
        &se,
        &max_filtered,
        &OrderFilter { mode: PadMode::Constant, cval: stats.min - 1.0, origins: None },
        &mut ws,
    )?;

    let border = settings.exclude_border.unwrap_or(settings.min_distance);
    let dims = image.dims();
    if dims.iter().any(|&d| d <= 2 * border) {
        return Ok(Vec::new());
    }

    let host_img = image.to_host(client);
    let mut candidates: Vec<(f64, usize, Vec<usize>)> = Vec::new();

    if image.ndim() == 2 {
        let flags = DeviceArray::<u32>::zeros(client, image.shape().clone())?;
        let geom = LaunchGeometry::elementwise(client, image.len());
        unsafe {
            peak_candidates_2d_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                image.binding(),
                max_filtered.binding(),
                flags.binding(),
                dims[0] as u32,
                dims[1] as u32,
                border as u32,
                cast::<F>(threshold),
                image.len() as u32,
            );
        }
        let host_flags = flags.to_host(client);
        let w = dims[1];
        for (idx, &f) in host_flags.iter().enumerate() {
            if f != 0 {
                let v = to_f64(host_img[idx]);
                candidates.push((v, idx, vec![idx / w, idx % w]));
            }
        }
    } else {
        check::ndim_in("local_peaks", image.shape(), &[1, 2, 3], "1, 2, or 3")?;
        let host_max = max_filtered.to_host(client);
        let strides = image.shape().strides();
        for idx in 0..image.len() {
            let v = to_f64(host_img[idx]);
            let mx = to_f64(host_max[idx]);
            if v >= mx && v > threshold {
                let mut coord = Vec::with_capacity(dims.len());
                let mut inside = true;
                for d in 0..dims.len() {
                    let c = (idx / strides[d]) % dims[d];
                    if c < border || c + border >= dims[d] {
                        inside = false;
                        break;
                    }
                    coord.push(c);
                }
                if inside {
                    candidates.push((v, idx, coord));
                }
            }
        }
    }

    // Sort by descending intensity, breaking ties by ascending raster index
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    // Greedy suppression within Chebyshev box of radius `min_distance` (handles flat plateaus)
    let min_d = settings.min_distance;
    let max_peaks = settings.num_peaks.unwrap_or(usize::MAX);
    let mut accepted: Vec<Vec<usize>> = Vec::new();
    'cand: for (_, _, coord) in candidates {
        if accepted.len() >= max_peaks {
            break;
        }
        for prev in &accepted {
            let within_box = coord
                .iter()
                .zip(prev.iter())
                .all(|(&c, &p)| c.abs_diff(p) <= min_d);
            if within_box {
                continue 'cand;
            }
        }
        accepted.push(coord);
    }
    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lucem_core::Shape;

    fn detects_isolated_peaks_and_suppresses_plateaus(client: &Client) {
        let (h, w) = (11usize, 11usize);
        let mut data = vec![0.0f32; h * w];
        data[3 * w + 3] = 10.0;
        data[7 * w + 7] = 5.0;
        data[7 * w + 8] = 5.0; // adjacent plateau within min_distance = 2
        let img = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();
        let peaks = local_peaks(
            client,
            &img,
            &LocalPeaks {
                min_distance: 2,
                threshold_abs: Some(1.0),
                exclude_border: Some(1),
                ..LocalPeaks::default()
            },
        )
        .unwrap();
        assert_eq!(peaks, vec![vec![3, 3], vec![7, 7]], "{}", client.name());
    }
    runtime_test!(test_local_peaks, detects_isolated_peaks_and_suppresses_plateaus);
}

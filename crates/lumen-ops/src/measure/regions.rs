//! Connected-component labeling, bounding boxes, and region feature extraction (`regionprops`).
//!
//! Built on [`crate::core::components`] so region labels and statistics are deterministic across
//! CPU and GPU backends.

use std::collections::BTreeMap;

use cubecl::prelude::*;
use lumen_core::element::{to_f64, ImageFloat};
use lumen_core::memory::DeviceArray;

use crate::core::check;
use crate::core::components::{
    self, ConnectedComponents, RegionSummary,
};
use crate::core::structuring::StructuringElement;
use crate::error::OpsResult;

/// Rich geometric and intensity features for a single labeled region (analogous to `regionprops`).
#[derive(Debug, Clone, PartialEq)]
pub struct RegionFeatures {
    pub label: u32,
    pub area: usize,
    pub bbox: Vec<(usize, usize)>,
    pub bbox_area: usize,
    pub extent: f64,
    pub centroid: Vec<f64>,
    pub weighted_centroid: Vec<f64>,
    pub equivalent_diameter: f64,
    pub major_axis_length: f64,
    pub minor_axis_length: f64,
    pub eccentricity: f64,
    pub orientation: f64,
    pub inertia_tensor_eigvals: Vec<f64>,
    pub perimeter: f64,
    pub mean_intensity: f64,
    pub min_intensity: f64,
    pub max_intensity: f64,
    pub std_intensity: f64,
    pub sum_intensity: f64,
}

/// Labels connected foreground (`!= 0`) components in `input` using `footprint` (or orthogonal
/// 1-connectivity when `None`), returning canonical raster-ordered labels `1..=num_labels`.
pub fn label_components<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    footprint: Option<&StructuringElement>,
) -> OpsResult<ConnectedComponents> {
    components::label_components(client, input, footprint)
}

/// Returns the tight axis-aligned bounding box `[(min_0, max_0), ...]` (`max` exclusive) for every
/// positive label in `labels`, sorted by ascending label ID.
pub fn bounding_boxes(
    client: &Client,
    labels: &DeviceArray<u32>,
) -> OpsResult<BTreeMap<u32, Vec<(usize, usize)>>> {
    let summaries = components::summarize_regions::<f32>(client, labels, None)?;
    Ok(summaries.into_iter().map(|s| (s.label, s.bbox)).collect())
}

/// Computes deterministic per-label intensity and coordinate summaries (`RegionSummary`) for every
/// positive label in `labels`.
pub fn region_stats<F: ImageFloat>(
    client: &Client,
    labels: &DeviceArray<u32>,
    intensity: Option<&DeviceArray<F>>,
) -> OpsResult<Vec<RegionSummary>> {
    components::summarize_regions(client, labels, intensity)
}

/// Computes comprehensive geometric and intensity features (`RegionFeatures`) for every positive
/// label in `labels`.
pub fn region_features<F: ImageFloat>(
    client: &Client,
    labels: &DeviceArray<u32>,
    intensity: Option<&DeviceArray<F>>,
) -> OpsResult<Vec<RegionFeatures>> {
    if let Some(img) = intensity {
        check::same_shape("labels", labels, "intensity", img)?;
    }
    let summaries = components::summarize_regions(client, labels, intensity)?;
    if summaries.is_empty() {
        return Ok(Vec::new());
    }

    let host_labels = labels.to_host(client);
    let _host_vals: Option<Vec<f64>> = intensity.map(|img| img.to_host(client).into_iter().map(to_f64).collect());
    let dims = labels.dims();
    let strides = labels.shape().strides();
    let ndim = dims.len();

    // Second pass for central second-order moments (covariance / inertia tensor) and boundary count.
    struct GeomAcc {
        mu20: f64,
        mu02: f64,
        mu11: f64,
        diag_cov: Vec<f64>,
        perimeter: f64,
    }

    let mut centers: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    let mut geom_map: BTreeMap<u32, GeomAcc> = BTreeMap::new();
    for s in &summaries {
        centers.insert(s.label, s.centroid.clone());
        geom_map.insert(
            s.label,
            GeomAcc {
                mu20: 0.0,
                mu02: 0.0,
                mu11: 0.0,
                diag_cov: vec![0.0; ndim],
                perimeter: 0.0,
            },
        );
    }

    for (idx, &lbl) in host_labels.iter().enumerate() {
        if lbl == 0 {
            continue;
        }
        let c_vec = &centers[&lbl];
        let g = geom_map.get_mut(&lbl).unwrap();

        let mut coords = vec![0usize; ndim];
        for d in 0..ndim {
            let c = (idx / strides[d]) % dims[d];
            coords[d] = c;
            let diff = (c as f64) - c_vec[d];
            g.diag_cov[d] += diff * diff;
        }
        if ndim >= 2 {
            let dy = (coords[0] as f64) - c_vec[0];
            let dx = (coords[1] as f64) - c_vec[1];
            g.mu20 += dy * dy;
            g.mu02 += dx * dx;
            g.mu11 += dy * dx;
        }

        // Exposed 4-connected / 2*ndim-connected faces
        let mut is_boundary = false;
        for d in 0..ndim {
            let c = coords[d];
            if c == 0 || host_labels[idx - strides[d]] != lbl {
                is_boundary = true;
            }
            if c + 1 >= dims[d] || host_labels[idx + strides[d]] != lbl {
                is_boundary = true;
            }
        }
        if is_boundary {
            g.perimeter += 1.0;
        }
    }

    let mut out = Vec::with_capacity(summaries.len());
    for s in summaries {
        let g = &geom_map[&s.label];
        let area = s.count;
        let area_f = (area.max(1)) as f64;
        let bbox_area: usize = s
            .bbox
            .iter()
            .map(|&(lo, hi)| hi.saturating_sub(lo).max(1))
            .product();
        let extent = (area as f64) / (bbox_area.max(1) as f64);
        let equivalent_diameter = if ndim == 2 {
            (4.0 * (area as f64) / std::f64::consts::PI).sqrt()
        } else {
            (area as f64).powf(1.0 / (ndim as f64))
        };

        let (major_axis_length, minor_axis_length, eccentricity, orientation, eigvals) = if ndim == 2 {
            let u20 = g.mu20 / area_f;
            let u02 = g.mu02 / area_f;
            let u11 = g.mu11 / area_f;
            let common = ((u20 - u02).powi(2) + 4.0 * u11 * u11).sqrt();
            let l1 = 0.5 * (u20 + u02 + common).max(0.0);
            let l2 = 0.5 * (u20 + u02 - common).max(0.0);
            // Inertia tensor diagonal is (u02, u20) with off-diagonal -u11; eigenvalues match l1, l2
            let major = 4.0 * l1.sqrt();
            let minor = 4.0 * l2.sqrt();
            let ecc = if l1 > 1e-12 {
                (1.0 - (l2 / l1).clamp(0.0, 1.0)).sqrt()
            } else {
                0.0
            };
            let orient = 0.5 * (2.0 * u11).atan2(u20 - u02);
            (major, minor, ecc, orient, vec![l1, l2])
        } else {
            let mut ev: Vec<f64> = g.diag_cov.iter().map(|&v| v / area_f).collect();
            ev.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let l1 = ev.first().copied().unwrap_or(0.0).max(0.0);
            let l2 = ev.last().copied().unwrap_or(0.0).max(0.0);
            let major = 4.0 * l1.sqrt();
            let minor = 4.0 * l2.sqrt();
            let ecc = if l1 > 1e-12 {
                (1.0 - (l2 / l1).clamp(0.0, 1.0)).sqrt()
            } else {
                0.0
            };
            (major, minor, ecc, 0.0, ev)
        };

        out.push(RegionFeatures {
            label: s.label,
            area,
            bbox: s.bbox,
            bbox_area,
            extent,
            centroid: s.centroid,
            weighted_centroid: s.weighted_centroid,
            equivalent_diameter,
            major_axis_length,
            minor_axis_length,
            eccentricity,
            orientation,
            inertia_tensor_eigvals: eigvals,
            perimeter: g.perimeter,
            mean_intensity: s.mean,
            min_intensity: s.min,
            max_intensity: s.max,
            std_intensity: s.std_dev,
            sum_intensity: s.sum,
        });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lumen_core::Shape;

    fn region_features_and_bboxes_match_known_shapes(client: &Client) {
        #[rustfmt::skip]
        let mask = vec![
            0u32, 1, 1, 0,
            0,    1, 1, 0,
            0,    0, 0, 1,
            0,    0, 0, 0,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([4, 4]), mask).unwrap();
        let cc = label_components(client, &arr, None).unwrap();
        assert_eq!(cc.num_labels, 2, "{}", client.name());

        let bboxes = bounding_boxes(client, &cc.labels).unwrap();
        assert_eq!(bboxes[&1], vec![(0, 2), (1, 3)]);
        assert_eq!(bboxes[&2], vec![(2, 3), (3, 4)]);

        let feats = region_features::<f32>(client, &cc.labels, None).unwrap();
        assert_eq!(feats.len(), 2);
        assert_eq!(feats[0].area, 4);
        assert_close(&[feats[0].extent], &[1.0], 1e-6, client.name());
        assert_close(&feats[0].centroid, &[0.5, 1.5], 1e-6, client.name());
        assert_close(&[feats[0].eccentricity], &[0.0], 1e-6, client.name());
    }
    runtime_test!(test_measure_regions, region_features_and_bboxes_match_known_shapes);
}

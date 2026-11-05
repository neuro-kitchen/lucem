//! Connected-component labeling, sequential relabeling, and deterministic per-region reductions.
//!
//! Assigns canonical `1..=num_labels` identifiers ordered by the first pixel of each component in
//! C raster order, and computes per-component statistics deterministically without float atomics.

use std::collections::BTreeMap;

use cubecl::prelude::*;
use lumen_core::element::{to_f64, ImageFloat};
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;

use super::compact::flag_nonzero;
use super::structuring::StructuringElement;
use crate::error::{OpsError, OpsResult};

/// Connected-component labeling result: the `u32` label map (`0` for background, `1..=num_labels`
/// in raster order of first appearance) and `num_labels`.
#[derive(Debug, Clone)]
pub struct ConnectedComponents {
    pub labels: DeviceArray<u32>,
    pub num_labels: usize,
}

/// Summary statistics for a single labeled region.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionSummary {
    pub label: u32,
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub min_index: usize,
    pub max_index: usize,
    /// Minimum and maximum coordinate along each axis `[(min_0, max_0), ...]`, where `max` is
    /// exclusive (`min..max` slice bounds).
    pub bbox: Vec<(usize, usize)>,
    /// Intensity-unweighted centroid coordinates along each axis.
    pub centroid: Vec<f64>,
    /// Intensity-weighted center of mass along each axis.
    pub weighted_centroid: Vec<f64>,
}

/// Finds connected components of non-zero elements in `input` using `footprint` (or orthogonal
/// 1-connectivity when `None`), returning canonical raster-ordered labels `1..=num_labels`.
pub fn label_components<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    footprint: Option<&StructuringElement>,
) -> OpsResult<ConnectedComponents> {
    if input.is_empty() {
        return Ok(ConnectedComponents {
            labels: DeviceArray::<u32>::zeros(client, input.shape().clone())?,
            num_labels: 0,
        });
    }
    let default_se;
    let se = match footprint {
        Some(s) => {
            if s.ndim() != input.ndim() {
                return Err(OpsError::InvalidDimension {
                    what: "footprint",
                    expected: "matching input ndim",
                    actual: s.ndim(),
                    shape: s.dims().to_vec(),
                });
            }
            s
        }
        None => {
            default_se = StructuringElement::connectivity(input.ndim(), 1)?;
            &default_se
        }
    };

    let mask = flag_nonzero(client, input)?.to_host(client);
    let n = mask.len();
    let dims = input.dims();
    let strides = input.shape().strides();
    let offsets = se.coordinate_offsets(None);

    // Disjoint-set union with path halving, canonicalized in a single second pass to raster order.
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = p;
        }
        x
    }

    for idx in 0..n {
        if mask[idx] == 0 {
            continue;
        }
        for off in &offsets {
            let mut neighbor_idx = 0usize;
            let mut in_bounds = true;
            for d in 0..dims.len() {
                let c = ((idx / strides[d]) % dims[d]) as isize + (off[d] as isize);
                if c < 0 || c >= dims[d] as isize {
                    in_bounds = false;
                    break;
                }
                neighbor_idx += (c as usize) * strides[d];
            }
            if in_bounds && neighbor_idx < idx && mask[neighbor_idx] != 0 {
                let ra = find(&mut parent, idx as u32);
                let rb = find(&mut parent, neighbor_idx as u32);
                if ra != rb {
                    let root = ra.min(rb);
                    let child = ra.max(rb);
                    parent[child as usize] = root;
                }
            }
        }
    }

    let mut canonical = vec![0u32; n];
    let mut labels = vec![0u32; n];
    let mut next_label = 0u32;
    for idx in 0..n {
        if mask[idx] != 0 {
            let root = find(&mut parent, idx as u32);
            let mut lbl = canonical[root as usize];
            if lbl == 0 {
                next_label += 1;
                lbl = next_label;
                canonical[root as usize] = lbl;
            }
            labels[idx] = lbl;
        }
    }

    Ok(ConnectedComponents {
        labels: DeviceArray::from_host(client, input.shape().clone(), labels)?,
        num_labels: next_label as usize,
    })
}

/// Relabels an arbitrary integer label array sequentially into `offset .. offset + num_unique` in
/// order of ascending positive label values (`0` remains `0`).
pub fn relabel_sequential(
    client: &Client,
    labels: &DeviceArray<u32>,
    offset: u32,
) -> OpsResult<(DeviceArray<u32>, BTreeMap<u32, u32>)> {
    if offset == 0 {
        return Err(OpsError::InvalidParameter {
            name: "offset",
            reason: "sequential label offset must be >= 1".to_string(),
        });
    }
    let host = labels.to_host(client);
    let mut forward = BTreeMap::new();
    for &lbl in &host {
        if lbl > 0 {
            forward.entry(lbl).or_insert(0u32);
        }
    }
    for (i, val) in forward.values_mut().enumerate() {
        *val = offset + (i as u32);
    }
    let mapped: Vec<u32> = host
        .into_iter()
        .map(|l| if l == 0 { 0 } else { forward[&l] })
        .collect();
    Ok((DeviceArray::from_host(client, labels.shape().clone(), mapped)?, forward))
}

/// Computes deterministic per-label region summaries in a single raster pass (avoiding non-associative
/// float atomic additions).
pub fn summarize_regions<F: ImageFloat>(
    client: &Client,
    labels: &DeviceArray<u32>,
    intensity: Option<&DeviceArray<F>>,
) -> OpsResult<Vec<RegionSummary>> {
    if let Some(img) = intensity {
        super::check::same_shape("labels", labels, "intensity", img)?;
    }
    let host_labels = labels.to_host(client);
    let host_vals: Vec<f64> = match intensity {
        Some(img) => img.to_host(client).into_iter().map(to_f64).collect(),
        None => vec![1.0; host_labels.len()],
    };
    let dims = labels.dims();
    let strides = labels.shape().strides();
    let ndim = dims.len();

    struct Acc {
        count: usize,
        sum: f64,
        mean: f64,
        m2: f64,
        min: f64,
        max: f64,
        min_index: usize,
        max_index: usize,
        bbox: Vec<(usize, usize)>,
        coord_sum: Vec<f64>,
        weighted_coord_sum: Vec<f64>,
    }

    let mut map: BTreeMap<u32, Acc> = BTreeMap::new();
    for (idx, (&lbl, &val)) in host_labels.iter().zip(host_vals.iter()).enumerate() {
        if lbl == 0 {
            continue;
        }
        let entry = map.entry(lbl).or_insert_with(|| Acc {
            count: 0,
            sum: 0.0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            min_index: idx,
            max_index: idx,
            bbox: vec![(usize::MAX, 0); ndim],
            coord_sum: vec![0.0; ndim],
            weighted_coord_sum: vec![0.0; ndim],
        });
        entry.count += 1;
        entry.sum += val;
        let delta = val - entry.mean;
        entry.mean += delta / (entry.count as f64);
        entry.m2 += delta * (val - entry.mean);
        if val < entry.min {
            entry.min = val;
            entry.min_index = idx;
        }
        if val > entry.max {
            entry.max = val;
            entry.max_index = idx;
        }
        for d in 0..ndim {
            let c = (idx / strides[d]) % dims[d];
            entry.bbox[d].0 = entry.bbox[d].0.min(c);
            entry.bbox[d].1 = entry.bbox[d].1.max(c + 1);
            entry.coord_sum[d] += c as f64;
            entry.weighted_coord_sum[d] += (c as f64) * val;
        }
    }

    Ok(map
        .into_iter()
        .map(|(label, acc)| {
            let n = acc.count.max(1) as f64;
            let variance = acc.m2 / n;
            let centroid = acc.coord_sum.into_iter().map(|s| s / n).collect();
            let weighted_centroid = if acc.sum.abs() > f64::EPSILON {
                acc.weighted_coord_sum.into_iter().map(|s| s / acc.sum).collect()
            } else {
                vec![0.0; ndim]
            };
            RegionSummary {
                label,
                count: acc.count,
                sum: acc.sum,
                mean: acc.mean,
                variance,
                std_dev: variance.sqrt(),
                min: acc.min,
                max: acc.max,
                min_index: acc.min_index,
                max_index: acc.max_index,
                bbox: acc.bbox,
                centroid,
                weighted_centroid,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn labels_and_summarizes_in_raster_order(client: &Client) {
        #[rustfmt::skip]
        let mask = vec![
            0u32, 1, 1, 0,
            0,    0, 1, 0,
            1,    0, 0, 1,
            1,    0, 0, 0,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([4, 4]), mask).unwrap();
        let cc = label_components(client, &arr, None).unwrap();
        assert_eq!(cc.num_labels, 3, "{}", client.name());
        #[rustfmt::skip]
        let want = vec![
            0u32, 1, 1, 0,
            0,    0, 1, 0,
            2,    0, 0, 3,
            2,    0, 0, 0,
        ];
        assert_eq!(cc.labels.to_host(client), want, "{}", client.name());

        let summaries = summarize_regions::<f32>(client, &cc.labels, None).unwrap();
        assert_eq!(summaries.len(), 3);
        assert_eq!(summaries[0].count, 3);
        assert_eq!(summaries[0].bbox, vec![(0, 2), (1, 3)]);
        assert_eq!(summaries[1].count, 2);
        assert_eq!(summaries[2].count, 1);
    }
    runtime_test!(test_connected_components, labels_and_summarizes_in_raster_order);
}

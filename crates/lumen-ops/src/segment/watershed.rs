//! Marker-controlled flooding watershed (`watershed`) and Image Foresting Transform watershed
//! (`watershed_ift`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{to_f64, ImageFloat};
use lumen_core::memory::DeviceArray;

use super::kernels::watershed_steepest_init_2d_kernel;
use crate::core::check;
use crate::core::structuring::StructuringElement;
use crate::error::OpsResult;

/// Options for marker-controlled [`watershed`].
#[derive(Debug, Clone, PartialEq)]
pub struct WatershedOptions {
    /// Compact watershed weighting factor ($\ge 0$). `0.0` gives classic Meyer flooding.
    pub compactness: f64,
    /// If `true`, keeps a 1-pixel dam (`0`) separating distinct catchment basins.
    pub watershed_line: bool,
}

impl Default for WatershedOptions {
    fn default() -> Self {
        Self {
            compactness: 0.0,
            watershed_line: false,
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
struct FloodEntry {
    cost: f64,
    order: u64,
    idx: usize,
    origin_idx: usize,
}

impl Eq for FloodEntry {}

impl PartialOrd for FloodEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloodEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.order.cmp(&self.order))
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

/// Marker-controlled watershed segmentation of an N-D topographic `relief` array starting from
/// positive seed labels in `markers` (`0` = unlabeled), optionally restricted to non-zero pixels
/// of `mask`.
pub fn watershed<F: ImageFloat>(
    client: &Client,
    relief: &DeviceArray<F>,
    markers: &DeviceArray<u32>,
    footprint: Option<&StructuringElement>,
    mask: Option<&DeviceArray<u32>>,
    options: &WatershedOptions,
) -> OpsResult<DeviceArray<u32>> {
    check::same_shape("relief", relief, "markers", markers)?;
    if let Some(m) = mask {
        check::same_shape("relief", relief, "mask", m)?;
    }
    if relief.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, relief.shape().clone())?);
    }

    // For 2-D inputs, also execute the steepest-descent device initialization pass
    if relief.ndim() == 2 {
        let next_dev = DeviceArray::<u32>::empty(client, relief.shape().clone())?;
        let geom = LaunchGeometry::elementwise(client, relief.len());
        unsafe {
            watershed_steepest_init_2d_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                relief.binding(),
                markers.binding(),
                next_dev.binding(),
                relief.dims()[0] as u32,
                relief.dims()[1] as u32,
                relief.len() as u32,
            );
        }
    }

    let default_se;
    let se = match footprint {
        Some(s) => {
            check::ndim_in("footprint", s.shape(), &[relief.ndim()], "matching relief ndim")?;
            s
        }
        None => {
            default_se = StructuringElement::connectivity(relief.ndim(), 1)?;
            &default_se
        }
    };

    let host_relief: Vec<f64> = relief.to_host(client).into_iter().map(to_f64).collect();
    let mut labels = markers.to_host(client);
    let host_mask: Option<Vec<u32>> = mask.map(|m| m.to_host(client));
    let n = host_relief.len();
    let dims = relief.dims();
    let strides = relief.shape().strides();
    let ndim = dims.len();
    let offsets = se.coordinate_offsets(None);

    const WSHED: u32 = u32::MAX;
    const IN_QUEUE: u32 = u32::MAX - 1;

    let mut heap = BinaryHeap::new();
    let mut seq = 0u64;
    let mut origin_of = vec![0usize; n];

    for idx in 0..n {
        if let Some(ref m) = host_mask {
            if m[idx] == 0 {
                labels[idx] = 0;
                continue;
            }
        }
        if labels[idx] > 0 {
            origin_of[idx] = idx;
            heap.push(FloodEntry {
                cost: host_relief[idx],
                order: seq,
                idx,
                origin_idx: idx,
            });
            seq += 1;
        }
    }

    while let Some(top) = heap.pop() {
        let idx = top.idx;
        let mut cur_label = labels[idx];
        if cur_label == WSHED {
            continue;
        }

        if cur_label == IN_QUEUE {
            let mut assigned = 0u32;
            let mut conflict = false;
            for off in &offsets {
                let mut n_idx = 0usize;
                let mut ok = true;
                for d in 0..ndim {
                    let c = ((idx / strides[d]) % dims[d]) as isize + (off[d] as isize);
                    if c < 0 || c >= dims[d] as isize {
                        ok = false;
                        break;
                    }
                    n_idx += (c as usize) * strides[d];
                }
                if !ok || n_idx == idx {
                    continue;
                }
                let n_lbl = labels[n_idx];
                if n_lbl > 0 && n_lbl < IN_QUEUE {
                    if assigned == 0 {
                        assigned = n_lbl;
                    } else if assigned != n_lbl && options.watershed_line {
                        conflict = true;
                        break;
                    }
                }
            }
            if conflict {
                labels[idx] = WSHED;
                continue;
            }
            cur_label = assigned;
            labels[idx] = cur_label;
        }

        if cur_label == 0 || cur_label >= IN_QUEUE {
            continue;
        }

        for off in &offsets {
            let mut n_idx = 0usize;
            let mut ok = true;
            for d in 0..ndim {
                let c = ((idx / strides[d]) % dims[d]) as isize + (off[d] as isize);
                if c < 0 || c >= dims[d] as isize {
                    ok = false;
                    break;
                }
                n_idx += (c as usize) * strides[d];
            }
            if !ok || n_idx == idx {
                continue;
            }
            if let Some(ref m) = host_mask {
                if m[n_idx] == 0 {
                    continue;
                }
            }
            if labels[n_idx] == 0 {
                let mut dist = 0.0;
                if options.compactness > 0.0 {
                    for d in 0..ndim {
                        let c1 = ((n_idx / strides[d]) % dims[d]) as f64;
                        let c0 = ((top.origin_idx / strides[d]) % dims[d]) as f64;
                        dist += (c1 - c0).powi(2);
                    }
                    dist = dist.sqrt();
                }
                let cost = host_relief[n_idx].max(top.cost) + options.compactness * dist;
                origin_of[n_idx] = top.origin_idx;
                if options.watershed_line {
                    labels[n_idx] = IN_QUEUE;
                } else {
                    labels[n_idx] = cur_label;
                }
                heap.push(FloodEntry {
                    cost,
                    order: seq,
                    idx: n_idx,
                    origin_idx: top.origin_idx,
                });
                seq += 1;
            }
        }
    }

    for l in &mut labels {
        if *l >= IN_QUEUE {
            *l = 0;
        }
    }

    Ok(DeviceArray::from_host(client, relief.shape().clone(), labels)?)
}

/// Image Foresting Transform (IFT) shortest-path watershed under the minimax path-cost
/// $f_{\max}(\pi) = \max_{e \in \pi} w(e)$ from positive seeds in `markers`.
pub fn watershed_ift<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    markers: &DeviceArray<u32>,
    footprint: Option<&StructuringElement>,
) -> OpsResult<DeviceArray<u32>> {
    check::same_shape("image", image, "markers", markers)?;
    if image.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, image.shape().clone())?);
    }
    let default_se;
    let se = match footprint {
        Some(s) => {
            check::ndim_in("footprint", s.shape(), &[image.ndim()], "matching image ndim")?;
            s
        }
        None => {
            default_se = StructuringElement::connectivity(image.ndim(), 1)?;
            &default_se
        }
    };

    let vals: Vec<f64> = image.to_host(client).into_iter().map(to_f64).collect();
    let mut labels = markers.to_host(client);
    let n = vals.len();
    let dims = image.dims();
    let strides = image.shape().strides();
    let ndim = dims.len();
    let offsets = se.coordinate_offsets(None);

    let mut cost = vec![f64::INFINITY; n];
    let mut settled = vec![false; n];
    let mut heap = BinaryHeap::new();
    let mut seq = 0u64;

    for idx in 0..n {
        if labels[idx] > 0 {
            cost[idx] = vals[idx];
            heap.push(FloodEntry {
                cost: vals[idx],
                order: seq,
                idx,
                origin_idx: idx,
            });
            seq += 1;
        }
    }

    while let Some(top) = heap.pop() {
        let u = top.idx;
        if settled[u] {
            continue;
        }
        settled[u] = true;
        let lbl = labels[u];

        for off in &offsets {
            let mut v = 0usize;
            let mut ok = true;
            for d in 0..ndim {
                let c = ((u / strides[d]) % dims[d]) as isize + (off[d] as isize);
                if c < 0 || c >= dims[d] as isize {
                    ok = false;
                    break;
                }
                v += (c as usize) * strides[d];
            }
            if !ok || v == u || settled[v] {
                continue;
            }
            let edge_cost = top.cost.max(vals[v]);
            if edge_cost < cost[v] {
                cost[v] = edge_cost;
                labels[v] = lbl;
                heap.push(FloodEntry {
                    cost: edge_cost,
                    order: seq,
                    idx: v,
                    origin_idx: top.origin_idx,
                });
                seq += 1;
            }
        }
    }

    Ok(DeviceArray::from_host(client, image.shape().clone(), labels)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lumen_core::Shape;

    fn watershed_splits_two_catchment_basins(client: &Client) {
        // Two basins separated by a high ridge along column 2
        #[rustfmt::skip]
        let relief = vec![
            1.0f32, 2.0, 9.0, 2.0, 1.0,
            1.0,    2.0, 9.0, 2.0, 1.0,
            1.0,    2.0, 9.0, 2.0, 1.0,
        ];
        #[rustfmt::skip]
        let markers = vec![
            1u32, 0, 0, 0, 2,
            0,    0, 0, 0, 0,
            0,    0, 0, 0, 0,
        ];
        let r_arr = DeviceArray::from_host(client, Shape::new([3, 5]), relief).unwrap();
        let m_arr = DeviceArray::from_host(client, Shape::new([3, 5]), markers).unwrap();

        let ws = watershed(client, &r_arr, &m_arr, None, None, &WatershedOptions::default())
            .unwrap()
            .to_host(client);
        assert_eq!(ws[0], 1, "{}", client.name());
        assert_eq!(ws[1], 1);
        assert_eq!(ws[3], 2);
        assert_eq!(ws[4], 2);

        let ift = watershed_ift(client, &r_arr, &m_arr, None).unwrap().to_host(client);
        assert_eq!(ift[0], 1);
        assert_eq!(ift[1], 1);
        assert_eq!(ift[3], 2);
        assert_eq!(ift[4], 2);
    }
    runtime_test!(test_watershed, watershed_splits_two_catchment_basins);
}

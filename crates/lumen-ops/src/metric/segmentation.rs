//! Segmentation and registration evaluation metrics: Hausdorff distance (`hausdorff`,
//! `hausdorff_pair`), Adapted Rand Error (`adapted_rand`), Variation of Information
//! (`variation_of_info`), and Normalized Mutual Information (`mutual_info`).

use std::collections::BTreeMap;

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{to_f64, ImageFloat};
use lumen_core::memory::DeviceArray;

use super::kernels::{invert_binary_mask_kernel, mask_at_edt_kernel};
use crate::core::{check, compact, distance, reduce};
use crate::error::{OpsError, OpsResult};

/// Variant of Hausdorff distance for [`hausdorff`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HausdorffMethod {
    /// Standard maximum directed Hausdorff distance:
    /// $\max\bigl(\max_{a \in A} d(a, B),\; \max_{b \in B} d(b, A)\bigr)$.
    #[default]
    Standard,
    /// Dubuisson & Jain (1994) modified (mean directed) Hausdorff distance:
    /// $\max\bigl(\frac{1}{|A|}\sum_{a \in A} d(a, B),\; \frac{1}{|B|}\sum_{b \in B} d(b, A)\bigr)$.
    Modified,
}

/// Computes the exact Euclidean Hausdorff distance (`Standard` or `Modified`) between non-zero
/// foreground elements of `mask_a` and `mask_b` on the device using separable Euclidean distance
/// transforms.
///
/// Returns `0.0` if both masks are empty (all zero), or `f64::INFINITY` if exactly one mask is
/// empty.
pub fn hausdorff<A: Numeric + CubeElement, B: Numeric + CubeElement>(
    client: &Client,
    mask_a: &DeviceArray<A>,
    mask_b: &DeviceArray<B>,
    method: HausdorffMethod,
) -> OpsResult<f64> {
    check::same_shape("mask_a", mask_a, "mask_b", mask_b)?;
    if mask_a.is_empty() {
        return Ok(0.0);
    }

    let n = mask_a.len();
    let shape = mask_a.shape();
    let geom = LaunchGeometry::elementwise(client, n);

    let inv_a = DeviceArray::<u32>::empty(client, shape.clone())?;
    let inv_b = DeviceArray::<u32>::empty(client, shape.clone())?;
    unsafe {
        invert_binary_mask_kernel::launch::<A>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim,
            mask_a.binding(),
            inv_a.binding(),
            n as u32,
        );
        invert_binary_mask_kernel::launch::<B>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim,
            mask_b.binding(),
            inv_b.binding(),
            n as u32,
        );
    }

    let stats_inv_a = reduce::sum_min_max(client, &DeviceArray::from_host(
        client,
        shape.clone(),
        inv_a.to_host(client).into_iter().map(|v| v as f32).collect(),
    )?)?;
    let stats_inv_b = reduce::sum_min_max(client, &DeviceArray::from_host(
        client,
        shape.clone(),
        inv_b.to_host(client).into_iter().map(|v| v as f32).collect(),
    )?)?;

    let count_a = n.saturating_sub(stats_inv_a.sum.round() as usize);
    let count_b = n.saturating_sub(stats_inv_b.sum.round() as usize);
    if count_a == 0 && count_b == 0 {
        return Ok(0.0);
    }
    if count_a == 0 || count_b == 0 {
        return Ok(f64::INFINITY);
    }

    let edt_to_b: DeviceArray<f32> = distance::euclidean_distance(client, &inv_b, None)?;
    let edt_to_a: DeviceArray<f32> = distance::euclidean_distance(client, &inv_a, None)?;

    let dist_ab = DeviceArray::<f32>::empty(client, shape.clone())?;
    let dist_ba = DeviceArray::<f32>::empty(client, shape.clone())?;
    unsafe {
        mask_at_edt_kernel::launch::<A, f32>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim,
            mask_a.binding(),
            edt_to_b.binding(),
            dist_ab.binding(),
            n as u32,
        );
        mask_at_edt_kernel::launch::<B, f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            mask_b.binding(),
            edt_to_a.binding(),
            dist_ba.binding(),
            n as u32,
        );
    }

    let s_ab = reduce::sum_min_max(client, &dist_ab)?;
    let s_ba = reduce::sum_min_max(client, &dist_ba)?;
    match method {
        HausdorffMethod::Standard => Ok(s_ab.max.max(s_ba.max)),
        HausdorffMethod::Modified => {
            let mean_ab = s_ab.sum / (count_a as f64);
            let mean_ba = s_ba.sum / (count_b as f64);
            Ok(mean_ab.max(mean_ba))
        }
    }
}

/// Returns the pair of N-D pixel coordinates `(coord_in_a, coord_in_b)` that realize the standard
/// Hausdorff distance between `mask_a` and `mask_b`, or `None` if either mask has no non-zero
/// foreground pixels.
pub fn hausdorff_pair<A: Numeric + CubeElement, B: Numeric + CubeElement>(
    client: &Client,
    mask_a: &DeviceArray<A>,
    mask_b: &DeviceArray<B>,
) -> OpsResult<Option<(Vec<usize>, Vec<usize>)>> {
    check::same_shape("mask_a", mask_a, "mask_b", mask_b)?;
    if mask_a.is_empty() {
        return Ok(None);
    }
    let flags_a = compact::flag_nonzero(client, mask_a)?.to_host(client);
    let flags_b = compact::flag_nonzero(client, mask_b)?.to_host(client);
    let dims = mask_a.dims();
    let strides = mask_a.shape().strides();
    let ndim = dims.len();

    let to_coords = |idx: usize| -> Vec<usize> {
        (0..ndim).map(|d| (idx / strides[d]) % dims[d]).collect()
    };

    let pts_a: Vec<(usize, Vec<usize>)> = flags_a
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| (f != 0).then(|| (i, to_coords(i))))
        .collect();
    let pts_b: Vec<(usize, Vec<usize>)> = flags_b
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| (f != 0).then(|| (i, to_coords(i))))
        .collect();

    if pts_a.is_empty() || pts_b.is_empty() {
        return Ok(None);
    }

    let dist_sq = |ca: &[usize], cb: &[usize]| -> u64 {
        ca.iter()
            .zip(cb.iter())
            .map(|(&a, &b)| {
                let d = (a as i64) - (b as i64);
                (d * d) as u64
            })
            .sum()
    };

    let mut best_d2 = 0u64;
    let mut best_pair = (pts_a[0].1.clone(), pts_b[0].1.clone());

    for (_, ca) in &pts_a {
        let mut min_d2 = u64::MAX;
        let mut nearest_b = &pts_b[0].1;
        for (_, cb) in &pts_b {
            let d2 = dist_sq(ca, cb);
            if d2 < min_d2 {
                min_d2 = d2;
                nearest_b = cb;
            }
        }
        if min_d2 > best_d2 {
            best_d2 = min_d2;
            best_pair = (ca.clone(), nearest_b.clone());
        }
    }

    for (_, cb) in &pts_b {
        let mut min_d2 = u64::MAX;
        let mut nearest_a = &pts_a[0].1;
        for (_, ca) in &pts_a {
            let d2 = dist_sq(ca, cb);
            if d2 < min_d2 {
                min_d2 = d2;
                nearest_a = ca;
            }
        }
        if min_d2 > best_d2 {
            best_d2 = min_d2;
            best_pair = (nearest_a.clone(), cb.clone());
        }
    }

    Ok(Some(best_pair))
}

/// Output of [`adapted_rand`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptedRandResult {
    /// Adapted Rand Error ($1 - F_1$ of pairwise same-cluster precision and recall).
    pub error: f64,
    /// Pairwise precision ($\sum_{i,j} p_{ij}^2 / \sum_j t_j^2$).
    pub precision: f64,
    /// Pairwise recall ($\sum_{i,j} p_{ij}^2 / \sum_i s_i^2$).
    pub recall: f64,
}

/// Computes the SNEMI3D Adapted Rand Error (`(error, precision, recall)`) between ground-truth
/// integer segmentation `labels_true` and candidate segmentation `labels_test`.
///
/// By default (`ignore_labels = None`), background label `0` in `labels_true` is ignored (pass
/// `Some(&[])` to include `0`).
pub fn adapted_rand(
    client: &Client,
    labels_true: &DeviceArray<u32>,
    labels_test: &DeviceArray<u32>,
    ignore_labels: Option<&[u32]>,
) -> OpsResult<AdaptedRandResult> {
    check::same_shape("labels_true", labels_true, "labels_test", labels_test)?;
    if labels_true.is_empty() {
        return Err(OpsError::EmptyInput("adapted_rand"));
    }
    let ignored: &[u32] = ignore_labels.unwrap_or(&[0]);
    let host_true = labels_true.to_host(client);
    let host_test = labels_test.to_host(client);

    let mut joint: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    let mut sum_true: BTreeMap<u32, u64> = BTreeMap::new();
    let mut sum_test: BTreeMap<u32, u64> = BTreeMap::new();
    let mut total = 0u64;

    for (&lt, &lp) in host_true.iter().zip(host_test.iter()) {
        if ignored.contains(&lt) {
            continue;
        }
        *joint.entry((lt, lp)).or_insert(0) += 1;
        *sum_true.entry(lt).or_insert(0) += 1;
        *sum_test.entry(lp).or_insert(0) += 1;
        total += 1;
    }

    if total == 0 {
        return Ok(AdaptedRandResult {
            error: 0.0,
            precision: 1.0,
            recall: 1.0,
        });
    }

    let n = total as f64;
    let sum_pij2: f64 = joint.values().map(|&c| ((c as f64) / n).powi(2)).sum();
    let sum_si2: f64 = sum_true.values().map(|&c| ((c as f64) / n).powi(2)).sum();
    let sum_tj2: f64 = sum_test.values().map(|&c| ((c as f64) / n).powi(2)).sum();

    let precision = if sum_tj2 > 0.0 { sum_pij2 / sum_tj2 } else { 0.0 };
    let recall = if sum_si2 > 0.0 { sum_pij2 / sum_si2 } else { 0.0 };
    let denom = precision + recall;
    let fscore = if denom > 0.0 {
        2.0 * precision * recall / denom
    } else {
        0.0
    };
    Ok(AdaptedRandResult {
        error: 1.0 - fscore,
        precision,
        recall,
    })
}

/// Computes Meilă's (2003) Variation of Information `(false_splits, false_merges)` in bits
/// ($\log_2$), where `false_splits = H(labels_test | labels_true)` (over-segmentation conditional
/// entropy) and `false_merges = H(labels_true | labels_test)` (under-segmentation conditional
/// entropy).
///
/// Labels listed in `ignore_labels` (defaults to `&[]`) are excluded if present in either array.
pub fn variation_of_info(
    client: &Client,
    labels_true: &DeviceArray<u32>,
    labels_test: &DeviceArray<u32>,
    ignore_labels: Option<&[u32]>,
) -> OpsResult<(f64, f64)> {
    check::same_shape("labels_true", labels_true, "labels_test", labels_test)?;
    if labels_true.is_empty() {
        return Err(OpsError::EmptyInput("variation_of_info"));
    }
    let ignored: &[u32] = ignore_labels.unwrap_or(&[]);
    let host_true = labels_true.to_host(client);
    let host_test = labels_test.to_host(client);

    let mut joint: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    let mut sum_true: BTreeMap<u32, u64> = BTreeMap::new();
    let mut sum_test: BTreeMap<u32, u64> = BTreeMap::new();
    let mut total = 0u64;

    for (&lt, &lp) in host_true.iter().zip(host_test.iter()) {
        if ignored.contains(&lt) || ignored.contains(&lp) {
            continue;
        }
        *joint.entry((lt, lp)).or_insert(0) += 1;
        *sum_true.entry(lt).or_insert(0) += 1;
        *sum_test.entry(lp).or_insert(0) += 1;
        total += 1;
    }

    if total == 0 {
        return Ok((0.0, 0.0));
    }

    let n = total as f64;
    let mut h_test_given_true = 0.0;
    let mut h_true_given_test = 0.0;
    for (&(lt, lp), &nij) in &joint {
        let pij = (nij as f64) / n;
        let pi = (sum_true[&lt] as f64) / n;
        let pj = (sum_test[&lp] as f64) / n;
        h_test_given_true -= pij * (pij / pi).log2();
        h_true_given_test -= pij * (pij / pj).log2();
    }
    Ok((h_test_given_true.max(0.0), h_true_given_test.max(0.0)))
}

/// Computes Studholme, Hill & Hawkes's (1999) Normalized Mutual Information (NMI) between two
/// images using a `bins x bins` joint intensity histogram:
/// $$\operatorname{NMI}(A, B) = \frac{H(A) + H(B)}{H(A, B)} \in [1.0, 2.0]$$
pub fn mutual_info<F: ImageFloat>(
    client: &Client,
    image_a: &DeviceArray<F>,
    image_b: &DeviceArray<F>,
    bins: usize,
) -> OpsResult<f64> {
    check::same_shape("image_a", image_a, "image_b", image_b)?;
    if image_a.is_empty() {
        return Err(OpsError::EmptyInput("mutual_info"));
    }
    if bins < 2 {
        return Err(OpsError::InvalidParameter {
            name: "bins",
            reason: format!("bins must be >= 2, got {bins}"),
        });
    }

    let stats_a = reduce::sum_min_max(client, image_a)?;
    let stats_b = reduce::sum_min_max(client, image_b)?;
    let host_a: Vec<f64> = image_a.to_host(client).into_iter().map(to_f64).collect();
    let host_b: Vec<f64> = image_b.to_host(client).into_iter().map(to_f64).collect();

    let span_a = stats_a.max - stats_a.min;
    let span_b = stats_b.max - stats_b.min;
    let scale_a = if span_a > f64::EPSILON { (bins as f64) / span_a } else { 0.0 };
    let scale_b = if span_b > f64::EPSILON { (bins as f64) / span_b } else { 0.0 };

    let mut joint = vec![0u64; bins * bins];
    let mut marg_a = vec![0u64; bins];
    let mut marg_b = vec![0u64; bins];
    for (&va, &vb) in host_a.iter().zip(host_b.iter()) {
        let ba = (((va - stats_a.min) * scale_a).floor() as isize).clamp(0, (bins - 1) as isize) as usize;
        let bb = (((vb - stats_b.min) * scale_b).floor() as isize).clamp(0, (bins - 1) as isize) as usize;
        joint[ba * bins + bb] += 1;
        marg_a[ba] += 1;
        marg_b[bb] += 1;
    }

    let n = host_a.len() as f64;
    let entropy = |counts: &[u64]| -> f64 {
        let mut h = 0.0;
        for &c in counts {
            if c > 0 {
                let p = (c as f64) / n;
                h -= p * p.ln();
            }
        }
        h
    };

    let ha = entropy(&marg_a);
    let hb = entropy(&marg_b);
    let hab = entropy(&joint);
    if hab <= f64::EPSILON {
        Ok(2.0)
    } else {
        Ok((ha + hb) / hab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test, sample_f64};
    use lumen_core::element::cast_all;
    use lumen_core::Shape;

    fn segmentation_metrics_match_analytic_cases(client: &Client) {
        #[rustfmt::skip]
        let a_data = vec![
            1u32, 0, 0, 0,
            0,    0, 0, 0,
            0,    0, 0, 0,
            0,    0, 0, 0,
        ];
        #[rustfmt::skip]
        let b_data = vec![
            0u32, 0, 0, 0,
            0,    0, 0, 0,
            0,    0, 0, 0,
            0,    0, 0, 1,
        ];
        let ma = DeviceArray::from_host(client, Shape::new([4, 4]), a_data).unwrap();
        let mb = DeviceArray::from_host(client, Shape::new([4, 4]), b_data).unwrap();

        let hd = hausdorff(client, &ma, &mb, HausdorffMethod::Standard).unwrap();
        let want_hd = 18.0f64.sqrt(); // sqrt(3^2 + 3^2)
        assert_close(&[hd], &[want_hd], 1e-5, client.name());

        let mhd = hausdorff(client, &ma, &mb, HausdorffMethod::Modified).unwrap();
        assert_close(&[mhd], &[want_hd], 1e-5, client.name());

        let pair = hausdorff_pair(client, &ma, &mb).unwrap().unwrap();
        assert_eq!(pair, (vec![0, 0], vec![3, 3]));

        // Identical segmentations -> 0 Adapted Rand Error, 0 VI
        let seg_a = DeviceArray::from_host(client, Shape::new([2, 4]), vec![0u32, 1, 1, 2, 0, 1, 2, 2]).unwrap();
        let ar_ident = adapted_rand(client, &seg_a, &seg_a, None).unwrap();
        assert_close(&[ar_ident.error, ar_ident.precision, ar_ident.recall], &[0.0, 1.0, 1.0], 1e-7, client.name());

        let (vi_split, vi_merge) = variation_of_info(client, &seg_a, &seg_a, None).unwrap();
        assert_close(&[vi_split, vi_merge], &[0.0, 0.0], 1e-7, client.name());

        // Over-segmentation: splitting label 1 into (1, 3) yields false_splits > 0 and false_merges == 0
        let seg_over = DeviceArray::from_host(client, Shape::new([2, 4]), vec![0u32, 1, 3, 2, 0, 1, 2, 2]).unwrap();
        let (over_split, over_merge) = variation_of_info(client, &seg_a, &seg_over, Some(&[0])).unwrap();
        assert!(over_split > 0.4, "over_split = {over_split}");
        assert_close(&[over_merge], &[0.0], 1e-7, client.name());

        // Normalized Mutual Information: identical arrays have NMI = 2.0
        let x = sample_f64(64);
        let img = DeviceArray::from_host(client, Shape::new([8, 8]), cast_all::<f32>(&x)).unwrap();
        let nmi_same = mutual_info(client, &img, &img, 16).unwrap();
        assert_close(&[nmi_same], &[2.0], 1e-6, client.name());
    }
    runtime_test!(test_segmentation_metrics, segmentation_metrics_match_analytic_cases);
}

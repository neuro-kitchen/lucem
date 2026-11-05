//! Global histogram- and reduction-based thresholding: Otsu, Multi-Otsu, ISODATA, Li (minimum
//! cross-entropy), Yen, Triangle, Minimum (bimodal valley), and Mean.
//!
//! Each threshold provides both a `*_device` entry returning a 1-element [`DeviceArray<F>`] (so
//! downstream device kernels never synchronize with the host) and a host scalar convenience entry.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, cast_all, to_f64, ImageFloat};
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;

use super::kernels::otsu_from_hist_kernel;
use crate::core::{histogram, reduce};
use crate::error::{OpsError, OpsResult};

/// Computes the Otsu threshold on the device as a 1-element [`DeviceArray<F>`].
pub fn otsu_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>> {
    let (counts, centers) = histogram::histogram(client, input, nbins, None)?;
    let n = centers.len();
    let min_val = centers.first().copied().unwrap_or(0.0);
    let step = if n > 1 { centers[1] - centers[0] } else { 1.0 };
    let min_edge = min_val - 0.5 * step;

    let out = DeviceArray::<F>::empty(client, Shape::vector(1))?;
    let geom = LaunchGeometry::elementwise(client, 1);
    unsafe {
        otsu_from_hist_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            counts.binding(),
            out.binding(),
            cast::<F>(min_edge),
            cast::<F>(step),
            n as u32,
        );
    }
    Ok(out)
}

/// Computes the global Otsu threshold maximizing inter-class variance.
pub fn otsu<F: ImageFloat>(client: &Client, input: &DeviceArray<F>, nbins: usize) -> OpsResult<f64> {
    let dev = otsu_device(client, input, nbins)?;
    Ok(to_f64(dev.to_host(client)[0]))
}

/// Multi-class Otsu thresholding for `classes` classes (`classes >= 2`), returning `classes - 1`
/// sorted thresholds maximizing between-class variance via dynamic programming over the histogram.
pub fn multi_otsu<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    classes: usize,
    nbins: usize,
) -> OpsResult<Vec<f64>> {
    if classes < 2 {
        return Err(OpsError::InvalidParameter {
            name: "classes",
            reason: format!("classes must be at least 2, got {classes}"),
        });
    }
    if classes == 2 {
        return Ok(vec![otsu(client, input, nbins)?]);
    }
    let (counts_dev, centers) = histogram::histogram(client, input, nbins, None)?;
    let counts = counts_dev.to_host(client);
    let total: f64 = counts.iter().map(|&c| c as f64).sum();
    if total <= 0.0 {
        return Ok(vec![centers[0]; classes - 1]);
    }

    let n = nbins;
    let mut p = vec![0.0; n + 1];
    let mut s = vec![0.0; n + 1];
    for i in 0..n {
        let prob = (counts[i] as f64) / total;
        p[i + 1] = p[i] + prob;
        s[i + 1] = s[i] + prob * centers[i];
    }

    let h = |lo: usize, hi: usize| -> f64 {
        let w = p[hi + 1] - p[lo];
        if w <= 1e-15 {
            0.0
        } else {
            let mu = s[hi + 1] - s[lo];
            (mu * mu) / w
        }
    };

    // DP: dp[c][j] = max variance partitioning bins 0..=j into c+1 classes
    let mut dp = vec![vec![f64::NEG_INFINITY; n]; classes];
    let mut split = vec![vec![0usize; n]; classes];
    for j in 0..n {
        dp[0][j] = h(0, j);
    }
    for c in 1..classes {
        for j in c..n {
            let mut best = f64::NEG_INFINITY;
            let mut best_k = c - 1;
            for k in (c - 1)..j {
                let cand = dp[c - 1][k] + h(k + 1, j);
                if cand > best {
                    best = cand;
                    best_k = k;
                }
            }
            dp[c][j] = best;
            split[c][j] = best_k;
        }
    }

    let mut thresholds = vec![0.0; classes - 1];
    let mut cur = n - 1;
    for c in (1..classes).rev() {
        let k = split[c][cur];
        thresholds[c - 1] = centers[k];
        cur = k;
    }
    Ok(thresholds)
}

/// Global mean threshold as a 1-element [`DeviceArray<F>`].
pub fn mean_threshold_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>> {
    let (mean, _) = reduce::mean_var_device(client, input, 0)?;
    Ok(mean)
}

/// Global mean threshold.
pub fn mean_threshold<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<f64> {
    let dev = mean_threshold_device(client, input)?;
    Ok(to_f64(dev.to_host(client)[0]))
}

/// ISODATA iterative threshold ($\theta = \frac{1}{2}(\mu_0(\theta) + \mu_1(\theta))$).
pub fn isodata<F: ImageFloat>(client: &Client, input: &DeviceArray<F>, nbins: usize) -> OpsResult<f64> {
    let (counts_dev, centers) = histogram::histogram(client, input, nbins, None)?;
    let counts: Vec<f64> = counts_dev.to_host(client).into_iter().map(f64::from).collect();
    let total_w: f64 = counts.iter().sum();
    if total_w <= 0.0 {
        return Ok(centers[0]);
    }
    let total_mu: f64 = counts.iter().zip(centers.iter()).map(|(&c, &x)| c * x).sum();
    let mut idx = nbins / 2;
    for _ in 0..nbins {
        let w0: f64 = counts[..=idx].iter().sum();
        let w1 = total_w - w0;
        if w0 <= 0.0 {
            idx = (idx + 1).min(nbins - 1);
            continue;
        }
        if w1 <= 0.0 {
            idx = idx.saturating_sub(1);
            continue;
        }
        let mu0 = counts[..=idx].iter().zip(centers[..=idx].iter()).map(|(&c, &x)| c * x).sum::<f64>() / w0;
        let mu1 = (total_mu - w0 * mu0) / w1;
        let mid = 0.5 * (mu0 + mu1);
        let new_idx = centers.partition_point(|&c| c < mid).min(nbins - 1);
        if new_idx == idx {
            return Ok(centers[idx]);
        }
        idx = new_idx;
    }
    Ok(centers[idx])
}

/// ISODATA threshold on the device.
pub fn isodata_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>> {
    let t = isodata(client, input, nbins)?;
    Ok(DeviceArray::from_host(client, Shape::vector(1), cast_all::<F>(&[t]))?)
}

/// Li's iterative minimum cross-entropy threshold.
pub fn li<F: ImageFloat>(client: &Client, input: &DeviceArray<F>, nbins: usize) -> OpsResult<f64> {
    let (counts_dev, centers) = histogram::histogram(client, input, nbins, None)?;
    let counts: Vec<f64> = counts_dev.to_host(client).into_iter().map(f64::from).collect();
    let min_c = centers[0];
    let shift = if min_c <= 0.0 { 1.0 - min_c } else { 0.0 };
    let shifted: Vec<f64> = centers.iter().map(|&c| c + shift).collect();

    let total_w: f64 = counts.iter().sum();
    if total_w <= 0.0 {
        return Ok(centers[0]);
    }
    let total_sum: f64 = counts.iter().zip(shifted.iter()).map(|(&c, &x)| c * x).sum();
    let mut t = total_sum / total_w;
    let tol = if nbins > 1 { 0.5 * (centers[1] - centers[0]).abs() } else { 1e-6 };

    for _ in 0..256 {
        let mut w_bg = 0.0;
        let mut s_bg = 0.0;
        for (&c, &x) in counts.iter().zip(shifted.iter()) {
            if x <= t {
                w_bg += c;
                s_bg += c * x;
            }
        }
        let w_fg = total_w - w_bg;
        if w_bg <= 0.0 || w_fg <= 0.0 {
            break;
        }
        let mu_bg = (s_bg / w_bg).max(1e-12);
        let mu_fg = ((total_sum - s_bg) / w_fg).max(1e-12);
        if (mu_fg - mu_bg).abs() <= 1e-12 {
            break;
        }
        let t_next = (mu_fg - mu_bg) / (mu_fg.ln() - mu_bg.ln());
        if (t_next - t).abs() <= tol {
            t = t_next;
            break;
        }
        t = t_next;
    }
    Ok(t - shift)
}

/// Li threshold as a 1-element [`DeviceArray<F>`].
pub fn li_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>> {
    let t = li(client, input, nbins)?;
    Ok(DeviceArray::from_host(client, Shape::vector(1), cast_all::<F>(&[t]))?)
}

/// Yen's maximum correlation criterion threshold.
pub fn yen<F: ImageFloat>(client: &Client, input: &DeviceArray<F>, nbins: usize) -> OpsResult<f64> {
    let (counts_dev, centers) = histogram::histogram(client, input, nbins, None)?;
    let counts: Vec<f64> = counts_dev.to_host(client).into_iter().map(f64::from).collect();
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return Ok(centers[0]);
    }
    let pmf: Vec<f64> = counts.into_iter().map(|c| c / total).collect();
    let mut p1 = vec![0.0; nbins];
    let mut p1_sq = vec![0.0; nbins];
    let mut p2_sq = vec![0.0; nbins];
    let mut acc_p = 0.0;
    let mut acc_sq = 0.0;
    for i in 0..nbins {
        acc_p += pmf[i];
        acc_sq += pmf[i] * pmf[i];
        p1[i] = acc_p;
        p1_sq[i] = acc_sq;
    }
    let mut tail_sq = 0.0;
    for i in (0..nbins).rev() {
        p2_sq[i] = tail_sq;
        tail_sq += pmf[i] * pmf[i];
    }

    let mut best_crit = f64::NEG_INFINITY;
    let mut best_idx = 0usize;
    for i in 0..nbins.saturating_sub(1) {
        let prod_sq = p1_sq[i] * p2_sq[i];
        let prod_p = p1[i] * (1.0 - p1[i]);
        if prod_sq > 0.0 && prod_p > 0.0 {
            let crit = -prod_sq.ln() + 2.0 * prod_p.ln();
            if crit > best_crit {
                best_crit = crit;
                best_idx = i;
            }
        }
    }
    Ok(centers[best_idx])
}

/// Yen threshold as a 1-element [`DeviceArray<F>`].
pub fn yen_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>> {
    let t = yen(client, input, nbins)?;
    Ok(DeviceArray::from_host(client, Shape::vector(1), cast_all::<F>(&[t]))?)
}

/// Geometric triangle threshold: constructs a chord from the histogram peak to the farthest
/// non-zero tail and picks the bin at maximum perpendicular distance.
pub fn triangle<F: ImageFloat>(client: &Client, input: &DeviceArray<F>, nbins: usize) -> OpsResult<f64> {
    let (counts_dev, centers) = histogram::histogram(client, input, nbins, None)?;
    let counts: Vec<f64> = counts_dev.to_host(client).into_iter().map(f64::from).collect();
    if counts.len() < 2 {
        return Ok(centers.first().copied().unwrap_or(0.0));
    }
    let first_nz = counts.iter().position(|&c| c > 0.0).unwrap_or(0);
    let last_nz = counts.iter().rposition(|&c| c > 0.0).unwrap_or(counts.len() - 1);
    let mut peak_idx = first_nz;
    for i in first_nz..=last_nz {
        if counts[i] > counts[peak_idx] {
            peak_idx = i;
        }
    }

    let (end_idx, range_iter): (usize, Box<dyn Iterator<Item = usize>>) =
        if (last_nz - peak_idx) >= (peak_idx - first_nz) {
            (last_nz, Box::new(peak_idx..=last_nz))
        } else {
            (first_nz, Box::new(first_nz..=peak_idx))
        };

    let x0 = peak_idx as f64;
    let y0 = counts[peak_idx];
    let x1 = end_idx as f64;
    let y1 = counts[end_idx];
    let dx = x1 - x0;
    let dy = y1 - y0;
    let norm = dx.hypot(dy).max(1e-12);

    let mut best_dist = -1.0;
    let mut best_bin = peak_idx;
    for b in range_iter {
        let bx = b as f64;
        let by = counts[b];
        let dist = ((dy * bx - dx * by + x1 * y0 - y1 * x0).abs()) / norm;
        if dist > best_dist {
            best_dist = dist;
            best_bin = b;
        }
    }
    Ok(centers[best_bin])
}

/// Triangle threshold as a 1-element [`DeviceArray<F>`].
pub fn triangle_device<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
) -> OpsResult<DeviceArray<F>> {
    let t = triangle(client, input, nbins)?;
    Ok(DeviceArray::from_host(client, Shape::vector(1), cast_all::<F>(&[t]))?)
}

/// Bimodal minimum threshold: iteratively smooths the histogram with a 3-point moving average
/// until at most two local maxima remain, then returns the valley minimum between them.
pub fn minimum_threshold<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
    max_num_iter: usize,
) -> OpsResult<f64> {
    let (counts_dev, centers) = histogram::histogram(client, input, nbins, None)?;
    let mut h: Vec<f64> = counts_dev.to_host(client).into_iter().map(f64::from).collect();
    if h.len() < 3 {
        return Ok(centers.first().copied().unwrap_or(0.0));
    }

    let find_peaks = |arr: &[f64]| -> Vec<usize> {
        (1..arr.len() - 1)
            .filter(|&i| arr[i] > arr[i - 1] && arr[i] >= arr[i + 1])
            .collect()
    };

    for _ in 0..max_num_iter.max(1) {
        let peaks = find_peaks(&h);
        if peaks.len() <= 2 {
            if peaks.len() == 2 {
                let (p0, p1) = (peaks[0], peaks[1]);
                let mut valley = p0;
                for i in p0..=p1 {
                    if h[i] < h[valley] {
                        valley = i;
                    }
                }
                return Ok(centers[valley]);
            }
            break;
        }
        let mut next = h.clone();
        next[0] = (h[0] + h[0] + h[1]) / 3.0;
        for i in 1..h.len() - 1 {
            next[i] = (h[i - 1] + h[i] + h[i + 1]) / 3.0;
        }
        let last = h.len() - 1;
        next[last] = (h[last - 1] + h[last] + h[last]) / 3.0;
        h = next;
    }
    otsu(client, input, nbins)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn global_thresholds_separate_bimodal_signal(client: &Client) {
        let mut data = Vec::with_capacity(200);
        for i in 0..100 {
            data.push(10.0f32 + (i % 5) as f32 * 0.2);
        }
        for i in 0..100 {
            data.push(90.0f32 + (i % 5) as f32 * 0.2);
        }
        let arr = DeviceArray::from_host(client, Shape::vector(data.len()), data).unwrap();
        for t in [
            otsu(client, &arr, 64).unwrap(),
            isodata(client, &arr, 64).unwrap(),
            li(client, &arr, 64).unwrap(),
            yen(client, &arr, 64).unwrap(),
            mean_threshold(client, &arr).unwrap(),
            minimum_threshold(client, &arr, 64, 1000).unwrap(),
        ] {
            assert!(t > 15.0 && t < 85.0, "{} threshold {t} failed to separate bimodal peaks", client.name());
        }
    }
    runtime_test!(test_global_thresholds, global_thresholds_separate_bimodal_signal);
}

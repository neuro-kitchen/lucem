//! Superpixel and graph-based image segmentation: SLIC (`slic`), Felzenszwalb-Huttenlocher minimum
//! spanning forest (`felzenszwalb`), Quickshift medoid mode-seeking (`quickshift`), and harmonic
//! random-walker segmentation (`random_walker`).

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, to_f64, ImageFloat};
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;

use super::kernels::slic_assign_2d_kernel;
use crate::core::check;
use crate::core::components::relabel_sequential;
use crate::error::{OpsError, OpsResult};

/// Parameters for Simple Linear Iterative Clustering ([`slic`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SlicOptions {
    pub n_segments: usize,
    pub compactness: f64,
    pub max_iter: usize,
}

impl Default for SlicOptions {
    fn default() -> Self {
        Self {
            n_segments: 64,
            compactness: 10.0,
            max_iter: 10,
        }
    }
}

/// Simple Linear Iterative Clustering (SLIC) superpixel over-segmentation on a 2-D image `[h, w]`.
/// Runs each pixel-to-cluster assignment iteration on the device via [`slic_assign_2d_kernel`].
pub fn slic<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    options: &SlicOptions,
) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("slic", image.shape(), &[2], "2")?;
    if options.n_segments == 0 {
        return Err(OpsError::InvalidParameter {
            name: "n_segments",
            reason: "n_segments must be >= 1".to_string(),
        });
    }
    if options.compactness <= 0.0 || !options.compactness.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "compactness",
            reason: "compactness must be finite and > 0".to_string(),
        });
    }
    let (h, w) = (image.dims()[0], image.dims()[1]);
    if h == 0 || w == 0 {
        return Ok(DeviceArray::<u32>::zeros(client, image.shape().clone())?);
    }

    let n_pixels = h * w;
    let step = ((n_pixels as f64) / (options.n_segments as f64)).sqrt().max(1.0);
    let host_img: Vec<f64> = image.to_host(client).into_iter().map(to_f64).collect();

    // Seed the cluster centres on a grid of about `n_segments` cells shaped like the image
    // (n_y × n_x, each seed at its cell's centre): a fixed step can leave fewer seeds than asked,
    // down to one on a small image
    let n_y = ((options.n_segments as f64 * h as f64 / w as f64).sqrt().round() as usize).clamp(1, h);
    let n_x = ((options.n_segments as f64 / n_y as f64).round() as usize).clamp(1, w);
    let mut centers_f64: Vec<[f64; 3]> = Vec::with_capacity(n_y * n_x);
    for i in 0..n_y {
        let r = (((i as f64 + 0.5) * h as f64 / n_y as f64) as usize).min(h - 1);
        for j in 0..n_x {
            let c = (((j as f64 + 0.5) * w as f64 / n_x as f64) as usize).min(w - 1);
            centers_f64.push([r as f64, c as f64, host_img[r * w + c]]);
        }
    }
    if centers_f64.is_empty() {
        centers_f64.push([0.0, 0.0, host_img[0]]);
    }

    let num_clusters = centers_f64.len();
    let labels = DeviceArray::<u32>::empty(client, image.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, n_pixels);
    let spatial_weight = options.compactness / step;
    let spatial_weight_sq = cast::<F>(spatial_weight * spatial_weight);
    let max_search_dist_sq = cast::<F>(4.0 * step * step);

    let iters = options.max_iter.max(1);
    for _ in 0..iters {
        let mut flat_centers = Vec::with_capacity(num_clusters * 3);
        for &[cy, cx, cv] in &centers_f64 {
            flat_centers.push(cast::<F>(cy));
            flat_centers.push(cast::<F>(cx));
            flat_centers.push(cast::<F>(cv));
        }
        let centers_dev =
            DeviceArray::from_host(client, Shape::new([num_clusters, 3]), flat_centers)?;

        unsafe {
            slic_assign_2d_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                image.binding(),
                centers_dev.binding(),
                labels.binding(),
                w as u32,
                num_clusters as u32,
                spatial_weight_sq,
                max_search_dist_sq,
                n_pixels as u32,
            );
        }

        let host_lbl = labels.to_host(client);
        let mut acc = vec![[0.0f64; 4]; num_clusters];
        for (idx, &lbl) in host_lbl.iter().enumerate() {
            let k = (lbl as usize).saturating_sub(1);
            if k < num_clusters {
                acc[k][0] += (idx / w) as f64;
                acc[k][1] += (idx % w) as f64;
                acc[k][2] += host_img[idx];
                acc[k][3] += 1.0;
            }
        }
        for (k, c) in centers_f64.iter_mut().enumerate() {
            if acc[k][3] > 0.0 {
                c[0] = acc[k][0] / acc[k][3];
                c[1] = acc[k][1] / acc[k][3];
                c[2] = acc[k][2] / acc[k][3];
            }
        }
    }

    let (relabeled, _) = relabel_sequential(client, &labels, 1)?;
    Ok(relabeled)
}

/// Graph-based minimum spanning forest segmentation (Felzenszwalb & Huttenlocher 2004) on a 2-D
/// image `[h, w]`.
pub fn felzenszwalb<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    scale: f64,
    min_size: usize,
) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("felzenszwalb", image.shape(), &[2], "2")?;
    if scale <= 0.0 || !scale.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "scale",
            reason: "scale must be finite and > 0".to_string(),
        });
    }
    let (h, w) = (image.dims()[0], image.dims()[1]);
    let n = h * w;
    if n == 0 {
        return Ok(DeviceArray::<u32>::zeros(client, image.shape().clone())?);
    }
    let vals: Vec<f64> = image.to_host(client).into_iter().map(to_f64).collect();

    let mut edges: Vec<(f64, u32, u32)> = Vec::with_capacity(n * 2);
    for r in 0..h {
        for c in 0..w {
            let u = r * w + c;
            if c + 1 < w {
                let v = r * w + c + 1;
                edges.push(((vals[u] - vals[v]).abs(), u as u32, v as u32));
            }
            if r + 1 < h {
                let v = (r + 1) * w + c;
                edges.push(((vals[u] - vals[v]).abs(), u as u32, v as u32));
            }
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));

    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut size: Vec<usize> = vec![1; n];
    let mut threshold: Vec<f64> = vec![scale; n];

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = p;
        }
        x
    }

    for &(w_e, u, v) in &edges {
        let ru = find(&mut parent, u) as usize;
        let rv = find(&mut parent, v) as usize;
        if ru != rv && w_e <= threshold[ru].min(threshold[rv]) {
            let (root, child) = if ru < rv { (ru, rv) } else { (rv, ru) };
            parent[child] = root as u32;
            size[root] += size[child];
            threshold[root] = w_e + scale / (size[root] as f64);
        }
    }

    // Merge components smaller than min_size
    if min_size > 1 {
        for &(_w_e, u, v) in &edges {
            let ru = find(&mut parent, u) as usize;
            let rv = find(&mut parent, v) as usize;
            if ru != rv && (size[ru] < min_size || size[rv] < min_size) {
                let (root, child) = if ru < rv { (ru, rv) } else { (rv, ru) };
                parent[child] = root as u32;
                size[root] += size[child];
            }
        }
    }

    let raw_labels: Vec<u32> = (0..n as u32).map(|i| find(&mut parent, i) + 1).collect();
    let dev = DeviceArray::from_host(client, image.shape().clone(), raw_labels)?;
    let (relabeled, _) = relabel_sequential(client, &dev, 1)?;
    Ok(relabeled)
}

/// Quickshift mode-seeking medoid shift segmentation (Vedaldi & Soatto 2008) on a 2-D image `[h, w]`.
pub fn quickshift<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    kernel_size: f64,
    max_dist: f64,
    ratio: f64,
) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("quickshift", image.shape(), &[2], "2")?;
    if kernel_size <= 0.0 || !kernel_size.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "kernel_size",
            reason: "kernel_size must be finite and > 0".to_string(),
        });
    }
    if max_dist < 0.0 || !max_dist.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "max_dist",
            reason: "max_dist must be finite and >= 0".to_string(),
        });
    }
    let (h, w) = (image.dims()[0], image.dims()[1]);
    let n = h * w;
    if n == 0 {
        return Ok(DeviceArray::<u32>::zeros(client, image.shape().clone())?);
    }

    let vals: Vec<f64> = image.to_host(client).into_iter().map(to_f64).collect();
    let inv_2sigma2 = 0.5 / (kernel_size * kernel_size);
    let rad = (3.0 * kernel_size).ceil() as isize;
    let max_dist_sq = max_dist * max_dist;

    // Parzen density estimate at each pixel
    let mut density = vec![0.0f64; n];
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            let v0 = vals[idx] * ratio;
            let mut sum = 0.0;
            let r0 = (r as isize - rad).max(0) as usize;
            let r1 = ((r as isize + rad).min(h as isize - 1)) as usize;
            let c0 = (c as isize - rad).max(0) as usize;
            let c1 = ((c as isize + rad).min(w as isize - 1)) as usize;
            for nr in r0..=r1 {
                for nc in c0..=c1 {
                    let dy = (nr as f64) - (r as f64);
                    let dx = (nc as f64) - (c as f64);
                    let dv = vals[nr * w + nc] * ratio - v0;
                    let d2 = dy * dy + dx * dx + dv * dv;
                    sum += (-d2 * inv_2sigma2).exp();
                }
            }
            density[idx] = sum;
        }
    }

    // Link each pixel to the nearest neighbor with strictly higher density within max_dist
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            let p0 = density[idx];
            let v0 = vals[idx] * ratio;
            let mut best_d2 = f64::INFINITY;
            let mut best_parent = idx as u32;

            let r0 = (r as isize - rad).max(0) as usize;
            let r1 = ((r as isize + rad).min(h as isize - 1)) as usize;
            let c0 = (c as isize - rad).max(0) as usize;
            let c1 = ((c as isize + rad).min(w as isize - 1)) as usize;
            for nr in r0..=r1 {
                for nc in c0..=c1 {
                    let n_idx = nr * w + nc;
                    if density[n_idx] > p0 {
                        let dy = (nr as f64) - (r as f64);
                        let dx = (nc as f64) - (c as f64);
                        let dv = vals[n_idx] * ratio - v0;
                        let d2 = dy * dy + dx * dx + dv * dv;
                        if d2 <= max_dist_sq && d2 < best_d2 {
                            best_d2 = d2;
                            best_parent = n_idx as u32;
                        }
                    }
                }
            }
            parent[idx] = best_parent;
        }
    }

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = p;
        }
        x
    }

    let raw_labels: Vec<u32> = (0..n as u32).map(|i| find(&mut parent, i) + 1).collect();
    let dev = DeviceArray::from_host(client, image.shape().clone(), raw_labels)?;
    let (relabeled, _) = relabel_sequential(client, &dev, 1)?;
    Ok(relabeled)
}

/// Anisotropic harmonic random-walker segmentation (Grady 2006) on a 2-D image `[h, w]` from
/// seed labels in `markers` (`> 0` = seed label, `0` = unlabeled), using edge weights
/// $w_{ij} = \exp(-\beta (I_i - I_j)^2)$.
pub fn random_walker<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    markers: &DeviceArray<u32>,
    beta: f64,
    tol: f64,
    max_iter: usize,
) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("random_walker", image.shape(), &[2], "2")?;
    check::same_shape("image", image, "markers", markers)?;
    if beta < 0.0 || !beta.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "beta",
            reason: "beta must be finite and >= 0".to_string(),
        });
    }
    let (h, w) = (image.dims()[0], image.dims()[1]);
    let n = h * w;
    if n == 0 {
        return Ok(DeviceArray::<u32>::zeros(client, image.shape().clone())?);
    }

    let vals: Vec<f64> = image.to_host(client).into_iter().map(to_f64).collect();
    let seed = markers.to_host(client);

    let mut unique_labels: Vec<u32> = seed.iter().copied().filter(|&l| l > 0).collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    if unique_labels.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, image.shape().clone())?);
    }
    if unique_labels.len() == 1 {
        let single = vec![unique_labels[0]; n];
        return Ok(DeviceArray::from_host(client, image.shape().clone(), single)?);
    }

    // Precompute 4-neighbor weights [up, down, left, right]
    let mut weights = vec![[0.0f64; 4]; n];
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            let v0 = vals[idx];
            if r > 0 {
                let d = v0 - vals[(r - 1) * w + c];
                weights[idx][0] = (-beta * d * d).exp();
            }
            if r + 1 < h {
                let d = v0 - vals[(r + 1) * w + c];
                weights[idx][1] = (-beta * d * d).exp();
            }
            if c > 0 {
                let d = v0 - vals[r * w + (c - 1)];
                weights[idx][2] = (-beta * d * d).exp();
            }
            if c + 1 < w {
                let d = v0 - vals[r * w + (c + 1)];
                weights[idx][3] = (-beta * d * d).exp();
            }
        }
    }

    let num_k = unique_labels.len();
    let mut probs = vec![vec![1.0 / (num_k as f64); n]; num_k];
    for (k, &lbl) in unique_labels.iter().enumerate() {
        for idx in 0..n {
            if seed[idx] > 0 {
                probs[k][idx] = if seed[idx] == lbl { 1.0 } else { 0.0 };
            }
        }
    }

    // Solve Dirichlet boundary problem for the first (num_k - 1) labels by Gauss-Seidel SOR
    let iters = max_iter.max(1);
    for p in probs.iter_mut().take(num_k - 1) {
        for _ in 0..iters {
            let mut max_diff = 0.0f64;
            for r in 0..h {
                for c in 0..w {
                    let idx = r * w + c;
                    if seed[idx] > 0 {
                        continue;
                    }
                    let w_arr = weights[idx];
                    let w_sum = w_arr[0] + w_arr[1] + w_arr[2] + w_arr[3];
                    if w_sum <= 1e-15 {
                        continue;
                    }
                    let mut num = 0.0;
                    if r > 0 {
                        num += w_arr[0] * p[(r - 1) * w + c];
                    }
                    if r + 1 < h {
                        num += w_arr[1] * p[(r + 1) * w + c];
                    }
                    if c > 0 {
                        num += w_arr[2] * p[r * w + (c - 1)];
                    }
                    if c + 1 < w {
                        num += w_arr[3] * p[r * w + (c + 1)];
                    }
                    let next_val = num / w_sum;
                    let diff = (next_val - p[idx]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    p[idx] = next_val;
                }
            }
            if max_diff <= tol {
                break;
            }
        }
    }

    let mut out = vec![0u32; n];
    for idx in 0..n {
        if seed[idx] > 0 {
            out[idx] = seed[idx];
            continue;
        }
        let mut rem = 1.0;
        let mut best_k = 0;
        let mut best_p = -1.0;
        for (k, prob_k) in probs.iter().enumerate().take(num_k - 1) {
            let pk = prob_k[idx];
            rem -= pk;
            if pk > best_p {
                best_p = pk;
                best_k = k;
            }
        }
        if rem > best_p {
            best_k = num_k - 1;
        }
        out[idx] = unique_labels[best_k];
    }

    Ok(DeviceArray::from_host(client, image.shape().clone(), out)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn superpixel_and_random_walker_segment_two_halves(client: &Client) {
        // 6x6 image with left half = 0.0, right half = 10.0
        let mut img = vec![0.0f32; 36];
        for r in 0..6 {
            for c in 3..6 {
                img[r * 6 + c] = 10.0;
            }
        }
        let arr = DeviceArray::from_host(client, Shape::new([6, 6]), img).unwrap();

        let s_lbl = slic(
            client,
            &arr,
            &SlicOptions {
                n_segments: 2,
                compactness: 1.0,
                max_iter: 5,
            },
        )
        .unwrap()
        .to_host(client);
        assert_ne!(s_lbl[0], s_lbl[5], "{}", client.name());

        let f_lbl = felzenszwalb(client, &arr, 1.0, 2).unwrap().to_host(client);
        assert_eq!(f_lbl[0], f_lbl[1]);
        assert_ne!(f_lbl[0], f_lbl[5]);

        let q_lbl = quickshift(client, &arr, 1.5, 4.0, 1.0).unwrap().to_host(client);
        assert_ne!(q_lbl[0], q_lbl[5]);

        let mut seeds = vec![0u32; 36];
        seeds[0] = 1;
        seeds[5] = 2;
        let seeds_dev = DeviceArray::from_host(client, Shape::new([6, 6]), seeds).unwrap();
        let rw = random_walker(client, &arr, &seeds_dev, 1.0, 1e-4, 200)
            .unwrap()
            .to_host(client);
        assert_eq!(rw[1], 1);
        assert_eq!(rw[4], 2);
    }
    runtime_test!(test_superpixel_and_random_walker, superpixel_and_random_walker_segment_two_halves);
}

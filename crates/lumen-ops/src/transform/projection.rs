//! Geometric projection transforms: straight-line Hough transform (`hough_lines`, `hough_line_peaks`),
//! circular and elliptical Hough transforms (`hough_circles`, `hough_ellipses`), and forward/inverse
//! Radon transforms (`radon`, `iradon`).

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, cast_all, to_f64, ImageFloat};
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;

use super::kernels::{iradon_backproject_kernel, radon_2d_kernel};
use crate::core::check;
use crate::core::compact::nonzero_coordinates;
use crate::error::{OpsError, OpsResult};

/// Output of the 2-D straight-line Hough transform ([`hough_lines`]).
#[derive(Debug, Clone)]
pub struct HoughLines {
    /// Accumulator array of shape `[distances.len(), angles.len()]`.
    pub accumulator: DeviceArray<u32>,
    /// Tested angles $\theta$ in radians.
    pub angles: Vec<f64>,
    /// Signed radial distance bins $\rho$ in pixels (`-diag ..= +diag`).
    pub distances: Vec<f64>,
}

/// Computes the straight-line Hough accumulator for non-zero edge pixels in a 2-D image over `angles`
/// (in radians; defaults to 180 uniformly spaced angles in $[-\pi/2, \pi/2)$ when `None`).
pub fn hough_lines<E: Numeric + CubeElement>(
    client: &Client,
    edges: &DeviceArray<E>,
    angles: Option<&[f64]>,
) -> OpsResult<HoughLines> {
    check::ndim_in("hough_lines", edges.shape(), &[2], "2")?;
    let thetas: Vec<f64> = match angles {
        Some(a) => {
            if a.is_empty() {
                return Err(OpsError::InvalidParameter {
                    name: "angles",
                    reason: "angles slice must not be empty".to_string(),
                });
            }
            a.to_vec()
        }
        None => (0..180)
            .map(|i| -std::f64::consts::FRAC_PI_2 + (i as f64) * std::f64::consts::PI / 180.0)
            .collect(),
    };

    let (h, w) = (edges.dims()[0], edges.dims()[1]);
    let diag = ((h as f64).hypot(w as f64)).ceil() as isize;
    let n_bins = (2 * diag + 1) as usize;
    let n_angles = thetas.len();
    let distances: Vec<f64> = (-diag..=diag).map(|d| d as f64).collect();

    let cos_t: Vec<f64> = thetas.iter().map(|&t| t.cos()).collect();
    let sin_t: Vec<f64> = thetas.iter().map(|&t| t.sin()).collect();
    let pts = nonzero_coordinates(client, edges)?;
    let mut acc = vec![0u32; n_bins * n_angles];

    for pt in &pts {
        let y = pt[0] as f64;
        let x = pt[1] as f64;
        for a in 0..n_angles {
            let rho = (x * cos_t[a] + y * sin_t[a]).round() as isize;
            let bin = (rho + diag) as usize;
            if bin < n_bins {
                acc[bin * n_angles + a] += 1;
            }
        }
    }

    Ok(HoughLines {
        accumulator: DeviceArray::from_host(client, Shape::new([n_bins, n_angles]), acc)?,
        angles: thetas,
        distances,
    })
}

/// A detected peak `(votes, angle_rad, distance_px)` in a Hough line accumulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoughLinePeak {
    pub votes: u32,
    pub angle: f64,
    pub distance: f64,
}

/// Extracts prominent peaks from a [`HoughLines`] accumulator with minimum `(min_xdistance, min_ydistance)`
/// bin separation between peaks.
pub fn hough_line_peaks(
    client: &Client,
    hough: &HoughLines,
    min_distance: usize,
    min_angle: usize,
    threshold: Option<u32>,
    num_peaks: Option<usize>,
) -> OpsResult<Vec<HoughLinePeak>> {
    let acc = hough.accumulator.to_host(client);
    let (n_bins, n_angles) = (hough.distances.len(), hough.angles.len());
    let max_val = acc.iter().copied().max().unwrap_or(0);
    if max_val == 0 {
        return Ok(Vec::new());
    }
    let thresh = threshold.unwrap_or((max_val / 2).max(1));

    let mut candidates: Vec<(u32, usize, usize)> = Vec::new();
    for r in 0..n_bins {
        for a in 0..n_angles {
            let v = acc[r * n_angles + a];
            if v >= thresh {
                candidates.push((v, r, a));
            }
        }
    }
    candidates.sort_by(|x, y| y.0.cmp(&x.0).then_with(|| x.1.cmp(&y.1)).then_with(|| x.2.cmp(&y.2)));

    let limit = num_peaks.unwrap_or(usize::MAX);
    let mut peaks: Vec<(u32, usize, usize)> = Vec::new();
    'cand: for (v, r, a) in candidates {
        if peaks.len() >= limit {
            break;
        }
        for &(_, pr, pa) in &peaks {
            if r.abs_diff(pr) <= min_distance && a.abs_diff(pa) <= min_angle {
                continue 'cand;
            }
        }
        peaks.push((v, r, a));
    }

    Ok(peaks
        .into_iter()
        .map(|(votes, r, a)| HoughLinePeak {
            votes,
            angle: hough.angles[a],
            distance: hough.distances[r],
        })
        .collect())
}

/// Computes a 3-D circular Hough accumulator `[radii.len(), height, width]` for non-zero edge pixels
/// in `edges` across candidate `radii`.
pub fn hough_circles<E: Numeric + CubeElement>(
    client: &Client,
    edges: &DeviceArray<E>,
    radii: &[f64],
    normalize: bool,
) -> OpsResult<DeviceArray<f32>> {
    check::ndim_in("hough_circles", edges.shape(), &[2], "2")?;
    if radii.is_empty() || radii.iter().any(|&r| !r.is_finite() || r <= 0.0) {
        return Err(OpsError::InvalidParameter {
            name: "radii",
            reason: "radii must be non-empty with positive finite values".to_string(),
        });
    }
    let (h, w) = (edges.dims()[0], edges.dims()[1]);
    let nr = radii.len();
    let pts = nonzero_coordinates(client, edges)?;
    let mut acc = vec![0.0f32; nr * h * w];

    for (ri, &r) in radii.iter().enumerate() {
        let n_samples = (2.0 * std::f64::consts::PI * r).ceil().max(8.0) as usize;
        let weight = if normalize { 1.0 / (n_samples as f32) } else { 1.0 };
        let offsets: Vec<(isize, isize)> = (0..n_samples)
            .map(|k| {
                let theta = 2.0 * std::f64::consts::PI * (k as f64) / (n_samples as f64);
                ((r * theta.sin()).round() as isize, (r * theta.cos()).round() as isize)
            })
            .collect();

        let slice_base = ri * h * w;
        for pt in &pts {
            let ey = pt[0] as isize;
            let ex = pt[1] as isize;
            for &(dy, dx) in &offsets {
                let cy = ey - dy;
                let cx = ex - dx;
                if cy >= 0 && cy < (h as isize) && cx >= 0 && cx < (w as isize) {
                    acc[slice_base + (cy as usize) * w + (cx as usize)] += weight;
                }
            }
        }
    }

    Ok(DeviceArray::from_host(client, Shape::new([nr, h, w]), acc)?)
}

/// Detected ellipse parameters from [`hough_ellipses`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoughEllipse {
    pub votes: usize,
    pub yc: f64,
    pub xc: f64,
    pub a: f64,
    pub b: f64,
    pub orientation: f64,
}

/// Detects ellipses in a 2-D binary edge image using Xie & Ji's randomized/pair-voting randomized
/// Hough transform (voting over minor semi-axis $b$ for candidate antipodal pairs).
pub fn hough_ellipses<E: Numeric + CubeElement>(
    client: &Client,
    edges: &DeviceArray<E>,
    min_size: usize,
    max_size: Option<usize>,
    accuracy: f64,
    threshold: usize,
) -> OpsResult<Vec<HoughEllipse>> {
    check::ndim_in("hough_ellipses", edges.shape(), &[2], "2")?;
    if !accuracy.is_finite() || accuracy <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "accuracy",
            reason: format!("accuracy must be positive and finite, got {accuracy}"),
        });
    }
    let pts = nonzero_coordinates(client, edges)?;
    if pts.len() < 5 {
        return Ok(Vec::new());
    }
    let max_a = max_size.unwrap_or_else(|| edges.dims()[0].max(edges.dims()[1])) as f64 * 0.5;
    let min_a = (min_size as f64 * 0.5).max(1.0);
    let n_bins = ((max_a / accuracy).ceil() as usize).max(1) + 1;
    let mut bins = vec![0usize; n_bins];
    let mut results = Vec::new();

    for i in 0..pts.len() {
        let (y1, x1) = (pts[i][0] as f64, pts[i][1] as f64);
        for j in (i + 1)..pts.len() {
            let (y2, x2) = (pts[j][0] as f64, pts[j][1] as f64);
            let dx = x2 - x1;
            let dy = y2 - y1;
            let a = 0.5 * dx.hypot(dy);
            if a < min_a || a > max_a {
                continue;
            }
            let xc = 0.5 * (x1 + x2);
            let yc = 0.5 * (y1 + y2);
            let a2 = a * a;

            bins.fill(0);
            for (k, pt) in pts.iter().enumerate() {
                if k == i || k == j {
                    continue;
                }
                let (yk, xk) = (pt[0] as f64, pt[1] as f64);
                let d2 = (xk - xc).powi(2) + (yk - yc).powi(2);
                if d2 < min_a * min_a || d2 > a2 {
                    continue;
                }
                let f2 = (xk - x2).powi(2) + (yk - y2).powi(2);
                let d = d2.sqrt();
                let cos_tau = ((a2 + d2 - f2) / (2.0 * a * d)).clamp(-1.0, 1.0);
                let cos2 = cos_tau * cos_tau;
                let sin2 = (1.0 - cos2).max(0.0);
                let denom = a2 - d2 * cos2;
                if denom > 1e-6 {
                    let b2 = (a2 * d2 * sin2) / denom;
                    let b = b2.max(0.0).sqrt();
                    let bin = (b / accuracy).round() as usize;
                    if bin > 0 && bin < n_bins {
                        bins[bin] += 1;
                    }
                }
            }

            if let Some((best_bin, &votes)) = bins.iter().enumerate().max_by_key(|&(_, &v)| v)
                && votes >= threshold {
                    let b = (best_bin as f64) * accuracy;
                    let orientation = dy.atan2(dx);
                    results.push(HoughEllipse { votes, yc, xc, a, b, orientation });
                }
        }
    }

    results.sort_by_key(|r| std::cmp::Reverse(r.votes));
    Ok(results)
}

/// Computes the 2-D forward Radon sinogram of `image` `[height, width]` across `thetas_deg` (in degrees,
/// defaulting to `0..180`), returning an array of shape `[n_bins, thetas_deg.len()]`.
pub fn radon<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    thetas_deg: Option<&[f64]>,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("radon", image.shape(), &[2], "2")?;
    let angles: Vec<f64> = match thetas_deg {
        Some(t) => {
            if t.is_empty() {
                return Err(OpsError::InvalidParameter {
                    name: "thetas_deg",
                    reason: "thetas_deg must not be empty".to_string(),
                });
            }
            t.to_vec()
        }
        None => (0..180).map(|i| i as f64).collect(),
    };

    let (h, w) = (image.dims()[0], image.dims()[1]);
    let half_diag = (((h as f64).hypot(w as f64) * 0.5).ceil() as usize).max(1);
    let n_bins = 2 * half_diag + 1;
    let n_angles = angles.len();

    let cos_vec: Vec<F> = angles.iter().map(|&deg| cast::<F>(deg.to_radians().cos())).collect();
    let sin_vec: Vec<F> = angles.iter().map(|&deg| cast::<F>(deg.to_radians().sin())).collect();
    let cos_dev = DeviceArray::from_host(client, Shape::vector(n_angles), cos_vec)?;
    let sin_dev = DeviceArray::from_host(client, Shape::vector(n_angles), sin_vec)?;
    let sinogram = DeviceArray::<F>::empty(client, Shape::new([n_bins, n_angles]))?;

    let geom = LaunchGeometry::elementwise(client, sinogram.len());
    unsafe {
        radon_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            image.binding(),
            cos_dev.binding(),
            sin_dev.binding(),
            sinogram.binding(),
            h as i32,
            w as i32,
            n_bins as u32,
            n_angles as u32,
            half_diag as i32,
            sinogram.len() as u32,
        );
    }
    Ok(sinogram)
}

/// Frequency-domain filter window for [`iradon`] filtered backprojection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackprojectionFilter {
    /// Unfiltered backprojection (lacks ramp weighting).
    None,
    /// Ideal Ram-Lak $|\omega|$ ramp filter.
    Ramp,
    /// Shepp-Logan $\operatorname{sinc}$-windowed ramp filter.
    SheppLogan,
    /// Cosine-windowed ramp filter.
    Cosine,
    /// Hamming-windowed ramp filter.
    Hamming,
    /// Hann-windowed ramp filter.
    Hann,
}

/// Reconstructs a 2-D `[output_size, output_size]` image from a `sinogram` `[n_bins, n_angles]`
/// using filtered backprojection (`filter` in spatial/detector domain + [`iradon_backproject_kernel`]).
pub fn iradon<F: ImageFloat>(
    client: &Client,
    sinogram: &DeviceArray<F>,
    thetas_deg: &[f64],
    output_size: Option<usize>,
    filter: BackprojectionFilter,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("iradon", sinogram.shape(), &[2], "2")?;
    let (n_bins, n_angles) = (sinogram.dims()[0], sinogram.dims()[1]);
    if thetas_deg.is_empty() || thetas_deg.len() != n_angles {
        return Err(OpsError::InvalidParameter {
            name: "thetas_deg",
            reason: format!("thetas_deg length ({}) must match sinogram columns ({n_angles})", thetas_deg.len()),
        });
    }
    let out_size = output_size.unwrap_or_else(|| {
        (((n_bins as f64) / std::f64::consts::SQRT_2).floor() as usize).max(1)
    });
    if out_size == 0 {
        return Err(OpsError::InvalidParameter {
            name: "output_size",
            reason: "output_size must be >= 1".to_string(),
        });
    }

    // Apply discrete spatial Ram-Lak (with optional spectral window) along the detector bin axis
    let filtered_dev = if filter == BackprojectionFilter::None {
        crate::core::elementwise::copy(client, sinogram)?
    } else {
        let radius = (n_bins / 2) as isize;
        let mut kernel = vec![0.0f64; 2 * (radius as usize) + 1];
        for n in -radius..=radius {
            let idx = (n + radius) as usize;
            let ram_lak = if n == 0 {
                0.25
            } else if n % 2 == 0 {
                0.0
            } else {
                -1.0 / ((std::f64::consts::PI * (n as f64)).powi(2))
            };
            let frac = if radius > 0 { (n.abs() as f64) / (radius as f64) } else { 0.0 };
            let win = match filter {
                BackprojectionFilter::None | BackprojectionFilter::Ramp => 1.0,
                BackprojectionFilter::SheppLogan => {
                    if frac == 0.0 {
                        1.0
                    } else {
                        let x = 0.5 * std::f64::consts::PI * frac;
                        x.sin() / x
                    }
                }
                BackprojectionFilter::Cosine => (0.5 * std::f64::consts::PI * frac).cos(),
                BackprojectionFilter::Hamming => 0.54 + 0.46 * (std::f64::consts::PI * frac).cos(),
                BackprojectionFilter::Hann => 0.5 * (1.0 + (std::f64::consts::PI * frac).cos()),
            };
            kernel[idx] = ram_lak * win;
        }

        let host_sino = sinogram.to_host(client);
        let mut filtered = vec![0.0f64; n_bins * n_angles];
        for a in 0..n_angles {
            for b in 0..n_bins {
                let mut acc = 0.0;
                for k in -radius..=radius {
                    let sb = (b as isize) - k;
                    if sb >= 0 && sb < (n_bins as isize) {
                        acc += to_f64(host_sino[(sb as usize) * n_angles + a])
                            * kernel[(k + radius) as usize];
                    }
                }
                filtered[b * n_angles + a] = acc;
            }
        }
        DeviceArray::from_host(client, sinogram.shape().clone(), cast_all::<F>(&filtered))?
    };

    let cos_vec: Vec<F> = thetas_deg.iter().map(|&deg| cast::<F>(deg.to_radians().cos())).collect();
    let sin_vec: Vec<F> = thetas_deg.iter().map(|&deg| cast::<F>(deg.to_radians().sin())).collect();
    let cos_dev = DeviceArray::from_host(client, Shape::vector(n_angles), cos_vec)?;
    let sin_dev = DeviceArray::from_host(client, Shape::vector(n_angles), sin_vec)?;

    let output = DeviceArray::<F>::empty(client, Shape::new([out_size, out_size]))?;
    let half_diag = ((n_bins - 1) as f64) * 0.5;
    let scale = std::f64::consts::PI / (2.0 * (n_angles as f64));
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        iradon_backproject_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            filtered_dev.binding(),
            cos_dev.binding(),
            sin_dev.binding(),
            output.binding(),
            n_bins as i32,
            n_angles as u32,
            out_size as u32,
            cast::<F>(half_diag),
            cast::<F>(scale),
            output.len() as u32,
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn hough_and_radon_detect_lines_and_reconstruct_spot(client: &Client) {
        // 1. Horizontal line at y = 5 in 11x11 image -> peak at angle = pi/2 (or -pi/2), |distance| = 5
        let (h, w) = (11usize, 11usize);
        let mut line_img = vec![0u32; h * w];
        for x in 0..w {
            line_img[5 * w + x] = 1;
        }
        let edges = DeviceArray::from_host(client, Shape::new([h, w]), line_img).unwrap();
        let hl = hough_lines(client, &edges, None).unwrap();
        let peaks = hough_line_peaks(client, &hl, 2, 2, None, Some(1)).unwrap();
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].votes, 11);
        assert!((peaks[0].distance.abs() - 5.0).abs() <= 1.0);

        // 2. Radon + iradon reconstructs a bright central phantom peak at the center of the image
        let mut phantom = vec![0.0f32; h * w];
        for y in 4..=6 {
            for x in 4..=6 {
                phantom[y * w + x] = 10.0;
            }
        }
        let ph_dev = DeviceArray::from_host(client, Shape::new([h, w]), phantom).unwrap();
        let thetas: Vec<f64> = (0..60).map(|i| (i * 3) as f64).collect();
        let sino = radon(client, &ph_dev, Some(&thetas)).unwrap();
        let recon = iradon(client, &sino, &thetas, Some(h), BackprojectionFilter::Ramp)
            .unwrap()
            .to_host(client);
        assert!(recon[5 * w + 5] > recon[0], "center must exceed corner ({})", client.name());
    }
    runtime_test!(test_hough_and_radon, hough_and_radon_detect_lines_and_reconstruct_spot);
}

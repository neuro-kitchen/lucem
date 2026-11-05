//! Multi-scale blob detectors: Laplacian of Gaussian (LoG), Difference of Gaussians (DoG), and
//! Determinant of Hessian (DoH).

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, to_f64, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::kernels::doh_scale_kernel;
use crate::core::{check, elementwise, reduce, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::filter::gaussian::{
    difference_of_gaussians_into, gaussian_blur_into, gaussian_laplace_into, GaussianBlur,
};

/// A detected multi-scale blob in a 2-D image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blob {
    pub y: usize,
    pub x: usize,
    pub sigma: f64,
    pub response: f64,
}

impl Blob {
    /// Approximate visual radius $\sqrt{2}\,\sigma$ of a 2-D isotropic Gaussian blob.
    pub fn radius(&self) -> f64 {
        std::f64::consts::SQRT_2 * self.sigma
    }
}

/// Settings for multi-scale blob detection ([`blobs_log`], [`blobs_dog`], [`blobs_doh`]).
#[derive(Debug, Clone, PartialEq)]
pub struct BlobDetector {
    pub min_sigma: f64,
    pub max_sigma: f64,
    /// Number of scales for LoG / DoH (`>= 1`).
    pub num_sigma: usize,
    /// Scale ratio $k > 1$ between successive Gaussians in [`blobs_dog`].
    pub sigma_ratio: f64,
    /// Absolute response threshold for scale-space maxima.
    pub threshold: f64,
    /// Maximum allowed area overlap fraction (`0.0..=1.0`); blobs overlapping more than `overlap`
    /// with a stronger blob are pruned.
    pub overlap: f64,
    /// Use logarithmic scale spacing when `true`, linear spacing when `false`.
    pub log_scale: bool,
    /// Exclude blobs within `exclude_border` pixels of the image boundary.
    pub exclude_border: usize,
}

impl Default for BlobDetector {
    fn default() -> Self {
        Self {
            min_sigma: 1.0,
            max_sigma: 30.0,
            num_sigma: 10,
            sigma_ratio: 1.6,
            threshold: 0.1,
            overlap: 0.5,
            log_scale: false,
            exclude_border: 1,
        }
    }
}

impl BlobDetector {
    fn validate(&self) -> OpsResult<()> {
        if !self.min_sigma.is_finite() || self.min_sigma <= 0.0 {
            return Err(OpsError::InvalidParameter {
                name: "min_sigma",
                reason: format!("min_sigma must be positive, got {}", self.min_sigma),
            });
        }
        if !self.max_sigma.is_finite() || self.max_sigma < self.min_sigma {
            return Err(OpsError::InvalidParameter {
                name: "max_sigma",
                reason: format!("max_sigma ({}) must be >= min_sigma ({})", self.max_sigma, self.min_sigma),
            });
        }
        if self.num_sigma == 0 {
            return Err(OpsError::InvalidParameter {
                name: "num_sigma",
                reason: "num_sigma must be at least 1".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.overlap) || !self.overlap.is_finite() {
            return Err(OpsError::InvalidParameter {
                name: "overlap",
                reason: format!("overlap must be in 0.0..=1.0, got {}", self.overlap),
            });
        }
        Ok(())
    }

    fn sigmas(&self) -> Vec<f64> {
        if self.num_sigma == 1 || (self.max_sigma - self.min_sigma).abs() <= f64::EPSILON {
            return vec![self.min_sigma];
        }
        (0..self.num_sigma)
            .map(|i| {
                let t = (i as f64) / ((self.num_sigma - 1) as f64);
                if self.log_scale {
                    self.min_sigma * (self.max_sigma / self.min_sigma).powf(t)
                } else {
                    self.min_sigma + t * (self.max_sigma - self.min_sigma)
                }
            })
            .collect()
    }
}

/// Computes the fractional area overlap between two 2-D circles divided by the area of the smaller circle.
fn circle_overlap_fraction(a: &Blob, b: &Blob) -> f64 {
    let r1 = a.radius();
    let r2 = b.radius();
    let d = ((a.y as f64 - b.y as f64).powi(2) + (a.x as f64 - b.x as f64).powi(2)).sqrt();
    if d >= r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        return 1.0;
    }
    let r_min = r1.min(r2);
    if r_min <= 0.0 {
        return 0.0;
    }
    let cos1 = ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).clamp(-1.0, 1.0);
    let cos2 = ((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).clamp(-1.0, 1.0);
    let term1 = r1 * r1 * cos1.acos();
    let term2 = r2 * r2 * cos2.acos();
    let s = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2);
    let area = term1 + term2 - 0.5 * s.max(0.0).sqrt();
    area / (std::f64::consts::PI * r_min * r_min)
}

/// Extracts 3-D `(scale, y, x)` local maxima from a stack of scale responses and prunes overlapping blobs.
fn extract_scale_space_blobs(
    cube: &[Vec<f64>],
    sigmas: &[f64],
    height: usize,
    width: usize,
    settings: &BlobDetector,
) -> Vec<Blob> {
    let num_scales = sigmas.len();
    let border = settings.exclude_border.max(1);
    if height <= 2 * border || width <= 2 * border {
        return Vec::new();
    }

    let mut raw: Vec<Blob> = Vec::new();
    for s in 0..num_scales {
        let s_lo = s.saturating_sub(1);
        let s_hi = (s + 1).min(num_scales - 1);
        for y in border..(height - border) {
            for x in border..(width - border) {
                let val = cube[s][y * width + x];
                if val <= settings.threshold {
                    continue;
                }
                let mut is_max = true;
                'nbr: for ns in s_lo..=s_hi {
                    for ny in (y - 1)..=(y + 1) {
                        for nx in (x - 1)..=(x + 1) {
                            if ns == s && ny == y && nx == x {
                                continue;
                            }
                            let nv = cube[ns][ny * width + nx];
                            if nv > val || (nv == val && (ns, ny, nx) < (s, y, x)) {
                                is_max = false;
                                break 'nbr;
                            }
                        }
                    }
                }
                if is_max {
                    raw.push(Blob { y, x, sigma: sigmas[s], response: val });
                }
            }
        }
    }

    raw.sort_by(|a, b| {
        b.response
            .total_cmp(&a.response)
            .then_with(|| a.y.cmp(&b.y))
            .then_with(|| a.x.cmp(&b.x))
    });

    let mut kept: Vec<Blob> = Vec::with_capacity(raw.len());
    'cand: for blob in raw {
        for prev in &kept {
            if circle_overlap_fraction(&blob, prev) > settings.overlap {
                continue 'cand;
            }
        }
        kept.push(blob);
    }
    kept
}

/// Scale-normalized Laplacian of Gaussian ($-\sigma^2 \nabla^2 G_\sigma * I$) blob detector on 2-D bright blobs.
pub fn blobs_log<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &BlobDetector,
) -> OpsResult<Vec<Blob>> {
    check::ndim_in("blobs_log", image.shape(), &[2], "2")?;
    settings.validate()?;
    if image.is_empty() {
        return Ok(Vec::new());
    }
    let sigmas = settings.sigmas();
    let mut ws = Workspace::new(client);
    let raw_log = DeviceArray::<F>::empty(client, image.shape().clone())?;
    let scaled = DeviceArray::<F>::empty(client, image.shape().clone())?;
    let mut cube = Vec::with_capacity(sigmas.len());

    for &sigma in &sigmas {
        gaussian_laplace_into(
            client,
            image,
            &[sigma, sigma],
            &raw_log,
            PadMode::Reflect,
            0.0,
            4.0,
            &mut ws,
        )?;
        elementwise::scale_offset_into(client, &raw_log, &scaled, -(sigma * sigma), 0.0)?;
        cube.push(scaled.to_host(client).into_iter().map(to_f64).collect());
    }
    Ok(extract_scale_space_blobs(&cube, &sigmas, image.dims()[0], image.dims()[1], settings))
}

/// Difference of Gaussians (DoG) multi-scale blob detector on 2-D bright blobs.
pub fn blobs_dog<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &BlobDetector,
) -> OpsResult<Vec<Blob>> {
    check::ndim_in("blobs_dog", image.shape(), &[2], "2")?;
    settings.validate()?;
    if !settings.sigma_ratio.is_finite() || settings.sigma_ratio <= 1.0 {
        return Err(OpsError::InvalidParameter {
            name: "sigma_ratio",
            reason: format!("sigma_ratio must be > 1.0, got {}", settings.sigma_ratio),
        });
    }
    if image.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = settings.sigma_ratio;
    let mut sigmas = Vec::new();
    let mut s = settings.min_sigma;
    while s <= settings.max_sigma * (1.0 + 1e-9) {
        sigmas.push(s);
        s *= ratio;
    }
    if sigmas.is_empty() {
        sigmas.push(settings.min_sigma);
    }

    let mut ws = Workspace::new(client);
    let dog = DeviceArray::<F>::empty(client, image.shape().clone())?;
    let scaled = DeviceArray::<F>::empty(client, image.shape().clone())?;
    let blur_cfg = GaussianBlur::default();
    let norm = 1.0 / (ratio - 1.0);
    let mut cube = Vec::with_capacity(sigmas.len());

    for &sigma in &sigmas {
        difference_of_gaussians_into(
            client,
            image,
            &[sigma, sigma],
            &[sigma * ratio, sigma * ratio],
            &dog,
            &blur_cfg,
            &mut ws,
        )?;
        elementwise::scale_offset_into(client, &dog, &scaled, norm, 0.0)?;
        cube.push(scaled.to_host(client).into_iter().map(to_f64).collect());
    }
    Ok(extract_scale_space_blobs(&cube, &sigmas, image.dims()[0], image.dims()[1], settings))
}

/// Scale-normalized Determinant of Hessian ($\sigma^4 \det H_\sigma$) blob detector on 2-D images.
pub fn blobs_doh<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &BlobDetector,
) -> OpsResult<Vec<Blob>> {
    check::ndim_in("blobs_doh", image.shape(), &[2], "2")?;
    settings.validate()?;
    if image.is_empty() {
        return Ok(Vec::new());
    }
    let sigmas = settings.sigmas();
    let mut ws = Workspace::new(client);
    let shape = image.shape().clone();
    let hyy = DeviceArray::<F>::empty(client, shape.clone())?;
    let hxy = DeviceArray::<F>::empty(client, shape.clone())?;
    let hxx = DeviceArray::<F>::empty(client, shape.clone())?;
    let doh = DeviceArray::<F>::empty(client, shape)?;
    let geom = LaunchGeometry::elementwise(client, image.len());
    let mut cube = Vec::with_capacity(sigmas.len());

    let stats = reduce::sum_min_max(client, image)?;
    let _ = stats;

    for &sigma in &sigmas {
        gaussian_blur_into(
            client,
            image,
            &[sigma, sigma],
            &hyy,
            &GaussianBlur { orders: vec![2, 0], ..GaussianBlur::default() },
            &mut ws,
        )?;
        gaussian_blur_into(
            client,
            image,
            &[sigma, sigma],
            &hxy,
            &GaussianBlur { orders: vec![1, 1], ..GaussianBlur::default() },
            &mut ws,
        )?;
        gaussian_blur_into(
            client,
            image,
            &[sigma, sigma],
            &hxx,
            &GaussianBlur { orders: vec![0, 2], ..GaussianBlur::default() },
            &mut ws,
        )?;
        unsafe {
            doh_scale_kernel::launch::<F>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim,
                hyy.binding(),
                hxy.binding(),
                hxx.binding(),
                doh.binding(),
                cast::<F>(sigma.powi(4)),
                image.len() as u32,
            );
        }
        cube.push(doh.to_host(client).into_iter().map(to_f64).collect());
    }
    Ok(extract_scale_space_blobs(&cube, &sigmas, image.dims()[0], image.dims()[1], settings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lumen_core::Shape;

    fn log_dog_doh_detect_synthetic_gaussian_blob(client: &Client) {
        let (h, w) = (25usize, 25usize);
        let (cy, cx, sigma) = (12.0f64, 12.0f64, 2.5f64);
        let mut data = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let r2 = (y as f64 - cy).powi(2) + (x as f64 - cx).powi(2);
                data[y * w + x] = (-r2 / (2.0 * sigma * sigma)).exp() as f32;
            }
        }
        let img = DeviceArray::from_host(client, Shape::new([h, w]), data).unwrap();
        let cfg = BlobDetector {
            min_sigma: 1.5,
            max_sigma: 4.0,
            num_sigma: 6,
            sigma_ratio: 1.3,
            threshold: 0.05,
            overlap: 0.5,
            log_scale: false,
            exclude_border: 2,
        };

        let log_blobs = blobs_log(client, &img, &cfg).unwrap();
        assert!(!log_blobs.is_empty(), "LoG found no blobs ({})", client.name());
        assert_eq!((log_blobs[0].y, log_blobs[0].x), (12, 12), "{}", client.name());

        let dog_blobs = blobs_dog(client, &img, &cfg).unwrap();
        assert!(!dog_blobs.is_empty(), "DoG found no blobs ({})", client.name());
        assert_eq!((dog_blobs[0].y, dog_blobs[0].x), (12, 12), "{}", client.name());

        let doh_blobs = blobs_doh(client, &img, & BlobDetector { threshold: 0.01, ..cfg }).unwrap();
        assert!(!doh_blobs.is_empty(), "DoH found no blobs ({})", client.name());
        assert_eq!((doh_blobs[0].y, doh_blobs[0].x), (12, 12), "{}", client.name());
    }
    runtime_test!(test_blob_detectors, log_dog_doh_detect_synthetic_gaussian_blob);
}

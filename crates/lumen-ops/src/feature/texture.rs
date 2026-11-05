//! Texture descriptors and patch matching: Zero-mean Normalized Cross-Correlation (`match_template`),
//! circular Local Binary Patterns (`lbp`), Gray-Level Co-occurrence Matrix (`glcm`, `glcm_props`),
//! multi-scale basic image features (`multiscale_features`), and Histogram of Oriented Gradients (`hog`).

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, cast_all, to_f64, ImageFloat, ImageInt};
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::kernels::{lbp_2d_kernel, ncc_template_2d_kernel};
use super::structure::hessian_eigvals;
use crate::core::{check, reduce, PadMode};
use crate::error::{OpsError, OpsResult};
use crate::filter::gaussian::{gaussian_blur, gaussian_grad_mag, GaussianBlur};
use crate::filter::gradient::{directional_stencil_into, StencilKind};

/// Zero-mean Normalized Cross-Correlation (ZNCC) template matching in 2-D.
///
/// When `pad_input = false`, returns the valid correlation map of shape
/// `[img_h - tpl_h + 1, img_w - tpl_w + 1]`; when `pad_input = true`, returns an array of the same
/// shape as `image` centered at `(tpl_h / 2, tpl_w / 2)`.
pub fn match_template<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    template: &DeviceArray<F>,
    pad_input: bool,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("match_template(image)", image.shape(), &[2], "2")?;
    check::ndim_in("match_template(template)", template.shape(), &[2], "2")?;
    let (img_h, img_w) = (image.dims()[0], image.dims()[1]);
    let (tpl_h, tpl_w) = (template.dims()[0], template.dims()[1]);
    if tpl_h == 0 || tpl_w == 0 || tpl_h > img_h || tpl_w > img_w {
        return Err(OpsError::InvalidParameter {
            name: "template",
            reason: format!("template shape [{tpl_h}, {tpl_w}] must be non-empty and fit inside image [{img_h}, {img_w}]"),
        });
    }

    let (mean_t, _) = reduce::mean_var(client, template, 0)?;
    let host_t: Vec<f64> = template.to_host(client).into_iter().map(|v| to_f64(v) - mean_t).collect();
    let tpl_norm = host_t.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let tpl_zm = DeviceArray::from_host(client, template.shape().clone(), cast_all::<F>(&host_t))?;

    let (out_h, out_w, oy, ox) = if pad_input {
        (img_h, img_w, (tpl_h / 2) as i32, (tpl_w / 2) as i32)
    } else {
        (img_h - tpl_h + 1, img_w - tpl_w + 1, 0i32, 0i32)
    };
    let out = DeviceArray::<F>::empty(client, Shape::new([out_h, out_w]))?;
    let geom = LaunchGeometry::elementwise(client, out.len());
    unsafe {
        ncc_template_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            image.binding(),
            tpl_zm.binding(),
            out.binding(),
            img_h as i32,
            img_w as i32,
            tpl_h as i32,
            tpl_w as i32,
            out_h as u32,
            out_w as u32,
            oy,
            ox,
            cast::<F>(tpl_norm),
            cast::<F>(1e-12),
            out.len() as u32,
        );
    }
    Ok(out)
}

/// Encoding variant for circular Local Binary Patterns ([`lbp`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbpMethod {
    /// Raw $P$-bit unsigned code $\sum_{p=0}^{P-1} s(g_p - g_c) 2^p$.
    Default = 0,
    /// Rotation-invariant minimum cyclic bit shift.
    Ror = 1,
    /// Uniform rotation-invariant pattern in `0..=P+1` ($\le 2$ bitwise transitions around the ring).
    Uniform = 2,
}

/// Computes circular 2-D Local Binary Patterns (`lbp`) with `num_points` (`1..=32`) at `radius > 0`.
pub fn lbp<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    num_points: usize,
    radius: f64,
    method: LbpMethod,
) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("lbp", image.shape(), &[2], "2")?;
    if !(1..=32).contains(&num_points) {
        return Err(OpsError::InvalidParameter {
            name: "num_points",
            reason: format!("num_points must be in 1..=32, got {num_points}"),
        });
    }
    if !radius.is_finite() || radius <= 0.0 {
        return Err(OpsError::InvalidParameter {
            name: "radius",
            reason: format!("radius must be positive and finite, got {radius}"),
        });
    }
    let out = DeviceArray::<u32>::zeros(client, image.shape().clone())?;
    if image.is_empty() {
        return Ok(out);
    }

    let mut dy = Vec::with_capacity(num_points);
    let mut dx = Vec::with_capacity(num_points);
    for p in 0..num_points {
        let theta = 2.0 * std::f64::consts::PI * (p as f64) / (num_points as f64);
        dy.push(cast::<F>(-radius * theta.sin()));
        dx.push(cast::<F>(radius * theta.cos()));
    }
    let off_y = DeviceArray::from_host(client, Shape::vector(num_points), dy)?;
    let off_x = DeviceArray::from_host(client, Shape::vector(num_points), dx)?;

    let geom = LaunchGeometry::elementwise(client, image.len());
    unsafe {
        lbp_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            image.binding(),
            off_y.binding(),
            off_x.binding(),
            out.binding(),
            image.dims()[0] as i32,
            image.dims()[1] as i32,
            num_points as u32,
            image.len() as u32,
            method as u32,
        );
    }
    Ok(out)
}

/// Haralick summary statistics of a single $(d, \theta)$ slice of a GLCM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlcmProps {
    pub contrast: f64,
    pub dissimilarity: f64,
    pub homogeneity: f64,
    pub asm: f64,
    pub energy: f64,
    pub correlation: f64,
}

/// Computes the 4-D Gray-Level Co-occurrence Matrix of shape `[levels, levels, distances.len(), angles.len()]`
/// from a 2-D integer image with values in `0..levels`.
pub fn glcm<I: ImageInt>(
    client: &Client,
    image: &DeviceArray<I>,
    distances: &[usize],
    angles: &[f64],
    levels: usize,
    symmetric: bool,
    normed: bool,
) -> OpsResult<DeviceArray<f64>> {
    check::ndim_in("glcm", image.shape(), &[2], "2")?;
    if distances.is_empty() || angles.is_empty() || levels == 0 {
        return Err(OpsError::InvalidParameter {
            name: "glcm",
            reason: "distances, angles, and levels must all be non-empty / positive".to_string(),
        });
    }
    let (h, w) = (image.dims()[0], image.dims()[1]);
    let host: Vec<usize> = image.to_host(client).into_iter().map(|v| to_f64(v) as usize).collect();
    if let Some(&bad) = host.iter().find(|&&v| v >= levels) {
        return Err(OpsError::InvalidParameter {
            name: "levels",
            reason: format!("pixel value {bad} exceeds levels = {levels}"),
        });
    }

    let nd = distances.len();
    let na = angles.len();
    let out_shape = Shape::new([levels, levels, nd, na]);
    let strides = out_shape.strides();
    let mut mat = vec![0.0f64; out_shape.len()];

    for (di, &dist) in distances.iter().enumerate() {
        let d = dist as f64;
        for (ai, &angle) in angles.iter().enumerate() {
            let dy = (-d * angle.sin()).round() as isize;
            let dx = (d * angle.cos()).round() as isize;
            for r in 0..(h as isize) {
                let nr = r + dy;
                if nr < 0 || nr >= (h as isize) {
                    continue;
                }
                for c in 0..(w as isize) {
                    let nc = c + dx;
                    if nc < 0 || nc >= (w as isize) {
                        continue;
                    }
                    let i = host[(r as usize) * w + (c as usize)];
                    let j = host[(nr as usize) * w + (nc as usize)];
                    mat[i * strides[0] + j * strides[1] + di * strides[2] + ai * strides[3]] += 1.0;
                    if symmetric {
                        mat[j * strides[0] + i * strides[1] + di * strides[2] + ai * strides[3]] += 1.0;
                    }
                }
            }
            if normed {
                let mut sum = 0.0;
                for i in 0..levels {
                    for j in 0..levels {
                        sum += mat[i * strides[0] + j * strides[1] + di * strides[2] + ai * strides[3]];
                    }
                }
                if sum > 0.0 {
                    for i in 0..levels {
                        for j in 0..levels {
                            mat[i * strides[0] + j * strides[1] + di * strides[2] + ai * strides[3]] /= sum;
                        }
                    }
                }
            }
        }
    }

    Ok(DeviceArray::from_host(client, out_shape, mat)?)
}

/// Extracts Haralick texture properties (`contrast`, `dissimilarity`, `homogeneity`, `asm`, `energy`,
/// `correlation`) for every `(distance, angle)` slice `[d_idx, a_idx]` of a 4-D `glcm` matrix.
pub fn glcm_props(client: &Client, p_matrix: &DeviceArray<f64>) -> OpsResult<Vec<Vec<GlcmProps>>> {
    check::ndim_in("glcm_props", p_matrix.shape(), &[4], "4")?;
    let dims = p_matrix.dims();
    let (levels, levels2, nd, na) = (dims[0], dims[1], dims[2], dims[3]);
    if levels != levels2 {
        return Err(OpsError::InvalidParameter {
            name: "p_matrix",
            reason: format!("first two dimensions of GLCM must match, got [{levels}, {levels2}]"),
        });
    }
    let strides = p_matrix.shape().strides();
    let host = p_matrix.to_host(client);

    let mut result = vec![Vec::with_capacity(na); nd];
    for di in 0..nd {
        for ai in 0..na {
            let mut sum = 0.0;
            for i in 0..levels {
                for j in 0..levels {
                    sum += host[i * strides[0] + j * strides[1] + di * strides[2] + ai * strides[3]];
                }
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

            let mut contrast = 0.0;
            let mut dissimilarity = 0.0;
            let mut homogeneity = 0.0;
            let mut asm = 0.0;
            let mut mu_i = 0.0;
            let mut mu_j = 0.0;

            for i in 0..levels {
                for j in 0..levels {
                    let p = host[i * strides[0] + j * strides[1] + di * strides[2] + ai * strides[3]] * inv_sum;
                    let diff = (i as f64) - (j as f64);
                    contrast += p * diff * diff;
                    dissimilarity += p * diff.abs();
                    homogeneity += p / (1.0 + diff * diff);
                    asm += p * p;
                    mu_i += (i as f64) * p;
                    mu_j += (j as f64) * p;
                }
            }

            let mut var_i = 0.0;
            let mut var_j = 0.0;
            let mut cov = 0.0;
            for i in 0..levels {
                for j in 0..levels {
                    let p = host[i * strides[0] + j * strides[1] + di * strides[2] + ai * strides[3]] * inv_sum;
                    let di_f = (i as f64) - mu_i;
                    let dj_f = (j as f64) - mu_j;
                    var_i += p * di_f * di_f;
                    var_j += p * dj_f * dj_f;
                    cov += p * di_f * dj_f;
                }
            }
            let denom = (var_i * var_j).sqrt();
            let correlation = if denom > 1e-12 { cov / denom } else { 1.0 };

            result[di].push(GlcmProps {
                contrast,
                dissimilarity,
                homogeneity,
                asm,
                energy: asm.sqrt(),
                correlation,
            });
        }
    }
    Ok(result)
}

/// Computes a bank of multi-scale basic features (intensity, gradient magnitude, and Hessian eigenvalues)
/// across each Gaussian scale in `sigmas`.
pub fn multiscale_features<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    sigmas: &[f64],
    include_intensity: bool,
    include_edges: bool,
    include_texture: bool,
) -> OpsResult<Vec<DeviceArray<F>>> {
    check::ndim_in("multiscale_features", image.shape(), &[2], "2")?;
    if sigmas.is_empty() {
        return Err(OpsError::InvalidParameter {
            name: "sigmas",
            reason: "at least one sigma scale is required".to_string(),
        });
    }
    let mut features = Vec::new();
    let blur_cfg = GaussianBlur::default();
    for &sigma in sigmas {
        if include_intensity {
            features.push(gaussian_blur(client, image, &[sigma, sigma], &blur_cfg)?);
        }
        if include_edges {
            features.push(gaussian_grad_mag(client, image, &[sigma, sigma], PadMode::Reflect, 0.0, 4.0)?);
        }
        if include_texture {
            let (e1, e2) = hessian_eigvals(client, image, sigma, PadMode::Reflect, 0.0)?;
            features.push(e1);
            features.push(e2);
        }
    }
    Ok(features)
}

/// Settings for Histogram of Oriented Gradients ([`hog`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Hog {
    pub orientations: usize,
    pub pixels_per_cell: (usize, usize),
    pub cells_per_block: (usize, usize),
}

impl Default for Hog {
    fn default() -> Self {
        Self { orientations: 9, pixels_per_cell: (8, 8), cells_per_block: (2, 2) }
    }
}

/// Computes the L2-Hys normalized Histogram of Oriented Gradients (HOG) feature vector for a 2-D image.
pub fn hog<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    settings: &Hog,
) -> OpsResult<Vec<f64>> {
    check::ndim_in("hog", image.shape(), &[2], "2")?;
    let (ch, cw) = settings.pixels_per_cell;
    let (bh, bw) = settings.cells_per_block;
    if settings.orientations == 0 || ch == 0 || cw == 0 || bh == 0 || bw == 0 {
        return Err(OpsError::InvalidParameter {
            name: "hog",
            reason: "orientations, pixels_per_cell, and cells_per_block must all be >= 1".to_string(),
        });
    }
    let (h, w) = (image.dims()[0], image.dims()[1]);
    let n_cells_y = h / ch;
    let n_cells_x = w / cw;
    if n_cells_y < bh || n_cells_x < bw {
        return Ok(Vec::new());
    }

    let mut ws = Workspace::new(client);
    let gy_dev = DeviceArray::<F>::empty(client, image.shape().clone())?;
    let gx_dev = DeviceArray::<F>::empty(client, image.shape().clone())?;
    directional_stencil_into(client, image, &gy_dev, 0, StencilKind::Sobel, PadMode::Reflect, 0.0, &mut ws)?;
    directional_stencil_into(client, image, &gx_dev, 1, StencilKind::Sobel, PadMode::Reflect, 0.0, &mut ws)?;
    let gy = gy_dev.to_host(client);
    let gx = gx_dev.to_host(client);

    let bins = settings.orientations;
    let mut cell_hist = vec![0.0f64; n_cells_y * n_cells_x * bins];
    for cy in 0..n_cells_y {
        for cx in 0..n_cells_x {
            let base = (cy * n_cells_x + cx) * bins;
            for py in (cy * ch)..((cy + 1) * ch) {
                for px in (cx * cw)..((cx + 1) * cw) {
                    let vy = to_f64(gy[py * w + px]);
                    let vx = to_f64(gx[py * w + px]);
                    let mag = (vy * vy + vx * vx).sqrt();
                    let mut angle = vy.atan2(vx).to_degrees();
                    if angle < 0.0 {
                        angle += 180.0;
                    }
                    if angle >= 180.0 {
                        angle -= 180.0;
                    }
                    let bin = ((angle / 180.0) * (bins as f64)).floor() as usize;
                    cell_hist[base + bin.min(bins - 1)] += mag;
                }
            }
        }
    }

    let n_blocks_y = n_cells_y - bh + 1;
    let n_blocks_x = n_cells_x - bw + 1;
    let block_len = bh * bw * bins;
    let mut descriptor = Vec::with_capacity(n_blocks_y * n_blocks_x * block_len);
    let eps = 1e-5;

    for by in 0..n_blocks_y {
        for bx in 0..n_blocks_x {
            let mut block = Vec::with_capacity(block_len);
            for dy in 0..bh {
                for dx in 0..bw {
                    let base = ((by + dy) * n_cells_x + (bx + dx)) * bins;
                    block.extend_from_slice(&cell_hist[base..(base + bins)]);
                }
            }
            // L2-Hys normalization (L2 -> clip at 0.2 -> L2)
            let norm1 = (block.iter().map(|&v| v * v).sum::<f64>() + eps * eps).sqrt();
            for v in &mut block {
                *v = (*v / norm1).min(0.2);
            }
            let norm2 = (block.iter().map(|&v| v * v).sum::<f64>() + eps * eps).sqrt();
            for v in &mut block {
                *v /= norm2;
            }
            descriptor.extend_from_slice(&block);
        }
    }
    Ok(descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};

    fn zncc_lbp_glcm_and_hog_work_on_known_patterns(client: &Client) {
        // 1. Template matching finds exact patch with correlation 1.0
        let mut img_data = vec![0.0f32; 64];
        img_data[2 * 8 + 3] = 5.0;
        img_data[2 * 8 + 4] = 10.0;
        img_data[3 * 8 + 3] = 15.0;
        img_data[3 * 8 + 4] = 20.0;
        let img = DeviceArray::from_host(client, Shape::new([8, 8]), img_data).unwrap();
        let tpl = DeviceArray::from_host(client, Shape::new([2, 2]), vec![5.0f32, 10.0, 15.0, 20.0]).unwrap();
        let ncc = match_template(client, &img, &tpl, false).unwrap().to_host(client);
        assert_close(&[ncc[2 * 7 + 3] as f64], &[1.0], 1e-5, client.name());

        // 2. LBP on constant image yields all-ones code (2^P - 1) in Default mode and P in Uniform mode
        let ones = DeviceArray::<f32>::filled(client, Shape::new([6, 6]), 3.0).unwrap();
        let u = lbp(client, &ones, 8, 1.0, LbpMethod::Uniform).unwrap().to_host(client);
        assert!(u.iter().all(|&v| v == 8), "{}", client.name());

        // 3. GLCM on horizontal checkerboard (0, 1, 0, 1) at d=1, angle=0 has only off-diagonal (0,1) and (1,0)
        let checker = DeviceArray::from_host(client, Shape::new([2, 4]), vec![0u32, 1, 0, 1, 0, 1, 0, 1]).unwrap();
        let g = glcm(client, &checker, &[1], &[0.0], 2, false, true).unwrap();
        let props = glcm_props(client, &g).unwrap();
        assert_close(&[props[0][0].contrast], &[1.0], 1e-6, client.name());

        // 4. HOG descriptor on 16x16 image produces 1 * 1 * (2*2*9) = 36 elements
        let hog_img = DeviceArray::<f32>::filled(client, Shape::new([16, 16]), 1.0).unwrap();
        let desc = hog(client, &hog_img, &Hog::default()).unwrap();
        assert_eq!(desc.len(), 36);
    }
    runtime_test!(test_texture_features, zncc_lbp_glcm_and_hog_work_on_known_patterns);
}

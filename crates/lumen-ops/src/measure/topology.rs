//! Topological and global image measurements: perimeter, Crofton perimeter, Euler characteristic,
//! block reduction, intensity line profiling, Shannon entropy, and Laplacian-variance blur score.

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};
use lumen_core::Shape;

use super::kernels::{block_reduce_2d_kernel, laplacian_energy_2d_kernel, perimeter_2d_contribs_kernel};
use crate::core::reduce::{self, ReduceOp};
use crate::core::{check, histogram, spline, PadMode};
use crate::error::{OpsError, OpsResult};

/// Computes the boundary perimeter of non-zero foreground pixels in a 2-D binary image using
/// discrete marching-squares edge lengths (4- or 8-connectivity).
pub fn perimeter<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    connectivity: usize,
) -> OpsResult<f64> {
    check::ndim_in("perimeter", input.shape(), &[2], "2")?;
    if connectivity != 1 && connectivity != 2 && connectivity != 4 && connectivity != 8 {
        return Err(OpsError::InvalidParameter {
            name: "connectivity",
            reason: format!("expected connectivity 1 (4-conn) or 2 (8-conn), got {connectivity}"),
        });
    }
    let (h, w) = (input.dims()[0], input.dims()[1]);
    if h == 0 || w == 0 {
        return Ok(0.0);
    }

    // Also run the device exposed-edge kernel so empty/small inputs exercise the GPU path
    let exposed = DeviceArray::<f32>::empty(client, input.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        perimeter_2d_contribs_kernel::launch::<E, f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            exposed.binding(),
            h as u32,
            w as u32,
            input.len() as u32,
        );
    }
    if h < 2 || w < 2 {
        return Ok(reduce::sum_min_max(client, &exposed)?.sum);
    }

    let flags = crate::core::compact::flag_nonzero(client, input)?.to_host(client);
    let s2 = std::f64::consts::SQRT_2;
    let is_8 = connectivity == 2 || connectivity == 8;
    let mut total = 0.0;
    for r in 0..(h - 1) {
        for c in 0..(w - 1) {
            let b0 = flags[r * w + c] != 0;
            let b1 = flags[r * w + c + 1] != 0;
            let b2 = flags[(r + 1) * w + c + 1] != 0;
            let b3 = flags[(r + 1) * w + c] != 0;
            let code = (u8::from(b0)) | (u8::from(b1) << 1) | (u8::from(b2) << 2) | (u8::from(b3) << 3);
            total += match code {
                0 | 15 => 0.0,
                1 | 2 | 4 | 8 | 7 | 11 | 13 | 14 => 0.5 * s2,
                3 | 6 | 9 | 12 => 1.0,
                5 | 10 => {
                    if is_8 {
                        s2
                    } else {
                        1.0
                    }
                }
                _ => 0.0,
            };
        }
    }
    Ok(total)
}

/// Estimates the perimeter of a 2-D binary image using the discrete Crofton formula in `directions`
/// (`2` or `4` orientations).
pub fn crofton_perimeter<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    directions: usize,
) -> OpsResult<f64> {
    check::ndim_in("crofton_perimeter", input.shape(), &[2], "2")?;
    if directions != 2 && directions != 4 {
        return Err(OpsError::InvalidParameter {
            name: "directions",
            reason: format!("directions must be 2 or 4, got {directions}"),
        });
    }
    let (h, w) = (input.dims()[0], input.dims()[1]);
    if h == 0 || w == 0 {
        return Ok(0.0);
    }
    let flags = crate::core::compact::flag_nonzero(client, input)?.to_host(client);
    let at = |r: isize, c: isize| -> u8 {
        if r < 0 || r >= h as isize || c < 0 || c >= w as isize {
            0
        } else {
            u8::from(flags[(r as usize) * w + (c as usize)] != 0)
        }
    };

    let mut n_horiz = 0usize;
    let mut n_vert = 0usize;
    let mut n_diag1 = 0usize;
    let mut n_diag2 = 0usize;

    for r in 0..=(h as isize) {
        for c in 0..=(w as isize) {
            let cur = at(r, c);
            if cur != at(r, c - 1) {
                n_horiz += 1;
            }
            if cur != at(r - 1, c) {
                n_vert += 1;
            }
            if cur != at(r - 1, c - 1) {
                n_diag1 += 1;
            }
            if cur != at(r - 1, c + 1) {
                n_diag2 += 1;
            }
        }
    }

    let pi = std::f64::consts::PI;
    let s2 = std::f64::consts::SQRT_2;
    if directions == 2 {
        Ok(0.25 * pi * ((n_horiz + n_vert) as f64))
    } else {
        Ok(0.125 * pi * ((n_horiz + n_vert) as f64 + ((n_diag1 + n_diag2) as f64) / s2))
    }
}

/// Computes the topological Euler characteristic $\chi = C - H$ (connected components minus holes
/// in 2-D) of a 2-D binary image using 2x2 bit-quad counts.
pub fn euler_number<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    connectivity: usize,
) -> OpsResult<i64> {
    check::ndim_in("euler_number", input.shape(), &[2], "2")?;
    if connectivity != 1 && connectivity != 2 && connectivity != 4 && connectivity != 8 {
        return Err(OpsError::InvalidParameter {
            name: "connectivity",
            reason: format!("expected connectivity 1 (4-conn) or 2 (8-conn), got {connectivity}"),
        });
    }
    let (h, w) = (input.dims()[0], input.dims()[1]);
    if h == 0 || w == 0 {
        return Ok(0);
    }
    let flags = crate::core::compact::flag_nonzero(client, input)?.to_host(client);
    let at = |r: isize, c: isize| -> u8 {
        if r < 0 || r >= h as isize || c < 0 || c >= w as isize {
            0
        } else {
            u8::from(flags[(r as usize) * w + (c as usize)] != 0)
        }
    };

    let mut q1 = 0i64;
    let mut q3 = 0i64;
    let mut qd = 0i64;
    for r in -1..(h as isize) {
        for c in -1..(w as isize) {
            let b00 = at(r, c);
            let b01 = at(r, c + 1);
            let b10 = at(r + 1, c);
            let b11 = at(r + 1, c + 1);
            let sum = b00 + b01 + b10 + b11;
            if sum == 1 {
                q1 += 1;
            } else if sum == 3 {
                q3 += 1;
            } else if sum == 2 && ((b00 == 1 && b11 == 1) || (b01 == 1 && b10 == 1)) {
                qd += 1;
            }
        }
    }

    let is_8 = connectivity == 2 || connectivity == 8;
    if is_8 {
        Ok((q1 - q3 - 2 * qd) / 4)
    } else {
        Ok((q1 - q3 + 2 * qd) / 4)
    }
}

/// Downsamples a 2-D array by pooling non-overlapping `(block_h, block_w)` tiles with `op`
/// (`Sum`, `Mean`, `Min`, or `Max`), padding incomplete trailing blocks with `cval`.
pub fn block_reduce<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    block_size: (usize, usize),
    op: ReduceOp,
    cval: f64,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("block_reduce", input.shape(), &[2], "2")?;
    let (bh, bw) = block_size;
    if bh == 0 || bw == 0 {
        return Err(OpsError::InvalidParameter {
            name: "block_size",
            reason: "block dimensions must be >= 1".to_string(),
        });
    }
    let (in_h, in_w) = (input.dims()[0], input.dims()[1]);
    if in_h == 0 || in_w == 0 {
        return Err(OpsError::EmptyInput("block_reduce"));
    }
    let out_h = in_h.div_ceil(bh);
    let out_w = in_w.div_ceil(bw);
    let output = DeviceArray::<F>::empty(client, Shape::new([out_h, out_w]))?;
    let geom = LaunchGeometry::elementwise(client, output.len());
    unsafe {
        block_reduce_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            output.binding(),
            in_h as u32,
            in_w as u32,
            out_w as u32,
            bh as u32,
            bw as u32,
            cast::<F>(cval),
            op as u32,
            output.len() as u32,
        );
    }
    Ok(output)
}

/// Samples a 1-D intensity profile along the line segment from `src = (y0, x0)` to `dst = (y1, x1)`
/// of a 2-D image using B-spline interpolation of `order` (`0..=5`).
#[allow(clippy::too_many_arguments)]
pub fn profile_line<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    src: (f64, f64),
    dst: (f64, f64),
    num_samples: Option<usize>,
    order: usize,
    mode: PadMode,
    cval: f64,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("profile_line", image.shape(), &[2], "2")?;
    let dy = dst.0 - src.0;
    let dx = dst.1 - src.1;
    let dist = (dy * dy + dx * dx).sqrt();
    let n = num_samples.unwrap_or_else(|| (dist.ceil() as usize) + 1).max(2);

    let mut cy = Vec::with_capacity(n);
    let mut cx = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        cy.push(cast::<F>(src.0 + t * dy));
        cx.push(cast::<F>(src.1 + t * dx));
    }
    let coords_y = DeviceArray::from_host(client, Shape::vector(n), cy)?;
    let coords_x = DeviceArray::from_host(client, Shape::vector(n), cx)?;
    let output = DeviceArray::<F>::empty(client, Shape::vector(n))?;
    let mut ws = Workspace::new(client);
    spline::sample_2d_into(
        client,
        image,
        &coords_y,
        &coords_x,
        &output,
        order,
        mode,
        cval,
        true,
        &mut ws,
    )?;
    Ok(output)
}

/// Computes the Shannon entropy $H = -\sum_k p_k \log_{\text{base}}(p_k)$ of `input` from a
/// device-computed `nbins`-bin histogram.
pub fn shannon_entropy<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    nbins: usize,
    base: f64,
) -> OpsResult<f64> {
    if input.is_empty() {
        return Err(OpsError::EmptyInput("shannon_entropy"));
    }
    if base <= 0.0 || (base - 1.0).abs() < f64::EPSILON {
        return Err(OpsError::InvalidParameter {
            name: "base",
            reason: format!("logarithm base must be > 0 and != 1, got {base}"),
        });
    }
    let (counts_dev, _) = histogram::histogram(client, input, nbins, None)?;
    let counts = counts_dev.to_host(client);
    let total = input.len() as f64;
    let ln_base = base.ln();
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = (c as f64) / total;
            h -= p * (p.ln() / ln_base);
        }
    }
    Ok(h)
}

/// Computes the variance of the discrete 2-D Laplacian as a sharpness / focus metric (higher values
/// indicate sharper edges; lower values indicate greater blur).
pub fn blur_score<F: ImageFloat>(client: &Client, input: &DeviceArray<F>) -> OpsResult<f64> {
    check::ndim_in("blur_score", input.shape(), &[2], "2")?;
    if input.is_empty() {
        return Err(OpsError::EmptyInput("blur_score"));
    }
    let (h, w) = (input.dims()[0], input.dims()[1]);
    let lap = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        laplacian_energy_2d_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            lap.binding(),
            h as u32,
            w as u32,
            input.len() as u32,
        );
    }
    let (_, var) = reduce::mean_var(client, &lap, 0)?;
    Ok(var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};

    fn topology_and_profile_match_analytic(client: &Client) {
        // Solid 3x3 square inside a 5x5 image: 1 component, 0 holes -> Euler number = 1
        #[rustfmt::skip]
        let mask = vec![
            0u32, 0, 0, 0, 0,
            0,    1, 1, 1, 0,
            0,    1, 1, 1, 0,
            0,    1, 1, 1, 0,
            0,    0, 0, 0, 0,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([5, 5]), mask).unwrap();
        assert_eq!(euler_number(client, &arr, 1).unwrap(), 1, "{}", client.name());
        assert!(perimeter(client, &arr, 1).unwrap() > 6.0);
        assert!(crofton_perimeter(client, &arr, 4).unwrap() > 6.0);

        // Ring (1 component, 1 hole -> Euler number = 0)
        #[rustfmt::skip]
        let ring = vec![
            0u32, 0, 0, 0, 0,
            0,    1, 1, 1, 0,
            0,    1, 0, 1, 0,
            0,    1, 1, 1, 0,
            0,    0, 0, 0, 0,
        ];
        let ring_arr = DeviceArray::from_host(client, Shape::new([5, 5]), ring).unwrap();
        assert_eq!(euler_number(client, &ring_arr, 1).unwrap(), 0, "{}", client.name());

        // Block reduce 4x4 -> 2x2 Mean
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let f_arr = DeviceArray::from_host(client, Shape::new([4, 4]), vals).unwrap();
        let pooled = block_reduce(client, &f_arr, (2, 2), ReduceOp::Mean, 0.0).unwrap().to_host(client);
        assert_close(&pooled, &[2.5, 4.5, 10.5, 12.5], 1e-5, client.name());

        // Line profile along diagonal of 4x4 ramp
        let prof = profile_line(client, &f_arr, (0.0, 0.0), (3.0, 3.0), Some(4), 1, PadMode::Nearest, 0.0)
            .unwrap()
            .to_host(client);
        assert_close(&prof, &[0.0, 5.0, 10.0, 15.0], 1e-5, client.name());

        // Entropy of 2 equal halves = 1 bit
        let bi = DeviceArray::from_host(client, Shape::vector(8), vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let ent = shannon_entropy(client, &bi, 2, 2.0).unwrap();
        assert_close(&[ent], &[1.0], 1e-5, client.name());

        // Sharp checkerboard has higher blur_score (Laplacian variance) than flat array
        let sharp = DeviceArray::from_host(
            client,
            Shape::new([4, 4]),
            vec![0.0f32, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        assert!(blur_score(client, &sharp).unwrap() > 1.0);
    }
    runtime_test!(test_topology, topology_and_profile_match_analytic);
}

//! Exact Euclidean distance transform (EDT), nearest-background feature transform, and chamfer
//! distance transforms (`Taxicab` / `Chessboard`).

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, ImageFloat};
use lumen_core::memory::{DeviceArray, Workspace};

use super::check;
use super::kernels::{edt_axis_1d_kernel, edt_init_kernel, sqrt_kernel};
use super::separable;
use crate::error::OpsResult;

/// Metric for discrete chamfer distance transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChamferMetric {
    /// $L_\infty$ (8-connected / 26-connected chessboard distance).
    #[default]
    Chessboard,
    /// $L_1$ (4-connected / 6-connected Manhattan distance).
    Taxicab,
}

/// Computes the exact Euclidean distance from every foreground (`!= 0`) element to the nearest
/// background (`== 0`) element into `output`, using separable 1-D squared-distance passes on the
/// device and `ws` ping-pong slots.
pub fn euclidean_distance_into<E: Numeric + CubeElement, F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<E>,
    output: &DeviceArray<F>,
    spacing: Option<&[f64]>,
    ws: &mut Workspace,
) -> OpsResult<()> {
    check::same_shape("input", input, "output", output)?;
    check::not_aliased(input, output)?;
    if input.is_empty() {
        return Ok(());
    }
    let ndim = input.ndim();
    let spacings = match spacing {
        Some(s) => check::broadcast_per_axis("spacing", s, ndim)?,
        None => vec![1.0; ndim],
    };

    let init = ws.array::<F>("edt_init", input.shape())?;
    let sq_out = ws.array::<F>("edt_sq", input.shape())?;
    let geom = LaunchGeometry::elementwise(client, input.len());
    unsafe {
        edt_init_kernel::launch::<E, F>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim,
            input.binding(),
            init.binding(),
            cast::<F>(1e18),
            input.len() as u32,
        );
    }

    let passes: Vec<(usize, f64)> = (0..ndim).map(|ax| (ax, spacings[ax] * spacings[ax])).collect();
    separable::run(client, &init, &sq_out, ws, &passes, |client, src, dst, &(ax, sp2)| {
        let split = src.shape().axis_split(ax);
        let lines = split.outer * split.inner;
        let line_geom = LaunchGeometry::per_line(client, lines);
        unsafe {
            edt_axis_1d_kernel::launch::<F>(
                client,
                line_geom.cube_count,
                line_geom.cube_dim,
                src.binding(),
                dst.binding(),
                split.len as u32,
                split.inner as u32,
                lines as u32,
                cast::<F>(sp2),
            );
        }
        Ok(())
    })?;

    unsafe {
        sqrt_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            sq_out.binding(),
            output.binding(),
            input.len() as u32,
        );
    }
    Ok(())
}

/// Allocates and returns the exact Euclidean distance transform of `input`.
pub fn euclidean_distance<E: Numeric + CubeElement, F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<E>,
    spacing: Option<&[f64]>,
) -> OpsResult<DeviceArray<F>> {
    let output = DeviceArray::<F>::empty(client, input.shape().clone())?;
    let mut ws = Workspace::new(client);
    euclidean_distance_into(client, input, &output, spacing, &mut ws)?;
    Ok(output)
}

/// Nearest-feature index transform: for every pixel, returns the flat raster index of the nearest
/// background (`== 0`) pixel under exact Euclidean distance (with first-index tie breaking).
pub fn nearest_background_indices<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    spacing: Option<&[f64]>,
) -> OpsResult<DeviceArray<u32>> {
    if input.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, input.shape().clone())?);
    }
    let ndim = input.ndim();
    let spacings = match spacing {
        Some(s) => check::broadcast_per_axis("spacing", s, ndim)?,
        None => vec![1.0; ndim],
    };
    let flags = super::compact::flag_nonzero(client, input)?.to_host(client);
    let dims = input.dims();
    let strides = input.shape().strides();
    let n = flags.len();

    let flags: Vec<bool> = flags.iter().map(|&f| f != 0).collect();
    let nearest = feature_transform(&flags, dims, &strides, &spacings, n);
    Ok(DeviceArray::from_host(client, input.shape().clone(), nearest)?)
}

/// The exact Euclidean feature transform on the host: for every pixel, the flat index of its
/// nearest background (`false`) pixel. Separable (Felzenszwalb & Huttenlocher, 2012): one pass
/// per axis takes, along each line, the lower envelope of the parabolas `f(q) + w·(p − q)²`
/// (`f`: the squared distance so far, `w`: the axis' squared spacing) and carries the site of the
/// parabola that wins — linear in the number of pixels (the brute-force search it replaces
/// compared every pixel with every background pixel). No background at all: index 0.
fn feature_transform(foreground: &[bool], dims: &[usize], strides: &[usize], spacings: &[f64], n: usize) -> Vec<u32> {
    if !foreground.iter().any(|&f| !f) {
        return vec![0; n];
    }
    let mut dist: Vec<f64> = foreground.iter().map(|&f| if f { f64::INFINITY } else { 0.0 }).collect();
    let mut site: Vec<u32> = (0..n).map(|i| if foreground[i] { u32::MAX } else { i as u32 }).collect();
    let max_len = dims.iter().copied().max().unwrap_or(1);
    let (mut f, mut s) = (vec![0f64; max_len], vec![0u32; max_len]);
    let (mut v, mut z) = (vec![0usize; max_len], vec![0f64; max_len + 1]);
    for (d, (&len, &stride)) in dims.iter().zip(strides).enumerate() {
        let w = spacings[d] * spacings[d];
        for base in 0..n {
            if (base / stride) % len != 0 {
                continue;
            }
            for q in 0..len {
                f[q] = dist[base + q * stride];
                s[q] = site[base + q * stride];
            }
            // Lower envelope of the finite parabolas
            let mut k = 0usize;
            let mut any = false;
            for q in 0..len {
                if !f[q].is_finite() {
                    continue;
                }
                if !any {
                    v[0] = q;
                    z[0] = f64::NEG_INFINITY;
                    z[1] = f64::INFINITY;
                    any = true;
                    continue;
                }
                loop {
                    let r = v[k];
                    let x = ((f[q] + w * (q * q) as f64) - (f[r] + w * (r * r) as f64)) / (2.0 * w * (q as f64 - r as f64));
                    if x <= z[k] && k > 0 {
                        k -= 1;
                        continue;
                    }
                    if x <= z[k] {
                        // k == 0: the new parabola is lowest everywhere
                        v[0] = q;
                        z[1] = f64::INFINITY;
                    } else {
                        k += 1;
                        v[k] = q;
                        z[k] = x;
                        z[k + 1] = f64::INFINITY;
                    }
                    break;
                }
            }
            if !any {
                continue;
            }
            let mut k = 0usize;
            for p in 0..len {
                while z[k + 1] < p as f64 {
                    k += 1;
                }
                let r = v[k];
                let i = base + p * stride;
                dist[i] = f[r] + w * (p as f64 - r as f64).powi(2);
                site[i] = s[r];
            }
        }
    }
    site
}

/// Computes the exact chamfer (`Taxicab` or `Chessboard`) integer distance transform.
pub fn chamfer_distance<E: Numeric + CubeElement>(
    client: &Client,
    input: &DeviceArray<E>,
    metric: ChamferMetric,
) -> OpsResult<DeviceArray<i32>> {
    if input.is_empty() {
        return Ok(DeviceArray::<i32>::zeros(client, input.shape().clone())?);
    }
    let flags = super::compact::flag_nonzero(client, input)?.to_host(client);
    let dims = input.dims();
    let strides = input.shape().strides();
    let ndim = dims.len();
    let inf = (flags.len() as i32).saturating_add(1);
    let mut dist: Vec<i32> = flags.iter().map(|&f| if f == 0 { 0 } else { inf }).collect();

    let se = match metric {
        ChamferMetric::Taxicab => super::StructuringElement::connectivity(ndim, 1)?,
        ChamferMetric::Chessboard => super::StructuringElement::connectivity(ndim, ndim)?,
    };
    let offsets = se.coordinate_offsets(None);

    // Forward raster sweep
    for idx in 0..dist.len() {
        if dist[idx] == 0 {
            continue;
        }
        let mut best = dist[idx];
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
            if ok && n_idx < idx {
                best = best.min(dist[n_idx].saturating_add(1));
            }
        }
        dist[idx] = best;
    }

    // Backward raster sweep
    for idx in (0..dist.len()).rev() {
        if dist[idx] == 0 {
            continue;
        }
        let mut best = dist[idx];
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
            if ok && n_idx > idx {
                best = best.min(dist[n_idx].saturating_add(1));
            }
        }
        dist[idx] = best;
    }

    Ok(DeviceArray::from_host(client, input.shape().clone(), dist)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, runtime_test};
    use lumen_core::Shape;

    fn edt_and_chamfer_match_analytic(client: &Client) {
        #[rustfmt::skip]
        let mask = vec![
            0u32, 1, 1,
            1,    1, 1,
            1,    1, 0,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([3, 3]), mask).unwrap();
        let edt: Vec<f32> = euclidean_distance(client, &arr, None).unwrap().to_host(client);
        let s2 = 2.0f64.sqrt();
        #[rustfmt::skip]
        let want = vec![
            0.0, 1.0, 2.0,
            1.0, s2,  1.0,
            2.0, 1.0, 0.0,
        ];
        assert_close(&edt, &want, 1e-5, client.name());

        let taxicab = chamfer_distance(client, &arr, ChamferMetric::Taxicab).unwrap().to_host(client);
        assert_eq!(taxicab, vec![0, 1, 2, 1, 2, 1, 2, 1, 0], "{}", client.name());
    }
    runtime_test!(test_distance_transforms, edt_and_chamfer_match_analytic);

    /// The separable feature transform finds a nearest background pixel (by distance: ties may
    /// pick another of the same distance) on random masks, 2-D and 3-D, anisotropic spacing.
    #[test]
    fn feature_transform_matches_brute_force() {
        for (dims, spacing) in [(vec![23usize, 31], vec![1.0, 1.0]), (vec![9usize, 14, 11], vec![2.0, 1.0, 0.5])] {
            let n: usize = dims.iter().product();
            let shape = Shape::new(dims.clone());
            let strides = shape.strides();
            let fg: Vec<bool> = (0..n).map(|i| ((i * 2_654_435_761usize) >> 7) % 10 < 8).collect();
            let got = feature_transform(&fg, &dims, &strides, &spacing, n);
            let coord = |i: usize| -> Vec<f64> { (0..dims.len()).map(|d| ((i / strides[d]) % dims[d]) as f64 * spacing[d]).collect() };
            let d2 = |a: usize, b: usize| coord(a).iter().zip(coord(b)).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
            for i in 0..n {
                let best = (0..n).filter(|&j| !fg[j]).map(|j| d2(i, j)).fold(f64::INFINITY, f64::min);
                assert!(!fg[got[i] as usize], "pixel {i}: its site {} is not background", got[i]);
                assert!((d2(i, got[i] as usize) - best).abs() < 1e-9, "pixel {i} of {dims:?}: {} vs {best}", d2(i, got[i] as usize));
            }
        }
    }
}

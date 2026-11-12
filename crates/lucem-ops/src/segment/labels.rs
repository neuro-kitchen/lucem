//! Label-map post-processing utilities: Euclidean label dilation (`expand_labels`), border object
//! removal (`clear_border`), boundary pixel detection (`find_boundaries`), boundary overlay
//! rendering (`mark_boundaries`), and label map intersection (`join_segmentations`).

use std::collections::BTreeSet;

use cubecl::prelude::*;
use lucem_core::compute::LaunchGeometry;
use lucem_core::element::{cast, ImageFloat};
use lucem_core::memory::DeviceArray;
use lucem_core::Shape;

use super::kernels::{
    expand_labels_gather_kernel, find_boundaries_2d_kernel, join_pair_key_kernel,
    mark_boundaries_gray_kernel, mark_boundaries_rgb_kernel,
};
use crate::core::components::relabel_sequential;
use crate::core::distance::{euclidean_distance, nearest_background_indices};
use crate::core::{check, elementwise, reduce};
use crate::error::{OpsError, OpsResult};

/// Boundary localization mode for [`find_boundaries`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryMode {
    /// Both inner and outer boundary pixels around label transitions.
    #[default]
    Thick = 0,
    /// Foreground pixels (`!= background`) adjacent to a different label.
    Inner = 1,
    /// Pixels adjacent to a different foreground label.
    Outer = 2,
}

/// Dilates positive labels in `labels` into background (`0`) pixels by at most `distance`
/// Euclidean distance without overwriting existing labels, using
/// [`crate::core::distance::nearest_background_indices`] on the inverted foreground mask.
pub fn expand_labels(
    client: &Client,
    labels: &DeviceArray<u32>,
    distance: f64,
    spacing: Option<&[f64]>,
) -> OpsResult<DeviceArray<u32>> {
    if distance < 0.0 || !distance.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "distance",
            reason: format!("distance must be finite and >= 0, got {distance}"),
        });
    }
    if labels.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, labels.shape().clone())?);
    }

    // Invert foreground/background so labeled pixels (`!= 0`) become `0` ("background" of the
    // inverted mask) and unlabeled pixels (`== 0`) become `1` ("foreground" to be queried).
    let host_lbl = labels.to_host(client);
    let inv: Vec<u32> = host_lbl.iter().map(|&l| u32::from(l == 0)).collect();
    let inv_dev = DeviceArray::from_host(client, labels.shape().clone(), inv)?;

    let nearest_idx = nearest_background_indices(client, &inv_dev, spacing)?;
    let edt = euclidean_distance::<u32, f32>(client, &inv_dev, spacing)?;
    let output = DeviceArray::<u32>::empty(client, labels.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, labels.len());
    unsafe {
        expand_labels_gather_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            labels.binding(),
            nearest_idx.binding(),
            edt.binding(),
            output.binding(),
            distance as f32,
            labels.len() as u32,
        );
    }
    Ok(output)
}

/// Clears any labeled region (`!= bgval`) that touches the outer `buffer_size`-pixel border of
/// `labels`, setting those regions to `bgval`.
pub fn clear_border(
    client: &Client,
    labels: &DeviceArray<u32>,
    buffer_size: usize,
    bgval: u32,
) -> OpsResult<DeviceArray<u32>> {
    if labels.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, labels.shape().clone())?);
    }
    let dims = labels.dims();
    if dims.iter().any(|&d| buffer_size * 2 >= d) {
        return Err(OpsError::InvalidParameter {
            name: "buffer_size",
            reason: format!("buffer_size {buffer_size} must be < half of every dimension {dims:?}"),
        });
    }

    let mut host = labels.to_host(client);
    let strides = labels.shape().strides();
    let ndim = dims.len();
    let mut border_labels = BTreeSet::new();

    for (idx, &lbl) in host.iter().enumerate() {
        if lbl == bgval {
            continue;
        }
        let mut on_border = false;
        for d in 0..ndim {
            let c = (idx / strides[d]) % dims[d];
            if c <= buffer_size || c + buffer_size >= dims[d] - 1 {
                on_border = true;
                break;
            }
        }
        if on_border {
            border_labels.insert(lbl);
        }
    }

    if !border_labels.is_empty() {
        for lbl in &mut host {
            if border_labels.contains(lbl) {
                *lbl = bgval;
            }
        }
    }

    Ok(DeviceArray::from_host(client, labels.shape().clone(), host)?)
}

/// Finds boundary pixels (`1` on boundaries, `0` elsewhere) between distinct regions in a 2-D
/// `labels` array.
pub fn find_boundaries(
    client: &Client,
    labels: &DeviceArray<u32>,
    connectivity: usize,
    mode: BoundaryMode,
    background: u32,
) -> OpsResult<DeviceArray<u32>> {
    check::ndim_in("find_boundaries", labels.shape(), &[2], "2")?;
    if connectivity != 1 && connectivity != 2 {
        return Err(OpsError::InvalidParameter {
            name: "connectivity",
            reason: format!("expected connectivity 1 or 2, got {connectivity}"),
        });
    }
    let output = DeviceArray::<u32>::zeros(client, labels.shape().clone())?;
    if labels.is_empty() {
        return Ok(output);
    }
    let (h, w) = (labels.dims()[0], labels.dims()[1]);
    let geom = LaunchGeometry::elementwise(client, labels.len());
    unsafe {
        find_boundaries_2d_kernel::launch(
            client,
            geom.cube_count,
            geom.cube_dim,
            labels.binding(),
            output.binding(),
            h as u32,
            w as u32,
            connectivity as u32,
            mode as u32,
            background,
            labels.len() as u32,
        );
    }
    Ok(output)
}

/// Returns an RGB `[h, w, 3]` visualization with boundaries of `labels` drawn in `color` on top of
/// a 2-D grayscale `[h, w]` or RGB `[h, w, 3]` `image`.
pub fn mark_boundaries<F: ImageFloat>(
    client: &Client,
    image: &DeviceArray<F>,
    labels: &DeviceArray<u32>,
    color: [f64; 3],
    mode: BoundaryMode,
) -> OpsResult<DeviceArray<F>> {
    check::ndim_in("labels", labels.shape(), &[2], "2")?;
    let (h, w) = (labels.dims()[0], labels.dims()[1]);
    let is_rgb = match image.ndim() {
        2 => {
            check::expect_shape("image", image.shape(), "labels", labels.shape())?;
            false
        }
        3 => {
            if image.dims() != [h, w, 3] {
                return Err(OpsError::ShapeMismatch {
                    first_name: "image",
                    first: image.dims().to_vec(),
                    second_name: "expected [h, w, 3]",
                    second: vec![h, w, 3],
                });
            }
            true
        }
        _ => {
            return Err(OpsError::InvalidDimension {
                what: "mark_boundaries",
                expected: "2 or 3",
                actual: image.ndim(),
                shape: image.dims().to_vec(),
            });
        }
    };

    let boundaries = find_boundaries(client, labels, 1, mode, 0)?;
    let output = DeviceArray::<F>::empty(client, Shape::new([h, w, 3]))?;
    let num_pixels = h * w;
    if num_pixels == 0 {
        return Ok(output);
    }
    let geom = LaunchGeometry::elementwise(client, num_pixels);
    let cr = cast::<F>(color[0]);
    let cg = cast::<F>(color[1]);
    let cb = cast::<F>(color[2]);

    unsafe {
        if is_rgb {
            mark_boundaries_rgb_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                image.binding(),
                boundaries.binding(),
                output.binding(),
                cr,
                cg,
                cb,
                num_pixels as u32,
            );
        } else {
            mark_boundaries_gray_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                image.binding(),
                boundaries.binding(),
                output.binding(),
                cr,
                cg,
                cb,
                num_pixels as u32,
            );
        }
    }
    Ok(output)
}

/// Computes the common refinement (overlay intersection) of two label maps `s1` and `s2`,
/// sequentially relabeling every non-zero combined pair into `1..=num_joined`.
pub fn join_segmentations(
    client: &Client,
    s1: &DeviceArray<u32>,
    s2: &DeviceArray<u32>,
) -> OpsResult<DeviceArray<u32>> {
    check::same_shape("s1", s1, "s2", s2)?;
    if s1.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, s1.shape().clone())?);
    }
    let s2_f = elementwise::cast::<u32, f32>(client, s2)?;
    let max_s2 = (reduce::sum_min_max(client, &s2_f)?.max as u32).saturating_add(1);
    let combined = DeviceArray::<u32>::empty(client, s1.shape().clone())?;
    let geom = LaunchGeometry::elementwise(client, s1.len());
    unsafe {
        join_pair_key_kernel::launch(
            client,
            geom.cube_count,
            geom.cube_dim,
            s1.binding(),
            s2.binding(),
            combined.binding(),
            max_s2,
            s1.len() as u32,
        );
    }
    let (relabeled, _) = relabel_sequential(client, &combined, 1)?;
    Ok(relabeled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn label_utilities_expand_clear_and_join(client: &Client) {
        #[rustfmt::skip]
        let lbl = vec![
            1u32, 0, 0, 0,
            0,    2, 0, 0,
            0,    0, 0, 0,
            0,    0, 0, 0,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([4, 4]), lbl).unwrap();

        // clear_border removes label 1 (on border) and preserves label 2 (interior)
        let cleared = clear_border(client, &arr, 0, 0).unwrap().to_host(client);
        assert_eq!(cleared[0], 0, "{}", client.name());
        assert_eq!(cleared[5], 2);

        // expand_labels by 1.0 dilates label 2 to its 4-neighbors
        let cleared_dev = DeviceArray::from_host(client, Shape::new([4, 4]), cleared).unwrap();
        let expanded = expand_labels(client, &cleared_dev, 1.0, None).unwrap().to_host(client);
        assert_eq!(expanded[1], 2);
        assert_eq!(expanded[4], 2);
        assert_eq!(expanded[5], 2);
        assert_eq!(expanded[6], 2);
        assert_eq!(expanded[9], 2);

        // find_boundaries and mark_boundaries
        let bounds = find_boundaries(client, &cleared_dev, 1, BoundaryMode::Inner, 0)
            .unwrap()
            .to_host(client);
        assert_eq!(bounds[5], 1);
        assert_eq!(bounds[0], 0);

        let gray = DeviceArray::<f32>::zeros(client, Shape::new([4, 4])).unwrap();
        let marked = mark_boundaries(client, &gray, &cleared_dev, [1.0, 0.5, 0.25], BoundaryMode::Inner)
            .unwrap()
            .to_host(client);
        assert_eq!(marked.len(), 4 * 4 * 3);
        assert!((marked[5 * 3] - 1.0).abs() < 1e-6);

        // join_segmentations
        let joined = join_segmentations(client, &arr, &cleared_dev).unwrap().to_host(client);
        assert!(joined[5] > 0);
    }
    runtime_test!(test_segment_labels, label_utilities_expand_clear_and_join);
}

//! Dual-threshold connected hysteresis binarization.

use std::collections::BTreeSet;

use cubecl::prelude::*;
use lumen_core::element::ImageFloat;
use lumen_core::memory::DeviceArray;

use crate::core::{components, elementwise, StructuringElement};
use crate::error::OpsResult;

/// Applies hysteresis thresholding with `low` and `high` thresholds: pixels `> high` are strong
/// foreground, and pixels `> low` connected (with full connectivity) to at least one strong pixel
/// are retained (`1u32`, else `0u32`).
pub fn hysteresis_threshold<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    low: f64,
    high: f64,
) -> OpsResult<DeviceArray<u32>> {
    if input.is_empty() {
        return Ok(DeviceArray::<u32>::zeros(client, input.shape().clone())?);
    }
    let lo = low.min(high);
    let hi = low.max(high);
    let weak = DeviceArray::<u32>::empty(client, input.shape().clone())?;
    let strong = DeviceArray::<u32>::empty(client, input.shape().clone())?;
    elementwise::compare_threshold_scalar_into(client, input, lo, &weak, false)?;
    elementwise::compare_threshold_scalar_into(client, input, hi, &strong, false)?;

    let se = StructuringElement::connectivity(input.ndim(), input.ndim())?;
    let cc = components::label_components(client, &weak, Some(&se))?;
    if cc.num_labels == 0 {
        return Ok(weak);
    }

    let labels_host = cc.labels.to_host(client);
    let strong_host = strong.to_host(client);
    let mut active_labels = BTreeSet::new();
    for (&lbl, &s) in labels_host.iter().zip(strong_host.iter()) {
        if lbl > 0 && s > 0 {
            active_labels.insert(lbl);
        }
    }

    let out: Vec<u32> = labels_host
        .into_iter()
        .map(|l| u32::from(l > 0 && active_labels.contains(&l)))
        .collect();
    Ok(DeviceArray::from_host(client, input.shape().clone(), out)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;
    use lumen_core::Shape;

    fn hysteresis_retains_connected_weak_and_drops_isolated(client: &Client) {
        #[rustfmt::skip]
        let img = vec![
            0.0f32, 0.4, 0.9, 0.4,
            0.0,    0.0, 0.0, 0.0,
            0.4,    0.4, 0.0, 0.0,
        ];
        let arr = DeviceArray::from_host(client, Shape::new([3, 4]), img).unwrap();
        let mask = hysteresis_threshold(client, &arr, 0.3, 0.8).unwrap().to_host(client);
        #[rustfmt::skip]
        let want = vec![
            0u32, 1, 1, 1,
            0,    0, 0, 0,
            0,    0, 0, 0,
        ];
        assert_eq!(mask, want, "{}", client.name());
    }
    runtime_test!(test_hysteresis_threshold, hysteresis_retains_connected_weak_and_drops_isolated);
}

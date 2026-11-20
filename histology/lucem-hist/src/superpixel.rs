//! Superpixels: SLIC (on the GPU through `lucem-ops`) cuts an image into pieces of about a given
//! size that follow its edges; each becomes a tile object with its measurements (area, mean and
//! std dev of every channel), ready for an object classifier — a classification by region
//! without detecting cells.

use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::DeviceArray;
use lucem_core::Shape;
use lucem_ops::segment::{slic, SlicOptions};
use serde::{Deserialize, Serialize};

use crate::error::{invalid, HistResult};
use crate::objects::{HistObject, ObjectKind};
use crate::raster::Grid;
use crate::polygon;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuperpixelSettings {
    /// About how wide a superpixel is (µm).
    pub spacing_um: f64,
    /// Higher: more square pieces; lower: closer to the edges.
    pub compactness: f64,
    pub iterations: usize,
}

impl Default for SuperpixelSettings {
    fn default() -> Self {
        Self { spacing_um: 50.0, compactness: 0.1, iterations: 10 }
    }
}

/// The superpixels of `channels` (SLIC runs on their mean) lying on the slide as `grid`, and
/// their tile objects; `pixel_um` µm per level-0 pixel.
pub fn superpixels(client: &Client, channels: &[(String, &[f32])], grid: Grid, pixel_um: f64, s: &SuperpixelSettings) -> HistResult<(Vec<u32>, Vec<HistObject>)> {
    let (w, h, origin, ds) = (grid.width, grid.height, grid.origin, grid.downsample);
    if channels.is_empty() || channels.iter().any(|(_, c)| c.len() != w * h) {
        return Err(invalid(format!("superpixels need channels of {w} × {h} values")));
    }
    let n = w * h;
    let shape = Shape::new([h, w]);
    let w_ch = 1.0 / channels.len() as f64;
    let mut dev = DeviceArray::<f32>::empty(client, shape.clone())?;
    let first = DeviceArray::from_host(client, shape.clone(), channels[0].1.to_vec())?;
    lucem_ops::core::elementwise::scale_offset_into(client, &first, &dev, w_ch, 0.0)?;
    for (_, ch) in &channels[1..] {
        let ch_dev = DeviceArray::from_host(client, shape.clone(), ch.to_vec())?;
        let next = DeviceArray::<f32>::empty(client, shape.clone())?;
        lucem_ops::core::elementwise::linear_combination_into(client, &dev, &ch_dev, &next, 1.0, w_ch, 0.0)?;
        dev = next;
    }
    let spacing_px = (s.spacing_um / (pixel_um * ds)).max(2.0);
    let segments = ((n as f64) / (spacing_px * spacing_px)).round().max(1.0) as usize;
    let raw = slic(client, &dev, &SlicOptions { n_segments: segments, compactness: s.compactness, max_iter: s.iterations })?.to_host(client);
    // Labels from 1, in order of appearance
    let mut map = std::collections::HashMap::new();
    let labels: Vec<u32> = raw
        .iter()
        .map(|l| {
            let next = map.len() as u32 + 1;
            *map.entry(*l).or_insert(next)
        })
        .collect();
    let k = map.len();
    let mut acc = vec![(0.0f64, 0.0f64, 0.0f64); (k + 1) * channels.len()];
    let mut count = vec![0usize; k + 1];
    for i in 0..n {
        let l = labels[i] as usize;
        count[l] += 1;
        for (c, (_, ch)) in channels.iter().enumerate() {
            let v = ch[i] as f64;
            let a = &mut acc[l * channels.len() + c];
            a.0 += 1.0;
            a.1 += v;
            a.2 += v * v;
        }
    }
    let px_area = (pixel_um * ds).powi(2);
    let objects = polygon::trace_labels(&labels, w, h, origin, ds)
        .into_iter()
        .map(|(l, poly)| {
            let mut o = HistObject::new(l, ObjectKind::Tile, poly);
            o.measurements.insert("Area µm^2".into(), count[l as usize] as f64 * px_area);
            for (c, (name, _)) in channels.iter().enumerate() {
                let a = acc[l as usize * channels.len() + c];
                let mean = a.1 / a.0.max(1.0);
                o.measurements.insert(format!("{name}: Mean"), mean);
                o.measurements.insert(format!("{name}: Std dev"), (a.2 / a.0.max(1.0) - mean * mean).max(0.0).sqrt());
            }
            o
        })
        .collect();
    Ok((labels, objects))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuts_an_image_into_tiles(client: &Client) {
        let (w, h) = (64, 64);
        let img: Vec<f32> = (0..w * h).map(|i| if i % w < 32 { 0.1 } else { 0.9 }).collect();
        let ch = [("Grey".to_string(), img.as_slice())];
        let (labels, tiles) = superpixels(client, &ch, Grid::new(w, h, [0.0, 0.0], 1.0), 1.0, &SuperpixelSettings { spacing_um: 16.0, ..SuperpixelSettings::default() }).unwrap();
        assert!(tiles.len() >= 8, "{}: {} tiles", client.name(), tiles.len());
        assert!(labels.iter().all(|&l| l > 0));
        // No tile mixes the two halves much: its std dev stays small
        let mixed = tiles.iter().filter(|t| t.get("Grey: Std dev").unwrap() > 0.2).count();
        assert!(mixed * 4 <= tiles.len(), "{}: {mixed} mixed of {}", client.name(), tiles.len());
    }

    lucem_core::runtime_test!(superpixels_become_tiles, cuts_an_image_into_tiles);
}

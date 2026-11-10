//! Pixel classifiers: every pixel of an image (at an analysis resolution) gets a class from
//! features of its neighbourhood at several scales — Gaussian-smoothed values, gradient
//! magnitude and Laplacian of Gaussian of each channel, computed on the GPU through
//! `lumen-ops` — and Random Trees trained on pixels of regions drawn and given a class. The
//! class map then measures areas ([`crate::area`]) or becomes regions ([`class_regions`]).

use lumen_core::cubecl::prelude::Client;
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;
use lumen_ops::core::PadMode;
use lumen_ops::filter::{gaussian_blur, gaussian_grad_mag, gaussian_laplace, GaussianBlur};
use serde::{Deserialize, Serialize};

use crate::error::{invalid, HistResult};
use crate::forest::{ForestSettings, RandomForest};
use crate::objects::Polygon;
use crate::raster::Grid;
use crate::{polygon, raster};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PixelFeatures {
    /// Scales (pixels of the analysed image).
    pub sigmas: Vec<f64>,
    pub smoothed: bool,
    pub gradient: bool,
    pub laplacian: bool,
}

impl Default for PixelFeatures {
    fn default() -> Self {
        Self { sigmas: vec![1.0, 2.0, 4.0], smoothed: true, gradient: true, laplacian: true }
    }
}

/// The feature planes of `channels` (each `w × h`): names and values.
pub fn features(client: &Client, channels: &[(String, &[f32])], w: usize, h: usize, f: &PixelFeatures) -> HistResult<Vec<(String, Vec<f32>)>> {
    let shape = Shape::new([h, w]);
    let mut out = Vec::new();
    for (name, values) in channels {
        if values.len() != w * h {
            return Err(invalid(format!("channel {name} does not hold {w} × {h} values")));
        }
        let dev = DeviceArray::from_host(client, shape.clone(), values.to_vec())?;
        for &s in &f.sigmas {
            if f.smoothed {
                out.push((format!("{name}: Gaussian σ {s}"), gaussian_blur(client, &dev, &[s, s], &GaussianBlur::default())?.to_host(client)));
            }
            if f.gradient {
                out.push((format!("{name}: Gradient magnitude σ {s}"), gaussian_grad_mag(client, &dev, &[s, s], PadMode::Reflect, 0.0, 4.0)?.to_host(client)));
            }
            if f.laplacian {
                out.push((format!("{name}: Laplacian of Gaussian σ {s}"), gaussian_laplace(client, &dev, &[s, s], PadMode::Reflect, 0.0, 4.0)?.to_host(client)));
            }
        }
    }
    if out.is_empty() {
        return Err(invalid("no feature chosen"));
    }
    Ok(out)
}

/// How a pixel classifier learns: its features, its trees, and at most how many pixels of each
/// class (evenly spread) it learns from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PixelTraining {
    pub features: PixelFeatures,
    pub forest: ForestSettings,
    pub max_per_class: usize,
}

impl Default for PixelTraining {
    fn default() -> Self {
        Self { features: PixelFeatures::default(), forest: ForestSettings::default(), max_per_class: 5_000 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PixelClassifier {
    pub features: PixelFeatures,
    /// The channels it was trained on, in order (the same must be given to classify).
    pub channels: Vec<String>,
    pub classes: Vec<String>,
    pub forest: RandomForest,
}

/// Trains on the pixels `labels` marks (`0`: none, `k`: `classes[k − 1]`).
pub fn train(client: &Client, channels: &[(String, &[f32])], w: usize, h: usize, labels: &[u8], classes: &[String], t: &PixelTraining) -> HistResult<PixelClassifier> {
    let (f, max_per_class) = (&t.features, t.max_per_class);
    if labels.len() != w * h {
        return Err(invalid("the training labels do not cover the image"));
    }
    let planes = features(client, channels, w, h, f)?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    for k in 1..=classes.len() {
        let px: Vec<usize> = (0..w * h).filter(|&i| labels[i] as usize == k).collect();
        let step = px.len().div_ceil(max_per_class.max(1)).max(1);
        for &i in px.iter().step_by(step) {
            x.push(planes.iter().map(|(_, p)| p[i] as f64).collect::<Vec<_>>());
            y.push(k - 1);
        }
    }
    let present = y.iter().copied().collect::<std::collections::BTreeSet<_>>().len();
    if present < 2 {
        return Err(invalid("draw regions of at least two classes to train a pixel classifier"));
    }
    let forest = RandomForest::train(&x, &y, classes.len(), &t.forest)?;
    Ok(PixelClassifier { features: f.clone(), channels: channels.iter().map(|(n, _)| n.clone()).collect(), classes: classes.to_vec(), forest })
}

/// The class map of an image (`k`: `classes[k − 1]`).
pub fn classify(client: &Client, c: &PixelClassifier, channels: &[(String, &[f32])], w: usize, h: usize) -> HistResult<Vec<u8>> {
    let names: Vec<&String> = channels.iter().map(|(n, _)| n).collect();
    if names.len() != c.channels.len() || names.iter().zip(&c.channels).any(|(a, b)| *a != b) {
        return Err(invalid(format!("the classifier was trained on {:?}", c.channels)));
    }
    let planes = features(client, channels, w, h, &c.features)?;
    let mut row = vec![0f64; planes.len()];
    Ok((0..w * h)
        .map(|i| {
            for (r, (_, p)) in row.iter_mut().zip(&planes) {
                *r = p[i] as f64;
            }
            (c.forest.predict(&row).0 + 1) as u8
        })
        .collect())
}

/// The regions of one class of a class map lying on the slide as `grid` (pieces under
/// `min_area_px` map pixels dropped, holes under `min_hole_px` filled), as level-0 outlines.
pub fn class_regions(map: &[u8], grid: Grid, class: u8, min_area_px: f64, min_hole_px: f64) -> Vec<Polygon> {
    let (w, h, origin, ds) = (grid.width, grid.height, grid.origin, grid.downsample);
    let mut mask: Vec<u8> = map.iter().map(|&c| u8::from(c == class)).collect();
    raster::clean(&mut mask, w, h, min_area_px, min_hole_px);
    let (labels, _) = raster::components(&mask, w, h);
    polygon::trace_labels(&labels, w, h, origin, ds).into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Left half smooth and dim, right half bright with texture.
    fn trains_and_classifies_two_textures(client: &Client) {
        let (w, h) = (48, 32);
        let img: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.2 } else if (i / w + i % w) % 2 == 0 { 0.9 } else { 0.5 }).collect();
        let mut labels = vec![0u8; w * h];
        for y in 4..10 {
            for x in 2..10 {
                labels[y * w + x] = 1;
                labels[y * w + x + 30] = 2;
            }
        }
        let ch = [("Grey".to_string(), img.as_slice())];
        let classes = ["Background".to_string(), "Tissue".to_string()];
        let c = train(client, &ch, w, h, &labels, &classes, &PixelTraining { forest: ForestSettings { trees: 10, ..ForestSettings::default() }, max_per_class: 500, ..PixelTraining::default() }).unwrap();
        let map = classify(client, &c, &ch, w, h).unwrap();
        let right = (0..w * h).filter(|&i| (i % w) >= 4 && (i % w) < w - 4 && (map[i] == 2) == (i % w >= w / 2)).count();
        let total = (0..w * h).filter(|&i| (i % w) >= 4 && (i % w) < w - 4).count();
        assert!(right as f64 / total as f64 > 0.9, "{}: {right} / {total}", client.name());
        let regions = class_regions(&map, Grid::new(w, h, [0.0, 0.0], 4.0), 2, 20.0, 20.0);
        assert_eq!(regions.len(), 1, "{}", client.name());
    }

    lumen_core::runtime_test!(pixel_classifier_learns_two_textures, trains_and_classifies_two_textures);
}

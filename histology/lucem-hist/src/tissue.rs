//! Tissue detection: on a low-resolution image of the whole slide, a smoothed value per pixel
//! (how much stain for brightfield, how bright for fluorescence) above a threshold (Otsu's when
//! none is given), pieces and holes smaller than given areas dropped and filled, then outlined.

use serde::{Deserialize, Serialize};

use crate::objects::{HistObject, ObjectKind};
use crate::raster::Grid;
use crate::{polygon, raster, stains};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TissueSettings {
    /// Brightfield (tissue absorbs: the optical density sum) or fluorescence (tissue shines: the
    /// mean of the channels).
    pub brightfield: bool,
    /// The threshold of the smoothed value; `None`: Otsu's.
    pub threshold: Option<f64>,
    /// Smoothing (µm).
    pub sigma_um: f64,
    /// Pieces under this area go, holes under this area fill (µm²).
    pub min_area_um2: f64,
    pub min_hole_um2: f64,
}

impl Default for TissueSettings {
    fn default() -> Self {
        Self { brightfield: true, threshold: None, sigma_um: 20.0, min_area_um2: 10_000.0, min_hole_um2: 5_000.0 }
    }
}

/// The tissue of a low-resolution image: its mask, the threshold used, and its regions.
#[derive(Debug, Clone, PartialEq)]
pub struct Tissue {
    pub mask: Vec<u8>,
    pub threshold: f64,
    pub regions: Vec<HistObject>,
}

/// The tissue of an interleaved image (`channels` values per pixel; RGB 0–255 for brightfield)
/// lying on the slide as `grid`, `pixel_um` µm per level-0 pixel; regions in level-0 pixels,
/// with their area (`Area µm^2`).
pub fn detect(image: &[f32], channels: usize, grid: Grid, pixel_um: f64, s: &TissueSettings) -> Tissue {
    let (w, h, origin, ds) = (grid.width, grid.height, grid.origin, grid.downsample);
    let values: Vec<f32> = if s.brightfield && channels >= 3 {
        let rgb: Vec<u8> = image.chunks_exact(channels).flat_map(|p| [p[0], p[1], p[2]].map(|v| v.clamp(0.0, 255.0) as u8)).collect();
        let bg = stains::estimate_background(&rgb);
        stains::od_sum(&rgb, bg)
    } else {
        image.chunks_exact(channels.max(1)).map(|p| p.iter().sum::<f32>() / p.len() as f32).collect()
    };
    let um_per_px = pixel_um * ds;
    let smooth = raster::gaussian(&values, w, h, s.sigma_um / um_per_px);
    let threshold = s.threshold.unwrap_or_else(|| raster::otsu(&smooth) as f64);
    let mut mask: Vec<u8> = smooth.iter().map(|&v| u8::from(v as f64 > threshold)).collect();
    let px_area = um_per_px * um_per_px;
    raster::clean(&mut mask, w, h, s.min_area_um2 / px_area, s.min_hole_um2 / px_area);
    let (labels, _) = raster::components(&mask, w, h);
    let regions = polygon::trace_labels(&labels, w, h, origin, ds)
        .into_iter()
        .enumerate()
        .map(|(i, (_, poly))| {
            let mut o = HistObject::new(i as u32 + 1, ObjectKind::Annotation, poly);
            o.class = Some("Tissue".into());
            o.measurements.insert("Area µm^2".into(), polygon::area(&o.roi) * pixel_um * pixel_um);
            o
        })
        .collect();
    Tissue { mask, threshold, regions }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stained_square_on_white_is_found() {
        let (w, h) = (64, 64);
        let mut img = vec![250f32; w * h * 3];
        for y in 16..48 {
            for x in 12..52 {
                let i = (y * w + x) * 3;
                img[i..i + 3].copy_from_slice(&[150.0, 90.0, 160.0]);
            }
        }
        // a speck of dust
        img[(3 * w + 3) * 3..(3 * w + 3) * 3 + 3].copy_from_slice(&[40.0, 40.0, 40.0]);
        let t = detect(&img, 3, Grid::new(w, h, [0.0, 0.0], 16.0), 0.5, &TissueSettings { sigma_um: 8.0, min_area_um2: 2_000.0, min_hole_um2: 1_000.0, ..TissueSettings::default() });
        assert_eq!(t.regions.len(), 1, "{:?}", t.regions.iter().map(|r| r.roi.len()).collect::<Vec<_>>());
        let area = t.regions[0].get("Area µm^2").unwrap();
        let want = (32.0 * 40.0) * (16.0 * 0.5f64).powi(2);
        assert!((area - want).abs() / want < 0.15, "{area} vs {want}");
    }
}

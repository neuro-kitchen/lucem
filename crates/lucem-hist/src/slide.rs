//! Whole-slide analysis: a region of any size is read tile by tile (with an overlap, so cells on
//! a tile's edge are whole in one of them), cells are detected in each, and a cell is kept by
//! the tile whose core (the tile without its overlap) holds its centroid — each cell once.
//! Inside a drawn region, only the cells whose centroid is in it are kept.
//!
//! The image's channels become what detection needs through [`Staining`]: brightfield RGB is
//! separated into stains (the first finds the nuclei, all are measured as optical densities);
//! fluorescence finds the nuclei on one channel and measures every channel.

use lucem_core::cubecl::prelude::Client;
use serde::{Deserialize, Serialize};

use crate::cells::{cells_from_nuclei, detect_cells, CellDetection, CellImage};
use crate::error::{invalid, HistResult};
use crate::objects::{HistObject, Polygon};
use crate::polygon;
use crate::stains::{deconvolve, StainVectors};

/// A tile ready for detection: the values nuclei are found on, and the channels measured.
pub type Prepared = (Vec<f32>, Vec<(String, Vec<f32>)>);

/// How the image's channels become a detection channel and the channels measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "staining", rename_all = "snake_case")]
pub enum Staining {
    /// RGB (the first three channels) separated with these stains; nuclei on the first stain.
    Brightfield { stains: StainVectors },
    /// Nuclei on channel `nuclear`; every channel measured under its name.
    Fluorescence { nuclear: usize },
}

impl Staining {
    /// The detection values and the measured channels of one tile, from its raw channels
    /// (`ranges`: each channel's full range, so 8-bit, 16-bit and float RGB all read as 0–255).
    pub fn prepare(&self, raw: &[Vec<f32>], names: &[String], ranges: &[f32]) -> HistResult<Prepared> {
        match self {
            Staining::Brightfield { stains } => {
                if raw.len() < 3 {
                    return Err(invalid("brightfield needs an RGB image (3 channels)"));
                }
                let n = raw[0].len();
                let scale: Vec<f32> = (0..3).map(|c| 255.0 / ranges.get(c).copied().filter(|r| *r > 0.0).unwrap_or(255.0)).collect();
                let rgb: Vec<u8> = (0..n).flat_map(|i| (0..3).map(move |c| (i, c))).map(|(i, c)| (raw[c][i] * scale[c]).round().clamp(0.0, 255.0) as u8).collect();
                let planes = deconvolve(&rgb, stains)?;
                let measured: Vec<(String, Vec<f32>)> = planes.iter().zip(&stains.stains).filter(|(_, s)| s.name != "Residual").map(|(p, s)| (format!("{} OD", s.name), p.clone())).collect();
                Ok((planes[0].clone(), measured))
            }
            Staining::Fluorescence { nuclear } => {
                let det = raw.get(*nuclear).ok_or_else(|| invalid(format!("no channel {nuclear}")))?.clone();
                Ok((det, raw.iter().zip(names).map(|(v, n)| (n.clone(), v.clone())).collect()))
            }
        }
    }
}

/// A detection threshold for a fluorescence channel from a coarse look at it: Otsu's, less the
/// median (the background) when a rolling ball takes the background away first. Returns the
/// threshold, Otsu's and the median.
pub fn auto_threshold(coarse: &[f32], background_subtracted: bool) -> (f64, f64, f64) {
    let otsu = crate::raster::otsu(coarse) as f64;
    let mut v: Vec<f32> = coarse.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return (0.0, otsu, 0.0);
    }
    v.sort_by(f32::total_cmp);
    let median = v[v.len() / 2] as f64;
    (if background_subtracted { (otsu - median).max(0.0) } else { otsu }, otsu, median)
}

/// A region of one pyramid level to analyse.
#[derive(Debug, Clone, PartialEq)]
pub struct SlideRegion {
    /// In pixels of the level read.
    pub x0: usize,
    pub y0: usize,
    pub width: usize,
    pub height: usize,
    /// Level-0 pixels per pixel of the level, and µm per level-0 pixel.
    pub downsample: f64,
    pub pixel_um: f64,
    /// Only cells whose centroid is inside (level-0 outline).
    pub within: Option<Polygon>,
}

/// Tiling of a region.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tiling {
    /// Tile side (pixels of the level read).
    pub tile: usize,
    /// Pixels shared with the neighbours on each side (more than a cell's diameter).
    pub overlap: usize,
}

impl Default for Tiling {
    fn default() -> Self {
        Self { tile: 1024, overlap: 64 }
    }
}

/// What reads the image: `(x0, y0, w, h)` in pixels of the level → one plane per channel.
pub type ReadTile<'a> = dyn FnMut(usize, usize, usize, usize) -> HistResult<Vec<Vec<f32>>> + 'a;

/// Nucleus labels of one tile from a model: the tile's raw channels, its width and height →
/// labels (`0`: none) of the same size.
pub type Labeller<'a> = dyn FnMut(&[Vec<f32>], usize, usize) -> HistResult<Vec<u32>> + 'a;

/// Detects the cells of a region, tile by tile; `progress(done, total)` after each tile. Cell
/// ids run `1..` over the whole region.
#[allow(clippy::too_many_arguments)]
pub fn detect_cells_in_region(client: &Client, read: &mut ReadTile, names: &[String], ranges: &[f32], staining: &Staining, region: &SlideRegion, settings: &CellDetection, tiling: Tiling, progress: &mut dyn FnMut(usize, usize)) -> HistResult<Vec<HistObject>> {
    cells_in_region(client, read, names, ranges, staining, region, settings, tiling, None, progress)
}

/// As [`detect_cells_in_region`], the nuclei found by `labeller` (a model) when given, else by
/// the threshold of `settings`; the cells grow from them and are measured the same way.
#[allow(clippy::too_many_arguments)]
pub fn cells_in_region(client: &Client, read: &mut ReadTile, names: &[String], ranges: &[f32], staining: &Staining, region: &SlideRegion, settings: &CellDetection, tiling: Tiling, mut labeller: Option<&mut Labeller>, progress: &mut dyn FnMut(usize, usize)) -> HistResult<Vec<HistObject>> {
    if region.width == 0 || region.height == 0 {
        return Ok(Vec::new());
    }
    let core = tiling.tile.saturating_sub(2 * tiling.overlap).max(16);
    let (nx, ny) = (region.width.div_ceil(core), region.height.div_ceil(core));
    let total = nx * ny;
    let mut out: Vec<HistObject> = Vec::new();
    for ty in 0..ny {
        for tx in 0..nx {
            // The core of this tile, and the tile read around it (level pixels)
            let (cx0, cy0) = (region.x0 + tx * core, region.y0 + ty * core);
            let (cx1, cy1) = ((cx0 + core).min(region.x0 + region.width), (cy0 + core).min(region.y0 + region.height));
            let (rx0, ry0) = (cx0.saturating_sub(tiling.overlap).max(region.x0.saturating_sub(tiling.overlap)), cy0.saturating_sub(tiling.overlap).max(region.y0.saturating_sub(tiling.overlap)));
            let (rx1, ry1) = (cx1 + tiling.overlap, cy1 + tiling.overlap);
            let (w, h) = (rx1 - rx0, ry1 - ry0);
            let raw = read(rx0, ry0, w, h)?;
            if raw.iter().any(|c| c.len() != w * h) {
                return Err(invalid("the image reader returned a tile of the wrong size"));
            }
            let (detection, channels) = staining.prepare(&raw, names, ranges)?;
            let img = CellImage { width: w, height: h, detection: &detection, channels: channels.iter().map(|(n, v)| (n.clone(), v.as_slice())).collect(), origin: [rx0 as f64 * region.downsample, ry0 as f64 * region.downsample], downsample: region.downsample, pixel_um: region.pixel_um };
            let found = match labeller.as_mut() {
                Some(l) => {
                    let nuclei = l(&raw, w, h)?;
                    if nuclei.len() != w * h {
                        return Err(invalid(format!("the model's labels hold {} pixels, not {w} × {h}", nuclei.len())));
                    }
                    cells_from_nuclei(client, &img, &nuclei, settings)?
                }
                None => detect_cells(client, &img, settings)?,
            };
            let ds = region.downsample;
            let (k0x, k0y, k1x, k1y) = (cx0 as f64 * ds, cy0 as f64 * ds, cx1 as f64 * ds, cy1 as f64 * ds);
            for mut o in found.objects {
                let [x, y] = o.centroid;
                if x < k0x || x >= k1x || y < k0y || y >= k1y {
                    continue;
                }
                if region.within.as_ref().is_some_and(|p| !polygon::contains(p, o.centroid)) {
                    continue;
                }
                o.id = out.len() as u32 + 1;
                out.push(o);
            }
            progress(ty * nx + tx + 1, total);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nuclei on a grid that tiles cut through: tiled and whole give the same cells.
    fn tiles_find_each_cell_once(client: &Client) {
        let (w, h) = (160usize, 120usize);
        let mut img = vec![0.02f32; w * h];
        let mut n = 0;
        for cy in (12..h - 8).step_by(22) {
            for cx in (12..w - 8).step_by(22) {
                n += 1;
                for y in 0..h {
                    for x in 0..w {
                        if (x as f64 + 0.5 - cx as f64).powi(2) + (y as f64 + 0.5 - cy as f64).powi(2) <= 25.0 {
                            img[y * w + x] = 1.0;
                        }
                    }
                }
            }
        }
        let s = CellDetection { background_radius_um: 0.0, sigma_um: 0.5, threshold: 0.5, min_area_um2: 3.0, max_area_um2: 60.0, cell_expansion_um: 1.0, ..CellDetection::default() };
        let region = SlideRegion { x0: 0, y0: 0, width: w, height: h, downsample: 1.0, pixel_um: 0.5, within: None };
        let mut read = |x0: usize, y0: usize, tw: usize, th: usize| -> HistResult<Vec<Vec<f32>>> {
            Ok(vec![(0..tw * th).map(|i| {
                let (x, y) = (x0 + i % tw, y0 + i / tw);
                if x < w && y < h { img[y * w + x] } else { 0.0 }
            }).collect()])
        };
        let staining = Staining::Fluorescence { nuclear: 0 };
        let names = ["DAPI".to_string()];
        let mut ticks = 0;
        let cells = detect_cells_in_region(client, &mut read, &names, &[1.0], &staining, &region, &s, Tiling { tile: 64, overlap: 12 }, &mut |_, _| ticks += 1).unwrap();
        assert!(ticks > 4);
        assert_eq!(cells.len(), n, "{}: {} cells, {n} nuclei", client.name(), cells.len());
        let areas: Vec<f64> = cells.iter().map(|c| c.get("Nucleus: Area µm^2").unwrap()).collect();
        assert!(areas.iter().all(|a| (a - areas[0]).abs() < 1.0), "{}: a cell was cut by a tile: {areas:?}", client.name());
        // Inside a drawn region only
        let within = SlideRegion { within: Some(vec![[0.0, 0.0], [60.0, 0.0], [60.0, 60.0], [0.0, 60.0]]), ..region };
        let some = detect_cells_in_region(client, &mut read, &names, &[1.0], &staining, &within, &s, Tiling { tile: 64, overlap: 12 }, &mut |_, _| {}).unwrap();
        assert_eq!(some.len(), 9, "{}", client.name());
    }

    lucem_core::runtime_test!(tiles_find_each_cell_once_test, tiles_find_each_cell_once);
}

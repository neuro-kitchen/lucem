//! The annotations of a volume as rasters, for the views that show more than the plane they were
//! drawn on: one plane across any axis (the 2D view) or every class over the whole box of a level
//! (the 3D view). An annotation belongs to its plane (`Annotation::z`, level-0 planes); one
//! without a plane belongs to every plane. Regions and strokes fill, eraser strokes clear, in
//! drawing order, as [`rasterize`] does on one plane.

use std::collections::BTreeSet;

use lucem_io::{LevelInfo, PlaneAxis};

use crate::domain::annotation::{rasterize, Annotation};

/// A class to draw: its name and colour.
pub type ClassColor = (String, [u8; 3]);

/// The plane of `level` that level-0 plane `z` falls in.
fn level_plane(z: usize, level: &LevelInfo) -> usize {
    ((z as f64 / level.z_downsample.max(1e-9)) as usize).min(level.depth.saturating_sub(1))
}

/// The level planes some annotation of `class` was drawn on, and whether one of them has no plane
/// (so it is on every plane).
fn planes_of(anns: &[Annotation], class: &str, level: &LevelInfo) -> (BTreeSet<usize>, bool) {
    let mut planes = BTreeSet::new();
    let mut everywhere = false;
    for a in anns.iter().filter(|a| a.class == class) {
        match a.z {
            Some(z) => {
                planes.insert(level_plane(z, level));
            }
            None => everywhere = true,
        }
    }
    (planes, everywhere)
}

/// The annotations of `class` on level plane `p` (with those on every plane), in drawing order.
fn on_plane<'a>(anns: &'a [Annotation], class: &str, level: &LevelInfo, p: usize) -> Vec<&'a Annotation> {
    anns.iter().filter(|a| a.class == class && a.z.is_none_or(|z| level_plane(z, level) == p)).collect()
}

/// For each level plane: the mask of `class` over the region `(x0, y0, w, h)` of that plane
/// (`None` where nothing is drawn). Planes without an annotation of their own share one raster of
/// the annotations on every plane.
fn class_planes(anns: &[Annotation], class: &str, level: &LevelInfo, (x0, y0, w, h): (usize, usize, usize, usize)) -> Vec<Option<Vec<u32>>> {
    let (planes, everywhere) = planes_of(anns, class, level);
    let shared = everywhere.then(|| {
        let all: Vec<&Annotation> = anns.iter().filter(|a| a.class == class && a.z.is_none()).collect();
        rasterize(&all, Some(class), x0, y0, w, h, level.downsample)
    });
    (0..level.depth)
        .map(|p| {
            if planes.contains(&p) {
                Some(rasterize(&on_plane(anns, class, level, p), Some(class), x0, y0, w, h, level.downsample))
            } else {
                shared.clone()
            }
        })
        .collect()
}

/// The annotations on one plane of `level` across `axis` at `index` (that level's pixels), as a
/// BGRA image with straight alpha (`alpha` where a class is, clear elsewhere), in the plane's
/// layout of `Pyramid::read_plane`: XY is `width × height`, XZ `width × depth`, YZ
/// `height × depth`. `None` when nothing is drawn there.
pub fn plane_overlay(anns: &[Annotation], classes: &[ClassColor], level: &LevelInfo, axis: PlaneAxis, index: usize, alpha: f32) -> Option<(Vec<u8>, usize, usize)> {
    if anns.is_empty() || classes.is_empty() {
        return None;
    }
    let (d, h, w) = (level.depth, level.height, level.width);
    let (ow, oh) = match axis {
        PlaneAxis::Z => (w, h),
        PlaneAxis::Y => (w, d),
        PlaneAxis::X => (h, d),
    };
    let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
    let mut out = vec![0u8; ow * oh * 4];
    let mut any = false;
    let mut put = |i: usize, c: [u8; 3]| {
        out[i * 4..i * 4 + 4].copy_from_slice(&[c[2], c[1], c[0], a]);
        any = true;
    };
    for (name, color) in classes {
        if !anns.iter().any(|a| &a.class == name) {
            continue;
        }
        match axis {
            PlaneAxis::Z => {
                let p = index.min(d.saturating_sub(1));
                let mask = rasterize(&on_plane(anns, name, level, p), Some(name), 0, 0, w, h, level.downsample);
                for (i, &m) in mask.iter().enumerate() {
                    if m != 0 {
                        put(i, *color);
                    }
                }
            }
            // One row (XZ) or one column (YZ) of every plane
            PlaneAxis::Y => {
                for (p, row) in class_planes(anns, name, level, (0, index.min(h.saturating_sub(1)), w, 1)).into_iter().enumerate() {
                    for (x, &m) in row.iter().flatten().enumerate() {
                        if m != 0 {
                            put(p * w + x, *color);
                        }
                    }
                }
            }
            PlaneAxis::X => {
                for (p, col) in class_planes(anns, name, level, (index.min(w.saturating_sub(1)), 0, 1, h)).into_iter().enumerate() {
                    for (y, &m) in col.iter().flatten().enumerate() {
                        if m != 0 {
                            put(p * h + y, *color);
                        }
                    }
                }
            }
        }
    }
    any.then_some((out, ow, oh))
}

/// Every class with annotations as one channel over the whole box of `level` (`[class, d, h, w]`,
/// 1 inside, 0 outside), with the colours of those classes. Empty when nothing is drawn.
pub fn volume_labels(anns: &[Annotation], classes: &[ClassColor], level: &LevelInfo) -> (Vec<f32>, Vec<[u8; 3]>) {
    let (d, h, w) = (level.depth, level.height, level.width);
    let mut values = Vec::new();
    let mut colors = Vec::new();
    for (name, color) in classes {
        if !anns.iter().any(|a| &a.class == name) {
            continue;
        }
        let start = values.len();
        values.resize(start + d * h * w, 0.0f32);
        for (p, mask) in class_planes(anns, name, level, (0, 0, w, h)).into_iter().enumerate() {
            let Some(mask) = mask else { continue };
            let plane = &mut values[start + p * h * w..start + (p + 1) * h * w];
            for (v, &m) in plane.iter_mut().zip(mask.iter()) {
                if m != 0 {
                    *v = 1.0;
                }
            }
        }
        colors.push(*color);
    }
    (values, colors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::annotation::Shape;

    fn level(d: usize, h: usize, w: usize) -> LevelInfo {
        LevelInfo { path: "test".into(), width: w, height: h, tile_w: w, tile_h: h, downsample: 1.0, depth: d, tile_d: 1, z_downsample: 1.0 }
    }

    fn rect(z: Option<usize>) -> Annotation {
        Annotation { id: "a".into(), image: "img".into(), shape: Shape::Rect { x: 2.0, y: 2.0, w: 4.0, h: 4.0 }, class: "Region".into(), note: String::new(), z }
    }

    #[test]
    fn an_annotation_shows_on_its_plane_only() {
        let l = level(5, 10, 10);
        let classes = vec![("Region".to_string(), [255, 0, 0])];
        let anns = vec![rect(Some(3))];
        assert!(plane_overlay(&anns, &classes, &l, PlaneAxis::Z, 3, 0.5).is_some());
        assert!(plane_overlay(&anns, &classes, &l, PlaneAxis::Z, 2, 0.5).is_none());
        // Across y = 4 (inside the rectangle): only row z = 3 of the XZ plane
        let (img, w, h) = plane_overlay(&anns, &classes, &l, PlaneAxis::Y, 4, 0.5).unwrap();
        assert_eq!((w, h), (10, 5));
        let lit: Vec<usize> = (0..w * h).filter(|i| img[i * 4 + 3] != 0).map(|i| i / w).collect();
        assert!(!lit.is_empty() && lit.iter().all(|&z| z == 3), "{lit:?}");
    }

    #[test]
    fn an_annotation_without_a_plane_fills_every_plane_of_the_volume() {
        let l = level(4, 10, 10);
        let classes = vec![("Region".to_string(), [0, 255, 0]), ("Other".to_string(), [0, 0, 255])];
        let (values, colors) = volume_labels(&[rect(None)], &classes, &l);
        assert_eq!(colors, vec![[0, 255, 0]], "only classes with annotations");
        assert_eq!(values.len(), 4 * 100);
        for p in 0..4 {
            assert_eq!(values[p * 100 + 4 * 10 + 4], 1.0, "plane {p}");
            assert_eq!(values[p * 100], 0.0);
        }
    }
}

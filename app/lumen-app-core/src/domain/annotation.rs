//! Annotations: shapes drawn on an image (regions, points, rulers, brush strokes), each of a
//! class. Coordinates are level-0 pixels of the image, so they hold at any zoom and pyramid level.
//!
//! A region class doubles as a mask: [`rasterize`] turns a class's shapes (filled regions, brush
//! strokes, erased by eraser strokes) into a mask over any region and level, which a pipeline reads
//! through an Annotations source.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A shape in level-0 pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "shape", rename_all = "snake_case", deny_unknown_fields)]
pub enum Shape {
    Rect { x: f64, y: f64, w: f64, h: f64 },
    Ellipse { cx: f64, cy: f64, rx: f64, ry: f64 },
    /// A closed polygon.
    Polygon { points: Vec<[f64; 2]> },
    /// A closed region drawn freehand.
    Freehand { points: Vec<[f64; 2]> },
    Point { x: f64, y: f64 },
    /// A measurement line (the ruler).
    Line { x0: f64, y0: f64, x1: f64, y1: f64 },
    /// A brush stroke of `radius` px; an eraser stroke (`erase`) clears its class's mask.
    Stroke {
        points: Vec<[f64; 2]>,
        radius: f64,
        #[serde(default)]
        erase: bool,
    },
}

/// One annotation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Annotation {
    /// Unique in the project (`a1`, `a2`, …); empty in a command means "give it one".
    #[serde(default)]
    pub id: String,
    /// The image it is drawn on (an image id of the project).
    pub image: String,
    pub shape: Shape,
    /// Its class (a name from the project's classes).
    pub class: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
    /// The plane of a volume it was drawn on (level-0 planes); `None`: every plane (2D images).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z: Option<usize>,
}

impl Annotation {
    /// Whether it shows on plane `z` (level-0 planes).
    pub fn on_plane(&self, z: usize) -> bool {
        self.z.is_none_or(|p| p == z)
    }
}

/// A class of annotations: a name and a colour.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClassDoc {
    pub name: String,
    pub color: [u8; 3],
    #[serde(default = "yes")]
    pub visible: bool,
    /// How opaque its annotations are drawn (0–1), times the overlays' opacity.
    #[serde(default = "opaque", skip_serializing_if = "is_opaque")]
    pub opacity: f32,
}

fn yes() -> bool {
    true
}

fn opaque() -> f32 {
    1.0
}

fn is_opaque(v: &f32) -> bool {
    *v >= 1.0
}

/// The classes a new project starts with (histology first).
pub fn default_classes() -> Vec<ClassDoc> {
    let c = |name: &str, color: [u8; 3]| ClassDoc { name: name.into(), color, visible: true, opacity: 1.0 };
    vec![
        c("Region", [255, 196, 0]),
        c("Tumor", [230, 70, 70]),
        c("Stroma", [80, 200, 120]),
        c("Nucleus", [80, 150, 255]),
        c("Immune", [190, 110, 240]),
        c("Ignore", [150, 150, 150]),
        c("Measurement", [255, 255, 255]),
    ]
}

impl Shape {
    /// What it is, for lists.
    pub fn kind(&self) -> &'static str {
        match self {
            Shape::Rect { .. } => "Rectangle",
            Shape::Ellipse { .. } => "Ellipse",
            Shape::Polygon { .. } => "Polygon",
            Shape::Freehand { .. } => "Freehand",
            Shape::Point { .. } => "Point",
            Shape::Line { .. } => "Ruler",
            Shape::Stroke { erase: false, .. } => "Brush",
            Shape::Stroke { erase: true, .. } => "Eraser",
        }
    }

    /// Whether it encloses an area (and so fills a mask).
    pub fn is_region(&self) -> bool {
        matches!(self, Shape::Rect { .. } | Shape::Ellipse { .. } | Shape::Polygon { .. } | Shape::Freehand { .. })
    }

    /// The outline of a region (closed), the path of a stroke or ruler.
    pub fn outline(&self) -> Vec<[f64; 2]> {
        match self {
            Shape::Rect { x, y, w, h } => vec![[*x, *y], [x + w, *y], [x + w, y + h], [*x, y + h]],
            Shape::Ellipse { cx, cy, rx, ry } => {
                let n = 72;
                (0..n).map(|i| {
                    let t = i as f64 / n as f64 * std::f64::consts::TAU;
                    [cx + rx * t.cos(), cy + ry * t.sin()]
                }).collect()
            }
            Shape::Polygon { points } | Shape::Freehand { points } | Shape::Stroke { points, .. } => points.clone(),
            Shape::Point { x, y } => vec![[*x, *y]],
            Shape::Line { x0, y0, x1, y1 } => vec![[*x0, *y0], [*x1, *y1]],
        }
    }

    /// `(x0, y0, x1, y1)` around it.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let pad = match self {
            Shape::Stroke { radius, .. } => *radius,
            _ => 0.0,
        };
        let pts = self.outline();
        let mut b = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for [x, y] in pts {
            b = (b.0.min(x), b.1.min(y), b.2.max(x), b.3.max(y));
        }
        (b.0 - pad, b.1 - pad, b.2 + pad, b.3 + pad)
    }

    /// Area in px² (regions).
    pub fn area(&self) -> Option<f64> {
        match self {
            Shape::Rect { w, h, .. } => Some((w * h).abs()),
            Shape::Ellipse { rx, ry, .. } => Some(std::f64::consts::PI * rx.abs() * ry.abs()),
            Shape::Polygon { points } | Shape::Freehand { points } => Some(polygon_area(points)),
            // A brush stroke covers what it painted (an eraser stroke covers nothing)
            Shape::Stroke { points, radius, erase: false } => Some(stroke_area(points, *radius)),
            _ => None,
        }
    }

    /// Perimeter of a region, length of a ruler or stroke (px).
    pub fn length(&self) -> Option<f64> {
        match self {
            Shape::Point { .. } => None,
            Shape::Line { x0, y0, x1, y1 } => Some((x1 - x0).hypot(y1 - y0)),
            Shape::Stroke { points, .. } => Some(path_length(points, false)),
            Shape::Ellipse { rx, ry, .. } => {
                // Ramanujan's approximation
                let (a, b) = (rx.abs(), ry.abs());
                Some(std::f64::consts::PI * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt()))
            }
            s => Some(path_length(&s.outline(), true)),
        }
    }

    /// Moved by `(dx, dy)`.
    pub fn translated(&self, dx: f64, dy: f64) -> Shape {
        let mv = |p: &Vec<[f64; 2]>| p.iter().map(|[x, y]| [x + dx, y + dy]).collect();
        match self {
            Shape::Rect { x, y, w, h } => Shape::Rect { x: x + dx, y: y + dy, w: *w, h: *h },
            Shape::Ellipse { cx, cy, rx, ry } => Shape::Ellipse { cx: cx + dx, cy: cy + dy, rx: *rx, ry: *ry },
            Shape::Polygon { points } => Shape::Polygon { points: mv(points) },
            Shape::Freehand { points } => Shape::Freehand { points: mv(points) },
            Shape::Point { x, y } => Shape::Point { x: x + dx, y: y + dy },
            Shape::Line { x0, y0, x1, y1 } => Shape::Line { x0: x0 + dx, y0: y0 + dy, x1: x1 + dx, y1: y1 + dy },
            Shape::Stroke { points, radius, erase } => Shape::Stroke { points: mv(points), radius: *radius, erase: *erase },
        }
    }

    /// Whether level-0 point `(x, y)` hits it, `tol` px around lines and points.
    pub fn hit(&self, x: f64, y: f64, tol: f64) -> bool {
        let (bx0, by0, bx1, by1) = self.bounds();
        if x < bx0 - tol || x > bx1 + tol || y < by0 - tol || y > by1 + tol {
            return false;
        }
        match self {
            s if s.is_region() => point_in_polygon(&s.outline(), x, y) || near_path(&s.outline(), true, x, y, tol),
            Shape::Point { x: px, y: py } => (x - px).hypot(y - py) <= tol * 1.5,
            Shape::Stroke { points, radius, .. } => near_path(points, false, x, y, tol + radius),
            s => near_path(&s.outline(), false, x, y, tol),
        }
    }

    /// Checks the shape makes sense (finite, not degenerate).
    pub fn validate(&self) -> Result<(), String> {
        let pts = self.outline();
        if pts.iter().any(|[x, y]| !x.is_finite() || !y.is_finite()) {
            return Err("annotation coordinates must be numbers".into());
        }
        match self {
            Shape::Polygon { points } | Shape::Freehand { points } if points.len() < 3 => Err("a region needs at least 3 points".into()),
            Shape::Stroke { points, radius, .. } if points.is_empty() || radius.is_nan() || *radius <= 0.0 => Err("a stroke needs points and a radius".into()),
            Shape::Rect { w, h, .. } if *w <= 0.0 || *h <= 0.0 => Err("a rectangle needs a size".into()),
            Shape::Ellipse { rx, ry, .. } if *rx <= 0.0 || *ry <= 0.0 => Err("an ellipse needs a size".into()),
            _ => Ok(()),
        }
    }
}

pub fn polygon_area(p: &[[f64; 2]]) -> f64 {
    if p.len() < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..p.len() {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        s += a[0] * b[1] - b[0] * a[1];
    }
    (s / 2.0).abs()
}

fn path_length(p: &[[f64; 2]], closed: bool) -> f64 {
    let mut s: f64 = p.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
    if closed && p.len() > 2 {
        let (a, b) = (p[p.len() - 1], p[0]);
        s += (b[0] - a[0]).hypot(b[1] - a[1]);
    }
    s
}

pub fn point_in_polygon(p: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut inside = false;
    let n = p.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi, xj, yj) = (p[i][0], p[i][1], p[j][0], p[j][1]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn seg_dist(a: [f64; 2], b: [f64; 2], x: f64, y: f64) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { (((x - a[0]) * dx + (y - a[1]) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (x - (a[0] + t * dx)).hypot(y - (a[1] + t * dy))
}

fn near_path(p: &[[f64; 2]], closed: bool, x: f64, y: f64, tol: f64) -> bool {
    if p.len() == 1 {
        return (x - p[0][0]).hypot(y - p[0][1]) <= tol;
    }
    let mut segs: Vec<([f64; 2], [f64; 2])> = p.windows(2).map(|w| (w[0], w[1])).collect();
    if closed && p.len() > 2 {
        segs.push((p[p.len() - 1], p[0]));
    }
    segs.into_iter().any(|(a, b)| seg_dist(a, b, x, y) <= tol)
}

/// The area a brush stroke paints (level-0 px²): rasterized at about 4 px across the brush.
fn stroke_area(points: &[[f64; 2]], radius: f64) -> f64 {
    if points.is_empty() || radius <= 0.0 {
        return 0.0;
    }
    let (x0, y0, x1, y1) = points.iter().fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |b, p| (b.0.min(p[0]), b.1.min(p[1]), b.2.max(p[0]), b.3.max(p[1])));
    let (bw, bh) = (x1 - x0 + 2.0 * radius + 2.0, y1 - y0 + 2.0 * radius + 2.0);
    let ds = (radius / 4.0).max(bw.max(bh) / 2048.0).max(1e-3);
    let (w, h) = (((bw / ds).ceil() as usize).max(1), ((bh / ds).ceil() as usize).max(1));
    let moved: Vec<[f64; 2]> = points.iter().map(|p| [p[0] - x0 + radius + 1.0, p[1] - y0 + radius + 1.0]).collect();
    let a = Annotation { id: String::new(), image: String::new(), shape: Shape::Stroke { points: moved, radius, erase: false }, class: String::new(), note: String::new(), z: None };
    rasterize(&[&a], None, 0, 0, w, h, ds).iter().filter(|&&v| v != 0).count() as f64 * ds * ds
}

/// A mask (`width × height`, 0 / 1) of the annotations of `class` (all regions and strokes when
/// `None`) over the region `(x0, y0)` of a level whose pixels are `downsample` level-0 px.
/// Regions and brush strokes fill it in drawing order; eraser strokes (of any class) clear it.
pub fn rasterize(annotations: &[&Annotation], class: Option<&str>, x0: usize, y0: usize, width: usize, height: usize, downsample: f64) -> Vec<u32> {
    let mut mask = vec![0u32; width * height];
    let ds = downsample.max(1e-9);
    // Level-0 point → region pixel (fractional)
    let to = |[x, y]: [f64; 2]| [x / ds - x0 as f64, y / ds - y0 as f64];
    for a in annotations {
        // An eraser clears every class (as on the canvas), a region or brush stroke fills its own
        let eraser = matches!(a.shape, Shape::Stroke { erase: true, .. });
        if !eraser && class.is_some_and(|c| c != a.class) {
            continue;
        }
        match &a.shape {
            s if s.is_region() => {
                let poly: Vec<[f64; 2]> = s.outline().into_iter().map(to).collect();
                fill_polygon(&mut mask, width, height, &poly, 1);
            }
            Shape::Stroke { points, radius, erase } => {
                let r = radius / ds;
                let v = if *erase { 0 } else { 1 };
                let pts: Vec<[f64; 2]> = points.iter().copied().map(to).collect();
                let step = (r * 0.5).max(0.5);
                let stamp = |mask: &mut Vec<u32>, c: [f64; 2]| stamp_disk(mask, width, height, c, r, v);
                if pts.len() == 1 {
                    stamp(&mut mask, pts[0]);
                }
                for w in pts.windows(2) {
                    let d = (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]);
                    let n = (d / step).ceil().max(1.0) as usize;
                    for k in 0..=n {
                        let t = k as f64 / n as f64;
                        stamp(&mut mask, [w[0][0] + t * (w[1][0] - w[0][0]), w[0][1] + t * (w[1][1] - w[0][1])]);
                    }
                }
            }
            _ => {}
        }
    }
    mask
}

fn stamp_disk(mask: &mut [u32], w: usize, h: usize, c: [f64; 2], r: f64, v: u32) {
    let (ya, yb) = ((c[1] - r).floor().max(0.0) as isize, (c[1] + r).ceil().min(h as f64 - 1.0) as isize);
    let (xa, xb) = ((c[0] - r).floor().max(0.0) as isize, (c[0] + r).ceil().min(w as f64 - 1.0) as isize);
    for y in ya..=yb {
        for x in xa..=xb {
            let (dx, dy) = (x as f64 + 0.5 - c[0], y as f64 + 0.5 - c[1]);
            if dx * dx + dy * dy <= r * r {
                mask[y as usize * w + x as usize] = v;
            }
        }
    }
}

/// Even-odd scanline fill of `poly` (pixel centres).
fn fill_polygon(mask: &mut [u32], w: usize, h: usize, poly: &[[f64; 2]], v: u32) {
    if poly.len() < 3 {
        return;
    }
    let (ymin, ymax) = poly.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p[1]), b.max(p[1])));
    let (ya, yb) = (ymin.floor().max(0.0) as usize, (ymax.ceil() as isize).min(h as isize - 1));
    if yb < 0 {
        return;
    }
    let mut xs: Vec<f64> = Vec::new();
    for y in ya..=(yb as usize) {
        let sy = y as f64 + 0.5;
        xs.clear();
        let n = poly.len();
        for i in 0..n {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            if (a[1] <= sy) != (b[1] <= sy) {
                xs.push(a[0] + (sy - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
            }
        }
        xs.sort_by(|a, b| a.total_cmp(b));
        for pair in xs.as_chunks::<2>().0 {
            let xa = (pair[0] - 0.5).ceil().max(0.0) as usize;
            let xb = ((pair[1] - 0.5).floor() as isize).min(w as isize - 1);
            if xb < 0 {
                continue;
            }
            for x in xa..=(xb as usize) {
                mask[y * w + x] = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ann(shape: Shape) -> Annotation {
        Annotation { id: "a".into(), image: "img1".into(), shape, class: "Region".into(), note: String::new(), z: None }
    }

    #[test]
    fn measurements() {
        let r = Shape::Rect { x: 0.0, y: 0.0, w: 10.0, h: 5.0 };
        assert_eq!(r.area(), Some(50.0));
        assert_eq!(r.length(), Some(30.0));
        assert_eq!(Shape::Line { x0: 0.0, y0: 0.0, x1: 3.0, y1: 4.0 }.length(), Some(5.0));
        assert!(r.hit(5.0, 2.0, 1.0) && !r.hit(20.0, 2.0, 1.0));
    }

    #[test]
    fn rasterize_fills_regions_and_erases_strokes() {
        let rect = ann(Shape::Rect { x: 2.0, y: 2.0, w: 6.0, h: 6.0 });
        let erase = ann(Shape::Stroke { points: vec![[5.0, 5.0]], radius: 1.0, erase: true });
        let m = rasterize(&[&rect, &erase], Some("Region"), 0, 0, 10, 10, 1.0);
        assert_eq!(m[3 * 10 + 3], 1);
        assert_eq!(m[4 * 10 + 4], 0, "erased");
        assert_eq!(m[0], 0);
        // At a level of half the resolution, the same rectangle is 3 px wide
        let m2 = rasterize(&[&rect], None, 0, 0, 5, 5, 2.0);
        assert_eq!(m2.iter().filter(|&&v| v == 1).count(), 9);
        // An eraser of another class clears too
        let other = Annotation { class: "Other".into(), ..erase.clone() };
        let m3 = rasterize(&[&rect, &other], Some("Region"), 0, 0, 10, 10, 1.0);
        assert_eq!(m3[4 * 10 + 4], 0, "erased by another class's eraser");
    }
}

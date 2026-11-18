//! Boolean operations on region shapes (Affinity's Add, Subtract, Intersect, Xor), with geo.
//!
//! A shape is one closed outline, so a result with holes is written as one outline that walks
//! out to each hole and back along the same line (a keyhole): filled even-odd and measured by
//! the shoelace formula, it is the region with its holes.

use geo::{BooleanOps, Coord, LineString, MultiPolygon, Polygon};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeOp {
    /// Everything any covers.
    Add,
    /// The first, without what the others cover.
    Subtract,
    /// What all of them cover.
    Intersect,
    /// What an odd number of them cover.
    Xor,
}

impl ShapeOp {
    pub const ALL: [ShapeOp; 4] = [ShapeOp::Add, ShapeOp::Subtract, ShapeOp::Intersect, ShapeOp::Xor];

    pub fn title(self) -> &'static str {
        match self {
            ShapeOp::Add => "Add",
            ShapeOp::Subtract => "Subtract",
            ShapeOp::Intersect => "Intersect",
            ShapeOp::Xor => "Xor",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            ShapeOp::Add => "Add: one shape covering all the selected",
            ShapeOp::Subtract => "Subtract the others from the first selected",
            ShapeOp::Intersect => "Intersect: only where all the selected overlap",
            ShapeOp::Xor => "Xor: where the selected don't overlap",
        }
    }
}

fn polygon(points: &[[f64; 2]]) -> MultiPolygon<f64> {
    let ring: Vec<Coord<f64>> = points.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
    MultiPolygon::new(vec![Polygon::new(LineString::new(ring), Vec::new())])
}

/// The outlines `op` makes of `outlines` (holes joined by keyholes).
pub fn combine(outlines: &[Vec<[f64; 2]>], op: ShapeOp) -> Vec<Vec<[f64; 2]>> {
    let mut shapes = outlines.iter().filter(|o| o.len() >= 3).map(|o| polygon(o));
    let Some(first) = shapes.next() else { return Vec::new() };
    let result = match op {
        ShapeOp::Add => shapes.fold(first, |acc, s| acc.union(&s)),
        ShapeOp::Intersect => shapes.fold(first, |acc, s| acc.intersection(&s)),
        ShapeOp::Xor => shapes.fold(first, |acc, s| acc.xor(&s)),
        ShapeOp::Subtract => {
            let others = shapes.reduce(|a, b| a.union(&b));
            match others {
                Some(o) => first.difference(&o),
                None => first,
            }
        }
    };
    result.0.iter().map(keyhole).filter(|o| o.len() >= 3).collect()
}

fn signed_area(r: &[[f64; 2]]) -> f64 {
    let n = r.len();
    (0..n).map(|i| r[i][0] * r[(i + 1) % n][1] - r[(i + 1) % n][0] * r[i][1]).sum::<f64>() / 2.0
}

fn ring(l: &LineString<f64>) -> Vec<[f64; 2]> {
    let mut v: Vec<[f64; 2]> = l.coords().map(|c| [c.x, c.y]).collect();
    if v.len() > 1 && v.first() == v.last() {
        v.pop();
    }
    v
}

/// One outline for a polygon with holes: from the outer ring's nearest corner into each hole,
/// round it (the other way round), and back.
fn keyhole(p: &Polygon<f64>) -> Vec<[f64; 2]> {
    let mut out = ring(p.exterior());
    let outer_sign = signed_area(&out).signum();
    for h in p.interiors() {
        let mut hole = ring(h);
        if hole.len() < 3 {
            continue;
        }
        if signed_area(&hole).signum() == outer_sign {
            hole.reverse();
        }
        // The closest pair (outer corner, hole corner) to join them
        let (mut bi, mut bj, mut bd) = (0, 0, f64::MAX);
        for (i, a) in out.iter().enumerate() {
            for (j, b) in hole.iter().enumerate() {
                let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
                if d < bd {
                    (bi, bj, bd) = (i, j, d);
                }
            }
        }
        let mut walk = Vec::with_capacity(hole.len() + 2);
        walk.extend(hole[bj..].iter().chain(hole[..bj].iter()).copied());
        walk.push(hole[bj]);
        walk.push(out[bi]);
        out.splice(bi + 1..bi + 1, walk);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, s: f64) -> Vec<[f64; 2]> {
        vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]]
    }

    #[test]
    fn add_subtract_intersect_xor() {
        let (a, b) = (square(0.0, 0.0, 10.0), square(5.0, 0.0, 10.0));
        let area = |o: &Vec<Vec<[f64; 2]>>| o.iter().map(|r| signed_area(r).abs()).sum::<f64>();
        let two = [a.clone(), b.clone()];
        assert!((area(&combine(&two, ShapeOp::Add)) - 150.0).abs() < 1e-6);
        assert!((area(&combine(&two, ShapeOp::Intersect)) - 50.0).abs() < 1e-6);
        assert!((area(&combine(&two, ShapeOp::Subtract)) - 50.0).abs() < 1e-6);
        assert!((area(&combine(&two, ShapeOp::Xor)) - 100.0).abs() < 1e-6);
        // A hole: one keyhole outline whose area is the ring's
        let ring = combine(&[square(0.0, 0.0, 10.0), square(4.0, 4.0, 2.0)], ShapeOp::Subtract);
        assert_eq!(ring.len(), 1);
        assert!((signed_area(&ring[0]).abs() - 96.0).abs() < 1e-6);
    }
}

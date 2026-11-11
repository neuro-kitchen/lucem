//! The object hierarchy of an image: the image is the root; a region's parent
//! is the smallest region that holds it; a detected object (a cell, a tile) belongs to the
//! smallest region that holds its centre. Computed from the document and the detections, never
//! stored: drawing, moving or deleting a region regroups everything at once.

use std::collections::HashMap;

use lumen_hist::HistObject;

use super::annotation::Annotation;

/// A region of the hierarchy: its id, class, outline (level-0) and area (px²).
#[derive(Debug, Clone)]
pub struct Region {
    pub id: String,
    pub class: String,
    pub outline: Vec<[f64; 2]>,
    pub area: f64,
    bounds: (f64, f64, f64, f64),
}

impl Region {
    fn holds(&self, p: [f64; 2]) -> bool {
        let (x0, y0, x1, y1) = self.bounds;
        p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1 && lumen_hist::polygon::contains(&self.outline, p)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Hierarchy {
    /// Regions, largest first.
    pub regions: Vec<Region>,
    /// Each region's parent region (`None`: the image).
    pub region_parent: HashMap<String, Option<String>>,
    /// Each object's region, by its index in the list given (`None`: the image).
    pub object_parent: Vec<Option<String>>,
    /// Objects in each region, with those of the regions inside it.
    pub in_subtree: HashMap<String, usize>,
}

impl Hierarchy {
    /// The hierarchy of `annotations` (their regions on `z`, every plane when `None`) and the
    /// detected `objects`.
    pub fn build<'a>(annotations: impl Iterator<Item = &'a Annotation>, objects: &[HistObject]) -> Self {
        let mut regions: Vec<Region> = annotations
            .filter(|a| a.shape.is_region())
            .map(|a| {
                let outline = a.shape.outline();
                let bounds = lumen_hist::polygon::bounds(&outline);
                Region { id: a.id.clone(), class: a.class.clone(), area: a.shape.area().unwrap_or(0.0), outline, bounds }
            })
            .collect();
        regions.sort_by(|a, b| b.area.total_cmp(&a.area));
        // The smallest region holding `p` (regions are largest first: the last that holds it)
        let smallest = |p: [f64; 2], skip: Option<usize>| regions.iter().enumerate().filter(|(i, r)| Some(*i) != skip && r.holds(p)).last().map(|(_, r)| r.id.clone());
        let mut region_parent = HashMap::new();
        for (i, r) in regions.iter().enumerate() {
            // A region is inside another when its centre and every corner of its bounds are,
            // and the other is larger
            let c = lumen_hist::polygon::centroid(&r.outline);
            let parent = regions.iter().enumerate().take(i).filter(|(_, o)| o.area > r.area && o.holds(c) && r.outline.iter().all(|p| o.holds(*p))).last().map(|(_, o)| o.id.clone());
            region_parent.insert(r.id.clone(), parent);
        }
        // Many objects: their regions found in parallel
        let object_parent: Vec<Option<String>> = {
            use rayon::prelude::*;
            objects.par_iter().map(|o| smallest(o.centroid, None)).collect()
        };
        // Totals with the regions inside: each object counts for its region and every ancestor
        let mut in_subtree: HashMap<String, usize> = HashMap::new();
        for p in object_parent.iter().flatten() {
            let mut cur = Some(p.clone());
            let mut guard = 0;
            while let Some(r) = cur {
                *in_subtree.entry(r.clone()).or_insert(0) += 1;
                cur = region_parent.get(&r).cloned().flatten();
                guard += 1;
                if guard > 256 {
                    break;
                }
            }
        }
        Self { regions, region_parent, object_parent, in_subtree }
    }

    /// The region and every region inside it, at any depth.
    pub fn subtree(&self, id: &str) -> Vec<String> {
        let mut out = vec![id.to_string()];
        let mut i = 0;
        while i < out.len() {
            let cur = out[i].clone();
            out.extend(self.region_parent.iter().filter(|(_, p)| p.as_deref() == Some(cur.as_str())).map(|(k, _)| k.clone()));
            i += 1;
        }
        out
    }

    /// How many objects each region holds directly.
    pub fn counts(&self) -> HashMap<String, usize> {
        let mut out = HashMap::new();
        for p in self.object_parent.iter().flatten() {
            *out.entry(p.clone()).or_insert(0) += 1;
        }
        out
    }

    /// The regions whose parent is `parent` (`None`: the image), largest first.
    pub fn children(&self, parent: Option<&str>) -> Vec<&Region> {
        self.regions.iter().filter(|r| self.region_parent.get(&r.id).map(|p| p.as_deref()) == Some(parent)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::annotation::Shape;

    fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Annotation {
        Annotation { id: id.into(), image: "img".into(), shape: Shape::Rect { x, y, w, h }, class: "Region".into(), note: String::new(), z: None }
    }

    #[test]
    fn regions_nest_and_cells_belong_to_the_smallest() {
        let anns = [rect("tissue", 0.0, 0.0, 100.0, 100.0), rect("tumor", 10.0, 10.0, 30.0, 30.0), rect("apart", 200.0, 0.0, 10.0, 10.0)];
        let cell = |x: f64, y: f64| HistObject::new(0, lumen_hist::ObjectKind::Cell, vec![[x - 1.0, y - 1.0], [x + 1.0, y - 1.0], [x + 1.0, y + 1.0], [x - 1.0, y + 1.0]]);
        let objects = [cell(20.0, 20.0), cell(80.0, 80.0), cell(500.0, 500.0)];
        let h = Hierarchy::build(anns.iter(), &objects);
        assert_eq!(h.region_parent["tumor"].as_deref(), Some("tissue"));
        assert_eq!(h.region_parent["tissue"], None);
        assert_eq!(h.region_parent["apart"], None);
        assert_eq!(h.object_parent, vec![Some("tumor".to_string()), Some("tissue".to_string()), None]);
        let mut sub = h.subtree("tissue");
        sub.sort();
        assert_eq!(sub, vec!["tissue".to_string(), "tumor".to_string()]);
        assert_eq!(h.children(None).iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["tissue", "apart"]);
    }
}

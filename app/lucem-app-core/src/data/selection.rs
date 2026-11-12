//! What is selected: shapes (annotation ids), detected objects (`c12`: cell 12, `t5`: tile 5)
//! of the shown image, rows of the Layers panel (`view:<id>`, `class:<name>`, `channel:<n>`, in
//! the order picked) and the Pipeline's node. Selecting is view state: never in the undo history
//! nor the project file.

use std::collections::{BTreeSet, HashSet};

use gpui_kit::Context;

use super::store::{AppEvent, Store};

#[derive(Debug, Default, Clone)]
pub struct Selection {
    /// Selected annotations (ids).
    pub shapes: BTreeSet<String>,
    /// Detected objects selected on the shown image.
    pub objects: Vec<String>,
    /// Selected rows of the Layers panel, in the order they were picked.
    pub layers: Vec<String>,
    /// The selected node in the Pipeline (shared so Explore can show it).
    pub node: Option<String>,
}

impl Selection {
    /// Whether nothing (no shape, no object) is selected.
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty() && self.objects.is_empty()
    }
}


impl Store {
    /// Selects a row of the Layers panel, or adds it to (or takes it out of) the selection.
    pub fn select_layer(&mut self, key: String, additive: bool, cx: &mut Context<Self>) {
        if additive {
            match self.sel.layers.iter().position(|k| *k == key) {
                Some(i) => {
                    self.sel.layers.remove(i);
                }
                None => self.sel.layers.push(key),
            }
        } else {
            self.sel.layers = vec![key];
        }
        cx.emit(AppEvent::Selection);
        cx.notify();
    }

    /// The detected object `key` (`c12`, `t5`) of the shown image.
    pub fn object(&self, key: &str) -> Option<&lucem_hist::HistObject> {
        let img = self.doc.active_image.as_deref()?;
        let (list, n) = match key.split_at(1) {
            ("c", n) => (self.hist.cells.get(img)?, n),
            ("t", n) => (self.hist.tiles.get(img)?, n),
            _ => return None,
        };
        let id: u32 = n.parse().ok()?;
        list.iter().find(|o| o.id == id)
    }

    /// Selects a detected object (Shift / Ctrl add or take it out) and shows it in Explore's view.
    pub fn select_object(&mut self, key: String, additive: bool, cx: &mut Context<Self>) {
        let at = self.object(&key).map(|o| o.centroid);
        if additive {
            match self.sel.objects.iter().position(|k| *k == key) {
                Some(i) => {
                    self.sel.objects.remove(i);
                }
                None => self.sel.objects.push(key),
            }
        } else {
            self.sel.objects = vec![key];
            self.sel.shapes.clear();
        }
        if let Some(p) = at {
            cx.emit(AppEvent::Reveal(p));
        }
        cx.emit(AppEvent::Selection);
        cx.notify();
    }

    /// The detected object under `p` (level-0) on the shown image: a cell holding it, else a tile.
    pub fn object_at(&self, p: [f64; 2]) -> Option<String> {
        let img = self.doc.active_image.as_deref()?;
        let hit = |o: &&lucem_hist::HistObject| {
            let (x0, y0, x1, y1) = lucem_hist::polygon::bounds(&o.roi);
            p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1 && lucem_hist::polygon::contains(&o.roi, p)
        };
        if let Some(c) = self.hist.cells.get(img).and_then(|l| l.iter().find(hit)) {
            return Some(format!("c{}", c.id));
        }
        self.hist.tiles.get(img).and_then(|l| l.iter().find(hit)).map(|t| format!("t{}", t.id))
    }

    pub fn select(&mut self, ids: impl IntoIterator<Item = String>, cx: &mut Context<Self>) {
        self.sel.shapes = ids.into_iter().collect();
        cx.emit(AppEvent::Selection);
        cx.notify();
    }

    /// The Delete key and the Delete buttons: the selected annotations, else the result layers
    /// (masks, labels, images of the pipeline) selected in Layers. Classes go only from their
    /// menu (they may hold many annotations).
    /// Selects the shapes (their bounds' centre) and cells or tiles (their centre) of the shown
    /// image inside `rect` (`[x0, y0, x1, y1]`, level-0 px); `add` keeps the selection there was.
    /// `None` (a click, no box) clears it unless `add`.
    pub fn select_in_rect(&mut self, rect: Option<[f64; 4]>, add: bool, cx: &mut Context<Self>) {
        let Some([x0, y0, x1, y1]) = rect else {
            if !add {
                self.sel.objects.clear();
                self.select([], cx);
            }
            return;
        };
        let inside = |x: f64, y: f64| x >= x0 && x <= x1 && y >= y0 && y <= y1;
        let Some(image) = self.doc.active_image.clone() else { return };
        let mut shapes: std::collections::BTreeSet<String> = if add { self.sel.shapes.iter().cloned().collect() } else { Default::default() };
        for a in self.doc.annotations_of(&image) {
            let (ax0, ay0, ax1, ay1) = a.shape.bounds();
            if inside((ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0) {
                shapes.insert(a.id.clone());
            }
        }
        let mut objects: Vec<String> = if add { self.sel.objects.clone() } else { Vec::new() };
        let known: HashSet<String> = objects.iter().cloned().collect();
        for (prefix, list) in [("c", self.hist.cells.get(&image)), ("t", self.hist.tiles.get(&image))] {
            for o in list.into_iter().flatten() {
                let key = format!("{prefix}{}", o.id);
                if inside(o.centroid[0], o.centroid[1]) && !known.contains(&key) {
                    objects.push(key);
                }
            }
        }
        let (ns, no) = (shapes.len(), objects.len());
        self.sel.objects = objects;
        self.select(shapes, cx);
        self.set_status(format!("{ns} shape(s) and {no} object(s) selected"), cx);
    }
}

//! Histology models as plugins.
//!
//! lumen never links a model: each upstream family is a crate `models/<family>` (its code under
//! its licence, weights fetched and verified through `lumen-hub`) and a program
//! `plugins/ml/lumen-ml-<family>` the app finds and runs (`lumen-ml`'s protocol). A histology
//! model says so in its manifest (`domains: ["histology"]`) and answers:
//!
//! - [`lumen_ml::Request::Instances`] (nucleus and cell segmentation): a label raster, and the
//!   class of each label when the model classifies (tumour, inflammatory, …). lumen turns the
//!   labels into cells with [`crate::cells::cells_from_nuclei`] — the same measurements,
//!   expansion and classifiers as its own detection — with [`labels_from_result`] and
//!   [`apply_instance_classes`].
//! - `Segment` (interactive: SAM) and, for patch encoders, an embedding per tile that object
//!   classifiers take as features.
//!
//! [`FAMILIES`] lists the families planned, in the order they are worth bringing in. Before one
//! is added, its licence and its weights' are checked at the source (`licence_checked`): some
//! weights are for non-commercial research only, or gated behind an agreement, and the plugin
//! must say so in its manifest.

use crate::error::{invalid, HistResult};
use crate::objects::HistObject;

/// A model family to bring in as a plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Family {
    /// `plugins/ml/lumen-ml-<id>`, `models/<id>`.
    pub id: &'static str,
    pub name: &'static str,
    pub task: lumen_ml::Task,
    /// Images it is trained for.
    pub domains: &'static [&'static str],
    /// The code's licence (SPDX) as last known; the weights' may differ.
    pub licence: &'static str,
    /// Whether code and weights licences were confirmed at the source for lumen's use.
    pub licence_checked: bool,
    pub what: &'static str,
}

pub const FAMILIES: &[Family] = &[
    Family {
        id: "stardist",
        name: "StarDist",
        task: lumen_ml::Task::Segmentation,
        domains: &["histology", "fluorescence"],
        licence: "BSD-3-Clause",
        licence_checked: false,
        what: "Nuclei as star-convex polygons; pretrained for H&E (versatile_he) and fluorescence (versatile_fluo). The first nucleus plugin.",
    },
    Family {
        id: "instanseg",
        name: "InstanSeg",
        task: lumen_ml::Task::Segmentation,
        domains: &["histology", "fluorescence"],
        licence: "Apache-2.0",
        licence_checked: false,
        what: "Nuclei and cells in brightfield and multiplexed fluorescence (any number of channels), fast on the CPU.",
    },
    Family {
        id: "cellpose",
        name: "Cellpose",
        task: lumen_ml::Task::Segmentation,
        domains: &["fluorescence", "histology"],
        licence: "BSD-3-Clause",
        licence_checked: false,
        what: "Cells and nuclei from flow fields; generalist models, strongest on fluorescence.",
    },
    Family {
        id: "hovernet",
        name: "HoVer-Net",
        task: lumen_ml::Task::Segmentation,
        domains: &["histology"],
        licence: "MIT",
        licence_checked: false,
        what: "Nuclei segmented and classified at once (neoplastic, inflammatory, connective, …) in H&E; check the weights' licence (trained on public datasets with their own terms).",
    },
    Family {
        id: "uni",
        name: "UNI",
        task: lumen_ml::Task::Embedding,
        domains: &["histology"],
        licence: "CC-BY-NC-ND-4.0",
        licence_checked: false,
        what: "A pathology foundation model: a feature vector per tile for tissue and region classifiers; gated weights, non-commercial.",
    },
    Family {
        id: "conch",
        name: "CONCH",
        task: lumen_ml::Task::Embedding,
        domains: &["histology"],
        licence: "CC-BY-NC-ND-4.0",
        licence_checked: false,
        what: "A vision-language pathology model: tile embeddings and text prompts for zero-shot tissue classes; gated weights, non-commercial.",
    },
];

/// The label raster a plugin wrote (`u32` little-endian, `width × height`).
pub fn labels_from_result(r: &lumen_ml::InstancesResult) -> HistResult<(Vec<u32>, usize, usize)> {
    let (w, h) = (r.labels.width as usize, r.labels.height as usize);
    let bytes = std::fs::read(&r.labels.path).map_err(|e| invalid(format!("{}: {e}", r.labels.path)))?;
    if bytes.len() != w * h * 4 {
        return Err(invalid(format!("the labels hold {} bytes, not {w} × {h} × 4", bytes.len())));
    }
    let labels: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
    if !labels.iter().any(|&l| l != 0) {
        return Err(invalid("the model found nothing (an empty label image is not taken as an answer)"));
    }
    Ok((labels, w, h))
}

/// Gives cells the class the model gave their nucleus. `original` maps each object's id (its
/// label after [`crate::cells::cells_from_nuclei`] renumbered them) back to the model's label:
/// the model's label of a pixel inside the object's nucleus.
pub fn apply_instance_classes(objects: &mut [HistObject], model_labels: &[u32], kept: &[u32], classes: &[lumen_ml::InstanceClass]) {
    let class_of: std::collections::HashMap<u32, &str> = classes.iter().map(|c| (c.label, c.class.as_str())).collect();
    let mut first: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for (&k, &m) in kept.iter().zip(model_labels) {
        if k != 0 && m != 0 {
            first.entry(k).or_insert(m);
        }
    }
    for o in objects.iter_mut() {
        if let Some(c) = first.get(&o.id).and_then(|m| class_of.get(m)) {
            o.class = Some((*c).to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::ObjectKind;

    #[test]
    fn model_labels_read_and_classes_follow() {
        let dir = std::env::temp_dir().join(format!("lumen-hist-models-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("labels.u32");
        let labels: Vec<u32> = vec![0, 7, 7, 0, 9, 9];
        std::fs::write(&path, labels.iter().flat_map(|l| l.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        let r = lumen_ml::InstancesResult { labels: lumen_ml::RasterRef { path: path.to_string_lossy().into(), width: 3, height: 2, channels: 1 }, classes: vec![lumen_ml::InstanceClass { label: 9, class: "Inflammatory".into(), score: None }], ms: 1.0 };
        let (l, w, h) = labels_from_result(&r).unwrap();
        assert_eq!((l, w, h), (labels.clone(), 3, 2));
        let kept = vec![0, 1, 1, 0, 2, 2];
        let mut objs = vec![HistObject::new(1, ObjectKind::Cell, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), HistObject::new(2, ObjectKind::Cell, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])];
        apply_instance_classes(&mut objs, &labels, &kept, &r.classes);
        assert_eq!((objs[0].class.as_deref(), objs[1].class.as_deref()), (None, Some("Inflammatory")));
        std::fs::write(&path, vec![0u8; 24]).unwrap();
        assert!(labels_from_result(&r).is_err(), "an empty answer is refused");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(FAMILIES.iter().all(|f| !f.licence_checked), "a family's licence is confirmed when its plugin is added");
    }
}

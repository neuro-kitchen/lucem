//! Multiplexed analysis: one classifier per marker (a measurement of its channel above a
//! threshold makes the cell positive for it), combined into a composite so a cell's class lists
//! its markers (`CD3: CD8`); phenotypes are named marker combinations counted over a region.

use serde::{Deserialize, Serialize};

use crate::classify::ObjectClassifier;
use crate::objects::HistObject;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkerRule {
    /// The marker (`CD8`): the class part a positive cell gets.
    pub marker: String,
    /// What decides it (`Cell: CD8 mean`).
    pub measurement: String,
    pub threshold: f64,
}

/// The composite classifier of the markers, in their order.
pub fn marker_classifier(name: &str, rules: &[MarkerRule]) -> ObjectClassifier {
    ObjectClassifier::Composite {
        name: name.into(),
        parts: rules.iter().map(|r| ObjectClassifier::SingleMeasurement { measurement: r.measurement.clone(), threshold: r.threshold, above: Some(r.marker.clone()), below: None, filter: None }).collect(),
    }
}

/// A named combination: the markers a cell must have and the ones it must not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phenotype {
    pub name: String,
    pub positive: Vec<String>,
    #[serde(default)]
    pub negative: Vec<String>,
}

impl Phenotype {
    pub fn matches(&self, o: &HistObject) -> bool {
        self.positive.iter().all(|m| o.has_class(m)) && !self.negative.iter().any(|m| o.has_class(m))
    }
}

/// How many objects each phenotype has (an object may count in several).
pub fn phenotype_counts<'a>(objects: impl IntoIterator<Item = &'a HistObject> + Clone, phenotypes: &[Phenotype]) -> Vec<(String, usize)> {
    phenotypes.iter().map(|p| (p.name.clone(), objects.clone().into_iter().filter(|o| p.matches(o)).count())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::ObjectKind;

    #[test]
    fn markers_make_phenotypes() {
        let rules = vec![
            MarkerRule { marker: "CD3".into(), measurement: "Cell: CD3 mean".into(), threshold: 0.5 },
            MarkerRule { marker: "CD8".into(), measurement: "Cell: CD8 mean".into(), threshold: 0.5 },
        ];
        let mut cells: Vec<HistObject> = [(0.9, 0.9), (0.9, 0.1), (0.1, 0.1)]
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| {
                let mut o = HistObject::new(i as u32, ObjectKind::Cell, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                o.measurements.insert("Cell: CD3 mean".into(), a);
                o.measurements.insert("Cell: CD8 mean".into(), b);
                o
            })
            .collect();
        marker_classifier("T cells", &rules).apply(&mut cells);
        assert_eq!(cells[0].class.as_deref(), Some("CD3: CD8"));
        assert_eq!(cells[1].class.as_deref(), Some("CD3"));
        assert_eq!(cells[2].class, None);
        let ph = [Phenotype { name: "Cytotoxic T".into(), positive: vec!["CD3".into(), "CD8".into()], negative: vec![] }, Phenotype { name: "Helper-like T".into(), positive: vec!["CD3".into()], negative: vec!["CD8".into()] }];
        assert_eq!(phenotype_counts(&cells, &ph), vec![("Cytotoxic T".into(), 1), ("Helper-like T".into(), 1)]);
    }
}

//! Object classifiers: a class for every object (cell, detection, tile) from its measurements.
//!
//! - [`ObjectClassifier::SingleMeasurement`]: one measurement against a threshold.
//! - [`ObjectClassifier::RandomTrees`]: trained from examples, [`train_random_trees`]: objects
//!   whose centroid lies in a region drawn and given a class (or under a point of that class)
//!   are that class's examples. The measurements used are all those the examples share, or the
//!   ones chosen; they may be normalized (mean 0, std dev 1), which trees do not need.
//! - [`ObjectClassifier::Composite`]: several applied in turn, their classes joined (`CD3: CD8`)
//!   — the multiplexed case.
//!
//! A classifier is saved as JSON ([`ObjectClassifier::to_json`]) and applied to other images.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::error::{invalid, HistResult};
use crate::forest::{ForestSettings, RandomForest};
use crate::objects::{HistObject, Point, Polygon};
use crate::polygon;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "classifier", rename_all = "snake_case")]
pub enum ObjectClassifier {
    SingleMeasurement {
        measurement: String,
        threshold: f64,
        /// The class above the threshold, and below it (`None`: no class).
        above: Option<String>,
        below: Option<String>,
        /// Only objects of this class are classified.
        #[serde(default)]
        filter: Option<String>,
    },
    RandomTrees {
        features: Vec<String>,
        classes: Vec<String>,
        /// Per feature `(mean, std dev)` when normalized.
        #[serde(default)]
        normalize: Option<Vec<(f64, f64)>>,
        forest: RandomForest,
        #[serde(default)]
        filter: Option<String>,
    },
    Composite {
        name: String,
        parts: Vec<ObjectClassifier>,
    },
}

/// Which measurements train a classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureChoice {
    /// Every measurement all the examples have.
    All,
    Selected(Vec<String>),
}

/// An example region or point and its class.
#[derive(Debug, Clone, PartialEq)]
pub enum Example {
    Region(Polygon, String),
    Point(Point, String),
}

fn passes(o: &HistObject, filter: &Option<String>) -> bool {
    filter.as_deref().is_none_or(|f| o.has_class(f))
}

impl ObjectClassifier {
    /// The class `o` gets (`None`: none, or not for this classifier).
    pub fn classify(&self, o: &HistObject) -> Option<String> {
        match self {
            ObjectClassifier::SingleMeasurement { measurement, threshold, above, below, filter } => {
                if !passes(o, filter) {
                    return None;
                }
                let v = o.get(measurement)?;
                if v > *threshold { above.clone() } else { below.clone() }
            }
            ObjectClassifier::RandomTrees { features, classes, normalize, forest, filter } => {
                if !passes(o, filter) {
                    return None;
                }
                let row = row_of(o, features, normalize.as_deref())?;
                Some(classes[forest.predict(&row).0].clone())
            }
            ObjectClassifier::Composite { parts, .. } => {
                let names: Vec<String> = parts.iter().filter_map(|p| p.classify(o)).collect();
                (!names.is_empty()).then(|| names.join(": "))
            }
        }
    }

    /// Classifies every object it applies to (the others keep their class); how many changed.
    pub fn apply(&self, objects: &mut [HistObject]) -> usize {
        let mut changed = 0;
        for o in objects.iter_mut() {
            let applies = match self {
                ObjectClassifier::SingleMeasurement { filter, .. } | ObjectClassifier::RandomTrees { filter, .. } => passes(o, filter),
                ObjectClassifier::Composite { .. } => true,
            };
            if !applies {
                continue;
            }
            let c = self.classify(o);
            if c != o.class {
                o.class = c;
                changed += 1;
            }
        }
        changed
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub fn from_json(text: &str) -> HistResult<Self> {
        serde_json::from_str(text).map_err(|e| invalid(format!("not a classifier: {e}")))
    }
}

/// The feature row of `o` (normalized when asked); `None` when one is missing.
fn row_of(o: &HistObject, features: &[String], normalize: Option<&[(f64, f64)]>) -> Option<Vec<f64>> {
    features
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let v = o.get(f)?;
            Some(match normalize {
                Some(n) => (v - n[i].0) / n[i].1.max(1e-12),
                None => v,
            })
        })
        .collect()
}

/// The class of the example covering `o`'s centroid (a region containing it, or a point in its
/// outline).
fn example_class<'a>(o: &HistObject, examples: &'a [Example]) -> Option<&'a str> {
    examples.iter().find_map(|e| match e {
        Example::Region(p, c) => polygon::contains(p, o.centroid).then_some(c.as_str()),
        Example::Point(q, c) => polygon::contains(&o.roi, *q).then_some(c.as_str()),
    })
}

/// Trains Random Trees from the objects the examples cover.
pub fn train_random_trees(objects: &[HistObject], examples: &[Example], features: &FeatureChoice, normalize: bool, filter: Option<String>, s: &ForestSettings) -> HistResult<ObjectClassifier> {
    let labelled: Vec<(&HistObject, &str)> = objects.iter().filter(|o| passes(o, &filter)).filter_map(|o| example_class(o, examples).map(|c| (o, c))).collect();
    let classes: Vec<String> = labelled.iter().map(|(_, c)| c.to_string()).collect::<BTreeSet<_>>().into_iter().collect();
    if classes.len() < 2 {
        return Err(invalid(format!("the examples cover objects of {} class(es); at least two are needed", classes.len())));
    }
    let features: Vec<String> = match features {
        FeatureChoice::Selected(f) => f.clone(),
        FeatureChoice::All => {
            let mut common: Option<BTreeSet<&String>> = None;
            for (o, _) in &labelled {
                let keys: BTreeSet<&String> = o.measurements.iter().filter(|(_, v)| v.is_finite()).map(|(k, _)| k).collect();
                common = Some(match common {
                    None => keys,
                    Some(c) => c.intersection(&keys).copied().collect(),
                });
            }
            common.unwrap_or_default().into_iter().cloned().collect()
        }
    };
    if features.is_empty() {
        return Err(invalid("the examples share no measurement to learn from"));
    }
    let rows: Vec<(Vec<f64>, usize)> = labelled.iter().filter_map(|(o, c)| Some((row_of(o, &features, None)?, classes.iter().position(|k| k == c)?))).collect();
    let norm = normalize.then(|| {
        (0..features.len())
            .map(|j| {
                let n = rows.len() as f64;
                let mean = rows.iter().map(|r| r.0[j]).sum::<f64>() / n;
                let sd = (rows.iter().map(|r| (r.0[j] - mean).powi(2)).sum::<f64>() / n).sqrt();
                (mean, if sd > 0.0 { sd } else { 1.0 })
            })
            .collect::<Vec<_>>()
    });
    let x: Vec<Vec<f64>> = rows.iter().map(|(r, _)| match &norm { Some(n) => r.iter().zip(n).map(|(v, (m, s))| (v - m) / s).collect(), None => r.clone() }).collect();
    let y: Vec<usize> = rows.iter().map(|r| r.1).collect();
    let forest = RandomForest::train(&x, &y, classes.len(), s)?;
    Ok(ObjectClassifier::RandomTrees { features, classes, normalize: norm, forest, filter })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::ObjectKind;

    fn cell(id: u32, x: f64, area: f64, dab: f64) -> HistObject {
        let mut o = HistObject::new(id, ObjectKind::Cell, vec![[x - 2.0, 0.0], [x + 2.0, 0.0], [x + 2.0, 4.0], [x - 2.0, 4.0]]);
        o.measurements.insert("Nucleus: Area µm^2".into(), area);
        o.measurements.insert("Nucleus: DAB OD mean".into(), dab);
        o
    }

    #[test]
    fn random_trees_learn_from_regions_and_classify_the_rest() {
        // Big nuclei on the left are tumour, small ones on the right stroma
        let mut cells: Vec<HistObject> = (0..60).map(|i| if i % 2 == 0 { cell(i, i as f64 * 10.0, 60.0 + (i % 7) as f64, 0.1) } else { cell(i, i as f64 * 10.0, 20.0 + (i % 5) as f64, 0.1) }).collect();
        let examples = vec![Example::Region(vec![[-5.0, -5.0], [205.0, -5.0], [205.0, 9.0], [-5.0, 9.0]], "Tumor".into())];
        assert!(train_random_trees(&cells, &examples, &FeatureChoice::All, false, None, &ForestSettings::default()).is_err(), "one class is not enough");
        // Examples: the first ten cells of each kind
        let mut ex = Vec::new();
        for o in cells.iter().take(20) {
            ex.push(Example::Point(o.centroid, if o.get("Nucleus: Area µm^2").unwrap() > 40.0 { "Tumor".into() } else { "Stroma".into() }));
        }
        let c = train_random_trees(&cells, &ex, &FeatureChoice::All, true, None, &ForestSettings { trees: 20, ..ForestSettings::default() }).unwrap();
        let c = ObjectClassifier::from_json(&c.to_json()).unwrap();
        c.apply(&mut cells);
        let right = cells.iter().filter(|o| (o.get("Nucleus: Area µm^2").unwrap() > 40.0) == (o.class.as_deref() == Some("Tumor"))).count();
        assert_eq!(right, cells.len());
    }

    #[test]
    fn composites_join_their_parts() {
        let mut o = cell(1, 0.0, 30.0, 0.6);
        o.measurements.insert("Cell: CD8 mean".into(), 0.9);
        let c = ObjectClassifier::Composite {
            name: "T cells".into(),
            parts: vec![
                ObjectClassifier::SingleMeasurement { measurement: "Nucleus: DAB OD mean".into(), threshold: 0.2, above: Some("CD3".into()), below: None, filter: None },
                ObjectClassifier::SingleMeasurement { measurement: "Cell: CD8 mean".into(), threshold: 0.5, above: Some("CD8".into()), below: None, filter: None },
            ],
        };
        assert_eq!(c.classify(&o).as_deref(), Some("CD3: CD8"));
        assert!(o.has_class("CD8") || c.classify(&o).is_some());
    }
}

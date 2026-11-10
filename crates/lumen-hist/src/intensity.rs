//! Intensity classes: one measurement (a cell's `Nucleus: DAB OD mean`, say) against one
//! threshold (negative / positive) or three (negative, 1+, 2+, 3+), and what a region reports
//! from them: Positive % (a labelling index such as Ki67's) and the H-score (0–300).

use serde::{Deserialize, Serialize};

use crate::objects::{HistObject, IntensityClass};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntensityThresholds {
    pub measurement: String,
    /// One threshold, or three increasing ones.
    pub thresholds: Vec<f64>,
    /// Only objects of this class (`Tumor`) get an intensity class.
    #[serde(default)]
    pub filter: Option<String>,
}

impl IntensityThresholds {
    pub fn class_of(&self, v: f64) -> IntensityClass {
        match self.thresholds.as_slice() {
            [t] => {
                if v > *t {
                    IntensityClass::Positive
                } else {
                    IntensityClass::Negative
                }
            }
            [t1, t2, t3, ..] => {
                if v > *t3 {
                    IntensityClass::ThreePlus
                } else if v > *t2 {
                    IntensityClass::TwoPlus
                } else if v > *t1 {
                    IntensityClass::OnePlus
                } else {
                    IntensityClass::Negative
                }
            }
            [t1, _] => {
                if v > *t1 {
                    IntensityClass::Positive
                } else {
                    IntensityClass::Negative
                }
            }
            [] => IntensityClass::Negative,
        }
    }
}

/// Sets the intensity class of the objects it applies to (their main class stays); the others
/// lose theirs. How many were classified.
pub fn classify(objects: &mut [HistObject], t: &IntensityThresholds) -> usize {
    let mut n = 0;
    for o in objects.iter_mut() {
        let applies = t.filter.as_deref().is_none_or(|f| o.has_class(f));
        o.intensity = match (applies, o.get(&t.measurement)) {
            (true, Some(v)) => {
                n += 1;
                Some(t.class_of(v))
            }
            _ => None,
        };
    }
    n
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct IntensitySummary {
    /// Objects with an intensity class.
    pub total: usize,
    pub negative: usize,
    pub positive: usize,
    pub one_plus: usize,
    pub two_plus: usize,
    pub three_plus: usize,
    pub positive_percent: f64,
    /// `1 · %1+ + 2 · %2+ + 3 · %3+` (0–300). With one threshold there are no 1+ / 2+ / 3+
    /// classes, so it stays 0.
    pub h_score: f64,
}

pub fn summarize<'a>(objects: impl IntoIterator<Item = &'a HistObject>) -> IntensitySummary {
    let mut s = IntensitySummary::default();
    for o in objects {
        let Some(c) = o.intensity else { continue };
        s.total += 1;
        match c {
            IntensityClass::Negative => s.negative += 1,
            IntensityClass::Positive => s.positive += 1,
            IntensityClass::OnePlus => s.one_plus += 1,
            IntensityClass::TwoPlus => s.two_plus += 1,
            IntensityClass::ThreePlus => s.three_plus += 1,
        }
    }
    if s.total > 0 {
        let t = s.total as f64;
        let pos = s.positive + s.one_plus + s.two_plus + s.three_plus;
        s.positive_percent = 100.0 * pos as f64 / t;
        s.h_score = 100.0 * (s.one_plus as f64 + 2.0 * s.two_plus as f64 + 3.0 * s.three_plus as f64) / t;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::ObjectKind;

    #[test]
    fn three_thresholds_give_the_h_score() {
        let mut cells: Vec<HistObject> = [0.05, 0.25, 0.45, 0.7, 0.1, 0.9]
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let mut o = HistObject::new(i as u32, ObjectKind::Cell, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                o.class = Some(if i == 5 { "Stroma".into() } else { "Tumor".into() });
                o.measurements.insert("Nucleus: DAB OD mean".into(), v);
                o
            })
            .collect();
        let t = IntensityThresholds { measurement: "Nucleus: DAB OD mean".into(), thresholds: vec![0.2, 0.4, 0.6], filter: Some("Tumor".into()) };
        assert_eq!(classify(&mut cells, &t), 5);
        let s = summarize(&cells);
        assert_eq!((s.total, s.negative, s.one_plus, s.two_plus, s.three_plus), (5, 2, 1, 1, 1));
        assert!((s.positive_percent - 60.0).abs() < 1e-9 && (s.h_score - 120.0).abs() < 1e-9);
        assert_eq!(cells[0].full_class().as_deref(), Some("Tumor: Negative"));
        assert_eq!(cells[3].full_class().as_deref(), Some("Tumor: 3+"));
    }
}

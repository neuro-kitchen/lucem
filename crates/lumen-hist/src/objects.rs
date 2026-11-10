//! The objects of an analysis: annotations, detections, cells and tiles, with their measurements.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::polygon;

/// A point in level-0 pixels.
pub type Point = [f64; 2];
/// A closed outline in level-0 pixels (the last point joins the first).
pub type Polygon = Vec<Point>;
/// Measurements by name (`Nucleus: Area µm^2`, `Cell: DAB OD mean`, …).
pub type Measurements = BTreeMap<String, f64>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind {
    /// A region drawn or found (tissue, a class region of a pixel classifier).
    Annotation,
    /// A detected object without a nucleus / cell split.
    Detection,
    /// A cell: its outline is the cell, `nucleus` its nucleus.
    Cell,
    /// A tile or superpixel.
    Tile,
}

/// The intensity class of an object (staining strength of one measurement).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntensityClass {
    Negative,
    /// One threshold: above it.
    Positive,
    OnePlus,
    TwoPlus,
    ThreePlus,
}

impl IntensityClass {
    pub fn name(self) -> &'static str {
        match self {
            IntensityClass::Negative => "Negative",
            IntensityClass::Positive => "Positive",
            IntensityClass::OnePlus => "1+",
            IntensityClass::TwoPlus => "2+",
            IntensityClass::ThreePlus => "3+",
        }
    }

    pub fn is_positive(self) -> bool {
        self != IntensityClass::Negative
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistObject {
    pub id: u32,
    pub kind: ObjectKind,
    /// The outline (a cell's whole cell).
    pub roi: Polygon,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nucleus: Option<Polygon>,
    pub centroid: Point,
    /// Its class (`Tumor`, `CD3: CD8`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intensity: Option<IntensityClass>,
    #[serde(default)]
    pub measurements: Measurements,
}

impl HistObject {
    /// An object from its outline (centroid from the outline).
    pub fn new(id: u32, kind: ObjectKind, roi: Polygon) -> Self {
        let centroid = polygon::centroid(&roi);
        Self { id, kind, roi, nucleus: None, centroid, class: None, intensity: None, measurements: Measurements::new() }
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.measurements.get(name).copied().filter(|v| v.is_finite())
    }

    /// The class with its intensity, as shown: `Tumor: 2+`, `Positive`, `Stroma`.
    pub fn full_class(&self) -> Option<String> {
        match (&self.class, self.intensity) {
            (Some(c), Some(i)) => Some(format!("{c}: {}", i.name())),
            (Some(c), None) => Some(c.clone()),
            (None, Some(i)) => Some(i.name().to_string()),
            (None, None) => None,
        }
    }

    /// Whether its class is `class` or starts with it (`CD3` matches `CD3: CD8`).
    pub fn has_class(&self, class: &str) -> bool {
        self.class.as_deref().is_some_and(|c| c == class || c.split(": ").any(|p| p == class))
    }
}

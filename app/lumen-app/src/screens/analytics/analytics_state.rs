//! Immutable UI state for the Polars & Arrow Analytics drawer (`AnalyticsUiState`).

/// A formatted row extracted from the Polars `DataFrame` / Arrow `RecordBatch` of `RegionFeatures`.
#[derive(Debug, Clone)]
pub struct RegionRowView {
    pub label: u32,
    pub area: u64,
    pub centroid_y: f64,
    pub centroid_x: f64,
    pub equivalent_diameter: f64,
    pub eccentricity: f64,
    pub perimeter: f64,
    pub mean_intensity: f64,
    pub max_intensity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionSortColumn {
    Label,
    Area,
    MeanIntensity,
    Eccentricity,
}

impl RegionSortColumn {
    pub fn label(self) -> &'static str {
        match self {
            Self::Label => "Label #",
            Self::Area => "Area (px)",
            Self::MeanIntensity => "Mean Int.",
            Self::Eccentricity => "Eccentricity",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnalyticsUiState {
    pub is_open: bool,
    pub sort_column: RegionSortColumn,
    pub sort_descending: bool,
    pub min_area_filter: u64,
    pub total_regions: usize,
    pub arrow_num_columns: usize,
    pub arrow_num_rows: usize,
    pub polars_summary_line: String,
    pub rows: Vec<RegionRowView>,
    pub hist_counts: Vec<u32>,
    pub otsu_level: f64,
    pub entropy_bits: f64,
    pub sharpness_score: f64,
}

impl Default for AnalyticsUiState {
    fn default() -> Self {
        Self {
            is_open: true,
            sort_column: RegionSortColumn::Area,
            sort_descending: true,
            min_area_filter: 4,
            total_regions: 0,
            arrow_num_columns: 21,
            arrow_num_rows: 0,
            polars_summary_line: "Polars DataFrame (0 × 10) · Arrow RecordBatch (0 × 21)".to_string(),
            rows: Vec::new(),
            hist_counts: vec![0; 64],
            otsu_level: 0.5,
            entropy_bits: 0.0,
            sharpness_score: 0.0,
        }
    }
}

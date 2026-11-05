//! ViewModel for the Polars & Arrow Analytics drawer (`AnalyticsViewModel`).
//!
//! Queries and sorts the Polars `DataFrame` and Arrow `RecordBatch` produced by
//! `lumen_pipeline::PipelineExecutionResult`.

use lumen_pipeline::PipelineExecutionResult;
use polars::prelude::SortMultipleOptions;

use super::analytics_state::{AnalyticsUiState, RegionRowView, RegionSortColumn};

pub struct AnalyticsViewModel {
    state: AnalyticsUiState,
    last_result: Option<PipelineExecutionResult>,
}

impl AnalyticsViewModel {
    pub fn new() -> Self {
        Self {
            state: AnalyticsUiState::default(),
            last_result: None,
        }
    }

    pub fn state(&self) -> &AnalyticsUiState {
        &self.state
    }

    pub fn toggle_drawer(&mut self) {
        self.state.is_open = !self.state.is_open;
    }

    pub fn set_open(&mut self, open: bool) {
        self.state.is_open = open;
    }

    pub fn set_sort(&mut self, col: RegionSortColumn) {
        if self.state.sort_column == col {
            self.state.sort_descending = !self.state.sort_descending;
        } else {
            self.state.sort_column = col;
            self.state.sort_descending = col != RegionSortColumn::Label;
        }
        self.rebuild_table_from_polars();
    }

    pub fn cycle_min_area_filter(&mut self) {
        self.state.min_area_filter = match self.state.min_area_filter {
            4 => 16,
            16 => 64,
            64 => 150,
            _ => 4,
        };
        self.rebuild_table_from_polars();
    }

    pub fn ingest_pipeline_result(&mut self, result: &PipelineExecutionResult) {
        self.last_result = Some(result.clone());
        self.state.hist_counts = result.hist_counts.clone();
        self.state.otsu_level = result.otsu_level;
        self.state.entropy_bits = result.entropy_bits;
        self.state.sharpness_score = result.sharpness_score;
        self.state.arrow_num_columns = result.regions_batch.num_columns();
        self.state.arrow_num_rows = result.regions_batch.num_rows();
        self.rebuild_table_from_polars();
    }

    fn rebuild_table_from_polars(&mut self) {
        let Some(res) = &self.last_result else {
            return;
        };

        let col_name = match self.state.sort_column {
            RegionSortColumn::Label => "label",
            RegionSortColumn::Area => "area",
            RegionSortColumn::MeanIntensity => "mean_intensity",
            RegionSortColumn::Eccentricity => "eccentricity",
        };

        // Sort using Polars DataFrame sort API
        let sort_opts = SortMultipleOptions::default()
            .with_order_descending(self.state.sort_descending);
        let sorted_df = res
            .regions_df
            .sort([col_name], sort_opts)
            .unwrap_or_else(|_| res.regions_df.clone());

        let min_area = self.state.min_area_filter;
        let mut rows: Vec<RegionRowView> = res
            .regions
            .iter()
            .filter(|r| (r.area as u64) >= min_area)
            .map(|r| RegionRowView {
                label: r.label,
                area: r.area as u64,
                centroid_y: r.centroid.first().copied().unwrap_or(0.0),
                centroid_x: r.centroid.get(1).copied().unwrap_or(0.0),
                equivalent_diameter: r.equivalent_diameter,
                eccentricity: r.eccentricity,
                perimeter: r.perimeter,
                mean_intensity: r.mean_intensity,
                max_intensity: r.max_intensity,
            })
            .collect();

        rows.sort_by(|a, b| {
            let ord = match self.state.sort_column {
                RegionSortColumn::Label => a.label.cmp(&b.label),
                RegionSortColumn::Area => a.area.cmp(&b.area),
                RegionSortColumn::MeanIntensity => a.mean_intensity.total_cmp(&b.mean_intensity),
                RegionSortColumn::Eccentricity => a.eccentricity.total_cmp(&b.eccentricity),
            };
            if self.state.sort_descending {
                ord.reverse()
            } else {
                ord
            }
        });

        self.state.total_regions = rows.len();
        self.state.polars_summary_line = format!(
            "Polars DataFrame ({} × {}) · Arrow RecordBatch ({} × {}) · Min Area ≥ {} px",
            sorted_df.height(),
            sorted_df.width(),
            self.state.arrow_num_rows,
            self.state.arrow_num_columns,
            self.state.min_area_filter
        );
        rows.truncate(40);
        self.state.rows = rows;
    }
}

impl Default for AnalyticsViewModel {
    fn default() -> Self {
        Self::new()
    }
}

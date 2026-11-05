//! User intents/events for the Polars & Arrow Analytics drawer (`AnalyticsEvent`).

use super::analytics_state::RegionSortColumn;

#[derive(Debug, Clone, Copy)]
pub enum AnalyticsEvent {
    ToggleDrawer,
    SortBy(RegionSortColumn),
    CycleMinAreaFilter,
}

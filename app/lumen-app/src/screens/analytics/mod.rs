//! Analytics & Polars/Arrow Inspector Screen (`screens/analytics`).

pub mod analytics_event;
pub mod analytics_screen;
pub mod analytics_state;
pub mod analytics_vm;
pub mod components;

pub use analytics_event::AnalyticsEvent;
pub use analytics_screen::AnalyticsScreen;
pub use analytics_state::{AnalyticsUiState, RegionRowView};
pub use analytics_vm::AnalyticsViewModel;

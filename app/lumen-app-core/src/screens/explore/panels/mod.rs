//! Explore's panels, in the side docks: Layers on the left, the studio on the right (Affinity
//! style: groups stacked top to bottom, each with its tabs, all on screen at once).

pub mod annotations;
pub mod display;
pub mod histology;
pub mod history;
pub mod layers;
pub mod navigator;
pub mod notes;
pub mod results;
pub mod selection;

/// A side tab: its key (saved in the preferences), name and icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Layers,
    Display,
    Annotations,
    Results,
    Selection,
    Notes,
    Navigator,
    History,
    Histology,
}

impl Tab {
    /// The right dock's groups, top to bottom, and the tabs of each (Layers is the left dock).
    /// Classes are managed in Layers, so the Annotations tab is not in a group (still reachable
    /// by key).
    pub const RIGHT_GROUPS: [&'static [Tab]; 3] = [&[Tab::Display], &[Tab::Results, Tab::Selection, Tab::Notes, Tab::Histology], &[Tab::Navigator, Tab::History]];
    /// Heights of the right dock's groups (px; `None`: what is left).
    pub const RIGHT_SIZES: [Option<f32>; 3] = [None, Some(220.), Some(260.)];

    pub fn key(self) -> &'static str {
        match self {
            Tab::Layers => "layers",
            Tab::Display => "display",
            Tab::Annotations => "annotations",
            Tab::Results => "results",
            Tab::Selection => "selection",
            Tab::Notes => "notes",
            Tab::Navigator => "navigator",
            Tab::History => "history",
            Tab::Histology => "histology",
        }
    }

    pub fn from_key(k: &str) -> Option<Tab> {
        [Tab::Layers, Tab::Annotations].iter().chain(Tab::RIGHT_GROUPS.iter().flat_map(|g| g.iter())).copied().find(|t| t.key() == k)
    }

    pub fn title(self) -> &'static str {
        match self {
            Tab::Layers => "Layers",
            Tab::Display => "Display",
            Tab::Annotations => "Annotations",
            Tab::Results => "Results",
            Tab::Selection => "Selection",
            Tab::Notes => "Notes",
            Tab::Navigator => "Navigator",
            Tab::History => "History",
            Tab::Histology => "Histology",
        }
    }

    pub fn icon(self) -> gpui_kit::assets::IconName {
        use gpui_kit::assets::IconName;
        match self {
            Tab::Layers => IconName::Layers2,
            Tab::Display => IconName::SlidersHorizontal,
            Tab::Annotations => IconName::Shapes,
            Tab::Results => IconName::ChartColumn,
            Tab::Selection => IconName::Info,
            Tab::Notes => IconName::StickyNote,
            Tab::Navigator => IconName::Map,
            Tab::History => IconName::Clock,
            Tab::Histology => IconName::Microscope,
        }
    }
}

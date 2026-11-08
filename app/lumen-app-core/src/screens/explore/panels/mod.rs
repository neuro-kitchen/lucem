//! Explore's side panels, one per tab (VS Code style): Layers, Display, Annotations and Results
//! on the left; Selection and Notes on the right.

pub mod annotations;
pub mod display;
pub mod layers;
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
}

impl Tab {
    pub const LEFT: [Tab; 4] = [Tab::Layers, Tab::Display, Tab::Annotations, Tab::Results];
    pub const RIGHT: [Tab; 2] = [Tab::Selection, Tab::Notes];

    pub fn key(self) -> &'static str {
        match self {
            Tab::Layers => "layers",
            Tab::Display => "display",
            Tab::Annotations => "annotations",
            Tab::Results => "results",
            Tab::Selection => "selection",
            Tab::Notes => "notes",
        }
    }

    pub fn from_key(k: &str) -> Option<Tab> {
        Tab::LEFT.iter().chain(Tab::RIGHT.iter()).copied().find(|t| t.key() == k)
    }

    pub fn title(self) -> &'static str {
        match self {
            Tab::Layers => "Layers",
            Tab::Display => "Display",
            Tab::Annotations => "Annotations",
            Tab::Results => "Results",
            Tab::Selection => "Selection",
            Tab::Notes => "Notes",
        }
    }

    pub fn icon(self) -> gpui_kit::assets::IconName {
        use gpui_kit::assets::IconName;
        match self {
            Tab::Layers => IconName::Layers2,
            Tab::Display => IconName::SlidersHorizontal,
            Tab::Annotations => IconName::Shapes,
            Tab::Results => IconName::Layers,
            Tab::Selection => IconName::Info,
            Tab::Notes => IconName::StickyNote,
        }
    }
}

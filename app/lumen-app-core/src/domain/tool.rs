//! The image tools (Photoshop style): one active at a time, each with its key, icon and group,
//! and the settings the options bar shows for it.

use gpui_kit::assets::IconName;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Tool {
    /// Select and move annotations.
    #[default]
    Select,
    Hand,
    Zoom,
    Rect,
    Ellipse,
    Polygon,
    /// Closed freehand region.
    Lasso,
    Point,
    Brush,
    Eraser,
    Ruler,
    /// Pixel values under the pointer.
    Probe,
    /// A brush that snaps to similar intensity (the wand): paints the pixels around the
    /// pointer that look like the one under it.
    Wand,
    /// Rapid segmentation: a live threshold of a channel, turned into annotations.
    Threshold,
}

impl Tool {
    /// Tool bar order; `None` marks a separator.
    pub const BAR: [Option<Tool>; 18] = [
        Some(Tool::Select),
        Some(Tool::Hand),
        Some(Tool::Zoom),
        None,
        Some(Tool::Rect),
        Some(Tool::Ellipse),
        Some(Tool::Polygon),
        Some(Tool::Lasso),
        Some(Tool::Point),
        None,
        Some(Tool::Brush),
        Some(Tool::Eraser),
        Some(Tool::Wand),
        Some(Tool::Threshold),
        None,
        Some(Tool::Ruler),
        Some(Tool::Probe),
        None,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Tool::Select => "Select",
            Tool::Hand => "Hand",
            Tool::Zoom => "Zoom",
            Tool::Rect => "Rectangle",
            Tool::Ellipse => "Ellipse",
            Tool::Polygon => "Polygon",
            Tool::Lasso => "Lasso",
            Tool::Point => "Points",
            Tool::Brush => "Brush",
            Tool::Eraser => "Eraser",
            Tool::Ruler => "Ruler",
            Tool::Probe => "Pixel probe",
            Tool::Wand => "Wand",
            Tool::Threshold => "Threshold",
        }
    }

    /// Its key (Photoshop's where there is one).
    pub fn key(self) -> &'static str {
        match self {
            Tool::Select => "v",
            Tool::Hand => "h",
            Tool::Zoom => "z",
            Tool::Rect => "m",
            Tool::Ellipse => "o",
            Tool::Polygon => "p",
            Tool::Lasso => "l",
            Tool::Point => "n",
            Tool::Brush => "b",
            Tool::Eraser => "e",
            Tool::Ruler => "r",
            Tool::Probe => "i",
            Tool::Wand => "w",
            Tool::Threshold => "t",
        }
    }

    pub fn from_key(key: &str) -> Option<Tool> {
        Tool::BAR.iter().flatten().copied().find(|t| t.key() == key)
    }

    pub fn icon(self) -> IconName {
        match self {
            Tool::Select => IconName::MousePointer2,
            Tool::Hand => IconName::Hand,
            Tool::Zoom => IconName::ZoomIn,
            Tool::Rect => IconName::Square,
            Tool::Ellipse => IconName::Circle,
            Tool::Polygon => IconName::Pentagon,
            Tool::Lasso => IconName::Lasso,
            Tool::Point => IconName::MapPin,
            Tool::Brush => IconName::Brush,
            Tool::Eraser => IconName::Eraser,
            Tool::Ruler => IconName::Ruler,
            Tool::Probe => IconName::Pipette,
            Tool::Wand => IconName::WandSparkles,
            Tool::Threshold => IconName::Contrast,
        }
    }

    /// One line for its tooltip.
    pub fn hint(self) -> &'static str {
        match self {
            Tool::Select => "Click an annotation to select it, drag to move it; Shift adds to the selection",
            Tool::Hand => "Drag to move the image (or hold Space with any tool)",
            Tool::Zoom => "Click to zoom in, Alt+click to zoom out",
            Tool::Rect => "Drag a rectangle",
            Tool::Ellipse => "Drag an ellipse",
            Tool::Polygon => "Click its corners; Enter or double-click closes it",
            Tool::Lasso => "Drag around a region",
            Tool::Point => "Click to place a point",
            Tool::Brush => "Paint the class; [ and ] change the size",
            Tool::Eraser => "Erase painted regions of the class; [ and ] change the size",
            Tool::Ruler => "Drag to measure a distance",
            Tool::Probe => "Shows the pixel values under the pointer",
            Tool::Wand => "Paint: the brush fills only pixels like the one under its centre (tolerance in the options bar)",
            Tool::Threshold => "Live threshold of a channel; Create annotations segments the whole image",
        }
    }

    /// Whether it draws annotations of a class.
    pub fn draws(self) -> bool {
        matches!(self, Tool::Rect | Tool::Ellipse | Tool::Polygon | Tool::Lasso | Tool::Point | Tool::Brush | Tool::Eraser | Tool::Ruler | Tool::Wand)
    }

    /// Whether it uses the brush size.
    pub fn uses_brush(self) -> bool {
        matches!(self, Tool::Brush | Tool::Eraser | Tool::Wand)
    }
}

/// What the options bar edits.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolState {
    pub tool: Tool,
    /// Brush radius in screen px (constant on screen, Photoshop style) …
    pub brush_px: f32,
    /// … or in image px (constant on the tissue).
    pub brush_in_image: bool,
    /// Brush radius in level-0 px, when `brush_in_image`.
    pub brush_image_px: f64,
    /// The class new annotations get.
    pub class: String,
    /// Rulers on the canvas edges.
    pub rulers: bool,
    /// Results of the pipeline drawn over the image.
    pub show_results: bool,
    /// Wand: how far from the seed a pixel may be (fraction of each channel's display window).
    pub wand_tolerance: f32,
    /// Threshold tool: channel, value (raw units), above or below, smoothing (level-0 px) and the
    /// smallest object kept (level-0 px²).
    pub threshold_channel: usize,
    pub threshold_value: f32,
    pub threshold_above: bool,
    pub threshold_sigma: f64,
    pub threshold_min_area: f64,
}

impl Default for ToolState {
    fn default() -> Self {
        Self { tool: Tool::Select, brush_px: 24.0, brush_in_image: false, brush_image_px: 40.0, class: "Region".into(), rulers: true, show_results: true, wand_tolerance: 0.15, threshold_channel: 0, threshold_value: 0.0, threshold_above: true, threshold_sigma: 2.0, threshold_min_area: 50.0 }
    }
}

/// Brush sizes `[` / `]` step through (screen px).
pub const BRUSH_STEPS: [f32; 14] = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0];

/// The next brush size up (`dir` > 0) or down.
pub fn step_brush(px: f32, dir: f32) -> f32 {
    if dir > 0.0 {
        BRUSH_STEPS.iter().copied().find(|&s| s > px + 0.01).unwrap_or(BRUSH_STEPS[BRUSH_STEPS.len() - 1])
    } else {
        BRUSH_STEPS.iter().rev().copied().find(|&s| s < px - 0.01).unwrap_or(BRUSH_STEPS[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique_and_found() {
        let tools: Vec<Tool> = Tool::BAR.iter().flatten().copied().collect();
        for t in &tools {
            assert_eq!(Tool::from_key(t.key()), Some(*t));
        }
        assert_eq!(step_brush(24.0, 1.0), 32.0);
        assert_eq!(step_brush(24.0, -1.0), 16.0);
        assert_eq!(step_brush(256.0, 1.0), 256.0);
    }
}

//! Interactive viewport tools (Pan/Explore, Zoom, Pixel Inspector, Line Profile, Vector Contour).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActiveTool {
    /// Hand / Pan & Scroll-Zoom exploration tool.
    ExplorePan,
    /// Click or drag to zoom in/out.
    ZoomLens,
    /// Inspect exact `(y, x)` coordinates, intensity, and local neighborhood statistics.
    PixelProbe,
    /// Toggle/inspect VectorCraft Marching Squares iso-contours.
    VectorContour,
}

impl ActiveTool {
    pub const ALL: &'static [Self] = &[
        Self::ExplorePan,
        Self::ZoomLens,
        Self::PixelProbe,
        Self::VectorContour,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ExplorePan => "Pan & Explore (H)",
            Self::ZoomLens => "Zoom Lens (Z)",
            Self::PixelProbe => "Pixel Probe (I)",
            Self::VectorContour => "Contour Trace (V)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ExplorePan => "Pan",
            Self::ZoomLens => "Zoom",
            Self::PixelProbe => "Probe",
            Self::VectorContour => "Vector",
        }
    }
}

//! Interactive viewport tools (Navigation, Painting/Masking, Vector Paths, and Measurement).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActiveTool {
    /// Hand / Pan & Scroll-Zoom exploration tool.
    ExplorePan,
    /// Click or drag to zoom in/out.
    ZoomLens,
    /// Freehand raster brush for painting masks / annotations.
    BrushPaint,
    /// Raster eraser for clearing painted strokes.
    Eraser,
    /// Cubic Bezier vector path pen tool.
    VectorPen,
    /// Polygon / freehand ROI lasso selection tool.
    PolygonRoi,
    /// Toggle/inspect Marching Squares iso-contours.
    VectorContour,
    /// Distance and angle measurement caliper.
    MeasureCaliper,
    /// Inspect exact `(y, x)` coordinates, intensity, and local neighborhood statistics.
    PixelProbe,
}

impl ActiveTool {
    pub const ALL: &'static [Self] = &[
        Self::ExplorePan,
        Self::ZoomLens,
        Self::BrushPaint,
        Self::Eraser,
        Self::VectorPen,
        Self::PolygonRoi,
        Self::VectorContour,
        Self::MeasureCaliper,
        Self::PixelProbe,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ExplorePan => "Pan & Explore",
            Self::ZoomLens => "Zoom Lens",
            Self::BrushPaint => "Brush Paint",
            Self::Eraser => "Eraser",
            Self::VectorPen => "Vector Pen (Bezier)",
            Self::PolygonRoi => "Polygon ROI Lasso",
            Self::VectorContour => "Iso-Contour Trace",
            Self::MeasureCaliper => "Measure Caliper",
            Self::PixelProbe => "Pixel Eyedropper",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ExplorePan => "Pan",
            Self::ZoomLens => "Zoom",
            Self::BrushPaint => "Brush",
            Self::Eraser => "Erase",
            Self::VectorPen => "Pen",
            Self::PolygonRoi => "ROI",
            Self::VectorContour => "Contour",
            Self::MeasureCaliper => "Measure",
            Self::PixelProbe => "Probe",
        }
    }

    pub fn category(self) -> &'static str {
        match self {
            Self::ExplorePan | Self::ZoomLens => "Navigation",
            Self::BrushPaint | Self::Eraser => "Paint & Mask",
            Self::VectorPen | Self::PolygonRoi | Self::VectorContour => "Vector & ROI",
            Self::MeasureCaliper | Self::PixelProbe => "Measure & Probe",
        }
    }
}

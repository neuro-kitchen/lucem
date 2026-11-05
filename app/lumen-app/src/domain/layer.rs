//! ArtCraft raster layers and VectorCraft contour/ROI overlay layers.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    RasterImage,
    VectorContours,
    RegionOverlay,
}

impl LayerKind {
    pub fn badge(self) -> &'static str {
        match self {
            Self::RasterImage => "RASTER",
            Self::VectorContours => "VECTOR",
            Self::RegionOverlay => "LABELS",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StudioLayer {
    pub id: u64,
    pub name: String,
    pub kind: LayerKind,
    pub visible: bool,
    pub opacity: f32,
}

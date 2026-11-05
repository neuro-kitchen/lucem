//! Project templates, presets, and blank-canvas specifications for the VectorCraft-style Startup Hub.

use lumen_io::PhantomKind;
use lumen_pipeline::ModifierKind;
use lumen_view::ColormapLut;

/// Specifications for creating a new blank canvas from the Startup Project Hub.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlankCanvasSpec {
    pub width: usize,
    pub height: usize,
    pub background_intensity: f32,
}

impl Default for BlankCanvasSpec {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            background_intensity: 0.08,
        }
    }
}

/// A curated project preset combining ArtCraft raster layers, VectorCraft contours, and Blender
/// non-destructive GPU modifiers.
#[derive(Debug, Clone)]
pub struct ProjectPreset {
    pub id: &'static str,
    pub title: &'static str,
    pub category: &'static str,
    pub description: &'static str,
    pub phantom: Option<PhantomKind>,
    pub colormap: ColormapLut,
    pub show_contours: bool,
    pub initial_modifiers: Vec<ModifierKind>,
}

impl ProjectPreset {
    pub fn curated_list() -> Vec<Self> {
        vec![
            Self {
                id: "bio-nuclei-segment",
                title: "Bio-Imaging: Nuclei & Contours",
                category: "VECTOR × SEGMENT",
                description: "Confocal brain-section phantom with Gaussian smoothing, Marching Squares vector contours, and Polars/Arrow RegionProps table.",
                phantom: Some(PhantomKind::FluorescenceNuclei),
                colormap: ColormapLut::CyanGfp,
                show_contours: true,
                initial_modifiers: vec![ModifierKind::GaussianBlur],
            },
            Self {
                id: "optics-restore-stack",
                title: "Optical Deconvolution & Denoise",
                category: "BLENDER PIPELINE",
                description: "Siemens resolution star with Total-Variation Chambolle denoising and local CLAHE contrast enhancement.",
                phantom: Some(PhantomKind::SiemensStarTarget),
                colormap: ColormapLut::Magma,
                show_contours: false,
                initial_modifiers: vec![ModifierKind::TvChambolleDenoise, ModifierKind::ClaheContrast],
            },
            Self {
                id: "vectorcraft-trace",
                title: "VectorCraft Iso-Contour Trace",
                category: "VECTORCRAFT",
                description: "Multi-scale geometric SDF primitives with live Marching Squares subpixel polyline extraction.",
                phantom: Some(PhantomKind::VectorContourShapes),
                colormap: ColormapLut::Viridis,
                show_contours: true,
                initial_modifiers: vec![],
            },
            Self {
                id: "holographic-fringes",
                title: "Rolling-Ball & Edge Lab",
                category: "ARTCRAFT × FILTER",
                description: "Modulated interference fringes with rolling-ball background subtraction and Sobel gradient magnitude.",
                phantom: Some(PhantomKind::InterferenceFringes),
                colormap: ColormapLut::Inferno,
                show_contours: false,
                initial_modifiers: vec![ModifierKind::RollingBallBackground],
            },
        ]
    }
}

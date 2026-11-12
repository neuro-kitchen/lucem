//! Blender-style non-destructive GPU modifier definitions (`ModifierKind`, `ModifierNode`).

use serde::{Deserialize, Serialize};

/// Supported non-destructive GPU operators in the Studio modifier stack and Node Editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModifierKind {
    GaussianBlur,
    SobelEdges,
    ClaheContrast,
    GammaCurve,
    TvChambolleDenoise,
    RollingBallBackground,
    CannyEdges,
    OtsuThreshold,
    MorphGradient,
}

impl ModifierKind {
    pub const ALL: &'static [Self] = &[
        Self::GaussianBlur,
        Self::SobelEdges,
        Self::ClaheContrast,
        Self::GammaCurve,
        Self::TvChambolleDenoise,
        Self::RollingBallBackground,
        Self::CannyEdges,
        Self::OtsuThreshold,
        Self::MorphGradient,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::GaussianBlur => "Gaussian Blur",
            Self::SobelEdges => "Sobel Gradient Magnitude",
            Self::ClaheContrast => "CLAHE Local Contrast",
            Self::GammaCurve => "Gamma & Gain Curve",
            Self::TvChambolleDenoise => "TV Chambolle Denoise",
            Self::RollingBallBackground => "Rolling-Ball Background",
            Self::CannyEdges => "Canny Edge Detector",
            Self::OtsuThreshold => "Otsu Auto-Threshold",
            Self::MorphGradient => "Morphological Gradient",
        }
    }

    pub fn category(self) -> &'static str {
        match self {
            Self::GaussianBlur | Self::SobelEdges => "FILTER",
            Self::ClaheContrast | Self::GammaCurve => "INTENSITY",
            Self::TvChambolleDenoise | Self::RollingBallBackground => "RESTORE",
            Self::CannyEdges => "FEATURE",
            Self::OtsuThreshold => "THRESHOLD",
            Self::MorphGradient => "MORPHOLOGY",
        }
    }

    pub fn primary_label(self) -> &'static str {
        match self {
            Self::GaussianBlur => "Sigma (px)",
            Self::SobelEdges => "Scale",
            Self::ClaheContrast => "Clip Limit",
            Self::GammaCurve => "Gamma",
            Self::TvChambolleDenoise => "Weight",
            Self::RollingBallBackground => "Radius (px)",
            Self::CannyEdges => "Sigma (px)",
            Self::OtsuThreshold => "Bins",
            Self::MorphGradient => "Radius (px)",
        }
    }

    pub fn default_params(self) -> (f64, f64) {
        match self {
            Self::GaussianBlur => (1.8, 4.0),
            Self::SobelEdges => (1.0, 0.0),
            Self::ClaheContrast => (0.03, 8.0),
            Self::GammaCurve => (0.85, 1.0),
            Self::TvChambolleDenoise => (0.12, 25.0),
            Self::RollingBallBackground => (12.0, 1.0),
            Self::CannyEdges => (1.4, 0.15),
            Self::OtsuThreshold => (256.0, 0.0),
            Self::MorphGradient => (1.0, 0.0),
        }
    }

    pub fn step(self) -> f64 {
        match self {
            Self::GaussianBlur => 0.25,
            Self::SobelEdges => 0.2,
            Self::ClaheContrast => 0.01,
            Self::GammaCurve => 0.1,
            Self::TvChambolleDenoise => 0.02,
            Self::RollingBallBackground => 2.0,
            Self::CannyEdges => 0.2,
            Self::OtsuThreshold => 32.0,
            Self::MorphGradient => 1.0,
        }
    }

    pub fn bounds(self) -> (f64, f64) {
        match self {
            Self::GaussianBlur => (0.2, 16.0),
            Self::SobelEdges => (0.1, 5.0),
            Self::ClaheContrast => (0.005, 0.25),
            Self::GammaCurve => (0.2, 3.0),
            Self::TvChambolleDenoise => (0.02, 0.8),
            Self::RollingBallBackground => (2.0, 40.0),
            Self::CannyEdges => (0.5, 6.0),
            Self::OtsuThreshold => (32.0, 512.0),
            Self::MorphGradient => (1.0, 8.0),
        }
    }
}

/// An instance of a non-destructive GPU modifier in the Studio stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModifierNode {
    pub id: u64,
    pub kind: ModifierKind,
    pub enabled: bool,
    pub primary_param: f64,
    pub secondary_param: f64,
}

impl ModifierNode {
    pub fn new(id: u64, kind: ModifierKind) -> Self {
        let (p1, p2) = kind.default_params();
        Self {
            id,
            kind,
            enabled: true,
            primary_param: p1,
            secondary_param: p2,
        }
    }
}
